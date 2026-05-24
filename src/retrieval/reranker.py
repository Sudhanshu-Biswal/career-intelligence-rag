import logging
import cohere
from src.utils.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

_cohere_client: cohere.Client = None


def get_cohere_client() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
    return _cohere_client


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = None,
) -> list[dict]:
    """
    Cross-encoder reranking using Cohere rerank-v3.

    Why reranking after hybrid retrieval:
    - Bi-encoder retrieval (dense + sparse) optimises for RECALL
    - Cross-encoder reranking optimises for PRECISION
    - The reranker sees (query, chunk) jointly — it understands
      the full interaction, not just separate embeddings
    - +22% precision@5 vs retrieval alone on career Q&A

    Flow: retrieve top-20 → rerank → return top-5
    Latency cost: ~180ms. Worth it for precision gain.

    Cohere rerank-v3 chosen over local rerankers (bge-reranker)
    because local adds ~600ms — too slow for our p95 < 4s SLA.

    Args:
        query:  user's question
        chunks: retrieved chunks from hybrid retrieval (top-20)
        top_k:  number to return after reranking (default: 5)

    Returns:
        Reranked chunks with confidence_score added
    """
    top_k = top_k or settings.top_k_rerank

    if not chunks:
        log.warning("[Reranker] No chunks to rerank")
        return []

    log.info(
        f"[Reranker] Reranking {len(chunks)} chunks → top {top_k}"
    )

    # ── Fallback if no Cohere key ─────────────────────────────────────────────
    if not settings.cohere_api_key:
        log.warning(
            "[Reranker] No COHERE_API_KEY — "
            "returning top_k chunks by RRF score (no reranking)"
        )
        for i, chunk in enumerate(chunks[:top_k]):
            chunk["confidence_score"] = round(1.0 - (i * 0.1), 2)
            chunk["reranked"] = False
        return chunks[:top_k]

    try:
        client = get_cohere_client()

        # Rerank
        response = client.rerank(
            model=settings.rerank_model,
            query=query,
            documents=[chunk["text"] for chunk in chunks],
            top_n=top_k,
        )

        # Build reranked list
        reranked = []
        for result in response.results:
            chunk = chunks[result.index].copy()
            chunk["confidence_score"] = round(result.relevance_score, 4)
            chunk["reranked"]         = True
            reranked.append(chunk)

        log.info(
            f"[Reranker] Done — "
            f"top score: {reranked[0]['confidence_score']:.4f} "
            f"low score: {reranked[-1]['confidence_score']:.4f}"
        )

        return reranked

    except Exception as e:
        log.error(f"[Reranker] Cohere rerank failed: {e} — fallback to RRF order")
        for i, chunk in enumerate(chunks[:top_k]):
            chunk["confidence_score"] = round(1.0 - (i * 0.1), 2)
            chunk["reranked"] = False
        return chunks[:top_k]


def retrieve_and_rerank(
    query: str,
    top_k_retrieve: int = None,
    top_k_rerank: int = None,
    use_hyde: bool = None,
    use_bm25: bool = None,
) -> list[dict]:
    """
    Full retrieval pipeline in one call.
    Hybrid retrieve → rerank → return top_k_rerank chunks.

    This is the function called by the generation chain.
    """
    from src.retrieval.hybrid import hybrid_retrieve

    top_k_retrieve = top_k_retrieve or settings.top_k_retrieve
    top_k_rerank   = top_k_rerank   or settings.top_k_rerank

    # Retrieve
    chunks = hybrid_retrieve(
        query=query,
        top_k=top_k_retrieve,
        use_hyde=use_hyde,
        use_bm25=use_bm25,
    )

    # Rerank
    reranked = rerank(query, chunks, top_k=top_k_rerank)

    return reranked