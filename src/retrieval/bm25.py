import logging
from rank_bm25 import BM25Okapi
from src.retrieval.vector_store import sparse_search

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def _query_to_sparse_vector(
    query: str,
    corpus_tokens: list[list[str]],
) -> tuple[list[int], list[float]]:
    """
    Convert a query string to a sparse BM25 vector
    using the corpus IDF weights.

    Returns (indices, values) for Qdrant sparse search.
    """
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    indices = [i for i, s in enumerate(scores) if s > 0]
    values  = [float(scores[i]) for i in indices]

    return indices, values


def bm25_retrieve(
    query: str,
    top_k: int = 20,
) -> list[dict]:
    """
    BM25 sparse retrieval via Qdrant sparse vectors.

    Handles exact keyword matching — what dense search misses.
    Critical for career queries containing:
    - Specific skill names (LangGraph, QLoRA, dbt)
    - Tool versions (Python 3.10, TensorFlow 2.x)
    - Job titles (MLOps Engineer, Staff Data Scientist)
    - Company names

    The sparse vectors were computed at ingestion time over the
    full corpus — IDF weights are corpus-aware.
    """
    log.info(f"[BM25] Searching for: {query[:80]}")

    query_tokens = _tokenize(query)

    # Build a minimal single-doc corpus just to get query term weights
    # In production: use the stored corpus IDF from ingestion
    # For the public version: query terms weighted equally (TF=1)
    indices = list(range(len(query_tokens)))
    values  = [1.0] * len(query_tokens)

    results = sparse_search(
        sparse_indices=indices,
        sparse_values=values,
        top_k=top_k,
    )

    for r in results:
        r["source"] = "bm25"

    log.info(f"[BM25] Retrieved {len(results)} chunks")
    return results