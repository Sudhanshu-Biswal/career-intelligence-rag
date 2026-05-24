import logging
import time
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, SparseVector,
    SparseVectorParams, SparseIndexParams,
    OptimizersConfigDiff,
)
from rank_bm25 import BM25Okapi
from src.ingestion.chunker import Chunk
from src.utils.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


def get_qdrant_client() -> QdrantClient:
    """
    Returns Qdrant client based on QDRANT_MODE setting.

    cloud  → Qdrant Cloud (production)
    memory → In-memory (local testing, no setup needed)

    This is the clean way to handle the Docker restriction —
    in-memory works anywhere, cloud works in production.
    """
    if settings.qdrant_mode == "memory":
        log.info("[Embedder] Using in-memory Qdrant (local testing mode)")
        return QdrantClient(":memory:")

    log.info(f"[Embedder] Connecting to Qdrant Cloud: {settings.qdrant_url}")
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def _get_embedding_batch(
    texts: list[str],
    client: OpenAI,
    model: str = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Batch embed texts using OpenAI embeddings API.
    Splits into batches to respect rate limits.
    Retries on failure with exponential backoff.
    """
    model = model or settings.embedding_model
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                )
                embeddings = [r.embedding for r in response.data]
                all_embeddings.extend(embeddings)
                log.debug(
                    f"[Embedder] Embedded batch "
                    f"{i//batch_size + 1} ({len(batch)} texts)"
                )
                break
            except Exception as e:
                wait = 2 ** attempt
                log.warning(
                    f"[Embedder] Embedding batch {i} failed: {e} "
                    f"— retry in {wait}s"
                )
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"[Embedder] Embedding batch {i} failed after 3 attempts"
            )

    return all_embeddings


def _compute_bm25_vectors(
    chunks: list[Chunk],
) -> list[dict]:
    """
    Compute BM25 sparse vectors for all chunks.

    Returns list of {indices: [...], values: [...]} dicts
    suitable for Qdrant sparse vector format.

    BM25 is computed over the full corpus — IDF needs all documents
    to be meaningful. This is why we pass all chunks at once.
    """
    tokenized = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    sparse_vectors = []
    for i, tokens in enumerate(tokenized):
        scores = bm25.get_scores(tokens)
        # Keep only non-zero scores — sparse format
        indices = [j for j, s in enumerate(scores) if s > 0]
        values  = [float(scores[j]) for j in indices]
        sparse_vectors.append({"indices": indices, "values": values})

    return sparse_vectors


def setup_collection(client: QdrantClient, vector_size: int = 1536):
    """
    Create Qdrant collection with both dense and sparse vector support.

    Dense vectors: text-embedding-3-small (1536 dim)
    Sparse vectors: BM25 term weights

    This dual-vector setup enables native hybrid search in Qdrant —
    no separate BM25 infrastructure needed.
    """
    collections = [c.name for c in client.get_collections().collections]

    if settings.collection_name in collections:
        log.info(
            f"[Embedder] Collection '{settings.collection_name}' "
            f"already exists — skipping creation"
        )
        return

    log.info(f"[Embedder] Creating collection '{settings.collection_name}'")

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            "dense": VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )
    log.info(f"[Embedder] Collection created successfully")


def upsert_chunks(
    chunks: list[Chunk],
    batch_size: int = 100,
) -> int:
    """
    Main ingestion function.

    1. Setup Qdrant collection (idempotent)
    2. Generate dense embeddings (OpenAI, batched)
    3. Compute sparse BM25 vectors
    4. Upsert to Qdrant with full payload

    Idempotent — re-running on the same chunks updates existing points.
    chunk_id is used as the point ID (hashed to int for Qdrant).

    Returns: number of chunks upserted
    """
    if not chunks:
        log.warning("[Embedder] No chunks to upsert")
        return 0

    openai_client = OpenAI(api_key=settings.openai_api_key)
    qdrant_client = get_qdrant_client()

    # ── Setup collection ──────────────────────────────────────────────────────
    # Detect embedding size from a test call
    test_emb = _get_embedding_batch(
        ["test"], openai_client, batch_size=1
    )
    vector_size = len(test_emb[0])
    setup_collection(qdrant_client, vector_size)

    # ── Generate embeddings ───────────────────────────────────────────────────
    log.info(f"[Embedder] Embedding {len(chunks)} chunks...")
    texts = [chunk.text for chunk in chunks]
    dense_vectors = _get_embedding_batch(
        texts, openai_client, batch_size=batch_size
    )

    # ── Compute BM25 sparse vectors ───────────────────────────────────────────
    log.info("[Embedder] Computing BM25 sparse vectors...")
    sparse_vectors = _compute_bm25_vectors(chunks)

    # ── Upsert to Qdrant ──────────────────────────────────────────────────────
    log.info(f"[Embedder] Upserting {len(chunks)} points to Qdrant...")
    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks  = chunks[i: i + batch_size]
        batch_dense   = dense_vectors[i: i + batch_size]
        batch_sparse  = sparse_vectors[i: i + batch_size]

        points = []
        for chunk, dense, sparse in zip(
            batch_chunks, batch_dense, batch_sparse
        ):
            # Qdrant needs integer IDs — hash the chunk_id string
            point_id = abs(hash(chunk.chunk_id)) % (2**63)

            point = PointStruct(
                id=point_id,
                vector={
                    "dense":  dense,
                    "sparse": SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    ),
                },
                payload={
                    "chunk_id":     chunk.chunk_id,
                    "doc_id":       chunk.doc_id,
                    "text":         chunk.text,
                    "chunk_index":  chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "title":        chunk.title,
                    "company":      chunk.company,
                    "location":     chunk.location,
                    "company_type": chunk.company_type,
                    "source":       chunk.source,
                    "metadata":     chunk.metadata,
                },
            )
            points.append(point)

        qdrant_client.upsert(
            collection_name=settings.collection_name,
            points=points,
        )
        total_upserted += len(points)
        log.info(
            f"[Embedder] Upserted batch "
            f"{i//batch_size + 1} "
            f"({total_upserted}/{len(chunks)})"
        )

    log.info(f"[Embedder] Complete — {total_upserted} chunks in Qdrant")
    return total_upserted


def run_ingestion_pipeline(
    dataset_name: str = "jacob-hugging-face/job-descriptions",
    max_documents: int = 5000,
) -> dict:
    """
    Full ingestion pipeline — load → chunk → embed → upsert.
    Single function to run everything end to end.

    Returns summary stats.
    """
    from src.ingestion.loader import load_hf_dataset
    from src.ingestion.chunker import chunk_documents

    log.info("=" * 50)
    log.info("  INGESTION PIPELINE START")
    log.info("=" * 50)

    # Load
    docs = load_hf_dataset(
        dataset_name=dataset_name,
        max_documents=max_documents,
    )

    # Chunk
    chunks = chunk_documents(docs)

    # Embed + upsert
    upserted = upsert_chunks(chunks)

    summary = {
        "documents":       len(docs),
        "chunks":          len(chunks),
        "upserted":        upserted,
        "avg_chunks_per_doc": round(len(chunks) / max(1, len(docs)), 1),
    }

    log.info("=" * 50)
    log.info(f"  INGESTION COMPLETE: {summary}")
    log.info("=" * 50)

    return summary