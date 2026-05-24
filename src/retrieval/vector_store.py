import logging
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest, NamedVector, NamedSparseVector, SparseVector
from openai import OpenAI
from src.utils.config import get_settings
from src.ingestion.embedder import get_qdrant_client

log = logging.getLogger(__name__)
settings = get_settings()

_qdrant: QdrantClient = None
_openai: OpenAI = None


def get_clients() -> tuple[QdrantClient, OpenAI]:
    """Lazy singleton clients — initialised once, reused across requests."""
    global _qdrant, _openai
    if _qdrant is None:
        _qdrant = get_qdrant_client()
    if _openai is None:
        _openai = OpenAI(api_key=settings.openai_api_key)
    return _qdrant, _openai


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    _, openai = get_clients()
    response = openai.embeddings.create(
        model=settings.embedding_model,
        input=[query],
    )
    return response.data[0].embedding


def dense_search(
    query_vector: list[float],
    top_k: int = None,
    filters: dict = None,
) -> list[dict]:
    """
    ANN search using HNSW dense vectors.
    Returns top_k chunks with scores and payloads.
    """
    top_k = top_k or settings.top_k_retrieve
    qdrant, _ = get_clients()

    results = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=NamedVector(name="dense", vector=query_vector),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "chunk_id":  r.payload.get("chunk_id"),
            "doc_id":    r.payload.get("doc_id"),
            "text":      r.payload.get("text"),
            "title":     r.payload.get("title"),
            "company":   r.payload.get("company"),
            "location":  r.payload.get("location"),
            "score":     r.score,
            "source":    "dense",
            "payload":   r.payload,
        }
        for r in results
    ]


def sparse_search(
    sparse_indices: list[int],
    sparse_values: list[float],
    top_k: int = None,
) -> list[dict]:
    """
    BM25 sparse vector search.
    Returns top_k chunks with scores and payloads.
    """
    top_k = top_k or settings.top_k_retrieve
    qdrant, _ = get_clients()

    results = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            ),
        ),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "chunk_id":  r.payload.get("chunk_id"),
            "doc_id":    r.payload.get("doc_id"),
            "text":      r.payload.get("text"),
            "title":     r.payload.get("title"),
            "company":   r.payload.get("company"),
            "location":  r.payload.get("location"),
            "score":     r.score,
            "source":    "sparse",
            "payload":   r.payload,
        }
        for r in results
    ]


def get_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    """
    Fetch specific chunks by chunk_id.
    Used for context expansion — fetch sibling chunks.
    """
    qdrant, _ = get_clients()
    point_ids = [abs(hash(cid)) % (2**63) for cid in chunk_ids]

    results = qdrant.retrieve(
        collection_name=settings.collection_name,
        ids=point_ids,
        with_payload=True,
    )

    return [
        {
            "chunk_id": r.payload.get("chunk_id"),
            "doc_id":   r.payload.get("doc_id"),
            "text":     r.payload.get("text"),
            "payload":  r.payload,
        }
        for r in results
    ]