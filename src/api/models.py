from pydantic import BaseModel, Field
from typing import Optional


# ── Request models ────────────────────────────────────────────────────────────

class CareerQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Career question to answer",
        example="What skills do I need for a Senior ML Engineer role?",
    )
    use_hyde: Optional[bool] = Field(
        None,
        description="Override HyDE setting. Default: from config.",
    )
    use_bm25: Optional[bool] = Field(
        None,
        description="Override BM25 setting. Default: from config.",
    )
    top_k_rerank: Optional[int] = Field(
        None,
        ge=1, le=20,
        description="Number of chunks to use for generation.",
    )


class ResumeGapRequest(BaseModel):
    resume: str = Field(
        ...,
        min_length=50,
        max_length=5000,
        description="Resume text to analyse",
    )
    target_role: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Target job role",
        example="Senior Data Engineer",
    )


# ── Response models ───────────────────────────────────────────────────────────

class CitationResponse(BaseModel):
    index:            int
    doc_id:           str
    chunk_id:         str
    title:            str
    company:          str
    location:         str
    confidence_score: float
    text_preview:     str


class CareerQueryResponse(BaseModel):
    query:            str
    answer:           str
    citations:        list[CitationResponse]
    chunks_retrieved: int
    chunks_used:      int
    low_confidence:   bool
    retrieval_method: str


class ResumeGapResponse(BaseModel):
    target_role:    str
    analysis:       str
    citations:      list[CitationResponse]
    n_listings:     int
    low_confidence: bool


class HealthResponse(BaseModel):
    status:     str
    collection: str
    qdrant_mode: str