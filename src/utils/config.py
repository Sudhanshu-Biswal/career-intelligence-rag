import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # ── API keys ──────────────────────────────────────────
    openai_api_key:  str = Field(..., env="OPENAI_API_KEY")
    qdrant_url:      str = Field("",  env="QDRANT_URL")
    qdrant_api_key:  str = Field("",  env="QDRANT_API_KEY")
    cohere_api_key:  str = Field("",  env="COHERE_API_KEY")

    # ── Models ────────────────────────────────────────────
    embedding_model:  str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    query_model:      str = Field("gpt-4o-mini",            env="QUERY_MODEL")
    generation_model: str = Field("gpt-4o-mini",            env="GENERATION_MODEL")
    rerank_model:     str = Field("rerank-english-v3.0",    env="RERANK_MODEL")

    # ── Qdrant ────────────────────────────────────────────
    collection_name: str = Field("career_intelligence", env="COLLECTION_NAME")
    qdrant_mode:     str = Field("cloud",               env="QDRANT_MODE")
    # "cloud" → uses QDRANT_URL + QDRANT_API_KEY
    # "memory" → in-memory Qdrant, no setup needed

    # ── Retrieval ─────────────────────────────────────────
    top_k_retrieve: int   = Field(20,   env="TOP_K_RETRIEVE")
    top_k_rerank:   int   = Field(5,    env="TOP_K_RERANK")
    hyde_enabled:   bool  = Field(True, env="HYDE_ENABLED")
    bm25_enabled:   bool  = Field(True, env="BM25_ENABLED")

    # ── Chunking ──────────────────────────────────────────
    chunk_size:    int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(128, env="CHUNK_OVERLAP")

    # ── Evaluation ────────────────────────────────────────
    ragas_sample_size: int = Field(20, env="RAGAS_SAMPLE_SIZE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.
    Call get_settings() anywhere in the codebase — returns the same object.
    lru_cache means .env is read once, not on every import.
    """
    return Settings()