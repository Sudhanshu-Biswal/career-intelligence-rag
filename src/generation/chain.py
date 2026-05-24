import logging
from src.generation.prompts import (
    CAREER_QA_SYSTEM,
    CAREER_QA_PROMPT,
    RESUME_GAP_SYSTEM,
    RESUME_GAP_PROMPT,
    format_context,
    build_citations,
)
from src.retrieval.reranker import retrieve_and_rerank
from src.utils.llm import call_mini
from src.utils.config import get_settings
from openai import OpenAI

log = logging.getLogger(__name__)
settings = get_settings()

_openai: OpenAI = None


def get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=settings.openai_api_key)
    return _openai


def _generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 500,
) -> str:
    """
    Generate a response using the system + user prompt pattern.
    Uses chat completions directly for system prompt support.
    """
    client = get_openai()
    response = client.chat.completions.create(
        model=settings.generation_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def answer_career_question(
    query: str,
    top_k_retrieve: int = None,
    top_k_rerank: int = None,
    use_hyde: bool = None,
    use_bm25: bool = None,
) -> dict:
    """
    Main RAG chain — career Q&A.

    Flow:
    1. Retrieve top-20 chunks (HyDE + BM25 + RRF)
    2. Rerank to top-5 (Cohere rerank-v3)
    3. Format context with source metadata
    4. Generate grounded, cited answer
    5. Return structured response with citations

    This is the function called by the FastAPI endpoint.

    Returns:
        {
            query:              str,
            answer:             str,
            citations:          list[dict],
            chunks_retrieved:   int,
            chunks_used:        int,
            low_confidence:     bool,
            retrieval_method:   str,
        }
    """
    log.info(f"[Chain] Career Q&A: {query[:80]!r}")

    # ── Retrieve + rerank ─────────────────────────────────────────────────────
    chunks = retrieve_and_rerank(
        query=query,
        top_k_retrieve=top_k_retrieve or settings.top_k_retrieve,
        top_k_rerank=top_k_rerank or settings.top_k_rerank,
        use_hyde=use_hyde,
        use_bm25=use_bm25,
    )

    if not chunks:
        log.warning("[Chain] No chunks retrieved")
        return {
            "query":            query,
            "answer":           (
                "I couldn't find relevant job listings to answer this question. "
                "Try searching for a specific role or skill on our platform."
            ),
            "citations":        [],
            "chunks_retrieved": 0,
            "chunks_used":      0,
            "low_confidence":   True,
            "retrieval_method": "none",
        }

    # ── Check confidence ──────────────────────────────────────────────────────
    avg_confidence = sum(
        c.get("confidence_score", 0) for c in chunks
    ) / len(chunks)
    low_confidence = avg_confidence < 0.4

    if low_confidence:
        log.warning(
            f"[Chain] Low confidence retrieval: "
            f"avg_score={avg_confidence:.3f}"
        )

    # ── Format context ────────────────────────────────────────────────────────
    context = format_context(chunks)

    # ── Generate answer ───────────────────────────────────────────────────────
    user_prompt = CAREER_QA_PROMPT.format(
        query=query,
        context=context,
    )

    answer = _generate(
        system_prompt=CAREER_QA_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=400,
    )

    # ── Build citations ───────────────────────────────────────────────────────
    citations = build_citations(chunks)

    # ── Detect low confidence in answer ──────────────────────────────────────
    # If the LLM said it doesn't have enough info — flag it
    uncertainty_phrases = [
        "don't have enough",
        "not enough",
        "cannot answer",
        "no information",
        "insufficient",
    ]
    answer_uncertain = any(
        p in answer.lower() for p in uncertainty_phrases
    )

    retrieval_method = []
    if use_hyde or settings.hyde_enabled:
        retrieval_method.append("hyde")
    if use_bm25 or settings.bm25_enabled:
        retrieval_method.append("bm25")
    retrieval_method.append("dense")

    log.info(
        f"[Chain] Done — "
        f"chunks={len(chunks)} "
        f"avg_conf={avg_confidence:.3f} "
        f"uncertain={answer_uncertain}"
    )

    return {
        "query":            query,
        "answer":           answer.strip(),
        "citations":        citations,
        "chunks_retrieved": settings.top_k_retrieve,
        "chunks_used":      len(chunks),
        "low_confidence":   low_confidence or answer_uncertain,
        "retrieval_method": "+".join(retrieval_method),
    }


def analyse_resume_gap(
    resume: str,
    target_role: str,
    top_k_retrieve: int = None,
    top_k_rerank: int = None,
) -> dict:
    """
    Resume gap analysis against real JD requirements.

    Retrieves JDs for the target role, then analyses
    what the resume is missing vs what listings require.

    Returns structured gap analysis with citations.
    """
    log.info(f"[Chain] Resume gap: role={target_role!r}")

    # Build a retrieval query from the target role
    query = f"requirements and skills for {target_role} position"

    # Retrieve
    chunks = retrieve_and_rerank(
        query=query,
        top_k_retrieve=top_k_retrieve or settings.top_k_retrieve,
        top_k_rerank=top_k_rerank or settings.top_k_rerank,
    )

    if not chunks:
        return {
            "target_role":  target_role,
            "analysis":     (
                "I couldn't find enough current listings for this role. "
                "Try a more common job title."
            ),
            "citations":    [],
            "n_listings":   0,
            "low_confidence": True,
        }

    context = format_context(chunks)

    user_prompt = RESUME_GAP_PROMPT.format(
        resume=resume[:2000],   # cap resume length
        target_role=target_role,
        n_listings=len(chunks),
        context=context,
    )

    analysis = _generate(
        system_prompt=RESUME_GAP_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=500,
    )

    citations = build_citations(chunks)

    log.info(f"[Chain] Resume gap done — {len(chunks)} listings used")

    return {
        "target_role":    target_role,
        "analysis":       analysis.strip(),
        "citations":      citations,
        "n_listings":     len(chunks),
        "low_confidence": False,
    }