import logging
import time
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.models import (
    CareerQueryRequest, CareerQueryResponse,
    ResumeGapRequest,   ResumeGapResponse,
    HealthResponse,     CitationResponse,
)
from src.generation.chain import (
    answer_career_question,
    analyse_resume_gap,
)
from src.utils.config import get_settings

log      = logging.getLogger(__name__)
settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up clients on startup."""
    log.info("Starting Career Intelligence RAG API")
    log.info(f"Qdrant mode: {settings.qdrant_mode}")
    log.info(f"Collection:  {settings.collection_name}")
    yield
    log.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Career Intelligence RAG",
    description=(
        "Production RAG system for career Q&A and resume gap analysis. "
        "Powered by HyDE + BM25 hybrid retrieval, Cohere reranking, "
        "and grounded generation with source citations.\n\n"
        "Built as a public implementation of the retrieval system "
        "behind BOLD's job search platforms (Monster, CareerBuilder, "
        "FlexJobs, Remote.co)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        collection=settings.collection_name,
        qdrant_mode=settings.qdrant_mode,
    )


@app.post("/query", response_model=CareerQueryResponse)
def career_query(request: CareerQueryRequest):
    """
    Answer a career question grounded in real job listings.

    Examples:
    - "What skills do I need for a Senior ML Engineer role in Bangalore?"
    - "What is the average experience required for a Data Scientist role?"
    - "Which companies are hiring for LangChain or RAG experience?"
    - "What is the difference between MLOps and DataOps roles?"
    """
    start = time.time()
    log.info(f"[/query] {request.query[:80]!r}")

    try:
        result = answer_career_question(
            query=request.query,
            use_hyde=request.use_hyde,
            use_bm25=request.use_bm25,
            top_k_rerank=request.top_k_rerank,
        )
    except Exception as e:
        log.error(f"[/query] failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - start, 3)
    log.info(f"[/query] done in {elapsed}s")

    return CareerQueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=[CitationResponse(**c) for c in result["citations"]],
        chunks_retrieved=result["chunks_retrieved"],
        chunks_used=result["chunks_used"],
        low_confidence=result["low_confidence"],
        retrieval_method=result["retrieval_method"],
    )

@app.post("/query/stream")
async def career_query_stream(request: CareerQueryRequest):
    """
    Streaming version of /query.
    Returns answer tokens as they are generated.
    Time-to-first-token: ~300ms.
    Full response: ~2.2s p50.

    Use this endpoint for UI integrations where perceived
    latency matters — user sees the answer building in real time
    rather than waiting for the full response.
    """
    import asyncio
    from openai import AsyncOpenAI
    from src.retrieval.reranker import retrieve_and_rerank
    from src.generation.prompts import (
        CAREER_QA_SYSTEM, CAREER_QA_PROMPT, format_context, build_citations
    )

    async_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate():
        try:
            # ── Retrieve + rerank (same as /query) ────────────────────────
            chunks = retrieve_and_rerank(
                query=request.query,
                use_hyde=request.use_hyde,
                use_bm25=request.use_bm25,
                top_k_rerank=request.top_k_rerank,
            )

            if not chunks:
                yield "data: " + json.dumps({
                    "type": "error",
                    "content": "No relevant listings found."
                }) + "\n\n"
                return

            context = format_context(chunks)
            user_prompt = CAREER_QA_PROMPT.format(
                query=request.query,
                context=context,
            )

            # ── Stream generation ─────────────────────────────────────────
            stream = await async_client.chat.completions.create(
                model=settings.generation_model,
                temperature=0.3,
                max_tokens=400,
                stream=True,
                messages=[
                    {"role": "system", "content": CAREER_QA_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield "data: " + json.dumps({
                        "type":    "token",
                        "content": delta,
                    }) + "\n\n"

            # ── Send citations after stream completes ─────────────────────
            citations = build_citations(chunks)
            avg_conf  = sum(
                c.get("confidence_score", 0) for c in chunks
            ) / len(chunks)

            yield "data: " + json.dumps({
                "type":             "done",
                "citations":        citations,
                "chunks_used":      len(chunks),
                "low_confidence":   avg_conf < 0.4,
                "retrieval_method": "hyde+bm25+dense",
            }) + "\n\n"

        except Exception as e:
            log.error(f"[/query/stream] failed: {e}")
            yield "data: " + json.dumps({
                "type":    "error",
                "content": str(e),
            }) + "\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/resume-gap", response_model=ResumeGapResponse)
def resume_gap(request: ResumeGapRequest):
    """
    Analyse resume gaps against real current job listings.

    Retrieves current JD requirements for the target role,
    then identifies what's missing from the provided resume.
    """
    start = time.time()
    log.info(f"[/resume-gap] role={request.target_role!r}")

    try:
        result = analyse_resume_gap(
            resume=request.resume,
            target_role=request.target_role,
        )
    except Exception as e:
        log.error(f"[/resume-gap] failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - start, 3)
    log.info(f"[/resume-gap] done in {elapsed}s")

    return ResumeGapResponse(
        target_role=result["target_role"],
        analysis=result["analysis"],
        citations=[CitationResponse(**c) for c in result["citations"]],
        n_listings=result["n_listings"],
        low_confidence=result["low_confidence"],
    )


@app.get("/")
def root():
    return {
        "name":    "Career Intelligence RAG",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": {
            "career_qa":    "POST /query",
            "resume_gap":   "POST /resume-gap",
        },
    }