# ── System prompt ─────────────────────────────────────────────────────────────
# Consumer-facing — answers must be clear, grounded, and cited.
# This is what a real user on Monster or CareerBuilder would read.

CAREER_QA_SYSTEM = """You are a career intelligence assistant for a major job search platform.
You help job seekers understand the current job market based on real,
live job listings.

YOUR RULES:
1. Answer ONLY from the provided job listing context.
   Never use your training knowledge to fill gaps.
2. Every specific claim must reference at least one source listing.
3. If the context does not contain enough information to answer,
   say exactly: "I don't have enough current job listings to answer
   this confidently. Try searching for [specific role] on our platform."
4. Be conversational and helpful — you are talking to a real job seeker.
5. Never hallucinate company names, salaries, or skill percentages
   unless they appear in the provided listings.
6. Quantify when possible — "mentioned in X of Y listings" is better
   than "commonly required".
"""


# ── RAG answer prompt ─────────────────────────────────────────────────────────
CAREER_QA_PROMPT = """Answer the following career question using ONLY
the job listing excerpts provided below.

QUESTION:
{query}

JOB LISTING CONTEXT:
{context}

INSTRUCTIONS:
- Lead with the most important insight directly
- Quantify trends where the data supports it
- Keep the answer under 200 words — job seekers want quick answers
- End with 2-3 source citations in this format:
  Sources: [1] Company · Role  [2] Company · Role

If the listings don't contain enough to answer confidently,
say so clearly and suggest a next step for the user.

Answer:"""


# ── Context formatter ─────────────────────────────────────────────────────────
def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Each chunk shows its source metadata so the LLM
    can generate accurate citations.
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        title   = chunk.get("title", "Unknown Role")
        company = chunk.get("company", "Unknown Company")
        conf    = chunk.get("confidence_score", 0.0)

        lines.append(
            f"[{i}] {company} · {title} "
            f"(relevance: {conf:.2f})\n"
            f"{chunk['text']}\n"
        )

    return "\n---\n".join(lines)


# ── Citation builder ──────────────────────────────────────────────────────────
def build_citations(chunks: list[dict]) -> list[dict]:
    """
    Build structured citation objects from reranked chunks.
    These are returned in the API response alongside the answer.
    """
    citations = []
    for i, chunk in enumerate(chunks, 1):
        citations.append({
            "index":            i,
            "doc_id":           chunk.get("doc_id", ""),
            "chunk_id":         chunk.get("chunk_id", ""),
            "title":            chunk.get("title", ""),
            "company":          chunk.get("company", ""),
            "location":         chunk.get("location", ""),
            "confidence_score": chunk.get("confidence_score", 0.0),
            "text_preview":     chunk["text"][:200] + "..."
                                if len(chunk["text"]) > 200
                                else chunk["text"],
        })
    return citations


# ── Resume gap analysis prompt ────────────────────────────────────────────────
# Secondary use case — analyse a resume against retrieved JDs

RESUME_GAP_SYSTEM = """You are a career coach helping a job seeker
understand gaps between their current resume and real job requirements.

YOUR RULES:
1. Base ALL gap analysis on the provided job listings only.
2. Be specific — name exact skills, tools, and qualifications missing.
3. Be encouraging — frame gaps as opportunities, not failures.
4. Never invent requirements not present in the listings.
5. Prioritise gaps by frequency — skills appearing in most listings first.
"""

RESUME_GAP_PROMPT = """Compare this resume against the following job listings
and identify the most important skill gaps.

RESUME:
{resume}

TARGET ROLE: {target_role}

JOB LISTING REQUIREMENTS (from {n_listings} real listings):
{context}

Provide:
1. TOP 3 GAPS — skills/experience missing from the resume but
   required in most listings
2. STRENGTHS — what the resume already has that matches
3. QUICK WINS — skills that can be added or learned quickly
   based on what listings emphasise

Keep the response under 250 words. Be specific and actionable.

Analysis:"""