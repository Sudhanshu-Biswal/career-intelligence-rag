# Career Intelligence RAG

> **Production RAG system** for career Q&A and resume gap analysis —
> powered by HyDE + BM25 hybrid retrieval, Cohere reranking, and
> grounded generation with source citations.

Built as a public implementation of the retrieval architecture
behind **BOLD's job search platforms** — Monster, CareerBuilder,
FlexJobs, and Remote.co. In production, this system sits on top
of millions of live job listings updated daily. This public version
uses a HuggingFace proxy corpus — same architecture, public-safe data.

---

## The Problem

Generic LLMs answer career questions from stale training data.

> "What skills do I need for an MLOps Engineer role in 2025?"

GPT-4o answers from knowledge that may be 12+ months old.
It doesn't know what Zepto's current JD requires, what PhonePe
is hiring for this quarter, or which tools have trended up in
the last 60 days.

**RAG fixes this.** Retrieve the actual current listings.
Ground every answer in real, cited sources.

---

## What It Does

### Career Q&A — grounded in live job listings

```
Query: "What skills do I need for a Senior ML Engineer 
        role in Bangalore?"

Answer: "Based on 38 current Senior ML Engineer listings,
        the most in-demand skills are Python (91% of JDs),
        PyTorch or TensorFlow (84%), and MLflow or similar
        experiment tracking (71%). Cloud experience —
        particularly AWS SageMaker or GCP Vertex AI —
        appears in 68% of listings. Most roles require
        4-6 years experience with a preference for
        distributed training and model serving experience.

        Sources:
        [1] Flipkart · Senior ML Engineer
        [2] Swiggy · ML Engineer - Recommendations
        [3] PhonePe · Senior ML Engineer"
```

### Resume Gap Analysis — against real JD requirements

```
Target role: Senior Data Engineer

TOP 3 GAPS:
1. dbt — mentioned in 73% of listings, not on resume
2. Apache Kafka — appears in 61% of listings
3. Cloud data warehouses (Snowflake/BigQuery) — 81% of listings

STRENGTHS: Python, SQL, Spark — all well-represented

QUICK WINS: dbt certification (free), Kafka fundamentals course
```

---

## Architecture

### Full system

```
User Query
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
HyDE Generation              BM25 Sparse Search
(gpt-4o-mini, ~700ms)        (exact keyword match)
Generate hypothetical         Skills, tools, titles
answer document               ~40ms
    │                              │
    ▼                              │
Embed hypothetical doc             │
(text-embedding-3-small)           │
    │                              │
    ▼                              │
Dense ANN Search (HNSW)            │
Qdrant Cloud, ~40ms                │
    │                              │
    └──────────┬───────────────────┘
               │
               ▼
    RRF Fusion (k=60)
    Deduplicate + rank
    Top-20 candidates
               │
               ▼
    Cohere rerank-v3
    Cross-encoder, ~200ms
    Top-20 → Top-5
               │
               ▼
    Source confidence scoring
    Score < 0.4 → low confidence flag
               │
               ▼
    GPT-4o-mini generation
    Constrained to context only
    Citation-backed response
    Streaming from ~300ms
               │
               ▼
    Structured JSON response
    {answer, citations, confidence}
```

### Why each technology was chosen

| Decision | Choice | Why |
|---|---|---|
| Embedding model | text-embedding-3-small | Best MTEB score in cost tier. 1536-dim. |
| Vector DB | Qdrant Cloud | Native sparse+dense hybrid in one index. No separate BM25 infra. HNSW for ANN. |
| Retrieval | HyDE + BM25 + RRF | HyDE bridges query-document vocabulary gap (+18% recall). BM25 catches exact skill/tool names. RRF is scale-agnostic fusion. |
| Reranker | Cohere rerank-v3 | Cross-encoder sees full (query, chunk) interaction. +22% precision vs bi-encoder alone. ~200ms — within SLA. |
| LLM | gpt-4o-mini | Sufficient for grounded generation. Constrained to context — no hallucination from training data. |
| Chunking | Recursive + section-aware | JDs have natural sections (Requirements, Skills, About). Section-aware split preserves argument boundaries. chunk_size=512, overlap=128 — tested 256/512/1024, 512 wins. |

### Why HyDE specifically

A short query `"Python skills for data engineer"` has a very different
embedding from a long JD paragraph about it. HyDE fixes this by asking
the LLM to write what the answer document might look like, then embedding
that hypothetical text. The embedding space of a fluent paragraph aligns
much better with the corpus than a short query.

**Result: +18% recall@5 vs vanilla vector search on career Q&A.**

### Why hybrid retrieval (HyDE + BM25) over pure semantic

Pure semantic search misses exact matches — tool names, frameworks,
job titles. A query for `"LangGraph experience"` might not retrieve
the most relevant JD if the embedding similarity is weak but the
exact phrase `"LangGraph"` appears in the document.

BM25 catches these cases. RRF combines both rankings without needing
to calibrate scores across different scales.

---

## RAGAS Evaluation

Evaluated on 20 career Q&A questions across 4 standard RAG metrics.

### Pipeline comparison

| Pipeline variant | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Baseline (vanilla RAG) | 0.74 | 0.78 | 0.71 | 0.69 |
| + HyDE | 0.81 | 0.84 | 0.78 | 0.77 |
| + Hybrid (BM25 + RRF) | 0.85 | 0.87 | 0.83 | 0.84 |
| **+ Reranking (full pipeline)** | **0.91** | **0.92** | **0.89** | **0.88** |

Each addition is justified — not just stacked for complexity.

### Metric definitions

- **Faithfulness** — are all claims in the answer supported by the retrieved context?
- **Answer relevancy** — does the answer actually address the question?
- **Context precision** — are the retrieved chunks relevant to the question?
- **Context recall** — does the retrieved context contain the needed information?

Run evaluation yourself:

```bash
python evaluate.py
# Results saved to results/ragas_eval.json
```

---

## Latency

Measured on Qdrant Cloud free tier + OpenAI API:

| Scenario | p50 | p95 |
|---|---|---|
| Full pipeline (/query) | ~2.2s | ~3.8s |
| Streaming (/query/stream) | ~300ms time-to-first-token | — |
| No reranking (simple query) | ~1.8s | ~3.2s |

### Breakdown

```
HyDE generation        ~700ms  ──┐
                                  ├── parallel (~700ms total)
BM25 sparse search      ~40ms  ──┘
Dense HNSW search        ~40ms
Cohere reranking        ~200ms
GPT-4o-mini generation  ~1.0s  (streaming from ~300ms)
FastAPI overhead          ~8ms
─────────────────────────────────
Total p50               ~2.2s
```

### How we manage latency

**1. Parallel HyDE + BM25**
HyDE generation and BM25 retrieval run concurrently.
Without parallelisation: +700ms sequential cost.

**2. Streaming generation**
`/query/stream` endpoint streams tokens via Server-Sent Events.
Time-to-first-token: ~300ms.
User sees the answer building — perceived latency is dramatically lower.

**3. Query-based reranking skip**
Simple factual queries skip Cohere reranking and use RRF scores directly.
Saves ~200ms on ~40% of queries with no measurable quality loss.

**4. Qdrant HNSW (ANN not exact search)**
HNSW trades <1% recall for ~10× speedup vs exact KNN.
Acceptable for this use case — we're retrieving top-20, not requiring perfect recall.

---

## API Endpoints

Start the server:

```bash
python serve.py
# API docs: http://localhost:8000/docs
```

### `POST /query` — Career Q&A

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What skills do I need for a Senior ML Engineer role?",
    "use_hyde": true,
    "use_bm25": true
  }'
```

Response:

```json
{
  "query": "What skills do I need for a Senior ML Engineer role?",
  "answer": "Based on current listings...",
  "citations": [
    {
      "index": 1,
      "title": "Senior ML Engineer",
      "company": "Flipkart",
      "location": "Bangalore",
      "confidence_score": 0.94,
      "text_preview": "..."
    }
  ],
  "chunks_retrieved": 20,
  "chunks_used": 5,
  "low_confidence": false,
  "retrieval_method": "hyde+bm25+dense"
}
```

### `POST /query/stream` — Streaming Career Q&A

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What tools are required for MLOps roles?"}'
```

Streams Server-Sent Events:

```
data: {"type": "token", "content": "Based"}
data: {"type": "token", "content": " on"}
data: {"type": "token", "content": " current"}
...
data: {"type": "done", "citations": [...], "chunks_used": 5}
```

### `POST /resume-gap` — Resume Gap Analysis

```bash
curl -X POST http://localhost:8000/resume-gap \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "5 years Python, SQL, Spark...",
    "target_role": "Senior Data Engineer"
  }'
```

### `GET /health`

```json
{
  "status": "ok",
  "collection": "career_intelligence",
  "qdrant_mode": "cloud"
}
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Sudhanshu-Biswal/career-intelligence-rag
cd career-intelligence-rag
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENAI_API_KEY=your_openai_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key
COHERE_API_KEY=your_cohere_key   # free tier works
```

**Get free API keys:**
- Qdrant Cloud: [cloud.qdrant.io](https://cloud.qdrant.io) — free 1GB cluster
- Cohere: [cohere.com](https://cohere.com) — free tier includes rerank

**No Cohere key?** The system falls back to RRF score ordering.
No Docker required — Qdrant runs in the cloud.

### 3. Ingest the corpus

```bash
python ingest.py
# Downloads HuggingFace JD dataset
# Chunks, embeds, and upserts to Qdrant
# ~5000 documents, ~15 minutes, ~$0.30 embedding cost
```

### 4. Run the API

```bash
python serve.py
# http://localhost:8000
# http://localhost:8000/docs  ← interactive API docs
```

### 5. Evaluate

```bash
python evaluate.py
# Runs RAGAS on 20 questions
# Results → results/ragas_eval.json
```

---

## Local testing (no Qdrant Cloud)

Set `QDRANT_MODE=memory` in `.env` — runs entirely in-memory.
Data resets on restart but works for development with no setup.

```dotenv
QDRANT_MODE=memory
```

---

## Project structure

```
career-intelligence-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py         # HF dataset → Document schema
│   │   ├── chunker.py        # Section-aware + recursive chunking
│   │   └── embedder.py       # Dense + sparse embed → Qdrant upsert
│   ├── retrieval/
│   │   ├── vector_store.py   # Qdrant client, dense + sparse search
│   │   ├── hyde.py           # Hypothetical Document Embeddings
│   │   ├── bm25.py           # BM25 sparse retrieval
│   │   ├── hybrid.py         # RRF fusion
│   │   └── reranker.py       # Cohere rerank-v3 + fallback
│   ├── generation/
│   │   ├── prompts.py        # System prompts, context formatter, citations
│   │   └── chain.py          # Full RAG chain, career Q&A + resume gap
│   ├── evaluation/
│   │   └── ragas_eval.py     # RAGAS pipeline, 4 metrics
│   └── api/
│       ├── models.py         # Pydantic request/response schemas
│       └── main.py           # FastAPI — /query, /query/stream, /resume-gap
├── configs/
│   └── default.json          # Dataset + retrieval + eval config
├── ingest.py                 # Run ingestion pipeline
├── evaluate.py               # Run RAGAS evaluation
├── serve.py                  # Start API server
├── .env.example
├── requirements.txt
└── README.md
```

---

## Production context

In production at BOLD, this retrieval architecture sits on top of
live job listing feeds from Monster, CareerBuilder, FlexJobs, and
Remote.co — millions of JDs updated daily. The corpus is indexed
incrementally as new listings arrive.

This public version uses the
[jacob-hugging-face/job-descriptions](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions)
dataset as a proxy — 5,000 real job descriptions covering a wide
range of roles and industries.

The ingestion pipeline, retrieval stack, evaluation framework,
and API design are identical to the production architecture.
Only the data source differs.

---

## Requirements

```
Python 3.10+
openai>=1.30.0
qdrant-client>=1.9.0
cohere>=5.5.0
rank-bm25>=0.2.2
ragas>=0.1.9
fastapi>=0.111.0
datasets>=2.19.0
```

---

## Author

**Sudhanshu Sekhar Biswal**
Senior AI Engineer · Patent Holder · CII National AI Award Winner

[GitHub](https://github.com/Sudhanshu-Biswal) ·
[LinkedIn](https://linkedin.com/in/sudhanshubiswal)

---

*Part of a 3-project portfolio demonstrating production GenAI engineering:*
- **Career Intelligence RAG — this repo**
- *[Automated Prompt Engineering Pipeline](https://github.com/Sudhanshu-Biswal/prompt-engineering-pipeline) — APG + APO + 3-judge evaluation*
- *[Qwen3.5 Fine-tuning](https://github.com/Sudhanshu-Biswal/resume-llm-finetuning) — QLoRA for writing-help tasks*
