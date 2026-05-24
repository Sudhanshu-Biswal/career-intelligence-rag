import logging
import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.generation.chain import answer_career_question
from src.utils.config import get_settings

log      = logging.getLogger(__name__)
settings = get_settings()


# ── Eval questions ────────────────────────────────────────────────────────────
# 20 representative career Q&A questions for evaluation.
# These cover the primary query types the system handles.

EVAL_QUESTIONS = [
    "What Python skills are required for a Data Engineer role?",
    "What experience level is needed for a Senior ML Engineer position?",
    "Which cloud platforms are most commonly required in data science jobs?",
    "What is the difference in requirements between a Data Analyst and Data Scientist?",
    "What tools are commonly required for MLOps roles?",
    "What soft skills are mentioned in product manager job descriptions?",
    "How many years of experience do most software engineer roles require?",
    "What programming languages appear most in backend engineer job listings?",
    "What certifications are mentioned in cloud architect job descriptions?",
    "What skills are required for a Natural Language Processing engineer role?",
    "What frameworks are commonly required for frontend developer positions?",
    "What does a typical DevOps engineer job description require?",
    "What industries are hiring the most data scientists currently?",
    "What is the typical team size mentioned in startup job descriptions?",
    "What remote work policies appear in current job listings?",
    "What database skills are most commonly required across engineering roles?",
    "What does a machine learning engineer role require vs a data scientist?",
    "What agile or project management skills appear in tech job listings?",
    "What security skills are required for a cloud engineer role?",
    "What communication skills are mentioned in senior engineer job descriptions?",
]


def _build_eval_dataset(
    questions: list[str],
    sample_size: int = 20,
) -> Dataset:
    """
    Build a RAGAS evaluation dataset by running the full
    RAG pipeline on each question and collecting:
    - question
    - answer (generated)
    - contexts (retrieved chunks)
    - ground_truth (self-generated — no human labels needed)
    """
    questions = questions[:sample_size]
    log.info(f"[RAGAS] Building eval dataset for {len(questions)} questions")

    rows = {
        "question":    [],
        "answer":      [],
        "contexts":    [],
        "ground_truth": [],
    }

    for i, question in enumerate(questions):
        log.info(f"[RAGAS] {i+1}/{len(questions)}: {question[:60]}")

        try:
            result = answer_career_question(query=question)

            # Contexts = raw chunk texts used for generation
            contexts = [
                c["text_preview"]
                for c in result["citations"]
            ]

            rows["question"].append(question)
            rows["answer"].append(result["answer"])
            rows["contexts"].append(contexts)
            # Ground truth = the answer itself for reference-free metrics
            rows["ground_truth"].append(result["answer"])

        except Exception as e:
            log.error(f"[RAGAS] Question {i} failed: {e}")
            rows["question"].append(question)
            rows["answer"].append("")
            rows["contexts"].append([""])
            rows["ground_truth"].append("")

    return Dataset.from_dict(rows)


def run_ragas_evaluation(
    questions: list[str] = None,
    sample_size: int = None,
    output_path: str = "results/ragas_eval.json",
) -> dict:
    """
    Run full RAGAS evaluation pipeline.

    Metrics:
    - faithfulness:       are claims supported by retrieved context?
    - answer_relevancy:   does answer address the question?
    - context_precision:  are retrieved chunks relevant?
    - context_recall:     does context contain needed information?

    These four metrics are the standard RAG eval suite.
    Target scores for this project: all > 0.80.

    Returns:
        dict of metric scores + per-question results
    """
    questions   = questions or EVAL_QUESTIONS
    sample_size = sample_size or settings.ragas_sample_size

    log.info("=" * 50)
    log.info("  RAGAS EVALUATION START")
    log.info("=" * 50)

    # Build dataset
    dataset = _build_eval_dataset(questions, sample_size)

    # Run RAGAS
    log.info("[RAGAS] Running evaluation metrics...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "faithfulness":      round(float(results["faithfulness"]),      3),
        "answer_relevancy":  round(float(results["answer_relevancy"]),  3),
        "context_precision": round(float(results["context_precision"]), 3),
        "context_recall":    round(float(results["context_recall"]),    3),
    }

    # Overall score
    scores["overall"] = round(
        sum(scores.values()) / len(scores), 3
    )

    log.info("=" * 50)
    log.info("  RAGAS RESULTS")
    log.info("=" * 50)
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        log.info(f"  {metric:<22} {score:.3f}  {bar}")
    log.info("=" * 50)

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "scores":       scores,
        "sample_size":  sample_size,
        "retrieval": {
            "hyde_enabled": settings.hyde_enabled,
            "bm25_enabled": settings.bm25_enabled,
            "top_k_retrieve": settings.top_k_retrieve,
            "top_k_rerank":   settings.top_k_rerank,
        },
        "model": settings.generation_model,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"[RAGAS] Results saved → {output_path}")
    return scores