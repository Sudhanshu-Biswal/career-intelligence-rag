"""Run RAGAS evaluation."""
import logging
from src.evaluation.ragas_eval import run_ragas_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

if __name__ == "__main__":
    scores = run_ragas_evaluation(sample_size=20)
    print(f"\nRAGAS scores: {scores}")