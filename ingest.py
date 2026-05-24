"""Run the full ingestion pipeline."""
import logging
from src.ingestion.embedder import run_ingestion_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

if __name__ == "__main__":
    summary = run_ingestion_pipeline(max_documents=5000)
    print(f"\nIngestion complete: {summary}")