import logging
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from datasets import load_dataset
from src.utils.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


# ── Document schema ───────────────────────────────────────────────────────────
# Every document in the system — regardless of source — is normalised
# to this schema before chunking or embedding.
# In production at BOLD this would come from Monster/CareerBuilder/FlexJobs.
# In this public version we use a HuggingFace proxy dataset.

@dataclass
class Document:
    doc_id:        str                      # SHA-256 of content — idempotent
    content:       str                      # cleaned full text
    title:         str   = ""              # job title
    company:       str   = ""              # company name
    location:      str   = ""              # city / remote
    company_type:  str   = ""              # industry / sector
    source:        str   = "huggingface"   # dataset source
    metadata:      dict  = field(default_factory=dict)


def _make_doc_id(content: str) -> str:
    """SHA-256 of content — same doc always gets same ID. Enables idempotent upsert."""
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _clean_text(text: str) -> str:
    """
    Basic text normalisation.
    Removes excessive whitespace, strips null bytes, normalises line breaks.
    In production: ftfy for encoding fixes, boilerplate removal per source.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")          # null bytes
    text = text.replace("\r\n", "\n")        # windows line endings
    text = "\n".join(                         # collapse blank lines > 2
        line for line in text.split("\n")
    )
    # Collapse whitespace within lines
    lines = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        lines.append(line)
    text = "\n".join(lines).strip()
    return text


def _is_valid(doc: Document, min_length: int = 100) -> bool:
    """Filter out documents too short to be useful."""
    return len(doc.content) >= min_length


def load_hf_dataset(
    dataset_name: str = "jacob-hugging-face/job-descriptions",
    split: str = "train",
    max_documents: int = 5000,
    cache_path: str = "data/raw/documents.json",
) -> list[Document]:
    """
    Load job descriptions from HuggingFace dataset.
    Normalises to Document schema.
    Caches to disk — re-running won't re-download.

    In production at BOLD: this function would call Monster/CareerBuilder
    ingestion APIs instead of HuggingFace. The schema and everything
    downstream stays identical.

    Args:
        dataset_name:   HuggingFace dataset identifier
        split:          dataset split to use
        max_documents:  cap to avoid excessive API costs during dev
        cache_path:     local cache location

    Returns:
        List of normalised Document objects
    """
    # ── Check cache ───────────────────────────────────────────────────────────
    cache = Path(cache_path)
    if cache.exists():
        log.info(f"[Loader] Loading from cache: {cache_path}")
        with open(cache, encoding="utf-8") as f:
            raw = json.load(f)
        docs = [Document(**d) for d in raw]
        log.info(f"[Loader] Loaded {len(docs)} documents from cache")
        return docs

    # ── Load from HuggingFace ─────────────────────────────────────────────────
    log.info(f"[Loader] Downloading {dataset_name} ({split}) from HuggingFace...")

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        log.error(f"[Loader] HuggingFace load failed: {e}")
        raise

    log.info(f"[Loader] Dataset loaded — {len(dataset)} rows")
    log.info(f"[Loader] Columns: {dataset.column_names}")

    # ── Normalise to Document schema ──────────────────────────────────────────
    docs = []
    skipped = 0

    for i, row in enumerate(dataset):
        if len(docs) >= max_documents:
            break

        # Primary content field
        content_raw = (
            row.get("job_description") or
            row.get("description") or
            row.get("text") or
            ""
        )
        content = _clean_text(str(content_raw))

        if not content:
            skipped += 1
            continue

        doc = Document(
            doc_id=_make_doc_id(content),
            content=content,
            title=_clean_text(str(row.get("position_title", ""))),
            company=_clean_text(str(row.get("company_name", ""))),
            location=_clean_text(str(row.get("location", ""))),
            company_type=_clean_text(str(row.get("company_type", ""))),
            source="huggingface",
            metadata={
                "row_index":   i,
                "dataset":     dataset_name,
                "split":       split,
            },
        )

        if _is_valid(doc):
            docs.append(doc)
        else:
            skipped += 1

    log.info(
        f"[Loader] Normalised {len(docs)} documents "
        f"(skipped {skipped} — too short or empty)"
    )

    # ── Deduplicate by doc_id ─────────────────────────────────────────────────
    seen = set()
    deduped = []
    for doc in docs:
        if doc.doc_id not in seen:
            seen.add(doc.doc_id)
            deduped.append(doc)

    dupes = len(docs) - len(deduped)
    if dupes:
        log.info(f"[Loader] Removed {dupes} duplicate documents")

    docs = deduped

    # ── Save to cache ─────────────────────────────────────────────────────────
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(
            [doc.__dict__ for doc in docs],
            f, indent=2, ensure_ascii=False,
        )
    log.info(f"[Loader] Cached {len(docs)} documents → {cache_path}")

    return docs