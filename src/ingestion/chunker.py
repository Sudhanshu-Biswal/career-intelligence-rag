import logging
import re
from dataclasses import dataclass, field
from src.ingestion.loader import Document
from src.utils.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


# ── Chunk schema ──────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:      str         # "{doc_id}_{chunk_index}"
    doc_id:        str         # parent document
    text:          str         # chunk content
    chunk_index:   int         # position within document
    total_chunks:  int         # total chunks in document
    title:         str = ""    # inherited from Document
    company:       str = ""
    location:      str = ""
    company_type:  str = ""
    source:        str = ""
    metadata:      dict = field(default_factory=dict)


def _split_recursive(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> list[str]:
    """
    Recursive character text splitter.
    Splits at natural boundaries in priority order:
    paragraphs → sentences → words → characters.

    This mirrors LangChain's RecursiveCharacterTextSplitter
    but without the dependency — keeps the ingestion pipeline lean.

    chunk_size=512, overlap=128 chosen after testing 256/512/1024:
    - 256: too granular, splits mid-argument frequently
    - 512: best balance of context and precision
    - 1024: too large, retrieval precision drops
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)

        chunks = []
        current = ""

        for split in splits:
            candidate = current + (sep if current else "") + split
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(split) > chunk_size:
                    # Recursively split oversized piece
                    sub = _split(split, separators[1:])
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = split

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if c.strip()]

    raw_chunks = _split(text, separators)

    # ── Apply overlap ─────────────────────────────────────────────────────────
    # Each chunk includes the tail of the previous chunk as context.
    # Prevents splitting mid-argument at chunk boundaries.
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-chunk_overlap:]
        overlapped.append(prev_tail + " " + raw_chunks[i])

    return overlapped


def _extract_sections(text: str) -> list[str]:
    """
    For JDs, try to split at section headers first.
    Recognises common JD sections: Responsibilities, Requirements,
    Qualifications, Skills, About, Benefits, etc.

    If no sections found, falls back to recursive splitting.
    """
    section_pattern = re.compile(
        r"(?:^|\n)(?:"
        r"responsibilities|requirements|qualifications|"
        r"skills|about|benefits|what you.ll do|"
        r"what we.re looking for|who you are|"
        r"nice to have|preferred|minimum qualifications"
        r")[:\s]*",
        re.IGNORECASE,
    )

    sections = section_pattern.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    if len(sections) <= 1:
        return []   # no sections found — caller falls back

    return sections


def chunk_document(
    doc: Document,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[Chunk]:
    """
    Chunk a single Document into Chunk objects.

    Strategy:
    1. Try section-aware splitting (JD headers)
    2. Within each section (or the full text if no sections),
       apply recursive character splitting
    3. Attach parent document metadata to every chunk

    Each chunk stores its position (chunk_index, total_chunks)
    enabling context expansion at retrieval time.
    """
    chunk_size    = chunk_size    or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # ── Try section-aware split first ────────────────────────────────────────
    sections = _extract_sections(doc.content)

    if sections:
        raw_texts = []
        for section in sections:
            if len(section) <= chunk_size:
                raw_texts.append(section)
            else:
                raw_texts.extend(
                    _split_recursive(section, chunk_size, chunk_overlap)
                )
    else:
        raw_texts = _split_recursive(doc.content, chunk_size, chunk_overlap)

    # Filter empty
    raw_texts = [t for t in raw_texts if len(t.strip()) >= 30]

    if not raw_texts:
        log.warning(f"[Chunker] doc_id={doc.doc_id} produced 0 chunks")
        return []

    total = len(raw_texts)

    chunks = []
    for i, text in enumerate(raw_texts):
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_{i:04d}",
            doc_id=doc.doc_id,
            text=text,
            chunk_index=i,
            total_chunks=total,
            title=doc.title,
            company=doc.company,
            location=doc.location,
            company_type=doc.company_type,
            source=doc.source,
            metadata={
                **doc.metadata,
                "chunk_index":  i,
                "total_chunks": total,
                "chunk_size":   len(text),
            },
        )
        chunks.append(chunk)

    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[Chunk]:
    """
    Chunk a list of documents.
    Logs progress every 500 documents.
    """
    all_chunks = []
    for i, doc in enumerate(docs):
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        if (i + 1) % 500 == 0:
            log.info(
                f"[Chunker] {i+1}/{len(docs)} docs → "
                f"{len(all_chunks)} chunks so far"
            )

    log.info(
        f"[Chunker] Complete — {len(docs)} docs → "
        f"{len(all_chunks)} chunks "
        f"(avg {len(all_chunks)/max(1,len(docs)):.1f} chunks/doc)"
    )
    return all_chunks