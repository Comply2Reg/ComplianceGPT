"""
Ontology-aware chunking: one chunk per Section or Clause with full semantic metadata.

No naive token-based splitting; boundaries follow legal hierarchy from parsing layer.
"""

import hashlib
import logging
from typing import Any, Optional

from src.obligation_pipeline.chunking.extractor import ExtractedDocument
from src.obligation_pipeline.chunking.models import Chunk
from src.obligation_pipeline.parsing.hierarchy import LegalSegment, parse_legal_hierarchy

logger = logging.getLogger(__name__)

CHUNK_LEVELS = ("section", "article", "subsection", "clause", "explanation", "proviso", "definition", "schedule")


def _document_id(local_path: str, s3_key: Optional[str] = None) -> str:
    raw = s3_key if s3_key else local_path
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def chunk_by_ontology(
    extracted: ExtractedDocument,
    document_id: str,
    doc_metadata: dict[str, Any],
    act_name: Optional[str] = None,
    effective_date: Optional[str] = None,
    amendment_info: Optional[str] = None,
    segments: Optional[list[LegalSegment]] = None,
) -> list[Chunk]:
    """Produce one chunk per section/clause with ontology metadata."""
    regulator = doc_metadata.get("regulator") or ""
    act_name = act_name or doc_metadata.get("title") or "Unknown Act"
    if segments is None:
        segments = parse_legal_hierarchy(
            extracted.full_text,
            get_page_at_offset=extracted.get_page_at_offset,
        )
    if not segments:
        logger.debug("No legal segments detected; returning single-doc fallback chunk")
        page = extracted.get_page_at_offset(0) if extracted.full_text else None
        return [
            Chunk(
                chunk_id=f"{document_id}_0",
                document_id=document_id,
                chunk_index=0,
                text=extracted.full_text[:12000].strip() or "(no text)",
                total_chunks=1,
                regulator=regulator,
                title=act_name,
                act_name=act_name,
                semantic_role="other",
                page_start=page,
                page_end=page,
                amendment_info=amendment_info,
                **{k: v for k, v in doc_metadata.items()
                   if k in ("jurisdiction", "document_type", "document_subtype", "language", "source_url", "source", "s3_key", "local_path")},
            )
        ]
    chunks: list[Chunk] = []
    chunk_index = 0
    current_chapter: Optional[str] = None
    for seg in segments:
        if seg.level == "chapter":
            current_chapter = seg.label
        if seg.level not in CHUNK_LEVELS:
            continue
        section_number = seg.section_number()
        clause_number = seg.clause_number() if seg.level in ("clause", "subsection") else None
        if seg.level == "section":
            clause_number = None
        cid = f"{document_id}_{chunk_index}"
        page_end = extracted.get_page_at_offset(seg.end_offset - 1) if seg.end_offset else seg.page
        chunk = Chunk(
            chunk_id=cid,
            document_id=document_id,
            chunk_index=chunk_index,
            text=seg.text.strip() or "(no content)",
            total_chunks=0,
            regulator=regulator,
            title=act_name,
            act_name=act_name,
            chapter=current_chapter,
            section_number=section_number,
            clause_number=clause_number,
            semantic_role=seg.semantic_role,
            parent_path=seg.hierarchy_path(),
            section_label=seg.label,
            page_start=seg.page,
            page_end=page_end,
            amendment_info=amendment_info,
            **{k: v for k, v in doc_metadata.items()
               if k in ("jurisdiction", "document_type", "document_subtype", "language", "source_url", "source", "s3_key", "local_path")},
        )
        chunks.append(chunk)
        chunk_index += 1
    for c in chunks:
        c.total_chunks = len(chunks)
    return chunks
