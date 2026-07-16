"""Data models for chunking pipeline."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Chunk:
    """A single chunk with text and metadata for vector DB / KG."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    # Document-level
    regulator: Optional[str] = None
    jurisdiction: Optional[str] = None
    title: Optional[str] = None
    document_type: Optional[str] = None
    document_subtype: Optional[str] = None
    language: Optional[str] = None
    source_url: Optional[str] = None
    source: Optional[str] = None
    s3_key: Optional[str] = None
    local_path: Optional[str] = None
    # Structure
    section_label: Optional[str] = None
    parent_path: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    # Ontology-aware
    act_name: Optional[str] = None
    chapter: Optional[str] = None
    section_number: Optional[str] = None
    clause_number: Optional[str] = None
    semantic_role: Optional[str] = None
    amendment_info: Optional[str] = None
    # Pipeline
    total_chunks: int = 0
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL output."""
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "total_chunks": self.total_chunks,
        }
        for field_name in (
            "regulator", "jurisdiction", "title", "document_type", "document_subtype",
            "language", "source_url", "source", "s3_key", "local_path",
            "section_label", "parent_path", "page_start", "page_end",
            "act_name", "chapter", "section_number", "clause_number",
            "semantic_role", "amendment_info",
        ):
            val = getattr(self, field_name, None)
            if val is not None:
                d[field_name] = val
        if self.extra:
            d["metadata"] = self.extra
        return d
