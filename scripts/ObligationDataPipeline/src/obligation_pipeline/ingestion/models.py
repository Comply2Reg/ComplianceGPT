"""Data models for the ingestion pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class DocumentRecord:
    """Single document metadata record from the database."""

    s3_key: str
    regulator: str
    s3_bucket: Optional[str] = None
    row_id: Optional[Any] = None
    title: Optional[str] = None
    document_type: Optional[str] = None
    document_subtype: Optional[str] = None
    effective_date: Optional[str] = None
    release_date: Optional[str] = None
    source_url: Optional[str] = None
    pdf_url: Optional[str] = None
    extra_metadata: Optional[dict[str, Any]] = None

    def to_manifest_entry(self, local_path: Optional[Path] = None) -> dict[str, Any]:
        """Export as manifest entry for chunking, vector DB, and knowledge graph."""
        entry: dict[str, Any] = {
            "s3_key": self.s3_key,
            "s3_bucket": self.s3_bucket,
            "regulator": self.regulator,
            "local_path": str(local_path) if local_path else None,
        }
        if self.title is not None:
            entry["title"] = self.title
        if self.document_type is not None:
            entry["document_type"] = self.document_type
        if self.document_subtype is not None:
            entry["document_subtype"] = self.document_subtype
        if self.effective_date is not None:
            entry["effective_date"] = self.effective_date
        if self.release_date is not None:
            entry["release_date"] = self.release_date
        if self.source_url is not None:
            entry["source_url"] = self.source_url
        if self.pdf_url is not None:
            entry["pdf_url"] = self.pdf_url
        if self.extra_metadata:
            entry["metadata"] = self.extra_metadata
        return entry


@dataclass
class IngestedDocument:
    """Document after download, with local path and record."""

    record: DocumentRecord
    local_path: Path
    success: bool
    error: Optional[str] = None
