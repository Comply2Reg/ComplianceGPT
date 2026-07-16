"""
Universal structural models and LLM output contract.

These models describe the JSON shape the LLM must return when performing
structure-aware parsing of legal / regulatory documents.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Universal structural node types for legal documents."""

    DOCUMENT = "document"
    PREAMBLE = "preamble"
    RECITAL = "recital"
    PART = "part"
    TITLE = "title"
    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLE = "article"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    POINT = "point"
    SUBPOINT = "subpoint"
    SCHEDULE = "schedule"
    ANNEX = "annex"
    DEFINITION = "definition"


class DocumentMetadata(BaseModel):
    """High-level document metadata the LLM can infer or be given as hints."""

    jurisdiction: Optional[str] = Field(default=None)
    document_type: Optional[str] = Field(default=None)
    source: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None)
    title: Optional[str] = None
    act_number: Optional[str] = None
    year: Optional[str] = None
    publication_date: Optional[str] = None
    effective_date: Optional[str] = None


class StructuredNode(BaseModel):
    """One structural node in the parsed hierarchy."""

    id_path: str = Field(description="Deterministic hierarchical ID path for this node.")
    type: NodeType
    label: Optional[str] = Field(default=None)
    numbering: Optional[List[str]] = Field(default=None)
    text: Optional[str] = Field(default=None)
    parent_id_path: Optional[str] = Field(default=None)
    order_index: Optional[int] = Field(default=None)
    page_start: Optional[int] = Field(default=None)
    page_end: Optional[int] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationshipKind(str, Enum):
    """Relationship type in LLM output."""

    CONTAINS = "CONTAINS"


class StructuredRelationship(BaseModel):
    """Relationship between two structural nodes."""

    source_id_path: str
    target_id_path: str
    kind: RelationshipKind = Field(default=RelationshipKind.CONTAINS)


class LLMParseResult(BaseModel):
    """Top-level contract expected from the LLM structural parser."""

    document_metadata: DocumentMetadata
    nodes: List[StructuredNode]
    relationships: List[StructuredRelationship]
