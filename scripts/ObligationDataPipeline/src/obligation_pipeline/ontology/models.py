"""Graph-ready node and edge models for the regulatory ontology."""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.obligation_pipeline.ontology.schema import EntityType, RelationType


@dataclass
class OntologyNode:
    """A node in the knowledge graph (Neo4j/RDF-ready)."""

    id: str
    type: EntityType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_neo4j_props(self) -> dict[str, Any]:
        out = {"id": self.id, "type": self.type.value, "label": self.label, **self.properties}
        return {k: v for k, v in out.items() if v is not None}


@dataclass
class OntologyEdge:
    """A directed edge in the knowledge graph."""

    source_id: str
    target_id: str
    type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)

    def to_neo4j_props(self) -> dict[str, Any]:
        return {"type": self.type.value, **self.properties}


@dataclass
class LegalHierarchy:
    """In-memory hierarchy for one Act: regulator, act, chapters, sections, clauses."""

    regulator: str
    act_id: str
    act_name: str
    document_id: str
    source_path: Optional[str] = None
    effective_date: Optional[str] = None
    amendment_info: Optional[str] = None
    chapters: list[dict[str, Any]] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    clauses: list[dict[str, Any]] = field(default_factory=list)
    definitions: list[dict[str, Any]] = field(default_factory=list)
