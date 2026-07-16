"""Formal ontology schema for regulatory/legal documents."""

from src.obligation_pipeline.ontology.schema import EntityType, RelationType, SEMANTIC_ROLES
from src.obligation_pipeline.ontology.models import OntologyNode, OntologyEdge, LegalHierarchy

__all__ = [
    "EntityType",
    "RelationType",
    "SEMANTIC_ROLES",
    "OntologyNode",
    "OntologyEdge",
    "LegalHierarchy",
]
