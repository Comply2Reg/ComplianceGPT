"""Legal document parsing: regex-based and LLM-assisted hierarchy extraction."""

from src.obligation_pipeline.parsing.hierarchy import parse_legal_hierarchy, build_hierarchy_from_parsed, LegalSegment
from src.obligation_pipeline.parsing.patterns import LEGAL_PATTERNS

__all__ = [
    "parse_legal_hierarchy",
    "build_hierarchy_from_parsed",
    "LegalSegment",
    "LEGAL_PATTERNS",
]
