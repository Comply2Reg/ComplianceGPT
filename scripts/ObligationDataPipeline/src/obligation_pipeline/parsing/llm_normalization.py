"""Validation and normalization for LLM structural parser output."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from pydantic import ValidationError

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.llm.client import run_structural_parse
from src.obligation_pipeline.llm.structure_models import (
    LLMParseResult,
    NodeType,
    RelationshipKind,
)

logger = logging.getLogger(__name__)


def _validate_llm_parse_result(result: LLMParseResult) -> LLMParseResult:
    """Apply additional invariants beyond schema validation."""
    id_paths = [n.id_path for n in result.nodes]
    if len(id_paths) != len(set(id_paths)):
        raise ValueError("LLM structural parse produced duplicate id_path values.")

    for rel in result.relationships:
        if rel.kind is not RelationshipKind.CONTAINS:
            raise ValueError(f"Unsupported relationship kind from LLM: {rel.kind}")

    roots = [n for n in result.nodes if n.type is NodeType.DOCUMENT]
    if not roots:
        raise ValueError("LLM structural parse must include at least one DOCUMENT node.")

    if len(roots) > 1:
        logger.warning("LLM structural parse returned multiple DOCUMENT nodes; using all as-is.")

    return result


def parse_with_llm(text: str, settings: Settings) -> LLMParseResult:
    """Run the LLM structural parser and return a validated result."""
    raw: Dict[str, Any] = run_structural_parse(text, settings)
    try:
        parsed = LLMParseResult.model_validate(raw)
    except ValidationError as exc:
        logger.error("LLM structural parse failed validation: %s", exc)
        raise ValueError("Invalid LLM structural parse output") from exc
    return _validate_llm_parse_result(parsed)


def try_parse_with_llm(text: str, settings: Settings) -> Tuple[LLMParseResult | None, Exception | None]:
    """Attempt LLM parsing; returns (result, error). Falls back gracefully."""
    try:
        result = parse_with_llm(text, settings)
        return result, None
    except Exception as exc:
        logger.warning("LLM structural parsing failed; falling back to regex: %s", exc)
        return None, exc
