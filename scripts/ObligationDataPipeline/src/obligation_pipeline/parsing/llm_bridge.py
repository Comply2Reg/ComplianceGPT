"""Bridge from LLM structural parse models into existing parsing / ontology models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from src.obligation_pipeline.llm.structure_models import LLMParseResult, NodeType
from src.obligation_pipeline.parsing.hierarchy import LegalSegment, build_hierarchy_from_parsed


@dataclass
class LLMBridgeContext:
    """Context parameters needed when mapping LLM output to ontology models."""

    regulator: str
    act_name: str
    document_id: str
    source_path: str | None = None
    effective_date: str | None = None
    amendment_info: str | None = None


def _node_type_to_level(node_type: NodeType) -> str | None:
    """Map universal NodeType to the 'level' string used by LegalSegment."""
    if node_type is NodeType.DOCUMENT:
        return None
    if node_type in {NodeType.PREAMBLE, NodeType.RECITAL, NodeType.TITLE}:
        return "section"
    if node_type is NodeType.PART:
        return "part"
    if node_type is NodeType.CHAPTER:
        return "chapter"
    if node_type in {NodeType.SECTION, NodeType.ARTICLE}:
        return "section"
    if node_type in {NodeType.SUBSECTION, NodeType.PARAGRAPH, NodeType.POINT, NodeType.SUBPOINT}:
        return "clause"
    if node_type is NodeType.SCHEDULE:
        return "schedule"
    if node_type is NodeType.ANNEX:
        return "annex"
    if node_type is NodeType.DEFINITION:
        return "definition"
    return None


def _make_label(node_id_path: str, label: str | None, numbering: list[str] | None) -> str:
    if label:
        return label
    if numbering:
        return "(" + ")(".join(numbering) + ")"
    return node_id_path.rsplit("/", 1)[-1]


def llm_to_segments(result: LLMParseResult) -> List[LegalSegment]:
    """Convert LLMParseResult nodes into a flat list of LegalSegment instances."""
    nodes_by_id = {n.id_path: n for n in result.nodes}
    segments: List[LegalSegment] = []

    for node in result.nodes:
        level = _node_type_to_level(node.type)
        if level is None:
            continue
        label = _make_label(node.id_path, node.label, node.numbering)

        parent_labels: list[tuple[str, str]] = []
        parent_id = node.parent_id_path
        while parent_id:
            parent_node = nodes_by_id.get(parent_id)
            if not parent_node:
                break
            parent_level = _node_type_to_level(parent_node.type)
            if parent_level:
                parent_label = _make_label(
                    parent_node.id_path,
                    parent_node.label,
                    parent_node.numbering,
                )
                parent_labels.append((parent_level, parent_label))
            parent_id = parent_node.parent_id_path

        parent_labels.reverse()
        seg = LegalSegment(
            level=level,
            label=label,
            title=node.label,
            text=(node.text or "").strip(),
            start_offset=0,
            end_offset=0,
            page=node.page_start,
            parent_labels=parent_labels,
        )
        segments.append(seg)

    return segments


def llm_to_hierarchy_and_segments(
    result: LLMParseResult,
    ctx: LLMBridgeContext,
) -> Tuple[List[LegalSegment], "LegalHierarchy"]:
    """Build LegalHierarchy + segments from LLM output."""
    segments = llm_to_segments(result)
    hierarchy = build_hierarchy_from_parsed(
        segments,
        regulator=ctx.regulator,
        act_name=ctx.act_name,
        document_id=ctx.document_id,
        source_path=ctx.source_path,
        effective_date=ctx.effective_date,
        amendment_info=ctx.amendment_info,
    )
    return segments, hierarchy
