"""Build ontology nodes and edges from parsed legal hierarchy."""

import logging
from typing import Any, Dict, Optional

from src.obligation_pipeline.ontology.schema import EntityType, RelationType
from src.obligation_pipeline.ontology.models import LegalHierarchy, OntologyNode, OntologyEdge
from src.obligation_pipeline.ontology.structural_ids import slug as _slug
from src.obligation_pipeline.ontology.structural_ids import structural_id as _structural_id
from src.obligation_pipeline.parsing.hierarchy import LegalSegment

logger = logging.getLogger(__name__)


def build_nodes_and_edges(
    hierarchy: LegalHierarchy,
    segments: list[LegalSegment],
) -> tuple[list[OntologyNode], list[OntologyEdge]]:
    """From LegalHierarchy and parsed segments, build ontology nodes and edges."""
    nodes: list[OntologyNode] = []
    edges: list[OntologyEdge] = []
    act_id = hierarchy.act_id
    reg_id = f"reg/{_slug(hierarchy.regulator)}" if hierarchy.regulator else "reg/unknown"

    # Regulator node
    nodes.append(OntologyNode(id=reg_id, type=EntityType.REGULATOR, label=hierarchy.regulator, properties={"name": hierarchy.regulator}))
    # Act node
    nodes.append(OntologyNode(
        id=act_id, type=EntityType.ACT, label=hierarchy.act_name,
        properties={"name": hierarchy.act_name, "document_id": hierarchy.document_id, "source_path": hierarchy.source_path,
                     "effective_date": hierarchy.effective_date, "amendment_info": hierarchy.amendment_info},
    ))
    edges.append(OntologyEdge(act_id, reg_id, RelationType.ACT_ISSUED_BY_REGULATOR))
    edges.append(OntologyEdge(reg_id, act_id, RelationType.CONTAINS))
    if hierarchy.effective_date:
        ed_id = _structural_id(act_id, "effective_date", hierarchy.effective_date)
        nodes.append(OntologyNode(id=ed_id, type=EntityType.EFFECTIVE_DATE, label=hierarchy.effective_date, properties={"date": hierarchy.effective_date}))
        edges.append(OntologyEdge(act_id, ed_id, RelationType.ACT_HAS_EFFECTIVE_DATE))
    if hierarchy.source_path:
        doc_id = _structural_id(act_id, "source_document", hierarchy.source_path)
        nodes.append(OntologyNode(id=doc_id, type=EntityType.SOURCE_DOCUMENT, label=hierarchy.source_path, properties={"path": hierarchy.source_path}))
        edges.append(OntologyEdge(act_id, doc_id, RelationType.ACT_SOURCE_DOCUMENT))

    part_ids: Dict[str, str] = {}
    chapter_ids: Dict[str, str] = {}
    schedule_ids: Dict[str, str] = {}
    annex_ids: Dict[str, str] = {}

    for seg in segments:
        if seg.level == "part":
            pid = _structural_id(act_id, "part", seg.label)
            part_ids[seg.label] = pid
            nodes.append(OntologyNode(id=pid, type=EntityType.PART, label=seg.label, properties={"title": seg.title, "act_id": act_id}))
            edges.append(OntologyEdge(act_id, pid, RelationType.DOCUMENT_HAS_PART))
            edges.append(OntologyEdge(act_id, pid, RelationType.CONTAINS))
        elif seg.level == "chapter":
            parent_id = act_id
            for plvl, plbl in seg.parent_labels:
                if plvl == "part" and plbl in part_ids:
                    parent_id = part_ids[plbl]
                    break
            cid = _structural_id(act_id, "chapter", seg.label)
            chapter_ids[seg.label] = cid
            nodes.append(OntologyNode(id=cid, type=EntityType.CHAPTER, label=seg.label, properties={"title": seg.title, "act_id": act_id}))
            if parent_id != act_id:
                edges.append(OntologyEdge(parent_id, cid, RelationType.PART_HAS_CHAPTER))
            else:
                edges.append(OntologyEdge(act_id, cid, RelationType.ACT_HAS_CHAPTER))
            edges.append(OntologyEdge(parent_id, cid, RelationType.CONTAINS))
        elif seg.level == "schedule":
            sid = _structural_id(act_id, "schedule", seg.label)
            schedule_ids[seg.label] = sid
            nodes.append(OntologyNode(id=sid, type=EntityType.SCHEDULE, label=seg.label, properties={"title": seg.title, "act_id": act_id}))
            edges.append(OntologyEdge(act_id, sid, RelationType.ACT_HAS_SCHEDULE))
            edges.append(OntologyEdge(act_id, sid, RelationType.CONTAINS))

    section_ids: Dict[str, str] = {}
    for seg in segments:
        if seg.level == "section":
            parent_id = act_id
            for plvl, plbl in reversed(seg.parent_labels):
                if plvl == "schedule" and plbl in schedule_ids:
                    parent_id = schedule_ids[plbl]
                    break
                if plvl == "chapter" and plbl in chapter_ids:
                    if parent_id == act_id:
                        parent_id = chapter_ids[plbl]
                    break
            sid = _structural_id(act_id, "section", seg.label)
            section_ids[seg.label] = sid
            nodes.append(OntologyNode(
                id=sid, type=EntityType.SECTION, label=seg.label,
                properties={"title": seg.title, "text_preview": seg.text[:300] if seg.text else None,
                             "page": seg.page, "semantic_role": seg.semantic_role, "act_id": act_id},
            ))
            if parent_id != act_id:
                edges.append(OntologyEdge(parent_id, sid, RelationType.CHAPTER_HAS_SECTION))
            else:
                edges.append(OntologyEdge(act_id, sid, RelationType.ACT_HAS_SECTION))
            edges.append(OntologyEdge(parent_id, sid, RelationType.CONTAINS))
        elif seg.level in ("subsection", "clause"):
            parent_section = seg.section_number()
            pid = section_ids.get(parent_section or "", act_id)
            cid = _structural_id(act_id, "clause", parent_section or "root", seg.label)
            node_type = EntityType.SUBSECTION if seg.level == "subsection" else EntityType.CLAUSE
            rel_type = RelationType.SECTION_HAS_SUBSECTION if seg.level == "subsection" else RelationType.SECTION_HAS_CLAUSE
            nodes.append(OntologyNode(id=cid, type=node_type, label=seg.label,
                                      properties={"title": seg.title, "text_preview": seg.text[:200], "page": seg.page, "semantic_role": seg.semantic_role}))
            edges.append(OntologyEdge(pid, cid, rel_type))
            edges.append(OntologyEdge(pid, cid, RelationType.CONTAINS))
        elif seg.level == "definition":
            parent_section = seg.section_number()
            pid = section_ids.get(parent_section or "", act_id)
            did = _structural_id(act_id, "definition", parent_section or "root", seg.label)
            nodes.append(OntologyNode(id=did, type=EntityType.LEGAL_DEFINITION, label=seg.label, properties={"title": seg.title, "text_preview": seg.text[:200], "page": seg.page}))
            edges.append(OntologyEdge(pid, did, RelationType.SECTION_CONTAINS_DEFINITION))
            edges.append(OntologyEdge(pid, did, RelationType.CONTAINS))
        elif seg.level == "explanation":
            parent_section = seg.section_number()
            if parent_section and parent_section in section_ids:
                eid = _structural_id(act_id, "explanation", parent_section, seg.label)
                nodes.append(OntologyNode(id=eid, type=EntityType.EXPLANATION, label=seg.label, properties={"text_preview": seg.text[:200], "page": seg.page}))
                edges.append(OntologyEdge(section_ids[parent_section], eid, RelationType.SECTION_HAS_EXPLANATION))
                edges.append(OntologyEdge(section_ids[parent_section], eid, RelationType.CONTAINS))
        elif seg.level == "proviso":
            parent_section = seg.section_number()
            if parent_section and parent_section in section_ids:
                prid = _structural_id(act_id, "proviso", parent_section, seg.label or "proviso")
                nodes.append(OntologyNode(id=prid, type=EntityType.PROVISO, label="Proviso", properties={"text_preview": seg.text[:200], "page": seg.page}))
                edges.append(OntologyEdge(section_ids[parent_section], prid, RelationType.SECTION_HAS_PROVISO))
                edges.append(OntologyEdge(section_ids[parent_section], prid, RelationType.CONTAINS))
    return nodes, edges
