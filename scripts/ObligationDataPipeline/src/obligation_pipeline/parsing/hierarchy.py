"""
Parse extracted PDF text into legal hierarchy segments.

Produces LegalSegment list aligned with ontology: act, chapter, section, subsection, clause, explanation, proviso, definition.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.obligation_pipeline.ontology.schema import SEMANTIC_ROLES

logger = logging.getLogger(__name__)

try:
    from src.obligation_pipeline.chunking.extractor import ExtractedDocument
except ImportError:
    ExtractedDocument = Any  # type: ignore


@dataclass
class LegalSegment:
    """One segment in the legal hierarchy (section, clause, explanation, etc.)."""

    level: str  # part, chapter, section, article, subsection, clause, explanation, proviso, definition, schedule, annex
    label: str
    title: Optional[str] = None
    text: str = ""
    start_offset: int = 0
    end_offset: int = 0
    page: Optional[int] = None
    parent_labels: list[tuple[str, str]] = field(default_factory=list)
    semantic_role: str = "other"

    def hierarchy_path(self) -> str:
        """E.g. Part I > Chapter 2 > Section 5 > Clause (a)."""
        parts = [f"{p[0].capitalize()} {p[1]}" for p in self.parent_labels] + [f"{self.level.capitalize()} {self.label}"]
        return " > ".join(parts)

    def section_number(self) -> Optional[str]:
        """Section/article number: this segment's section/article or the parent one."""
        if self.level in ("section", "article"):
            return self.label
        for lev, lbl in self.parent_labels:
            if lev in ("section", "article"):
                return lbl
        return None

    def clause_number(self) -> Optional[str]:
        """Clause label if this is a clause or under a clause."""
        if self.level == "clause":
            return self.label
        for lev, lbl in reversed(self.parent_labels):
            if lev == "clause":
                return lbl
        return None


def _normalize_semantic_role(role: str) -> str:
    r = (role or "").strip().lower()
    return r if r in SEMANTIC_ROLES else "other"


def _detect_semantic_role(segment: "LegalSegment") -> str:
    """Heuristic: infer semantic role from text and level."""
    t = (segment.text or "")[:500].lower()
    if segment.level == "definition":
        return "definition"
    if segment.level == "explanation":
        return "explanation"
    if segment.level == "proviso":
        return "proviso"
    if segment.level == "schedule":
        return "schedule"
    if "shall be punishable" in t or "punishment" in t or "penalty" in t or "imprisonment" in t:
        return "penalty"
    if "shall " in t and ("not " in t or "no " in t):
        return "prohibition"
    if "shall " in t or "must " in t:
        return "obligation"
    if "means " in t or "definition" in t:
        return "definition"
    if "provided that" in t or "proviso" in t:
        return "proviso"
    if "procedure" in t or "apply" in t or "application" in t:
        return "procedure"
    if "power" in t or "authority" in t or "may " in t:
        return "authority"
    if "exception" in t or "except" in t:
        return "exception"
    return "other"


SECTION_LEVELS = ("part", "chapter", "section", "article", "schedule")
CLAUSE_LEVELS = ("subsection", "clause", "explanation", "proviso", "definition")


def parse_legal_hierarchy(
    full_text: str,
    get_page_at_offset: Optional[Any] = None,
    patterns: Optional[dict[str, list]] = None,
) -> list[LegalSegment]:
    """
    Parse full document text into legal segments with hierarchy.

    Uses regex patterns for part, chapter, section, subsection, clause, explanation, proviso, definition.
    Returns list of LegalSegment ordered by start_offset, with parent_labels set.
    """
    from src.obligation_pipeline.parsing.patterns import LEGAL_PATTERNS

    if patterns is None:
        patterns = LEGAL_PATTERNS
    text = full_text
    segments: list[LegalSegment] = []
    seen: set[tuple[str, str, int]] = set()

    def page_at(offset: int) -> Optional[int]:
        if get_page_at_offset is not None and callable(get_page_at_offset):
            return get_page_at_offset(offset)
        return None

    raw: list[tuple[str, str, Optional[str], int, int]] = []
    for level, config_list in patterns.items():
        for cfg in config_list:
            re_obj = cfg.get("re") or cfg.get("pattern")
            if not re_obj:
                continue
            label_grp = cfg.get("label_grp", 1)
            title_grp = cfg.get("title_grp", 2)
            for m in re_obj.finditer(text):
                label = m.group(label_grp).strip() if label_grp and m.lastindex >= label_grp else ""
                title = m.group(title_grp).strip() if title_grp and m.lastindex >= title_grp else None
                if level == "proviso":
                    label = "Proviso"
                    if m.lastindex >= 1 and m.group(1):
                        title = m.group(1).strip()[:300]
                key = (level, label, m.start())
                if key in seen:
                    continue
                seen.add(key)
                raw.append((level, label or level, title, m.start(), m.end()))

    raw.sort(key=lambda x: (x[3], x[0]))
    for i in range(len(raw)):
        start = raw[i][3]
        end = raw[i][4]
        if i + 1 < len(raw):
            end = raw[i + 1][3]
        else:
            end = len(text)
        level, label, title, _, _ = raw[i]
        segment_text = text[start:end].strip()
        parent_labels: list[tuple[str, str]] = []
        for j in range(i):
            pl, plbl, _, _, _ = raw[j]
            if pl in SECTION_LEVELS or pl in CLAUSE_LEVELS or pl == "part":
                parent_labels.append((pl, plbl))
        seg = LegalSegment(
            level=level,
            label=label,
            title=title,
            text=segment_text,
            start_offset=start,
            end_offset=end,
            page=page_at(start),
            parent_labels=parent_labels.copy(),
        )
        seg.semantic_role = _normalize_semantic_role(_detect_semantic_role(seg))
        segments.append(seg)
    return segments


def build_hierarchy_from_parsed(
    segments: list[LegalSegment],
    regulator: str,
    act_name: str,
    document_id: str,
    source_path: Optional[str] = None,
    effective_date: Optional[str] = None,
    amendment_info: Optional[str] = None,
) -> "LegalHierarchy":
    """Build LegalHierarchy from parsed segments for ontology/graph."""
    from src.obligation_pipeline.ontology.models import LegalHierarchy

    chapters: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    clauses: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    for s in segments:
        if s.level == "chapter":
            chapters.append({"label": s.label, "title": s.title, "text": s.text[:500], "page": s.page})
        elif s.level == "section":
            sections.append({
                "label": s.label, "title": s.title, "text": s.text[:1000],
                "page": s.page, "semantic_role": s.semantic_role, "hierarchy_path": s.hierarchy_path(),
            })
        elif s.level == "clause" or s.level == "subsection":
            clauses.append({
                "label": s.label, "title": s.title, "text": s.text[:500],
                "page": s.page, "section_number": s.section_number(), "semantic_role": s.semantic_role,
            })
        elif s.level == "definition":
            definitions.append({"label": s.label, "title": s.title, "text": s.text[:500], "page": s.page})
    return LegalHierarchy(
        regulator=regulator,
        act_id=document_id,
        act_name=act_name,
        document_id=document_id,
        source_path=source_path,
        effective_date=effective_date,
        amendment_info=amendment_info,
        chapters=chapters,
        sections=sections,
        clauses=clauses,
        definitions=definitions,
    )
