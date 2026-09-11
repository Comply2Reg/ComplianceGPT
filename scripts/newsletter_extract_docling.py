"""Docling-based ESMA Spotlight newsletter extraction (format-aware, not page-hardcoded).

Produces source-derived plain text only — no artificial Markdown section markers.
Region/section identity is returned as metadata alongside the text.

Classification uses Docling labels + content/layout signals so the same logic
works across issues with different page counts (e.g. 20-page and 33-page PDFs).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


FURNITURE_LABELS = {
    "page_header",
    "page_footer",
    "footnote",
    "furniture",
}

TOC_LABELS = {
    "document_index",
    "toc",
    "table_of_contents",
}

RUNNING_HEADER_RE = re.compile(
    r"^spotlight on markets(?:\s*[–—-]\s*.*)?$",
    re.IGNORECASE,
)
ISSUE_DATE_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"(?:\s*&\s*(january|february|march|april|may|june|july|august|september|october|november|december))?"
    r"\s+\d{4}$",
    re.IGNORECASE,
)
PAGE_NUM_ONLY_RE = re.compile(r"^\d{1,3}$")
GETTY_CREDIT_RE = re.compile(r"gettyimages|^\u00a9\s*\w+", re.IGNORECASE)
EDITION_RE = re.compile(r"^edition$|^n[º°o]\s*\d+$", re.IGNORECASE)

# TOC entry: title text with a trailing page number (common in ESMA ToC).
TOC_ENTRY_RE = re.compile(r"^(.{12,200}?)\s+(\d{1,3})$")
# Glued TOC form produced by weak extractors: "12ESMA ..."
TOC_GLUED_RE = re.compile(r"^(\d{1,3})([A-Z].{8,})$")

PROMO_MARKERS = (
    "esma is on instagram",
    "@esmacomms",
    "empowering retail investors",
    "smart choices make for",
    "smarter investing",
    "brought this commitment to instagram",
    "practical tips, fun polls",
    "connecting with retail investors",
    "click on the event",
    "click on the consultation",
    "press contact information",
    "info@esma.europa.eu",
    "press@esma.europa.eu",
)

CONTACT_HEADINGS = {
    "contact info",
    "contact information",
    "press contact information",
    "social media",
}

TOC_HEADINGS = {
    "in this issue",
    "in this edition",
    "look ahead",
}

EVENTS_HEADINGS = {"events"}
CONSULTATIONS_HEADINGS = {"consultations", "consultation"}
CALENDAR_HEADINGS = {
    "consultations and events",
    "events and consultations",
    "events & consultations",
    "consultations & events",
}

# Newsletter subsections (not legislation CHAPTER/ARTICLE).
KNOWN_SUBSECTIONS = {
    "next steps",
    "background",
    "market developments",
    "role and responsibilities",
    "applications",
    "consultation process",
    "legal basis and background",
    "a strategic and multicultural working environment",
}

_DATE_TOKEN_RE = re.compile(
    r"^(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"|\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)$",
    re.IGNORECASE,
)


@dataclass
class TextBlock:
    text: str
    label: str
    page_no: Optional[int]
    self_ref: str
    level: Optional[int] = None
    bbox_top: float = 0.0
    bbox_left: float = 0.0


@dataclass
class TableBlock:
    page_no: Optional[int]
    self_ref: str
    label: str
    rows: list[list[str]]
    bbox_top: float = 0.0


@dataclass
class RegionSpan:
    region: str
    start: int
    end: int
    title: str = ""
    section: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class ExtractResult:
    canonical_text: str
    regions: list[RegionSpan] = field(default_factory=list)
    doc_hash: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2011", "-")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    return "\n".join(out).strip() + "\n"


def _page_no(obj: dict) -> Optional[int]:
    prov = obj.get("prov") or []
    if prov:
        return prov[0].get("page_no")
    return None


def _bbox(obj: dict) -> tuple[float, float]:
    prov = obj.get("prov") or []
    if prov:
        bbox = prov[0].get("bbox") or {}
        try:
            return float(bbox.get("t") or 0.0), float(bbox.get("l") or 0.0)
        except (TypeError, ValueError):
            return 0.0, 0.0
    return 0.0, 0.0


def _is_chrome_text(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if PAGE_NUM_ONLY_RE.match(t):
        return True
    if RUNNING_HEADER_RE.match(t):
        return True
    if ISSUE_DATE_RE.match(t):
        return True
    if GETTY_CREDIT_RE.search(t):
        return True
    if EDITION_RE.match(t):
        return True
    low = t.lower()
    if low in {"date", "deadline", "click on the event", "click on the consultation"}:
        return True
    return False


def _is_promo_text(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in PROMO_MARKERS)


def _is_contact_heading(text: str) -> bool:
    return text.strip().lower() in CONTACT_HEADINGS


def _is_toc_heading(text: str) -> bool:
    return text.strip().lower() in TOC_HEADINGS


def _is_toc_entry(text: str) -> bool:
    t = text.strip()
    if TOC_ENTRY_RE.match(t):
        return True
    if TOC_GLUED_RE.match(t):
        return True
    return False


def _is_events_heading(text: str) -> bool:
    return text.strip().lower() in EVENTS_HEADINGS


def _is_consultations_heading(text: str) -> bool:
    return text.strip().lower() in CONSULTATIONS_HEADINGS


def _is_calendar_heading(text: str) -> bool:
    low = text.strip().lower()
    if low in CALENDAR_HEADINGS:
        return True
    # Tolerate trailing page crumbs: "Consultations and events 32"
    return bool(re.match(r"^consultations and events\b", low) or re.match(r"^events and consultations\b", low))


def _route_calendar_line(line: str) -> str:
    """Return 'consultations' or 'events' for a dated calendar line."""
    low = line.lower()
    if "consultation" in low or "call for evidence" in low or "deadline" in low:
        return "consultations"
    return "events"


def _is_known_subsection(text: str) -> bool:
    return text.strip().lower() in KNOWN_SUBSECTIONS


def _looks_like_cover_teaser(text: str, label: str) -> bool:
    t = text.strip()
    low = t.lower()
    if low.startswith("in this edition"):
        return True
    if RUNNING_HEADER_RE.match(t):
        return True
    if EDITION_RE.match(t):
        return True
    # Short cover headlines before TOC
    if label == "section_header" and len(t) <= 90 and t.isupper():
        return True
    return False


def _looks_like_article_title(block: TextBlock) -> bool:
    """Conservative: Docling section_header that is not a known special heading."""
    t = block.text.strip()
    low = t.lower()
    if block.label != "section_header":
        return False
    if _is_toc_heading(t) or _is_events_heading(t) or _is_consultations_heading(t):
        return False
    if _is_calendar_heading(t):
        return False
    if _is_contact_heading(t) or _is_known_subsection(t):
        return False
    if _is_chrome_text(t) or _is_promo_text(t):
        return False
    if len(t) < 12 or len(t) > 220:
        return False
    return True


def _render_table_rows(table_obj: dict) -> list[list[str]]:
    data = table_obj.get("data") or {}
    cells = data.get("table_cells") or []
    if not cells:
        grid = data.get("grid")
        if isinstance(grid, list):
            rows = []
            for row in grid:
                if isinstance(row, list):
                    rows.append(
                        [
                            (c.get("text") if isinstance(c, dict) else str(c or "")).strip()
                            for c in row
                        ]
                    )
            return rows
        return []

    by_row: dict[int, dict[int, str]] = {}
    for cell in cells:
        r = int(cell.get("start_row_offset_idx") or 0)
        c = int(cell.get("start_col_offset_idx") or 0)
        by_row.setdefault(r, {})[c] = (cell.get("text") or "").strip()

    rows: list[list[str]] = []
    for r in sorted(by_row):
        cols = by_row[r]
        width = max(cols) + 1 if cols else 0
        rows.append([cols.get(i, "") for i in range(width)])
    return rows


def _format_dated_rows(rows: list[list[str]]) -> list[str]:
    """Deterministic lines: '<title> | <date_or_deadline>'."""
    lines: list[str] = []
    for row in rows:
        cells = [c.strip() for c in row if c and c.strip()]
        if not cells:
            continue
        joined_low = " ".join(cells).lower()
        if joined_low in {"event date", "events date", "date", "deadline", "consultation deadline"}:
            continue
        headerish = {x.lower() for x in cells}
        if headerish <= {"event", "events", "date", "deadline", "consultation", "consultations"}:
            continue
        if len(cells) >= 2:
            date_cell = cells[-1]
            title = " ".join(cells[:-1]).strip()
            if _DATE_TOKEN_RE.match(date_cell) or (
                re.search(r"\d", date_cell) and len(date_cell) <= 48
            ):
                lines.append(f"{title} | {date_cell}")
            else:
                # Sometimes date is first column
                if _DATE_TOKEN_RE.match(cells[0]) or (
                    re.search(r"\d", cells[0]) and len(cells[0]) <= 48
                ):
                    lines.append(f"{' '.join(cells[1:]).strip()} | {cells[0]}")
                else:
                    lines.append(" | ".join(cells))
        else:
            only = cells[0]
            m = re.match(
                r"^(.*?)(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)$",
                only,
                re.IGNORECASE,
            )
            if m and m.group(1).strip():
                lines.append(f"{m.group(1).strip()} | {m.group(2).strip()}")
            else:
                lines.append(only)
    return lines


def _pair_dated_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if not cur or _is_promo_text(cur):
            i += 1
            continue
        if " | " in cur:
            out.append(cur)
            i += 1
            continue
        if i + 1 < len(lines) and _DATE_TOKEN_RE.match(lines[i + 1].strip()):
            out.append(f"{cur} | {lines[i + 1].strip()}")
            i += 2
            continue
        if _DATE_TOKEN_RE.match(cur):
            i += 1
            continue
        out.append(cur)
        i += 1
    return out


def convert_pdf_to_dict(pdf_path: Path) -> dict:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
        TableFormerMode,
    )

    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE,
        ),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(str(pdf_path))
    try:
        from hierarchical.postprocessor import ResultPostprocessor

        ResultPostprocessor(result).process()
    except Exception:
        pass
    return result.document.export_to_dict()


def convert_pdf_to_dict_with_fallback(pdf_path: Path) -> tuple[dict, str]:
    """Return (doc_dict, extractor_name). Falls back to a minimal pypdf pseudo-doc."""
    try:
        return convert_pdf_to_dict(pdf_path), "docling"
    except Exception as exc:
        # Minimal fallback structure so callers still get plain text (degraded).
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        texts = []
        for i, page in enumerate(reader.pages, start=1):
            raw = (page.extract_text() or "").strip()
            if not raw:
                continue
            texts.append(
                {
                    "self_ref": f"#/texts/{i}",
                    "label": "text",
                    "text": raw,
                    "prov": [{"page_no": i, "bbox": {"l": 0, "t": 0, "r": 0, "b": 0}}],
                }
            )
        return {"texts": texts, "tables": [], "body": {"children": []}}, f"pypdf_fallback:{type(exc).__name__}"


def _collect_blocks(doc: dict) -> tuple[list[TextBlock], list[TableBlock]]:
    texts: list[TextBlock] = []
    for t in doc.get("texts") or []:
        label = (t.get("label") or "text").lower()
        text = (t.get("text") or "").strip()
        if not text:
            continue
        top, left = _bbox(t)
        texts.append(
            TextBlock(
                text=text,
                label=label,
                page_no=_page_no(t),
                self_ref=str(t.get("self_ref") or ""),
                level=t.get("level"),
                bbox_top=top,
                bbox_left=left,
            )
        )

    tables: list[TableBlock] = []
    for tb in doc.get("tables") or []:
        label = (tb.get("label") or "table").lower()
        top, _ = _bbox(tb)
        tables.append(
            TableBlock(
                page_no=_page_no(tb),
                self_ref=str(tb.get("self_ref") or ""),
                label=label,
                rows=_render_table_rows(tb),
                bbox_top=top,
            )
        )
    return texts, tables


def _table_bboxes(doc: dict) -> list[tuple[Optional[int], float, float, float, float]]:
    boxes = []
    for tb in doc.get("tables") or []:
        prov = tb.get("prov") or []
        if not prov:
            continue
        bbox = prov[0].get("bbox") or {}
        try:
            boxes.append(
                (
                    prov[0].get("page_no"),
                    float(bbox.get("l", 0)),
                    float(bbox.get("t", 0)),
                    float(bbox.get("r", 0)),
                    float(bbox.get("b", 0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return boxes


def _inside_table(block: TextBlock, boxes: list[tuple]) -> bool:
    if block.page_no is None:
        return False
    for page, l, t, r, b in boxes:
        if page != block.page_no:
            continue
        if l - 2 <= block.bbox_left <= r + 2 and min(t, b) - 2 <= block.bbox_top <= max(t, b) + 2:
            return True
    return False


def _frequent_running_lines(texts: list[TextBlock], min_count: int = 4) -> set[str]:
    """Conservative frequency filter for repeated headers/footers."""
    counts: Counter[str] = Counter()
    for b in texts:
        t = b.text.strip()
        if not t or len(t) > 80:
            continue
        if b.label in FURNITURE_LABELS or PAGE_NUM_ONLY_RE.match(t):
            counts[t] += 1
            continue
        if RUNNING_HEADER_RE.match(t) or ISSUE_DATE_RE.match(t):
            counts[t] += 1
    return {t for t, n in counts.items() if n >= min_count}


def _dedupe(seq: list[str]) -> list[str]:
    out: list[str] = []
    for s in seq:
        s = s.strip()
        if not s:
            continue
        if out and out[-1] == s:
            continue
        out.append(s)
    return out


def _append_segment(
    segments: list[tuple[str, str, str, Optional[int], Optional[int]]],
    region: str,
    title: str,
    parts: list[str],
    pages: list[int],
) -> None:
    text = "\n".join(_dedupe(parts)).strip()
    if not text:
        return
    p0 = min(pages) if pages else None
    p1 = max(pages) if pages else None
    segments.append((region, title, text, p0, p1))


def extract_newsletter(pdf_path: Path) -> ExtractResult:
    doc, extractor = convert_pdf_to_dict_with_fallback(pdf_path)
    texts, tables = _collect_blocks(doc)
    boxes = _table_bboxes(doc)
    frequent = _frequent_running_lines(texts)

    filtered: list[TextBlock] = []
    for b in texts:
        if b.label in FURNITURE_LABELS:
            continue
        if b.text.strip() in frequent:
            continue
        if _is_chrome_text(b.text):
            continue
        if b.label in TOC_LABELS:
            filtered.append(b)
            continue
        if b.label != "section_header" and _inside_table(b, boxes):
            continue
        filtered.append(b)
    texts = filtered

    texts.sort(key=lambda b: (b.page_no or 0, -b.bbox_top, b.bbox_left))
    tables.sort(key=lambda t: (t.page_no or 0, -t.bbox_top))

    # Streaming modes — no fixed page numbers.
    mode = "cover"  # cover -> toc -> article -> events/consultations/contact
    toc_parts: list[str] = []
    toc_pages: list[int] = []
    cover_parts: list[str] = []
    events_parts: list[str] = []
    events_pages: list[int] = []
    consultations_parts: list[str] = []
    consultations_pages: list[int] = []
    contact_skipped = 0
    promo_skipped = 0

    article_regions: list[dict[str, Any]] = []
    current_title = ""
    current_section = "body"
    article_buf: list[str] = []
    article_pages: list[int] = []
    saw_toc = False
    article_count = 0

    def flush_article() -> None:
        nonlocal article_buf, current_title, current_section, article_pages, article_count
        text = "\n".join(article_buf).strip()
        if text:
            article_count += 1
            article_regions.append(
                {
                    "region": "article",
                    "title": current_title,
                    "section": current_title or current_section or "article",
                    "text": text,
                    "page_start": min(article_pages) if article_pages else None,
                    "page_end": max(article_pages) if article_pages else None,
                }
            )
        article_buf = []
        article_pages = []

    def add_article_line(b: TextBlock) -> None:
        article_buf.append(b.text)
        if b.page_no:
            article_pages.append(b.page_no)

    for b in texts:
        t = b.text.strip()
        low = t.lower()

        if _is_promo_text(t):
            promo_skipped += 1
            continue

        # Mode transitions by content/heading signals.
        # Do not treat TOC index lines (e.g. "Consultations and events 32") as
        # section starts — only transition after leaving cover/toc.
        if mode not in {"toc", "cover"}:
            if _is_events_heading(t):
                flush_article()
                mode = "events"
                continue

            if _is_calendar_heading(t):
                flush_article()
                mode = "calendar"
                continue

            if _is_consultations_heading(t):
                flush_article()
                mode = "consultations"
                continue

            if _is_contact_heading(t):
                flush_article()
                mode = "contact"
                contact_skipped += 1
                continue

        if mode == "contact":
            contact_skipped += 1
            continue

        if mode == "events":
            events_parts.append(t)
            if b.page_no:
                events_pages.append(b.page_no)
            continue

        if mode == "consultations":
            consultations_parts.append(t)
            if b.page_no:
                consultations_pages.append(b.page_no)
            continue

        if mode == "calendar":
            dest = _route_calendar_line(t)
            if dest == "consultations":
                consultations_parts.append(t)
                if b.page_no:
                    consultations_pages.append(b.page_no)
            else:
                events_parts.append(t)
                if b.page_no:
                    events_pages.append(b.page_no)
            continue

        # TOC label / heading can start (or continue) TOC from cover
        if b.label in TOC_LABELS or _is_toc_heading(t):
            if mode == "article":
                flush_article()
            mode = "toc"
            saw_toc = True
            toc_parts.append(t)
            if b.page_no:
                toc_pages.append(b.page_no)
            continue

        if mode == "cover":
            # Enter TOC on first TOC-like signal; else stay on cover until article/TOC
            if _is_toc_entry(t) or _is_toc_heading(t):
                mode = "toc"
                saw_toc = True
                toc_parts.append(t)
                if b.page_no:
                    toc_pages.append(b.page_no)
                continue
            if _looks_like_cover_teaser(t, b.label) and not saw_toc and article_count == 0:
                cover_parts.append(t)
                continue
            # First real article title after cover (some issues have thin TOC)
            if _looks_like_article_title(b):
                mode = "article"
                current_title = t
                current_section = t
                add_article_line(b)
                continue
            # Ignore residual cover noise
            if _looks_like_cover_teaser(t, b.label):
                cover_parts.append(t)
                continue
            # Fall through to article body only after TOC seen or title started
            if saw_toc:
                mode = "article"
            else:
                cover_parts.append(t)
                continue

        if mode == "toc":
            if _is_toc_entry(t) or _is_toc_heading(t) or b.label in TOC_LABELS:
                toc_parts.append(t)
                if b.page_no:
                    toc_pages.append(b.page_no)
                continue
            # Leaving TOC: first article title (or substantial non-entry header)
            if _looks_like_article_title(b):
                mode = "article"
                current_title = t
                current_section = t
                add_article_line(b)
                continue
            # Short TOC leftovers
            if len(t) < 80 and not t.endswith("."):
                toc_parts.append(t)
                if b.page_no:
                    toc_pages.append(b.page_no)
                continue
            # Otherwise start article stream without a detected title
            mode = "article"
            add_article_line(b)
            continue

        # mode == article (and multi-page continuation: page change does not flush)
        if _looks_like_article_title(b):
            flush_article()
            current_title = t
            current_section = t
            add_article_line(b)
            continue

        if _is_known_subsection(t) or (
            b.label == "section_header" and _is_known_subsection(t)
        ):
            add_article_line(b)
            current_section = t
            continue

        # list_item / quote / body — keep reading order
        add_article_line(b)

    flush_article()

    # Assign tables using labels + nearest mode pages (not hardcoded page indexes)
    events_page_set = set(events_pages)
    consultations_page_set = set(consultations_pages)
    toc_page_set = set(toc_pages)

    for tb in tables:
        if tb.label in TOC_LABELS:
            for row in tb.rows:
                line = " ".join(c for c in row if c).strip()
                if line and not _is_chrome_text(line) and not _is_promo_text(line):
                    toc_parts.append(line)
                    if tb.page_no:
                        toc_pages.append(tb.page_no)
            continue

        page = tb.page_no
        dated = _format_dated_rows(tb.rows)
        dated = [x for x in dated if not _is_promo_text(x) and not _is_chrome_text(x)]
        if not dated:
            continue

        # Prefer events/consultations when the table is on pages where those
        # sections were detected via headings (not via hardcoded page indexes).
        if page is not None and page in events_page_set and page not in consultations_page_set:
            events_parts.extend(dated)
            events_pages.append(page)
            continue
        if page is not None and page in consultations_page_set and page not in events_page_set:
            consultations_parts.extend(dated)
            consultations_pages.append(page)
            continue
        if page is not None and page in events_page_set and page in consultations_page_set:
            for line in dated:
                if _route_calendar_line(line) == "consultations":
                    consultations_parts.append(line)
                else:
                    events_parts.append(line)
            continue

        # Header-row / content inference when heading page not yet linked
        flat = " ".join(" ".join(r) for r in tb.rows).lower()
        has_consultation = "deadline" in flat or "consultation" in flat
        dateish = sum(
            1
            for r in tb.rows
            for c in r
            if c
            and (
                _DATE_TOKEN_RE.match(c.strip())
                or re.search(r"\b\d{1,2}\s*[A-Za-z]{3}\b", c)
            )
        )
        has_eventish = "event" in flat or "webinar" in flat or "workshop" in flat or "conference" in flat

        if has_consultation and (has_eventish or dateish >= 2):
            # Mixed consultations/events table — route each line by content.
            for line in dated:
                if _route_calendar_line(line) == "consultations":
                    consultations_parts.append(line)
                    if page:
                        consultations_pages.append(page)
                else:
                    events_parts.append(line)
                    if page:
                        events_pages.append(page)
            continue

        if has_consultation:
            consultations_parts.extend(dated)
            if page:
                consultations_pages.append(page)
            continue

        if has_eventish or dateish >= 2:
            if events_page_set or has_eventish:
                events_parts.extend(dated)
                if page:
                    events_pages.append(page)
                continue

        # Otherwise attach as article table lines (deterministic)
        line_block = "\n".join(dated)
        article_regions.append(
            {
                "region": "article",
                "title": "",
                "section": "table",
                "text": line_block,
                "page_start": page,
                "page_end": page,
            }
        )

    segments: list[tuple[str, str, str, Optional[int], Optional[int]]] = []
    _append_segment(segments, "toc", "In this issue", toc_parts, toc_pages)

    for r in article_regions:
        if r["text"].strip():
            segments.append(
                (
                    "article",
                    r.get("title") or "",
                    r["text"],
                    r.get("page_start"),
                    r.get("page_end"),
                )
            )

    events_clean = _pair_dated_lines([e for e in events_parts if not _is_promo_text(e)])
    _append_segment(segments, "events", "Events", events_clean, events_pages)

    consultations_clean = _pair_dated_lines(
        [c for c in consultations_parts if not _is_promo_text(c)]
    )
    _append_segment(
        segments,
        "consultations",
        "Consultations",
        consultations_clean,
        consultations_pages,
    )

    # Build canonical + region spans on normalized text
    pieces_n: list[str] = []
    regions_n: list[RegionSpan] = []
    cursor = 0
    for region, title, text, p0, p1 in segments:
        text_n = normalize_whitespace(text).strip()
        if not text_n:
            continue
        if pieces_n:
            sep = "\n\n"
            pieces_n.append(sep)
            cursor += len(sep)
        start = cursor
        pieces_n.append(text_n)
        cursor += len(text_n)
        regions_n.append(
            RegionSpan(
                region=region,
                start=start,
                end=cursor,
                title=title,
                section=title or region,
                page_start=p0,
                page_end=p1,
            )
        )
    canonical = "".join(pieces_n)
    if canonical and not canonical.endswith("\n"):
        canonical += "\n"
        if regions_n:
            last = regions_n[-1]
            regions_n[-1] = RegionSpan(
                region=last.region,
                start=last.start,
                end=len(canonical),
                title=last.title,
                section=last.section,
                page_start=last.page_start,
                page_end=last.page_end,
            )

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    stats = {
        "extractor": extractor,
        "n_text_blocks": len(texts),
        "n_tables": len(tables),
        "n_toc_lines": len(toc_parts),
        "n_article_regions": sum(1 for r in regions_n if r.region == "article"),
        "n_event_lines": len(events_clean),
        "n_consultation_lines": len(consultations_clean),
        "canonical_chars": len(canonical),
        "cover_lines_skipped": len(cover_parts),
        "promo_skipped": promo_skipped,
        "contact_skipped": contact_skipped,
        "saw_toc": saw_toc,
        "frequent_chrome_lines": len(frequent),
    }
    return ExtractResult(
        canonical_text=canonical,
        regions=regions_n,
        doc_hash=digest,
        stats=stats,
    )


_SUBSECTION_SPLIT_RE = re.compile(
    r"(?m)^(Next steps|Background|Market developments|Role and Responsibilities|"
    r"Applications|Consultation process|Legal basis and background)\s*$"
)


def regions_to_chunks(
    doc_id: int,
    extract: ExtractResult,
    source_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Build offset-valid chunks from region spans; split articles on subsections."""
    text = extract.canonical_text
    chunks: list[dict[str, Any]] = []

    for i, reg in enumerate(extract.regions):
        span = text[reg.start : reg.end]

        if reg.region in {"toc", "events", "consultations"}:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}#{reg.region}-{i+1}",
                    "doc_id": doc_id,
                    "region": reg.region,
                    "section": reg.title or reg.region,
                    "text": span,
                    "start": reg.start,
                    "end": reg.end,
                    "source_url": source_url,
                    "doc_hash": extract.doc_hash,
                    "page_start": reg.page_start,
                    "page_end": reg.page_end,
                }
            )
            continue

        matches = list(_SUBSECTION_SPLIT_RE.finditer(span))
        if not matches:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}#article-{i+1}",
                    "doc_id": doc_id,
                    "region": "article",
                    "section": reg.title or reg.section or "article",
                    "text": span,
                    "start": reg.start,
                    "end": reg.end,
                    "source_url": source_url,
                    "doc_hash": extract.doc_hash,
                    "page_start": reg.page_start,
                    "page_end": reg.page_end,
                }
            )
            continue

        boundaries = [0] + [m.start() for m in matches] + [len(span)]
        titles = [reg.title or "article"] + [m.group(1) for m in matches]
        for j in range(len(boundaries) - 1):
            a, b = boundaries[j], boundaries[j + 1]
            if a == b:
                continue
            piece = span[a:b]
            if not piece.strip():
                continue
            start = reg.start + a
            end = reg.start + b
            chunks.append(
                {
                    "chunk_id": f"{doc_id}#article-{i+1}-{j+1}",
                    "doc_id": doc_id,
                    "region": "article",
                    "section": titles[j],
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "source_url": source_url,
                    "doc_hash": extract.doc_hash,
                    "page_start": reg.page_start,
                    "page_end": reg.page_end,
                }
            )

    for c in chunks:
        if text[c["start"] : c["end"]] != c["text"]:
            raise AssertionError(
                f"Offset validation failed for {c['chunk_id']}: "
                f"start={c['start']} end={c['end']}"
            )
    return chunks
