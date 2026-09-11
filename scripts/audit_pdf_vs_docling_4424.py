"""Audit-only: compare existing Stage-A Docling raw JSON vs 4424.pdf.

Does NOT regenerate Docling extraction.
Does NOT modify parsers, canonicalizers, chunkers, or source files.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "raw" / "4424.pdf"
RAW_JSON = ROOT / "out" / "audit_parse_quality" / "4424.docling.raw.json"
OUT_MD = ROOT / "out" / "audit_parse_quality" / "4424.pdf_vs_docling_audit.md"

WORD_JOIN_RE = re.compile(r"[a-z][A-Z]")
PUNCT_JOIN_RE = re.compile(r"[a-zA-Z]\d|\d[A-Za-z]{2,}")
DATE_TOKEN_RE = re.compile(
    r"^\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*$",
    re.I,
)
GLUED_DATE_RE = re.compile(
    r"[A-Za-z]{3,}\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    re.I,
)
REPLACEMENT_RE = re.compile(r"\uFFFD")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def load_raw() -> dict:
    return json.loads(RAW_JSON.read_text(encoding="utf-8"))


def pdf_page_texts() -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(PDF_PATH))
    return [(p.extract_text() or "") for p in reader.pages]


def significant_tokens(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower())
    stop = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "their",
        "which",
        "about",
        "into",
        "esma",
        "markets",
        "market",
        "authority",
        "european",
        "securities",
        "financial",
        "january",
        "february",
        "spotlight",
    }
    return {t for t in toks if t not in stop}


def bbox_top(block: dict) -> float:
    bbox = block.get("bbox") or {}
    try:
        return float(bbox.get("t") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def analyze_page(
    page_no: int,
    texts: list[dict],
    tables: list[dict],
    pdf_text: str,
) -> dict:
    labels = Counter((t.get("label") or "unknown") for t in texts)
    for tb in tables:
        labels[f"table:{tb.get('label') or 'table'}"] += 1

    n_blocks = len(texts) + len(tables)
    has_content = n_blocks > 0

    notes: list[str] = []
    flags = {
        "reading_order": "ok",
        "headings": "ok",
        "lists": "n/a",
        "tables_columns": "n/a",
        "dates_events": "n/a",
        "missing_or_dup": "ok",
        "headers_footers": "ok",
        "corruption": "ok",
        "whitespace_joins": "ok",
    }

    # --- headers/footers ---
    n_hdr = labels.get("page_header", 0)
    n_ftr = labels.get("page_footer", 0)
    if n_hdr or n_ftr:
        flags["headers_footers"] = "identifiable"
        notes.append(f"Headers/footers labeled: page_header={n_hdr}, page_footer={n_ftr}.")
    else:
        flags["headers_footers"] = "not_labeled"
        # still may contain chrome as plain text
        chrome = [
            t
            for t in texts
            if re.search(r"spotlight on markets", (t.get("text") or ""), re.I)
        ]
        if chrome:
            notes.append(
                "No page_header/page_footer labels, but Spotlight chrome appears as plain text."
            )
            flags["headers_footers"] = "present_as_text"

    # --- headings ---
    headers = [t for t in texts if t.get("label") == "section_header"]
    if headers:
        flags["headings"] = "present"
        notes.append(
            "section_header(s): "
            + "; ".join((h.get("text") or "")[:80] for h in headers[:4])
        )
    else:
        # cover/TOC may still be fine
        if page_no in (1, 2, 3):
            flags["headings"] = "sparse_or_via_table"
        else:
            flags["headings"] = "none_detected"
            notes.append("No section_header labels on this page.")

    # --- lists ---
    list_items = [t for t in texts if t.get("label") == "list_item"]
    if list_items:
        flags["lists"] = "preserved"
        notes.append(f"{len(list_items)} list_item block(s).")
        # check if a section_header appears after list_items (order issue)
        header_idxs = [t["order_index"] for t in headers]
        list_idxs = [t["order_index"] for t in list_items]
        if header_idxs and list_idxs:
            # Role and Responsibilities should precede its bullets
            role = [
                t
                for t in headers
                if "role and responsibilities" in (t.get("text") or "").lower()
            ]
            if role and list_idxs:
                if min(list_idxs) < role[0]["order_index"]:
                    flags["lists"] = "order_suspicious"
                    notes.append(
                        "MAJOR?: list_item appears before 'Role and Responsibilities' header by order_index."
                    )
    else:
        flags["lists"] = "none"

    # --- tables ---
    if tables:
        flags["tables_columns"] = "present"
        for tb in tables:
            n_cells = tb.get("n_cells") or len(tb.get("table_cells") or [])
            label = tb.get("label")
            notes.append(f"Table label={label!r}, cells={n_cells}.")
            cells = tb.get("table_cells") or []
            rows = {}
            for c in cells:
                rows.setdefault(c.get("start_row_offset_idx"), []).append(c)
            # check title/page-number style TOC rows
            if label == "document_index":
                flags["tables_columns"] = "toc_document_index"
                notes.append(
                    "TOC represented as document_index table (title/page columns)."
                )
                # sample row integrity
                bad_rows = 0
                for r, cols in rows.items():
                    texts_c = [(c.get("text") or "").strip() for c in cols]
                    if any(re.fullmatch(r"\d{1,3}", x) for x in texts_c) and any(
                        len(x) > 8 for x in texts_c
                    ):
                        continue
                    if texts_c:
                        bad_rows += 0
                notes.append(f"TOC table row groups: {len(rows)}.")
    else:
        # Events page often has no table
        if page_no >= 18:
            flags["tables_columns"] = "no_table_object"
            notes.append(
                "No Docling table object on this page (calendar may be free text)."
            )

    # --- dates / events association ---
    page_texts = [(t.get("text") or "").strip() for t in texts if (t.get("text") or "").strip()]
    date_blocks = [t for t in page_texts if DATE_TOKEN_RE.match(t)]
    glued = [t for t in page_texts if GLUED_DATE_RE.search(t)]
    if date_blocks:
        flags["dates_events"] = "dates_as_separate_blocks"
        notes.append(
            f"{len(date_blocks)} date-only block(s); association with titles requires later pairing."
        )
        # check if a date-only block is immediately adjacent in order to a title-like block
        orders = {t["order_index"]: t for t in texts}
        paired_hint = 0
        for t in texts:
            if DATE_TOKEN_RE.match((t.get("text") or "").strip()):
                prev = orders.get(t["order_index"] - 1)
                nxt = orders.get(t["order_index"] + 1)
                if prev and len((prev.get("text") or "")) > 10:
                    paired_hint += 1
                elif nxt and len((nxt.get("text") or "")) > 10:
                    paired_hint += 1
        notes.append(f"Date blocks with adjacent non-empty neighbor: {paired_hint}/{len(date_blocks)}.")
    if glued:
        flags["dates_events"] = "glued_date_in_text"
        notes.append(f"Suspicious glued date pattern in text: {glued[:2]!r}")

    # --- reading order via bbox tops (BOTTOMLEFT: higher t = higher on page) ---
    content_blocks = [
        t
        for t in texts
        if t.get("label") not in {"page_header", "page_footer"} and t.get("bbox")
    ]
    if len(content_blocks) >= 3:
        tops = [bbox_top(t) for t in content_blocks]
        # In reading order, tops should generally be non-increasing for BOTTOMLEFT
        inversions = sum(1 for i in range(1, len(tops)) if tops[i] > tops[i - 1] + 25)
        if inversions >= max(2, len(tops) // 4):
            flags["reading_order"] = "suspicious_vs_bbox"
            notes.append(
                f"Reading-order vs bbox: {inversions} upward jumps (possible column/layout reorder)."
            )
        else:
            flags["reading_order"] = "reasonable"
            notes.append(
                f"Reading-order vs bbox mostly top-down ({inversions} minor upward jumps)."
            )

    # Special: page 8 historically had title-after-body in pypdf
    if page_no == 8:
        title = next(
            (
                t
                for t in texts
                if t.get("label") == "section_header"
                and "reserve bank" in (t.get("text") or "").lower()
            ),
            None,
        )
        agreement = next(
            (
                t
                for t in texts
                if "this agreement marks" in (t.get("text") or "").lower()
            ),
            None,
        )
        if title and agreement:
            if title["order_index"] < agreement["order_index"]:
                notes.append(
                    "PASS signal: MoU title precedes 'This agreement marks...' in Docling order."
                )
            else:
                flags["reading_order"] = "misordered"
                notes.append(
                    "MAJOR: MoU title appears after agreement paragraph in Docling order."
                )

    # --- missing / duplicate vs PDF reference ---
    docling_blob = "\n".join(page_texts)
    for tb in tables:
        for c in tb.get("table_cells") or []:
            if c.get("text"):
                docling_blob += "\n" + c["text"]

    pdf_toks = significant_tokens(pdf_text)
    dl_toks = significant_tokens(docling_blob)
    if pdf_toks:
        missing = sorted(pdf_toks - dl_toks)
        # ignore tiny set noise
        miss_ratio = len(missing) / max(1, len(pdf_toks))
        if miss_ratio > 0.35 and len(pdf_text.strip()) > 80:
            flags["missing_or_dup"] = "possible_missing"
            notes.append(
                f"Token overlap low vs pypdf reference: ~{miss_ratio:.0%} PDF tokens absent "
                f"(sample missing: {missing[:8]})."
            )
        elif miss_ratio > 0.18:
            flags["missing_or_dup"] = "minor_gap"
            notes.append(
                f"Some PDF tokens not seen in Docling (~{miss_ratio:.0%}); may be layout/OCR variance."
            )

    # duplicates within page
    norm = [(re.sub(r"\s+", " ", (t.get("text") or "")).strip().lower()) for t in texts]
    norm = [n for n in norm if len(n) > 20]
    dup = [n for n, c in Counter(norm).items() if c > 1]
    if dup:
        flags["missing_or_dup"] = "duplicates"
        notes.append(f"Duplicated text blocks on page: {len(dup)} (e.g. {dup[0][:70]!r}).")

    if not has_content and pdf_text.strip():
        flags["missing_or_dup"] = "empty_extraction"
        notes.append("MAJOR: PDF page has text but Docling has zero blocks.")

    # --- corruption / joins ---
    joined = []
    for t in page_texts:
        if WORD_JOIN_RE.search(t) or PUNCT_JOIN_RE.search(t) or "betterinformed" in t.replace(" ", "").lower():
            if re.search(r"[a-z][A-Z]|betterinformed|[A-Za-z]{4}\d{1,2}\s*[A-Z][a-z]", t):
                joined.append(t[:80])
        if "  " in t and len(t) > 40:
            pass  # double spaces common in Docling; minor
    if any("\ufffd" in t or CONTROL_RE.search(t) for t in page_texts):
        flags["corruption"] = "control_or_replacement"
        notes.append("Replacement/control characters detected.")
    if joined:
        flags["whitespace_joins"] = "word_join_or_glue"
        notes.append(f"Word-join / glue examples: {joined[:3]!r}")
    # double-space density
    if sum(t.count("  ") for t in page_texts) >= 8:
        flags["whitespace_joins"] = "extra_internal_spaces"
        notes.append("Frequent internal double-spaces in extracted text (minor).")

    # --- page-type notes ---
    if page_no in (2, 3):
        notes.append("DIFFICULT PAGE TYPE: TOC / multi-column index.")
        if not tables:
            notes.append("MINOR/MAJOR?: TOC page without document_index table.")
    if page_no == 19:
        notes.append("DIFFICULT PAGE TYPE: Events + promo; dates may be separate blocks.")
    if any(t.get("label") == "list_item" for t in texts):
        notes.append("DIFFICULT PAGE TYPE: contains lists.")

    # continuation heuristic: page starts mid-sentence?
    first_content = next(
        (
            t
            for t in texts
            if t.get("label") not in {"page_header", "page_footer"}
            and (t.get("text") or "").strip()
        ),
        None,
    )
    if first_content:
        ft = (first_content.get("text") or "").strip()
        if ft and ft[0].islower():
            notes.append(
                "Page may continue previous article (starts with lowercase)."
            )

    # --- severity ---
    severity = "PASS"
    reasons = []
    if flags["missing_or_dup"] == "empty_extraction" or flags["reading_order"] == "misordered":
        severity = "MAJOR ISSUE"
        reasons.append(flags["missing_or_dup"] if flags["missing_or_dup"] == "empty_extraction" else "reading_order")
    elif flags["lists"] == "order_suspicious" or flags["dates_events"] == "glued_date_in_text":
        severity = "MAJOR ISSUE"
        reasons.append("list/date structure")
    elif flags["missing_or_dup"] in {"possible_missing", "duplicates"}:
        severity = "MINOR ISSUE"
        reasons.append(flags["missing_or_dup"])
    elif flags["reading_order"] == "suspicious_vs_bbox":
        severity = "MINOR ISSUE"
        reasons.append("bbox/order")
    elif flags["whitespace_joins"] != "ok" or flags["dates_events"] == "dates_as_separate_blocks":
        severity = "MINOR ISSUE"
        reasons.append("normalization later")
    elif flags["tables_columns"] == "no_table_object" and page_no == 19:
        severity = "MINOR ISSUE"
        reasons.append("events not as table object")
    elif not has_content:
        severity = "UNKNOWN"
        reasons.append("no blocks")
    elif flags["headings"] == "none_detected" and page_no not in (1, 20) and len(pdf_text) > 200:
        severity = "MINOR ISSUE"
        reasons.append("no section_header")

    # If almost no PDF text and no issues
    if severity == "PASS" and not pdf_text.strip() and not has_content:
        severity = "UNKNOWN"
        reasons.append("blank-looking page")

    return {
        "page_no": page_no,
        "has_content": has_content,
        "n_text_blocks": len(texts),
        "n_tables": len(tables),
        "n_blocks": n_blocks,
        "labels": dict(labels),
        "flags": flags,
        "notes": notes,
        "severity": severity,
        "reasons": reasons,
        "preview": [
            {
                "order": t.get("order_index"),
                "label": t.get("label"),
                "text": ((t.get("text") or "")[:100]),
            }
            for t in texts[:6]
        ],
    }


def render_report(pages: list[dict], raw: dict) -> str:
    lines: list[str] = []
    lines.append("# 4424 PDF vs raw Docling — Stage A quality audit")
    lines.append("")
    lines.append("**Audit only.** Uses existing `4424.docling.raw.json` and pypdf text as reference.")
    lines.append("Does not regenerate Docling extraction or modify any pipeline code.")
    lines.append("")
    lines.append(f"- PDF: `data/raw/4424.pdf`")
    lines.append(f"- Raw JSON: `out/audit_parse_quality/4424.docling.raw.json`")
    lines.append(
        f"- Raw counts: texts={raw.get('counts', {}).get('n_texts')}, "
        f"tables={raw.get('counts', {}).get('n_tables')}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for p in pages:
        lines.append(f"## Page {p['page_no']}")
        lines.append("")
        lines.append(f"- **Severity:** {p['severity']}")
        lines.append(f"- **Docling content present:** {'yes' if p['has_content'] else 'NO'}")
        lines.append(
            f"- **Blocks:** texts={p['n_text_blocks']}, tables={p['n_tables']} "
            f"(total={p['n_blocks']})"
        )
        lines.append(f"- **Labels:** `{p['labels']}`")
        lines.append("")
        lines.append("### Structure checks")
        for k, v in p["flags"].items():
            lines.append(f"- {k}: **{v}**")
        lines.append("")
        if p["notes"]:
            lines.append("### Notes")
            for n in p["notes"]:
                lines.append(f"- {n}")
            lines.append("")
        if p["preview"]:
            lines.append("### First Docling blocks (preview)")
            for b in p["preview"]:
                lines.append(
                    f"- `[order={b['order']}] [{b['label']}] {b['text']}`"
                )
            lines.append("")
        lines.append("---")
        lines.append("")

    # summary buckets
    buckets = {"PASS": [], "MINOR ISSUE": [], "MAJOR ISSUE": [], "UNKNOWN": []}
    for p in pages:
        buckets.setdefault(p["severity"], []).append(p["page_no"])

    lines.append("# Summary")
    lines.append("")
    lines.append("## A. PASS — clearly correct")
    lines.append(
        ", ".join(str(x) for x in buckets["PASS"]) if buckets["PASS"] else "_none_"
    )
    lines.append("")
    lines.append("## B. MINOR ISSUE — works but later normalization/layout handling needed")
    lines.append(
        ", ".join(str(x) for x in buckets["MINOR ISSUE"])
        if buckets["MINOR ISSUE"]
        else "_none_"
    )
    lines.append("")
    lines.append("## C. MAJOR ISSUE — important content/structure lost or misordered")
    lines.append(
        ", ".join(str(x) for x in buckets["MAJOR ISSUE"])
        if buckets["MAJOR ISSUE"]
        else "_none_"
    )
    lines.append("")
    lines.append("## D. UNKNOWN — cannot confidently determine automatically")
    lines.append(
        ", ".join(str(x) for x in buckets["UNKNOWN"])
        if buckets["UNKNOWN"]
        else "_none_"
    )
    lines.append("")
    lines.append("## Counts")
    lines.append(f"- Total pages audited: {len(pages)}")
    lines.append(f"- Pages with major issues: {len(buckets['MAJOR ISSUE'])}")
    lines.append(f"- Pages with minor issues: {len(buckets['MINOR ISSUE'])}")
    lines.append(f"- Pages with no obvious issues (PASS): {len(buckets['PASS'])}")
    lines.append(f"- Pages unknown: {len(buckets['UNKNOWN'])}")
    lines.append("")

    # recommendation
    if buckets["MAJOR ISSUE"]:
        rec = "FIX PARSER FIRST"
        rec_detail = (
            "Major extraction/order issues detected on one or more pages; "
            "review those pages before promoting Stage A."
        )
    elif len(buckets["MINOR ISSUE"]) >= 8:
        rec = "APPROVE RAW PARSING"
        rec_detail = (
            "No major losses/misorders detected. Remaining issues are mostly "
            "normalization/layout concerns for later stages (furniture removal, "
            "date pairing, whitespace)."
        )
    elif buckets["MINOR ISSUE"]:
        rec = "APPROVE RAW PARSING"
        rec_detail = (
            "Raw Docling extraction looks structurally faithful enough for Stage A. "
            "Minor issues should be handled downstream (classification/canonicalization), "
            "not by rejecting the raw parse."
        )
    else:
        rec = "APPROVE RAW PARSING"
        rec_detail = "No automatic major or minor structural failures detected."

    lines.append("## Overall recommendation")
    lines.append("")
    lines.append(f"**{rec}**")
    lines.append("")
    lines.append(rec_detail)
    lines.append("")
    lines.append(
        "_Note: Automatic audit cannot fully verify visual layout fidelity; "
        "TOC/Events/multi-column pages should still get a quick human PDF skim._"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not RAW_JSON.is_file():
        print(f"Missing raw JSON: {RAW_JSON}")
        return 1
    if not PDF_PATH.is_file():
        print(f"Missing PDF: {PDF_PATH}")
        return 1

    raw = load_raw()
    pdf_pages = pdf_page_texts()
    n_pdf = len(pdf_pages)

    texts = raw.get("texts") or []
    tables = raw.get("tables") or []

    by_page_texts: dict[int, list] = {}
    by_page_tables: dict[int, list] = {}
    for t in texts:
        p = t.get("page_no")
        if p is None:
            continue
        by_page_texts.setdefault(int(p), []).append(t)
    for tb in tables:
        p = tb.get("page_no")
        if p is None:
            continue
        by_page_tables.setdefault(int(p), []).append(tb)

    results = []
    for i in range(1, n_pdf + 1):
        results.append(
            analyze_page(
                i,
                sorted(by_page_texts.get(i, []), key=lambda x: x.get("order_index", 0)),
                by_page_tables.get(i, []),
                pdf_pages[i - 1],
            )
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_report(results, raw), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(
        "PASS",
        sum(1 for r in results if r["severity"] == "PASS"),
        "MINOR",
        sum(1 for r in results if r["severity"] == "MINOR ISSUE"),
        "MAJOR",
        sum(1 for r in results if r["severity"] == "MAJOR ISSUE"),
        "UNKNOWN",
        sum(1 for r in results if r["severity"] == "UNKNOWN"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
