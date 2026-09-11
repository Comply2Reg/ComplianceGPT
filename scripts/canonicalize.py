"""Convert raw documents under data/raw/ into deterministic canonical text.

Usage:
  python scripts/canonicalize.py --ids 4424
  python scripts/canonicalize.py
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from lxml import etree, html as lxml_html

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CANONICAL_DIR, RAW_DIR, ensure_data_dirs, get_source_selection, load_env

# Tags whose text is typically navigation / chrome in HTML pages.
_HTML_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "svg",
}


def normalize_whitespace(text: str) -> str:
    """Deterministic whitespace normalization without paraphrasing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    # Collapse spaces/tabs within lines; preserve blank-line paragraph breaks.
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    # Collapse 3+ blank lines to a single blank line.
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


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "pypdf is required for PDF canonicalization. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        parts.append(page_text)
    return normalize_whitespace("\n\n".join(parts))


def extract_html(path: Path) -> str:
    raw = path.read_bytes()
    # Prefer encoding from meta/declared; fall back utf-8.
    doc = lxml_html.fromstring(raw)
    for tag in list(doc.iter()):
        if isinstance(tag.tag, str) and tag.tag.lower() in _HTML_DROP_TAGS:
            parent = tag.getparent()
            if parent is not None:
                parent.remove(tag)

    # Prefer <main> / <article> if present.
    bodies = doc.xpath("//main|//article")
    root = bodies[0] if bodies else doc

    texts: list[str] = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "pre"}:
            t = " ".join(el.text_content().split())
            if t:
                texts.append(t)
        elif tag == "br":
            texts.append("")

    if not texts:
        # Fallback: whole body text
        body = doc.find(".//body")
        target = body if body is not None else doc
        texts = [" ".join(target.text_content().split())]

    return normalize_whitespace("\n\n".join(texts))


def extract_xml(path: Path) -> str:
    tree = etree.parse(str(path))
    texts = [
        " ".join(t.split())
        for t in tree.xpath("//text()")
        if t and t.strip()
    ]
    return normalize_whitespace("\n\n".join(texts))


def extract_txt(path: Path) -> str:
    return normalize_whitespace(path.read_text(encoding="utf-8", errors="replace"))


def detect_and_extract(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf", extract_pdf(path)
    if ext in {".html", ".htm"}:
        return "html", extract_html(path)
    if ext == ".xml":
        return "xml", extract_xml(path)
    if ext in {".txt", ".md"}:
        return "text", extract_txt(path)
    raise RuntimeError(
        f"Unsupported or unknown format for {path.name} (ext={ext!r}). "
        "Add deliberate handling before processing."
    )


def find_raw_for_id(doc_id: int) -> Path:
    matches = sorted(RAW_DIR.glob(f"{doc_id}.*"))
    matches = [p for p in matches if p.is_file() and p.name != "manifest.json"]
    if not matches:
        raise FileNotFoundError(f"No raw file for id={doc_id} under {RAW_DIR}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple raw files for id={doc_id}: {[p.name for p in matches]}"
        )
    return matches[0]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_one(doc_id: int, *, force: bool) -> Path:
    raw_path = find_raw_for_id(doc_id)
    out_path = CANONICAL_DIR / f"{doc_id}.txt"
    if out_path.exists() and not force:
        print(f"id={doc_id}: skip existing {out_path.name}")
        return out_path

    fmt, text = detect_and_extract(raw_path)
    if not text.strip():
        raise RuntimeError(f"id={doc_id}: canonical text empty after {fmt} extraction")

    out_path.write_text(text, encoding="utf-8")
    digest = sha256_text(text)
    print(
        f"id={doc_id}: {raw_path.name} ({fmt}) -> {out_path.name} "
        f"chars={len(text)} sha256={digest[:16]}..."
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize raw documents")
    parser.add_argument("--source", default=None)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_env()
    ensure_data_dirs()
    selection = get_source_selection(args.source, args.ids)

    failed = 0
    for doc_id in selection.ids:
        try:
            canonicalize_one(doc_id, force=args.force)
        except Exception as exc:
            print(f"ERROR id={doc_id}: {exc}")
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
