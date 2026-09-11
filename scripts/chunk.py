"""Deterministic newsletter chunking with character-offset validation.

Writes data/chunks.jsonl.

Usage:
  python scripts/chunk.py --ids 4424
  python scripts/chunk.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CANONICAL_DIR,
    CHUNKS_PATH,
    RAW_DIR,
    ensure_data_dirs,
    get_source_selection,
    load_env,
)

# Soft max for newsletter paragraph groups (characters).
MAX_CHUNK_CHARS = 4000
MIN_CHUNK_CHARS = 80

HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s+.+"  # markdown heading
    r"|[A-Z][A-Z0-9][A-Z0-9\s,&/\-]{2,120}"  # ALL-CAPS-ish line
    r"|(?:Article|Section|Chapter)\s+\d+[A-Za-z0-9.\-]*"  # numbered article/section
    r"|\d+\.\s+[A-Z].{0,120}"  # numbered heading
    r")$"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_source_url(doc_id: int) -> str | None:
    manifest = RAW_DIR / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for entry in data:
        if int(entry.get("doc_id", -1)) == doc_id:
            return entry.get("source_url") or entry.get("pdf_url")
    return None


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 160:
        return False
    if HEADING_RE.match(s):
        return True
    # Short title-case lines ending without period often mark sections.
    if len(s) <= 80 and s[0].isupper() and not s.endswith((".", ",", ";", ":")):
        words = s.split()
        if 2 <= len(words) <= 12 and sum(w[0].isupper() for w in words if w) >= max(2, len(words) // 2):
            return True
    return False


def split_paragraphs(canonical: str) -> list[tuple[int, int, str]]:
    """Return (start, end, text) for each non-empty paragraph in canonical."""
    paras: list[tuple[int, int, str]] = []
    # Paragraph = run of non-empty content bounded by blank lines.
    for m in re.finditer(r"(?:[^\n]|\n(?!\n))+", canonical):
        start, end = m.start(), m.end()
        # Trim outer newlines so offsets still refer to canonical.
        while start < end and canonical[start] == "\n":
            start += 1
        while end > start and canonical[end - 1] == "\n":
            end -= 1
        if start < end and canonical[start:end].strip():
            paras.append((start, end, canonical[start:end]))
    return paras


def group_newsletter_chunks(
    canonical: str,
) -> list[dict[str, Any]]:
    """
    Chunk newsletters by heading / paragraph groups.

    Extensible schema: uses `section` (not `provision`) for newsletters.
    """
    paragraphs = split_paragraphs(canonical)
    if not paragraphs:
        return []

    groups: list[dict[str, Any]] = []
    current_section = "body"
    current_parts: list[tuple[int, int, str]] = []

    def flush() -> None:
        nonlocal current_parts
        if not current_parts:
            return
        start = current_parts[0][0]
        end = current_parts[-1][1]
        # Include intervening whitespace from canonical between first and last.
        text = canonical[start:end]
        if text.strip():
            groups.append(
                {
                    "section": current_section,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
        current_parts = []

    for start, end, text in paragraphs:
        heading = is_heading(text) and len(text) < 160 and "\n" not in text.strip()
        if heading:
            flush()
            current_section = re.sub(r"\s+", " ", text.strip())[:120]
            current_parts = [(start, end, text)]
            flush()
            continue

        current_parts.append((start, end, text))
        size = current_parts[-1][1] - current_parts[0][0]
        if size >= MAX_CHUNK_CHARS:
            flush()

    flush()

    # Merge tiny trailing fragments into previous chunk when possible.
    # Never merge across section boundaries (preserves newsletter headings).
    merged: list[dict[str, Any]] = []
    for g in groups:
        if (
            merged
            and g["section"] == merged[-1]["section"]
            and len(g["text"]) < MIN_CHUNK_CHARS
            and (len(merged[-1]["text"]) + (g["end"] - merged[-1]["end"]))
            < MAX_CHUNK_CHARS * 1.25
            and g["start"] >= merged[-1]["end"]
        ):
            prev = merged[-1]
            new_end = g["end"]
            prev["end"] = new_end
            prev["text"] = canonical[prev["start"]:new_end]
        else:
            merged.append(g)
    return merged


def validate_offsets(canonical: str, chunks: list[dict[str, Any]], doc_id: int) -> None:
    for i, c in enumerate(chunks):
        start, end, text = c["start"], c["end"], c["text"]
        if not (0 <= start <= end <= len(canonical)):
            raise AssertionError(
                f"Offset range invalid for doc_id={doc_id} chunk_index={i} "
                f"chunk_id={c.get('chunk_id')} start={start} end={end} "
                f"canonical_len={len(canonical)}"
            )
        sliced = canonical[start:end]
        if sliced != text:
            raise AssertionError(
                f"Offset validation failed for doc_id={doc_id} chunk_index={i} "
                f"chunk_id={c.get('chunk_id')} start={start} end={end}: "
                f"canonical[start:end] != chunk['text']"
            )


def chunk_document(doc_id: int) -> list[dict[str, Any]]:
    path = CANONICAL_DIR / f"{doc_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical file: {path}")

    canonical = path.read_text(encoding="utf-8")
    doc_hash = sha256_text(canonical)
    source_url = load_source_url(doc_id)

    groups = group_newsletter_chunks(canonical)
    if not groups:
        # Single full-document chunk as last resort (still offset-valid).
        text = canonical
        groups = [{"section": "document", "start": 0, "end": len(text), "text": text}]

    chunks: list[dict[str, Any]] = []
    for i, g in enumerate(groups):
        section = g["section"] or "body"
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", section).strip("-").lower()[:40] or "section"
        chunk = {
            "chunk_id": f"{doc_id}#{slug}-{i+1}",
            "doc_id": doc_id,
            "section": section,
            "text": g["text"],
            "start": g["start"],
            "end": g["end"],
            "source_url": source_url,
            "doc_hash": doc_hash,
        }
        chunks.append(chunk)

    validate_offsets(canonical, chunks, doc_id)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk canonical newsletter texts")
    parser.add_argument("--source", default=None)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing chunks.jsonl instead of rewriting selected docs",
    )
    args = parser.parse_args()

    load_env()
    ensure_data_dirs()
    selection = get_source_selection(args.source, args.ids)

    all_chunks: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []
    if args.append and CHUNKS_PATH.is_file():
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
        skip_ids = set(selection.ids)
        existing = [c for c in existing if int(c.get("doc_id", -1)) not in skip_ids]

    failed = 0
    for doc_id in selection.ids:
        try:
            chunks = chunk_document(doc_id)
            print(f"id={doc_id}: {len(chunks)} chunks, offsets verified")
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"ERROR id={doc_id}: {exc}")
            failed += 1
            # Hard stop on offset/validation failures for safety.
            if "Offset" in str(exc) or isinstance(exc, AssertionError):
                return 1

    if failed:
        return 1

    output = existing + all_chunks if args.append else all_chunks
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in output:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"DONE — {len(output)} chunks -> {CHUNKS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
