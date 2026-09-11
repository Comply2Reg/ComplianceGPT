"""Validate raw, canonical, and chunk artifacts.

Usage:
  python scripts/validate_data.py
  python scripts/validate_data.py --ids 4424
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CANONICAL_DIR,
    CHUNKS_PATH,
    RAW_DIR,
    get_source_selection,
    load_env,
)

REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "doc_id",
    "section",
    "text",
    "start",
    "end",
    "source_url",
    "doc_hash",
)


def find_raw(doc_id: int) -> list[Path]:
    return [
        p
        for p in sorted(RAW_DIR.glob(f"{doc_id}.*"))
        if p.is_file() and p.suffix.lower() != ".json"
    ]


def validate_doc(doc_id: int) -> list[str]:
    errors: list[str] = []
    raws = find_raw(doc_id)
    if not raws:
        errors.append(f"id={doc_id}: missing raw file")
    elif len(raws) > 1:
        errors.append(f"id={doc_id}: multiple raw files {[p.name for p in raws]}")
    else:
        if raws[0].stat().st_size <= 0:
            errors.append(f"id={doc_id}: raw file empty")

    canonical = CANONICAL_DIR / f"{doc_id}.txt"
    if not canonical.is_file():
        errors.append(f"id={doc_id}: missing canonical file")
        return errors
    text = canonical.read_text(encoding="utf-8")
    if not text.strip():
        errors.append(f"id={doc_id}: canonical file empty")

    return errors


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.is_file():
        return []
    chunks = []
    for i, line in enumerate(CHUNKS_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {CHUNKS_PATH} line {i}: {exc}") from exc
    return chunks


def validate_chunks_for_ids(doc_ids: list[int] | tuple[int, ...]) -> list[str]:
    errors: list[str] = []
    chunks = load_chunks()
    if not chunks:
        errors.append("chunks.jsonl missing or empty")
        return errors

    by_doc: dict[int, list[dict[str, Any]]] = {}
    for c in chunks:
        for field in REQUIRED_CHUNK_FIELDS:
            if field not in c:
                errors.append(f"chunk {c.get('chunk_id')}: missing field {field}")
        try:
            did = int(c["doc_id"])
        except Exception:
            errors.append(f"chunk {c.get('chunk_id')}: invalid doc_id")
            continue
        by_doc.setdefault(did, []).append(c)

    for doc_id in doc_ids:
        canonical_path = CANONICAL_DIR / f"{doc_id}.txt"
        if not canonical_path.is_file():
            errors.append(f"id={doc_id}: cannot validate offsets (no canonical)")
            continue
        canonical = canonical_path.read_text(encoding="utf-8")
        doc_chunks = by_doc.get(doc_id, [])
        if not doc_chunks:
            errors.append(f"id={doc_id}: no chunks in chunks.jsonl")
            continue
        for c in doc_chunks:
            start, end, text = c["start"], c["end"], c["text"]
            if not (isinstance(start, int) and isinstance(end, int)):
                errors.append(f"{c.get('chunk_id')}: start/end must be int")
                continue
            if not (0 <= start <= end <= len(canonical)):
                errors.append(
                    f"{c.get('chunk_id')}: bad range start={start} end={end} "
                    f"len={len(canonical)}"
                )
                continue
            if canonical[start:end] != text:
                errors.append(
                    f"{c.get('chunk_id')}: offset mismatch "
                    f"canonical[start:end] != text (doc_id={doc_id})"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage-1 data artifacts")
    parser.add_argument("--source", default=None)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    args = parser.parse_args()

    load_env()
    selection = get_source_selection(args.source, args.ids)

    errors: list[str] = []
    for doc_id in selection.ids:
        errors.extend(validate_doc(doc_id))
    errors.extend(validate_chunks_for_ids(selection.ids))

    raw_count = len([p for p in RAW_DIR.glob("*") if p.is_file() and p.name != "manifest.json"])
    canon_count = len(list(CANONICAL_DIR.glob("*.txt")))
    chunk_count = len(load_chunks()) if CHUNKS_PATH.is_file() else 0

    print("=== Validation summary ===")
    print(f"raw files: {raw_count}")
    print(f"canonical files: {canon_count}")
    print(f"chunks: {chunk_count}")
    print(f"checked ids: {list(selection.ids)}")

    if errors:
        print(f"\nFAILED ({len(errors)} issues):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK — raw, canonical, chunks, and character offsets validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
