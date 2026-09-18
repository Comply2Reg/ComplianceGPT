"""Docling-based newsletter chunking with character-offset validation.

Writes data/canonical/{id}.txt and data/chunks.jsonl.

Usage:
  python scripts/chunk.py --ids 4424
  python scripts/chunk.py --from-fixtures
  python scripts/chunk.py --allow-degraded
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
    ROOT,
    ensure_data_dirs,
    get_source_selection,
    load_env,
)
from newsletter_extract_docling import (
    extract_from_doc,
    extract_newsletter,
    regions_to_chunks,
)

FIXTURES_DOCLING_DIR = ROOT / "tests" / "fixtures" / "docling"


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


def load_fixture_doc(doc_id: int) -> dict[str, Any]:
    path = FIXTURES_DOCLING_DIR / f"{doc_id}.raw.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Docling fixture: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_document(
    doc_id: int,
    *,
    allow_degraded: bool,
    from_fixtures: bool,
) -> tuple[Any, str]:
    """Return (ExtractResult, extractor). Prefer live PDF, else fixture."""
    pdf_path = RAW_DIR / f"{doc_id}.pdf"
    fixture_path = FIXTURES_DOCLING_DIR / f"{doc_id}.raw.json"

    if from_fixtures:
        result = extract_from_doc(load_fixture_doc(doc_id), extractor="docling")
        return result, "docling"

    if pdf_path.is_file():
        result = extract_newsletter(pdf_path, allow_degraded=allow_degraded)
        extractor = result.stats.get("extractor") or "docling"
        return result, extractor

    if fixture_path.is_file():
        result = extract_from_doc(load_fixture_doc(doc_id), extractor="docling")
        return result, "docling"

    raise FileNotFoundError(
        f"No PDF at {pdf_path} and no fixture at {fixture_path}"
    )


def chunk_document(
    doc_id: int,
    *,
    allow_degraded: bool = False,
    from_fixtures: bool = False,
    write_canonical: bool = True,
) -> list[dict[str, Any]]:
    result, extractor = extract_document(
        doc_id, allow_degraded=allow_degraded, from_fixtures=from_fixtures
    )
    if extractor != "docling" and not allow_degraded:
        raise RuntimeError(
            f"id={doc_id}: extractor={extractor!r} is not docling "
            "(pass --allow-degraded to accept pypdf)"
        )

    if write_canonical:
        CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
        (CANONICAL_DIR / f"{doc_id}.txt").write_text(
            result.canonical_text, encoding="utf-8"
        )

    source_url = load_source_url(doc_id)
    chunks = regions_to_chunks(
        doc_id, result, source_url=source_url, extractor=extractor
    )
    return chunks


def load_existing_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.is_file():
        return []
    existing: list[dict[str, Any]] = []
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            existing.append(json.loads(line))
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk ESMA newsletters via Docling (not pypdf headings)"
    )
    parser.add_argument("--source", default=None)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep chunks for docs not in --ids (default: also keep them)",
    )
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="Rewrite chunks.jsonl with only the selected ids",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Allow silent-quality pypdf fallback if Docling fails",
    )
    parser.add_argument(
        "--from-fixtures",
        action="store_true",
        help="Use tests/fixtures/docling/{id}.raw.json instead of PDFs",
    )
    args = parser.parse_args()

    load_env()
    ensure_data_dirs()
    selection = get_source_selection(args.source, args.ids)

    existing = [] if args.replace_all else load_existing_chunks()
    skip_ids = set(selection.ids)
    existing = [c for c in existing if int(c.get("doc_id", -1)) not in skip_ids]

    all_chunks: list[dict[str, Any]] = []
    failed = 0
    for doc_id in selection.ids:
        try:
            chunks = chunk_document(
                doc_id,
                allow_degraded=args.allow_degraded,
                from_fixtures=args.from_fixtures,
            )
            print(
                f"id={doc_id}: {len(chunks)} chunks, "
                f"extractor=docling, offsets verified"
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"ERROR id={doc_id}: {exc}")
            failed += 1
            if "Offset" in str(exc):
                return 1

    if failed:
        return 1

    # Default preserves other docs so --ids 4424 does not drop 7673/8162.
    output = existing + all_chunks
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in output:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"DONE — {len(output)} chunks -> {CHUNKS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
