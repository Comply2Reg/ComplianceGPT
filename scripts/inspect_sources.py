"""Inspect selected source documents (safe metadata only).

Usage:
  python scripts/inspect_sources.py
  python scripts/inspect_sources.py --ids 4424
  python scripts/inspect_sources.py --source esma_newsletter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/inspect_sources.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    bucket_env_status,
    env_present,
    get_source_selection,
    load_env,
    missing_required_db_config,
)
from db import fetch_documents_by_ids
from s3_uri import key_extension, redact_s3_location


def guess_format(meta) -> str:
    if meta.s3_key:
        ext = key_extension(meta.s3_key)
        if ext:
            return ext.lstrip(".")
    if meta.pdf_url:
        return "pdf?"
    if meta.url and str(meta.url).lower().endswith(".html"):
        return "html?"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect selected regulation documents")
    parser.add_argument(
        "--source",
        default=None,
        help="Source key from config.SOURCES (default: ACTIVE_SOURCE)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        default=None,
        help="Override document IDs (subset of the source)",
    )
    args = parser.parse_args()

    load_env()
    selection = get_source_selection(args.source, args.ids)

    print("=== Source selection ===")
    print(f"source: {selection.source_key} ({selection.label})")
    print(f"document_type: {selection.document_type}")
    print(f"ids: {list(selection.ids)}")
    print()

    print("=== Config status (names only) ===")
    db_missing = missing_required_db_config()
    if db_missing:
        print(f"database: INCOMPLETE — missing {', '.join(db_missing)}")
    else:
        print("database: configured")
    print(f"aws_access_key_id set: {env_present('AWS_ACCESS_KEY_ID')}")
    print(f"aws_secret_access_key set: {env_present('AWS_SECRET_ACCESS_KEY')}")
    print(f"aws_region set: {env_present('AWS_REGION')}")
    print(f"default bucket env: {bucket_env_status()}")
    print()

    if db_missing:
        print(
            "Cannot query database until DB env vars are set "
            "(same names as GraphRAG: DATABASE_URL or DB_*). "
            "Add them to .env locally - do not paste secrets here."
        )
        return 1

    rows = fetch_documents_by_ids(
        selection.ids,
        expected_document_type=selection.document_type,
    )
    found = {r.id: r for r in rows}

    print("=== Documents ===")
    errors = 0
    for doc_id in selection.ids:
        meta = found.get(doc_id)
        if meta is None:
            print(f"id={doc_id}  MISSING (no matching row for type={selection.document_type!r})")
            errors += 1
            continue
        fmt = guess_format(meta)
        loc = redact_s3_location(meta.s3_bucket, meta.s3_key)
        has_path = bool(meta.file_path)
        print(
            f"id={meta.id}  title={meta.title!r}  regulator={meta.regulator!r}  "
            f"document_type={meta.document_type!r}  format={fmt}  "
            f"file_path_present={has_path}  {loc}"
        )

    print()
    print(f"found={len(found)}/{len(selection.ids)}  errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
