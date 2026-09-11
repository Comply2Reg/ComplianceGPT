"""Download selected documents from S3 into data/raw/ (read-only S3).

Usage:
  python scripts/ingest_s3.py --ids 4424
  python scripts/ingest_s3.py
  python scripts/ingest_s3.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAW_DIR, ensure_data_dirs, get_source_selection, load_env
from db import DocumentMeta, fetch_documents_by_ids
from s3_client import download_object, get_s3_client, resolve_bucket_and_key
from s3_uri import key_extension, redact_s3_location


def detect_extension(meta: DocumentMeta, content_type: str | None = None) -> str:
    if meta.s3_key:
        ext = key_extension(meta.s3_key)
        if ext:
            return ext
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "html" in ct:
        return ".html"
    if "xml" in ct:
        return ".xml"
    if "text/plain" in ct:
        return ".txt"
    # Last resort: pdf_url hint
    if meta.pdf_url:
        return ".pdf"
    raise RuntimeError(
        f"Cannot identify format for document id={meta.id} "
        f"({redact_s3_location(meta.s3_bucket, meta.s3_key)}; "
        f"content_type={content_type!r})"
    )


def validate_record(meta: DocumentMeta, expected_type: str) -> None:
    if meta.document_type != expected_type:
        raise RuntimeError(
            f"Document id={meta.id} has document_type={meta.document_type!r}, "
            f"expected {expected_type!r}"
        )
    if not meta.file_path:
        raise RuntimeError(f"Document id={meta.id} has empty file_path")
    if not meta.s3_key:
        raise RuntimeError(f"Document id={meta.id} has no resolvable S3 key")


def ingest_one(
    meta: DocumentMeta,
    expected_type: str,
    *,
    dry_run: bool,
    force: bool,
    s3_client,
) -> Path:
    validate_record(meta, expected_type)
    bucket, key = resolve_bucket_and_key(meta.s3_bucket, meta.s3_key)
    loc = redact_s3_location(bucket, key)

    # Probe format via head when possible
    content_type = None
    if not dry_run:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        content_type = head.get("ContentType")
        size = int(head.get("ContentLength") or 0)
        if size <= 0:
            raise RuntimeError(f"Document id={meta.id}: S3 object empty; {loc}")

    ext = detect_extension(meta, content_type)
    out_path = RAW_DIR / f"{meta.id}{ext}"

    if out_path.exists() and not force:
        print(f"id={meta.id}: skip existing {out_path.name} ({loc})")
        return out_path

    if dry_run:
        print(f"id={meta.id}: DRY-RUN would download -> {out_path.name} ({loc})")
        return out_path

    size = download_object(bucket, key, out_path, s3_client=s3_client)
    print(f"id={meta.id}: wrote {out_path.name} ({size} bytes)")
    return out_path


def write_manifest(paths: dict[int, Path], metas: dict[int, DocumentMeta]) -> Path:
    manifest_path = RAW_DIR / "manifest.json"
    payload = []
    for doc_id, path in sorted(paths.items()):
        meta = metas[doc_id]
        payload.append(
            {
                "doc_id": doc_id,
                "raw_path": str(path.relative_to(Path(__file__).resolve().parents[1])),
                "title": meta.title,
                "regulator": meta.regulator,
                "document_type": meta.document_type,
                "source_url": meta.url,
                "pdf_url": meta.pdf_url,
                "extension": path.suffix.lower(),
            }
        )
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest selected docs from S3 (read-only)")
    parser.add_argument("--source", default=None)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing raw files (default: never overwrite)",
    )
    args = parser.parse_args()

    load_env()
    ensure_data_dirs()
    selection = get_source_selection(args.source, args.ids)

    rows = fetch_documents_by_ids(
        selection.ids,
        expected_document_type=selection.document_type,
    )
    found = {r.id: r for r in rows}
    s3_client = get_s3_client()

    paths: dict[int, Path] = {}
    metas: dict[int, DocumentMeta] = {}
    failed = 0

    for doc_id in selection.ids:
        meta = found.get(doc_id)
        if meta is None:
            print(
                f"ERROR id={doc_id}: no DB row with "
                f"document_type={selection.document_type!r}"
            )
            failed += 1
            continue
        try:
            path = ingest_one(
                meta,
                selection.document_type,
                dry_run=args.dry_run,
                force=args.force,
                s3_client=s3_client,
            )
            paths[doc_id] = path
            metas[doc_id] = meta
        except Exception as exc:
            print(f"ERROR id={doc_id}: {exc}")
            failed += 1

    if paths and not args.dry_run:
        manifest = write_manifest(paths, metas)
        print(f"manifest -> {manifest}")

    print(f"done: ok={len(paths)} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
