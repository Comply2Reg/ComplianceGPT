"""Run Stage-1 pipeline for selected document IDs.

Usage:
  python scripts/run_pipeline.py --ids 4424
  python scripts/run_pipeline.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    print(f"\n>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(SCRIPTS.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inspect → ingest → docling-chunk → validate")
    parser.add_argument("--source", default=None)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Inspect + ingest dry-run only")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--from-fixtures", action="store_true")
    args = parser.parse_args()

    extra: list[str] = []
    if args.source:
        extra += ["--source", args.source]
    if args.ids:
        extra += ["--ids", *[str(i) for i in args.ids]]

    if args.from_fixtures:
        chunk_extra = list(extra) + ["--append", "--from-fixtures"]
        if args.allow_degraded:
            chunk_extra.append("--allow-degraded")
        run("chunk.py", chunk_extra)
        validate_extra = list(extra) + ["--strict", "--skip-raw"]
        if args.allow_degraded:
            validate_extra.append("--allow-degraded")
        run("validate_data.py", validate_extra)
        return 0

    run("inspect_sources.py", extra)
    ingest_extra = list(extra)
    if args.dry_run:
        ingest_extra.append("--dry-run")
    if args.force:
        ingest_extra.append("--force")
    run("ingest_s3.py", ingest_extra)

    if args.dry_run:
        print("Dry-run complete (skipped chunk/validate)")
        return 0

    # Docling path writes canonical text. pypdf canonicalize.py is not used for ESMA PDFs.
    chunk_extra = list(extra)
    chunk_extra.append("--append")
    if args.allow_degraded:
        chunk_extra.append("--allow-degraded")
    if args.from_fixtures:
        chunk_extra.append("--from-fixtures")
    run("chunk.py", chunk_extra)

    validate_extra = list(extra)
    validate_extra.append("--strict")
    if args.allow_degraded:
        validate_extra.append("--allow-degraded")
    if args.from_fixtures:
        validate_extra.append("--skip-raw")
    run("validate_data.py", validate_extra)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
