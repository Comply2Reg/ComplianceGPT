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
    parser = argparse.ArgumentParser(description="Run inspect → ingest → canonicalize → chunk → validate")
    parser.add_argument("--source", default=None)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Inspect + ingest dry-run only")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    extra: list[str] = []
    if args.source:
        extra += ["--source", args.source]
    if args.ids:
        extra += ["--ids", *[str(i) for i in args.ids]]

    run("inspect_sources.py", extra)
    ingest_extra = list(extra)
    if args.dry_run:
        ingest_extra.append("--dry-run")
    if args.force:
        ingest_extra.append("--force")
    run("ingest_s3.py", ingest_extra)

    if args.dry_run:
        print("Dry-run complete (skipped canonicalize/chunk/validate)")
        return 0

    canon_extra = list(extra)
    if args.force:
        canon_extra.append("--force")
    run("canonicalize.py", canon_extra)
    run("chunk.py", extra)
    run("validate_data.py", extra)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
