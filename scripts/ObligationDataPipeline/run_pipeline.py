"""Run the full 5-stage obligation data pipeline.

Usage:
  python run_pipeline.py --version v1.0.0
  python run_pipeline.py --version v1.0.0 --skip-ingest --skip-vector --limit 5
"""

import argparse
import logging
import sys

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Obligation Data Pipeline — Full 5-Stage Run")
    parser.add_argument("--version", default="v1.0.0", help="Dataset version tag (default: v1.0.0)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip DB+S3 ingestion (use existing manifest)")
    parser.add_argument("--skip-parse", action="store_true", help="Skip parse/chunk (use existing chunks)")
    parser.add_argument("--skip-vector", action="store_true", help="Skip vector DB ingest")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph DB ingest")
    args = parser.parse_args()

    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    summary = run_full_pipeline(
        settings,
        version=args.version,
        skip_ingest=args.skip_ingest,
        skip_parse=args.skip_parse,
        skip_vector=args.skip_vector,
        skip_graph=args.skip_graph,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
