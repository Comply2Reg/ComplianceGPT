"""Run Stage 2: Parse PDFs into legal hierarchy and produce ontology-aware chunks.

Usage:
  python run_parse.py
  python run_parse.py --limit 5
"""

import argparse
import logging

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.stages import run_parse_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Structural Parse & Ontology Chunk")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--manifest", default=None, help="Path to ingestion manifest")
    args = parser.parse_args()

    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from pathlib import Path
    manifest_path = Path(args.manifest) if args.manifest else None
    chunks_path = run_parse_stage(settings, manifest_path=manifest_path, limit=args.limit)
    print(f"Parse complete. Chunks: {chunks_path}")


if __name__ == "__main__":
    main()
