"""Run Stage 3: Extract obligations from ontology chunks using OpenAI.

Usage:
  python run_extract.py --version v1.0.0
  python run_extract.py --version v1.0.0 --limit 50
"""

import argparse
import logging

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.stages import run_extraction_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Obligation Extraction")
    parser.add_argument("--version", default="v1.0.0", help="Dataset version tag")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks")
    parser.add_argument("--chunks", default=None, help="Path to chunks JSONL")
    args = parser.parse_args()

    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    
    # Silence noisy third-party network retries to keep the terminal clean
    #logging.getLogger("httpx").setLevel(logging.WARNING)
    #logging.getLogger("openai").setLevel(logging.WARNING)

    from pathlib import Path
    chunks_path = Path(args.chunks) if args.chunks else None
    records, raw_path = run_extraction_stage(settings, chunks_path=chunks_path, version=args.version, limit=args.limit)
    print(f"Extraction complete: {len(records)} records → {raw_path}")


if __name__ == "__main__":
    main()
