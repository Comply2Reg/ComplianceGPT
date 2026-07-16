"""Run Stage 5: Ingest obligation records into Neo4j.

Usage:
  python run_ingest_graph.py --version v1.0.0
"""

import argparse
import json
import logging
from pathlib import Path

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.schema import ObligationRecord
from src.obligation_pipeline.stages import run_graph_ingest_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: Graph DB Ingest")
    parser.add_argument("--version", default="v1.0.0", help="Dataset version to ingest")
    parser.add_argument("--input", default=None, help="Path to raw JSONL")
    args = parser.parse_args()

    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    input_path = Path(args.input) if args.input else Path(f"data/output/raw_{args.version}.jsonl")
    records: list[ObligationRecord] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ObligationRecord.model_validate_json(line))

    count = run_graph_ingest_stage(records, settings)
    print(f"Graph ingest complete: {count} records")


if __name__ == "__main__":
    main()
