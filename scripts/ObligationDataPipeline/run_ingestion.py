"""Run Stage 1: Fetch document metadata from DB and download PDFs from S3.

Usage:
  python run_ingestion.py
"""

import logging

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.stages import run_ingestion_stage


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    manifest_path = run_ingestion_stage(settings)
    print(f"Ingestion complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
