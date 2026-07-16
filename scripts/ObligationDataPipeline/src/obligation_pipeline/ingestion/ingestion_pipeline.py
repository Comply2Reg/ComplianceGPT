"""Orchestrate fetch from DB, download from S3, and regulator-wise organization."""

import logging
from pathlib import Path

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.ingestion.db import fetch_document_metadata
from src.obligation_pipeline.ingestion.models import DocumentRecord, IngestedDocument
from src.obligation_pipeline.ingestion.organizer import organize_by_regulator, write_manifest
from src.obligation_pipeline.ingestion.s3 import download_pdf, get_s3_client

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(settings: Settings) -> list[IngestedDocument]:
    """
    Run the full ingestion pipeline:
    1. Fetch document metadata (S3 keys) from SQL database
    2. Download PDF files from S3
    3. Organize by regulator
    4. Write manifest for downstream chunking and extraction
    """
    Path(settings.output_base).mkdir(parents=True, exist_ok=True)
    records = list(fetch_document_metadata(settings))
    logger.info("Fetched %d document records from database", len(records))

    organized = list(organize_by_regulator(settings, iter(records)))
    s3_client = get_s3_client(settings)
    ingested: list[IngestedDocument] = []

    for record, local_path in organized:
        success, error = download_pdf(settings, record, local_path, s3_client=s3_client)
        ingested.append(
            IngestedDocument(record=record, local_path=local_path, success=success, error=error)
        )

    manifest_path = Path(settings.output_base) / "manifest.json"
    write_manifest(settings, ingested, manifest_path)

    success_count = sum(1 for d in ingested if d.success)
    logger.info("Pipeline complete: %d succeeded, %d failed", success_count, len(ingested) - success_count)
    return ingested
