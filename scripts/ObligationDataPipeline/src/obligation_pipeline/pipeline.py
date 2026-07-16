"""Full 5-stage obligation data pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.stages import (
    run_ingestion_stage,
    run_parse_stage,
    run_extraction_stage,
    run_vector_ingest_stage,
    run_graph_ingest_stage,
)

logger = logging.getLogger(__name__)


def run_full_pipeline(
    settings: Settings,
    version: str = "v1.0.0",
    skip_ingest: bool = False,
    skip_parse: bool = False,
    skip_vector: bool = False,
    skip_graph: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """
    Run all 5 stages of the obligation data pipeline.

    Stages:
      1. Ingestion: DB + S3 → manifest.json + PDFs
      2. Parse & Chunk: PDFs → legal hierarchy → ontology chunks
      3. Extraction: chunks → OpenAI → obligation JSONL
      4. Vector Ingest: records → ChromaDB
      5. Graph Ingest: records → Neo4j

    Returns a summary dict with stage results.
    """
    summary: dict = {"version": version, "stages": {}}

    # Stage 1: Ingestion
    if not skip_ingest:
        logger.info("=== Stage 1: Ingestion (DB + S3) ===")
        manifest_path = run_ingestion_stage(settings)
        summary["stages"]["ingestion"] = {"manifest": str(manifest_path)}
    else:
        manifest_path = settings.ingest_manifest_path or Path(settings.output_base) / "manifest.json"
        logger.info("Skipping ingestion; using manifest: %s", manifest_path)

    # Stage 2: Parse & Chunk
    if not skip_parse:
        logger.info("=== Stage 2: Parse & Chunk ===")
        chunks_path = run_parse_stage(settings, manifest_path=manifest_path, limit=limit)
        summary["stages"]["parse"] = {"chunks_path": str(chunks_path)}
    else:
        chunks_path = settings.chunks_ontology_path or Path(settings.chunks_output_dir) / "chunks_ontology.jsonl"
        logger.info("Skipping parse; using chunks: %s", chunks_path)

    # Stage 3: Extraction
    logger.info("=== Stage 3: Obligation Extraction ===")
    records, raw_path = run_extraction_stage(settings, chunks_path=chunks_path, version=version, limit=limit)
    summary["stages"]["extraction"] = {
        "total_records": len(records),
        "output_path": str(raw_path),
    }

    # Stage 4: Vector Ingest
    if not skip_vector:
        logger.info("=== Stage 4: Vector Ingest ===")
        vector_count = run_vector_ingest_stage(records, settings)
        summary["stages"]["vector_ingest"] = {"count": vector_count}
    else:
        logger.info("Skipping vector ingest")

    # Stage 5: Graph Ingest
    if not skip_graph:
        logger.info("=== Stage 5: Graph Ingest ===")
        graph_count = run_graph_ingest_stage(records, settings)
        summary["stages"]["graph_ingest"] = {"count": graph_count}
    else:
        logger.info("Skipping graph ingest")

    logger.info("Pipeline complete: %s", summary)
    return summary
