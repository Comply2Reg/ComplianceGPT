"""
Pipeline stages: ingestion, parse/chunk, extraction, vector ingest, graph ingest.

Each stage is independently callable and produces well-defined outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.schema import ObligationRecord

logger = logging.getLogger(__name__)


# ── Stage 1: Ingestion (DB + S3) ───────────────────────────────────────


def run_ingestion_stage(settings: Settings) -> Path:
    """Fetch document metadata from DB, download PDFs from S3, write manifest."""
    from src.obligation_pipeline.ingestion.ingestion_pipeline import run_ingestion_pipeline

    ingested = run_ingestion_pipeline(settings)
    manifest_path = Path(settings.output_base) / "manifest.json"
    logger.info("Ingestion complete: %d documents processed", len(ingested))
    return manifest_path


# ── Stage 2: Parse & Chunk (ontology-aware) ─────────────────────────────


def run_parse_stage(
    settings: Settings,
    manifest_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Path:
    """Parse PDFs into legal hierarchy segments, then chunk by ontology."""
    from src.obligation_pipeline.chunking.extractor import extract_pdf_pages
    from src.obligation_pipeline.chunking.ontology_chunker import chunk_by_ontology, _document_id
    from src.obligation_pipeline.parsing.hierarchy import parse_legal_hierarchy, build_hierarchy_from_parsed
    from src.obligation_pipeline.ontology.graph_builder import build_nodes_and_edges

    manifest_path = manifest_path or settings.ingest_manifest_path or Path(settings.output_base) / "manifest.json"
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    documents = [d for d in manifest.get("documents", []) if d.get("ingested") and d.get("local_path")]
    if limit is not None:
        documents = documents[:limit]

    out_dir = Path(settings.chunks_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = Path(settings.graph_output_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    all_nodes = []
    all_edges = []

    for entry in documents:
        local_path = entry.get("local_path")
        path = Path(local_path)
        if not path.exists():
            logger.warning("PDF not found: %s", path)
            continue

        doc_id = _document_id(local_path, entry.get("s3_key"))
        extracted = extract_pdf_pages(path, skip_pages=settings.skip_pages)
        if not extracted or not extracted.full_text.strip():
            logger.warning("No text extracted: %s", path)
            continue

        doc_meta: dict[str, Any] = {}
        for k in ("regulator", "title", "document_type", "document_subtype", "source_url", "s3_key"):
            if entry.get(k) is not None:
                doc_meta[k] = entry[k]
        doc_meta["local_path"] = local_path

        act_name = entry.get("title")
        effective_date = entry.get("release_date")
        regulator_code = doc_meta.get("regulator", "")

        # Structural parsing (regex by default, LLM if configured)
        segments = None
        if settings.use_llm_structure_parser:
            from src.obligation_pipeline.parsing.llm_normalization import try_parse_with_llm
            from src.obligation_pipeline.parsing.llm_bridge import LLMBridgeContext, llm_to_hierarchy_and_segments

            llm_result, error = try_parse_with_llm(extracted.full_text, settings)
            if llm_result is not None:
                bridge_ctx = LLMBridgeContext(
                    regulator=regulator_code, act_name=act_name or "Unknown",
                    document_id=doc_id, source_path=local_path, effective_date=effective_date,
                )
                segments, hierarchy = llm_to_hierarchy_and_segments(llm_result, bridge_ctx)
            else:
                logger.info("Falling back to regex parsing for %s", path.name)

        if segments is None:
            segments = parse_legal_hierarchy(
                extracted.full_text,
                get_page_at_offset=extracted.get_page_at_offset,
            )
            hierarchy = build_hierarchy_from_parsed(
                segments, regulator=regulator_code, act_name=act_name or "Unknown",
                document_id=doc_id, source_path=local_path, effective_date=effective_date,
            )

        # Ontology-aware chunking
        chunks = chunk_by_ontology(
            extracted, document_id=doc_id, doc_metadata=doc_meta,
            act_name=act_name, effective_date=effective_date, segments=segments,
        )
        all_chunks.extend(chunks)

        # Build graph nodes/edges
        nodes, edges = build_nodes_and_edges(hierarchy, segments)
        all_nodes.extend(nodes)
        all_edges.extend(edges)
        logger.info("Processed %s: %d chunks, %d graph nodes", path.name, len(chunks), len(nodes))

    # Write chunks JSONL
    chunks_path = out_dir / "chunks_ontology.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks to %s", len(all_chunks), chunks_path)

    # Write graph export
    if all_nodes:
        nodes_path = graph_dir / "nodes.jsonl"
        with open(nodes_path, "w", encoding="utf-8") as f:
            for n in all_nodes:
                f.write(json.dumps(n.to_neo4j_props(), ensure_ascii=False) + "\n")
        edges_path = graph_dir / "edges.jsonl"
        with open(edges_path, "w", encoding="utf-8") as f:
            for e in all_edges:
                row = {"source": e.source_id, "target": e.target_id, **e.to_neo4j_props()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d nodes, %d edges to %s", len(all_nodes), len(all_edges), graph_dir)

    return chunks_path


# ── Stage 3: Obligation Extraction ──────────────────────────────────────


def run_extraction_stage(
    settings: Settings,
    chunks_path: Optional[Path] = None,
    version: str = "v1.0.0",
    limit: Optional[int] = None,
) -> tuple[list[ObligationRecord], Path]:
    """Read ontology chunks, extract obligations with OpenAI, write JSONL output."""
    import asyncio
    from src.obligation_pipeline.extraction import extract_record, ExtractInput
    from src.obligation_pipeline.validate import validate_record
    from src.obligation_pipeline.balance import balance_records

    chunks_path = chunks_path or settings.chunks_ontology_path or Path(settings.chunks_output_dir) / "chunks_ontology.jsonl"
    if not Path(chunks_path).exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")

    # Load chunks
    chunks: list[dict[str, Any]] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if limit is not None:
        chunks = chunks[:limit]
    logger.info("Loaded %d chunks for extraction", len(chunks))

    # Build OpenAI client if available
    openai_client = None
    if settings.use_openai and settings.openai_api_key:
        try:
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        except ImportError:
            logger.warning("openai not installed; falling back to rule-based extraction")

    # Extract
    records: list[ObligationRecord] = []
    rejects: list[dict[str, Any]] = []

    semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def safe_extract(inp: ExtractInput):
        async with semaphore:
            try:
                rec = await extract_record(inp, settings, client=openai_client)
                valid, err = validate_record(rec)
                if valid:
                    records.append(rec)
                    print(f"Extracted: {inp.chunk_id} -> {rec.classification}")
                else:
                    rejects.append({"chunk_id": inp.chunk_id, "error": err})
                    print(f"Validation failed: {inp.chunk_id} -> {err}")
            except Exception as e:
                logger.warning("Extraction failed for %s: %s", inp.chunk_id, e)
                rejects.append({"chunk_id": inp.chunk_id, "error": str(e)})
                print(f"Error processing {inp.chunk_id}: {e}")
            finally:
                await asyncio.sleep(1.5)

    async def _extract_all():
        tasks = []
        for chunk in chunks:
            doc_title = chunk.get("act_name") or chunk.get("title") or "Unknown"
            filename = (chunk.get("local_path") or chunk.get("s3_key") or "unknown.pdf")
            filename = filename.split("/")[-1].split("\\")[-1]
            inp = ExtractInput(
                dataset_version=version,
                document_id=chunk["document_id"],
                document_title=doc_title,
                chunk_id=chunk["chunk_id"],
                chunk_text=chunk["text"],
                page_number=chunk.get("page_start") or 0,
                char_start=0,
                char_end=len(chunk["text"]),
                section_number=chunk.get("section_number"),
                clause_number=chunk.get("clause_number"),
                semantic_role=chunk.get("semantic_role"),
                hierarchy_path=chunk.get("parent_path"),
                act_name=chunk.get("act_name"),
                chapter=chunk.get("chapter"),
                regulator=chunk.get("regulator"),
                jurisdiction=chunk.get("jurisdiction"),
            )
            tasks.append(safe_extract(inp))
        if tasks:
            await asyncio.gather(*tasks)

    asyncio.run(_extract_all())

    # Write outputs
    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"raw_{version}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    # Balance
    balanced = balance_records(
        records,
        settings.target_obligation_ratio,
        settings.target_non_obligation_ratio,
        settings.target_neutral_ratio,
    )
    balanced_path = out_dir / f"balanced_{version}.jsonl"
    with open(balanced_path, "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(r.model_dump_json() + "\n")

    # Rejects
    if rejects:
        rejects_path = out_dir / f"rejects_{version}.jsonl"
        with open(rejects_path, "w", encoding="utf-8") as f:
            for r in rejects:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Run report
    report = {
        "version": version,
        "total_chunks": len(chunks),
        "total_records": len(records),
        "rejects": len(rejects),
        "balanced_records": len(balanced),
        "classifications": {
            "obligation": sum(1 for r in records if r.classification == "obligation"),
            "non_obligation": sum(1 for r in records if r.classification == "non_obligation"),
            "neutral": sum(1 for r in records if r.classification == "neutral"),
        }
    }
    report_path = out_dir / f"run_report_{version}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Extraction complete: %d records (%d obligations, %d rejected)",
        len(records), report["classifications"]["obligation"], len(rejects)
    )
    return records, raw_path


# ── Stage 4: Vector Ingest ──────────────────────────────────────────────


def run_vector_ingest_stage(records: list[ObligationRecord], settings: Settings) -> int:
    """Ingest records into ChromaDB."""
    if not settings.enable_vector_ingest:
        logger.info("Vector ingest disabled")
        return 0
    from src.obligation_pipeline.storage_vector import ingest_records_to_chroma

    count = ingest_records_to_chroma(records, settings)
    logger.info("Ingested %d records into ChromaDB", count)
    return count


# ── Stage 5: Graph Ingest ───────────────────────────────────────────────


def run_graph_ingest_stage(records: list[ObligationRecord], settings: Settings) -> int:
    """Ingest records into Neo4j."""
    if not settings.enable_graph_ingest:
        logger.info("Graph ingest disabled")
        return 0
    from src.obligation_pipeline.storage_graph import ingest_records_to_neo4j

    count = ingest_records_to_neo4j(records, settings)
    logger.info("Ingested %d records into Neo4j", count)
    return count

