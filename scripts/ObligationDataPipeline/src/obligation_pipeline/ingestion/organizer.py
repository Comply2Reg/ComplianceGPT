"""Organize downloaded PDFs regulator-wise and prepare manifest for chunking/vector DB/KG."""

import json
import logging
from pathlib import Path
from typing import Iterator

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.ingestion.models import DocumentRecord, IngestedDocument

logger = logging.getLogger(__name__)


def organize_by_regulator(
    settings: Settings,
    records: Iterator[DocumentRecord],
) -> Iterator[tuple[DocumentRecord, Path]]:
    """
    Yield (record, local_path) with local_path under {OUTPUT_BASE}/{regulator}/.
    """
    base = Path(settings.output_base)
    for record in records:
        if not settings.all_regulators and record.regulator not in settings.REGULATORS:
            continue
        safe_name = _s3_key_to_safe_filename(record.s3_key)
        local_path = base / record.regulator / safe_name
        yield record, local_path


def _s3_key_to_safe_filename(s3_key: str) -> str:
    """Use last component of S3 key as filename; ensure .pdf extension."""
    name = s3_key.replace("\\", "/").split("/")[-1] or "document"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return _sanitize_filename(name)


def _sanitize_filename(name: str) -> str:
    """Remove or replace characters unsafe for filesystems."""
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
    return safe.strip("._ ") or "document.pdf"


def write_manifest(settings: Settings, ingested: list[IngestedDocument], path: Path) -> None:
    """Write a JSON manifest of ingested documents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for doc in ingested:
        if not doc.success:
            continue
        entry = doc.record.to_manifest_entry(doc.local_path)
        entry["ingested"] = True
        entries.append(entry)
    failed = [d for d in ingested if not d.success]
    regulators_list = (
        sorted({d.record.regulator for d in ingested})
        if settings.all_regulators
        else list(settings.REGULATORS)
    )
    manifest = {
        "version": "1.0",
        "output_base": str(settings.output_base),
        "regulators": regulators_list,
        "documents": entries,
        "count": len(entries),
        "failed_count": len(failed),
        "failed": [
            {"s3_key": d.record.s3_key, "regulator": d.record.regulator, "error": d.error}
            for d in failed
        ],
    }
    def _json_default(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
    logger.info("Wrote manifest to %s (%d documents, %d failed)", path, len(entries), len(failed))
