"""Docling spike runner for ESMA newsletters (side files only).

Usage:
  python scripts/spike_docling_4424.py
  python scripts/spike_docling_4424.py --ids 4424
  python scripts/spike_docling_4424.py --ids 4424 7673

Writes per id:
  data/canonical/{id}.docling.txt
  data/canonical/{id}.docling.regions.json
  data/canonical/{id}.docling.chunks.jsonl

Does not modify raw PDFs, reference-parser/, production canonical, or data/chunks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CANONICAL_DIR, RAW_DIR, ensure_data_dirs, load_env
from newsletter_extract_docling import extract_newsletter, regions_to_chunks


def run_one(doc_id: int) -> dict:
    pdf_path = RAW_DIR / f"{doc_id}.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing raw PDF: {pdf_path}")

    print(f"\n=== Docling spike id={doc_id} ({pdf_path.name}) ===")
    result = extract_newsletter(pdf_path)
    print("stats:", json.dumps(result.stats, ensure_ascii=False))

    out_txt = CANONICAL_DIR / f"{doc_id}.docling.txt"
    out_regions = CANONICAL_DIR / f"{doc_id}.docling.regions.json"
    out_chunks = CANONICAL_DIR / f"{doc_id}.docling.chunks.jsonl"

    out_txt.write_text(result.canonical_text, encoding="utf-8")

    regions_payload = [
        {
            "region": r.region,
            "section": r.section,
            "title": r.title,
            "start": r.start,
            "end": r.end,
            "page_start": r.page_start,
            "page_end": r.page_end,
            "preview": result.canonical_text[r.start : r.end][:240],
        }
        for r in result.regions
    ]
    out_regions.write_text(
        json.dumps(
            {"doc_id": doc_id, "doc_hash": result.doc_hash, "regions": regions_payload},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    chunks = regions_to_chunks(doc_id, result)
    with out_chunks.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    text = result.canonical_text
    probes = {
        "chars": len(text),
        "spotlight_count": text.lower().count("spotlight on markets"),
        "has_providers31": "Providers31" in text or bool(
            __import__("re").search(r"[A-Za-z]{4}\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", text)
        ),
        "has_instagram": "instagram" in text.lower(),
        "has_next_steps": "Next steps" in text,
        "has_pipe_dates": " | " in text,
        "toc_regions": sum(1 for r in result.regions if r.region == "toc"),
        "events_regions": sum(1 for r in result.regions if r.region == "events"),
        "consultations_regions": sum(1 for r in result.regions if r.region == "consultations"),
        "article_regions": sum(1 for r in result.regions if r.region == "article"),
        "chunks": len(chunks),
        "unique_sections": sorted({c["section"] for c in chunks})[:12],
    }
    print(f"wrote {out_txt.name}, {out_regions.name}, {out_chunks.name}")
    print("probes:", json.dumps(probes, ensure_ascii=False))
    return {"doc_id": doc_id, "stats": result.stats, "probes": probes}


def main() -> int:
    parser = argparse.ArgumentParser(description="ESMA newsletter Docling spike")
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        default=[4424],
        help="Document IDs to process (default: 4424 only)",
    )
    args = parser.parse_args()

    load_env()
    ensure_data_dirs()

    failed = 0
    for doc_id in args.ids:
        try:
            run_one(doc_id)
        except Exception as exc:
            print(f"ERROR id={doc_id}: {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
