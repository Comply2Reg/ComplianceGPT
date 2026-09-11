"""Stage A only: PDF → Docling → raw JSON (+ raw inspection markdown).

No region classification, canonicalization, chunking, cleaning,
or hierarchical post-processing. Pure DocumentConverter export.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "raw" / "4424.pdf"
OUT_DIR = ROOT / "out" / "audit_parse_quality"
RAW_JSON = OUT_DIR / "4424.docling.raw.json"
REPORT_MD = OUT_DIR / "4424.docling.raw_inspection.md"


def convert_pdf_raw(pdf_path: Path) -> dict:
    """Docling DocumentConverter only — no our filters or postprocessors."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
        TableFormerMode,
    )

    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE,
        ),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(str(pdf_path))
    return result.document.export_to_dict()


def _page_no(obj: dict):
    prov = obj.get("prov") or []
    if prov:
        return prov[0].get("page_no")
    return None


def _bbox(obj: dict):
    prov = obj.get("prov") or []
    if prov:
        return prov[0].get("bbox")
    return None


def build_payload(doc: dict) -> dict:
    texts = doc.get("texts") or []
    tables = doc.get("tables") or []
    return {
        "doc_id": 4424,
        "source_pdf": "data/raw/4424.pdf",
        "stage": "A_raw_docling_extraction",
        "note": (
            "Raw Docling export ONLY via DocumentConverter.export_to_dict(). "
            "No ResultPostprocessor, no region classification, no canonicalization, "
            "no furniture filtering, no newsletter rules, no chunking."
        ),
        "counts": {
            "n_texts": len(texts),
            "n_tables": len(tables),
            "n_groups": len(doc.get("groups") or []),
        },
        "texts": [
            {
                "order_index": i,
                "self_ref": t.get("self_ref"),
                "label": t.get("label"),
                "level": t.get("level"),
                "page_no": _page_no(t),
                "bbox": _bbox(t),
                "text": t.get("text"),
                "prov": t.get("prov"),
                "parent": t.get("parent"),
            }
            for i, t in enumerate(texts)
        ],
        "tables": [
            {
                "order_index": i,
                "self_ref": tb.get("self_ref"),
                "label": tb.get("label"),
                "page_no": _page_no(tb),
                "bbox": _bbox(tb),
                "prov": tb.get("prov"),
                "n_cells": len((tb.get("data") or {}).get("table_cells") or []),
                "table_cells": [
                    {
                        "text": c.get("text"),
                        "start_row_offset_idx": c.get("start_row_offset_idx"),
                        "start_col_offset_idx": c.get("start_col_offset_idx"),
                        "end_row_offset_idx": c.get("end_row_offset_idx"),
                        "end_col_offset_idx": c.get("end_col_offset_idx"),
                    }
                    for c in ((tb.get("data") or {}).get("table_cells") or [])
                ],
            }
            for i, tb in enumerate(tables)
        ],
        "body": doc.get("body"),
        "schema_name": doc.get("schema_name"),
        "version": doc.get("version"),
        "name": doc.get("name"),
        "origin": doc.get("origin"),
    }


def write_inspection(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# 4424 — raw Docling inspection (Stage A only)")
    lines.append("")
    lines.append("Derived ONLY from `4424.docling.raw.json`.")
    lines.append("No classification, cleaning, or canonicalization.")
    lines.append("")
    lines.append(f"Source PDF: `{payload['source_pdf']}`")
    lines.append(
        f"Counts: texts={payload['counts']['n_texts']}, "
        f"tables={payload['counts']['n_tables']}"
    )
    lines.append("")

    texts = payload["texts"]
    tables = payload["tables"]
    pages = sorted(
        {
            *(t.get("page_no") for t in texts if t.get("page_no") is not None),
            *(t.get("page_no") for t in tables if t.get("page_no") is not None),
        }
    )

    for page in pages:
        lines.append(f"PAGE {page}")
        page_texts = [t for t in texts if t.get("page_no") == page]
        page_tables = [t for t in tables if t.get("page_no") == page]
        for t in page_texts:
            text = (t.get("text") or "").replace("\n", "\\n")
            lines.append(
                f"[order={t['order_index']}] [label={t.get('label')}] {text}"
            )
        for tb in page_tables:
            lines.append(
                f"[table order={tb['order_index']}] [label={tb.get('label')}] "
                f"n_cells={tb.get('n_cells')} ref={tb.get('self_ref')}"
            )
            for c in tb.get("table_cells") or []:
                cell_text = (c.get("text") or "").replace("\n", "\\n")
                lines.append(
                    f"  [r={c.get('start_row_offset_idx')} "
                    f"c={c.get('start_col_offset_idx')}] {cell_text}"
                )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    if not PDF_PATH.is_file():
        print(f"Missing PDF: {PDF_PATH}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage A only: DocumentConverter → export_to_dict (no downstream)")
    doc = convert_pdf_raw(PDF_PATH)
    payload = build_payload(doc)

    RAW_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(write_inspection(payload), encoding="utf-8")
    print(f"created {RAW_JSON}")
    print(f"created {REPORT_MD}")
    print(
        f"texts={payload['counts']['n_texts']} "
        f"tables={payload['counts']['n_tables']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
