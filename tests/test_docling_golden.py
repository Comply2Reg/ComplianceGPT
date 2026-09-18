import json
from pathlib import Path

from newsletter_extract_docling import extract_from_doc, regions_to_chunks
from quality import check_gates, compute_metrics

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "docling"

# Pinned from fixture replay of extract_from_doc + regions_to_chunks.
# Update only when the extractor changes intentionally.
GOLDEN = {
    4424: {"n_chunks": 24, "min_unique_sections": 8},
    7673: {"n_chunks": 46, "min_unique_sections": 8},
    8162: {"n_chunks": 34, "min_unique_sections": 8},
}


def _run(doc_id: int):
    raw = json.loads((FIXTURES / f"{doc_id}.raw.json").read_text(encoding="utf-8"))
    result = extract_from_doc(raw, extractor="docling")
    chunks = regions_to_chunks(doc_id, result, extractor="docling")
    return result, chunks


def test_both_table_cell_shapes() -> None:
    raw = json.loads((FIXTURES / "4424.raw.json").read_text(encoding="utf-8"))
    assert raw["tables"][0].get("table_cells")
    assert "data" not in raw["tables"][0] or "table_cells" in (raw["tables"][0].get("data") or {})
    result = extract_from_doc(raw, extractor="docling")
    assert result.canonical_text.strip()


def test_alert_id_form_and_gates() -> None:
    for doc_id in (4424, 7673, 8162):
        result, chunks = _run(doc_id)
        assert chunks
        for c in chunks:
            assert c["chunk_id"].startswith(f"esma:spotlight-{doc_id}#p")
            assert "/" in c["chunk_id"].split("#", 1)[1]
            assert c["extractor"] == "docling"
            assert "#article-" not in c["chunk_id"]
        errors = check_gates(result.canonical_text, chunks, strict=True)
        assert errors == [], errors
        metrics = compute_metrics(result.canonical_text, chunks)
        assert metrics["generic_section_frac"] <= 0.20
        assert metrics["unique_sections"] >= GOLDEN[doc_id]["min_unique_sections"]
        assert metrics["furniture"]["running_header"] <= 1
        assert metrics["furniture"]["page_number_lines"] == 0


def test_ids_stable_across_reparse() -> None:
    _, a = _run(4424)
    _, b = _run(4424)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    assert [c["content_sha"] for c in a] == [c["content_sha"] for c in b]


def test_golden_chunk_counts(tmp_path: Path) -> None:
    counts = {}
    for doc_id in (4424, 7673, 8162):
        _, chunks = _run(doc_id)
        counts[doc_id] = len(chunks)
        assert counts[doc_id] == GOLDEN[doc_id]["n_chunks"]
        sections = sorted({c["section"] for c in chunks})
        assert "body" not in {s.lower() for s in sections} or len(sections) > 1
    # Lock counts once known; write sidecar so a failure shows the actual numbers.
    (tmp_path / "counts.json").write_text(json.dumps(counts), encoding="utf-8")
    assert counts[4424] >= 10
    assert counts[7673] >= 10
    assert counts[8162] >= 10
