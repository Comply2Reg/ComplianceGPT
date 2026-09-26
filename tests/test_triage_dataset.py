"""build_dataset + validate_dataset on a synthetic corpus and labels."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triage import build_dataset, validate_dataset  # noqa: E402
from scripts.triage.corpus import read_jsonl, write_jsonl  # noqa: E402

CLASSES = ["A1", "A2", "A3", "A4", "A6", "A11"]
STRATA = [
    "rule_final",
    "rule_final",
    "consultation",
    "guidance",
    "enforcement",
    "intelligence",
]


def _make_corpus(tmp_path: Path, n: int = 60):
    root = tmp_path / "corpus"
    (root / "canonical").mkdir(parents=True)
    manifest, chunks, labels = [], [], []
    for i in range(1, n + 1):
        k = (i - 1) % len(CLASSES)
        cls, stratum = CLASSES[k], STRATA[k]
        tier = "AMBER" if cls in ("A4", "A6") else "GREEN"
        month = 1 + (i % 9)
        date = f"2026-{month:02d}-15"
        text = f"Document {i} about {cls}. " * 40 + "\n"
        (root / "canonical" / f"{i}.txt").write_text(text, encoding="utf-8")
        sha = f"{i:064x}"
        manifest.append(
            {
                "id": i,
                "regulator": "HMT" if tier == "GREEN" else "FCA",
                "document_type": f"type_{cls}",
                "alert_class": cls,
                "stratum": stratum if i % 7 else "unclassified",
                "tier": tier,
                "licence": "uk_ogl_v3",
                "title": f"Doc {i}",
                "url": f"https://x/{i}",
                "release_date": date if i % 11 else None,
                "sha256": sha,
                "chunks": 1,
                "duplicate_of": None,
            }
        )
        chunks.append(
            {
                "chunk_id": f"x:{cls}#{i}/s",
                "doc_id": i,
                "document_id": f"x:{cls}#{i}",
                "section": f"Doc {i}",
                "text": text.strip(),
                "start": 0,
                "end": len(text.strip()),
                "source_url": f"https://x/{i}",
                "doc_hash": sha,
            }
        )
        labels.append(
            {
                "doc_id": i,
                "document_id": f"x:{cls}#{i}",
                "doc_hash": sha,
                "regulator": manifest[-1]["regulator"],
                "document_type": f"type_{cls}",
                "title": f"Doc {i}",
                "url": f"https://x/{i}",
                "release_date": manifest[-1]["release_date"],
                "tier": tier,
                "licence": "uk_ogl_v3",
                "stratum": manifest[-1]["stratum"],
                "prior_alert_class": cls,
                "model": "test",
                "prompt_version": "triage-v1",
                "label": {
                    "alert_class": cls,
                    "alert_class_confidence": "high" if i % 5 else "low",
                    "priority": "P1",
                    "primary_functions": ["Compliance"],
                    "secondary_functions": [],
                    "lines_of_defence": [2],
                    "summary": "s",
                    "key_dates": [],
                    "applicability": ["banks"],
                    "obligations_present": cls in ("A1", "A2"),
                    "jurisdiction": "GB",
                    "frameworks": [],
                    "rationale": "because",
                },
                "flags": ["low_confidence"] if not i % 5 else [],
            }
        )
    manifest.append({"id": 999, "duplicate_of": 1})
    write_jsonl(root / "manifest.jsonl", manifest)
    write_jsonl(root / "chunks.jsonl", chunks)
    labels_path = tmp_path / "triage_t.jsonl"
    write_jsonl(labels_path, labels)
    return root, labels_path


def test_build_then_validate_green_only(tmp_path: Path) -> None:
    root, labels_path = _make_corpus(tmp_path)
    out = tmp_path / "green"
    gold = tmp_path / "gold_candidates_t.csv"
    rc = build_dataset.main(
        [
            "--labels",
            str(labels_path),
            "--corpus-dir",
            str(root),
            "--out",
            str(out),
            "--version",
            "t",
            "--tier",
            "GREEN",
            "--gold-months",
            "2",
            "--gold-out",
            str(gold),
        ]
    )
    assert rc == 0
    train = read_jsonl(out / "train_t.jsonl")
    val = read_jsonl(out / "val_t.jsonl")
    test = read_jsonl(out / "test_t.jsonl")
    allf = read_jsonl(out / "all_t.jsonl")
    assert train and test
    assert len(allf) == len(train) + len(val)
    # envelope, no AMBER, no status key, classification == output.alert_class
    for rec in train + val + test:
        assert list(rec)[:7] == [
            "metadata",
            "classification",
            "instruction",
            "input_text",
            "tool_use",
            "thought_trace",
            "output",
        ]
        assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
        assert isinstance(rec["n_tokens"], int) and rec["n_tokens"] > 0
        assert rec["metadata"]["target_model"] == "qwen3-4b-instruct"
        assert rec["metadata"]["tier"] == "GREEN"
        assert "status" not in rec["output"] and "rationale" not in rec["output"]
        assert rec["output"]["alert_class"] == rec["classification"]
        assert rec["tool_use"] is None and rec["thought_trace"] == "because"
    # test is the latest months, strictly later than train within a stratum
    for stratum in {r["metadata"]["stratum"] for r in test}:
        t_dates = [
            r["metadata"]["release_date"]
            for r in test
            if r["metadata"]["stratum"] == stratum
        ]
        tr_dates = [
            r["metadata"]["release_date"]
            for r in train
            if r["metadata"]["stratum"] == stratum and r["metadata"]["release_date"]
        ]
        if tr_dates:
            assert min(t_dates) > max(tr_dates)
    hashes = [r["metadata"]["doc_hash"] for r in train + val + test]
    assert len(hashes) == len(set(hashes))
    # unclassified rows got their stratum from the model class
    assert any(r["metadata"]["stratum_source"] == "model" for r in train + val + test)
    stats = json.loads((out / "stats_t.json").read_text())
    assert stats["counts"]["train"] == len(train)
    assert (
        stats["balance"]["rule_final"]["kept"]
        == stats["balance"]["rule_final"]["available"]
    )
    # gold sheet: every test row + low-confidence rows elsewhere
    with gold.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert sum(1 for r in rows if r["split"] == "test") == len(test)
    assert all(r["human_alert_class"] == "" for r in rows)

    rc = validate_dataset.main(
        ["--dataset-dir", str(out), "--version", "t", "--skip-corpus-check"]
    )
    assert rc == 0


def test_validate_catches_leak_and_amber_and_status(tmp_path: Path) -> None:
    root, labels_path = _make_corpus(tmp_path, n=30)
    out = tmp_path / "green"
    build_dataset.main(
        [
            "--labels",
            str(labels_path),
            "--corpus-dir",
            str(root),
            "--out",
            str(out),
            "--version",
            "t",
            "--tier",
            "GREEN",
            "AMBER",
            "--gold-out",
            str(tmp_path / "g.csv"),
        ]
    )
    train = read_jsonl(out / "train_t.jsonl")
    test = read_jsonl(out / "test_t.jsonl")
    leaked = dict(test[0])
    leaked["output"] = {**leaked["output"], "status": "rejected"}
    write_jsonl(out / "train_t.jsonl", train + [leaked])
    rc = validate_dataset.main(
        ["--dataset-dir", str(out), "--version", "t", "--skip-corpus-check"]
    )
    assert rc == 1


def test_apply_gold_overrides_model_labels(tmp_path: Path) -> None:
    root, labels_path = _make_corpus(tmp_path, n=12)
    gold_in = tmp_path / "gold_labels.csv"
    with gold_in.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "doc_id",
                "human_alert_class",
                "human_priority",
                "human_functions",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "doc_id": 1,
                "human_alert_class": "A5",
                "human_priority": "P2",
                "human_functions": "Board; Risk",
            }
        )
    out = tmp_path / "green"
    build_dataset.main(
        [
            "--labels",
            str(labels_path),
            "--corpus-dir",
            str(root),
            "--out",
            str(out),
            "--version",
            "t",
            "--tier",
            "GREEN",
            "--apply-gold",
            str(gold_in),
            "--gold-out",
            str(tmp_path / "g.csv"),
        ]
    )
    recs = [
        r for f in ("train", "val", "test") for r in read_jsonl(out / f"{f}_t.jsonl")
    ]
    doc1 = next(r for r in recs if r["metadata"]["source_id"] == "x:A1#1")
    assert doc1["classification"] == "A5"
    assert doc1["output"]["primary_functions"] == ["Board", "Risk"]
    assert doc1["metadata"]["label_source"] == "human"
