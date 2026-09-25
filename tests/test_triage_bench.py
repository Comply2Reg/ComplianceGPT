"""External benchmark scoring, prompt building, and re-scoring saved runs.

No network and no model: the benchmark adapters are exercised on hand-built
rows shaped like the real datasets (verified against the Hugging Face datasets
server), and the re-scorer on a small synthetic evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triage import bench, rescore  # noqa: E402
from scripts.triage.corpus import write_jsonl  # noqa: E402

TOS_LABELS = [
    "Limitation of liability",
    "Unilateral termination",
    "Unilateral change",
    "Content removal",
    "Contract by using",
    "Choice of law",
    "Jurisdiction",
    "Arbitration",
]


# ── scorers ─────────────────────────────────────────────────────────────────


def test_single_label_separates_wrong_from_unanswered() -> None:
    s = bench.score_single_label(["A", "B", "C", "D"], ["A", "X", None, None])
    assert s["n"] == 4
    assert s["accuracy"] == pytest.approx(0.25)
    assert s["answered_rate"] == pytest.approx(0.5)
    # Half the items produced no label at all; of those that did, one of two
    # was right. Conflating the two would hide format lock.
    assert s["accuracy_when_answered"] == pytest.approx(0.5)


def test_multi_label_micro_f1_counts_partial_credit() -> None:
    golds = [{"Arbitration", "Jurisdiction"}, {"Content removal"}]
    preds = [["Arbitration"], ["Content removal", "Choice of law"]]
    s = bench.score_multi_label(golds, preds)
    # tp=2 (Arbitration, Content removal), fp=1 (Choice of law), fn=1 (Jurisdiction)
    assert s["micro_precision"] == pytest.approx(2 / 3, abs=5e-5)
    assert s["micro_recall"] == pytest.approx(2 / 3, abs=5e-5)
    # Neither row matches exactly: the first misses a label, the second adds
    # one. Partial credit shows up in F1, not here.
    assert s["exact_set_match"] == 0.0


def test_multi_label_counts_an_unanswered_row_as_missed_not_skipped() -> None:
    s = bench.score_multi_label([{"Arbitration"}], [None])
    assert s["micro_recall"] == 0.0
    assert s["answered_rate"] == 0.0


def test_schema_scorer_reports_validity_even_without_jsonschema() -> None:
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    s = bench.score_schema([schema, schema], [{"a": 1}, None])
    assert s["json_valid_rate"] == pytest.approx(0.5)
    if s["schema_checked"]:
        assert s["schema_conformance"] == pytest.approx(1.0)


def test_probe_is_descriptive_and_says_so() -> None:
    preds = [
        {"alert_class": "A11", "obligations_present": False},
        {"alert_class": "A11", "obligations_present": True},
        None,
    ]
    s = bench.score_probe(["q1", "q2", "q3"], preds)
    assert s["descriptive_only"] is True
    assert s["produced_triage_json"] == pytest.approx(2 / 3, abs=5e-5)
    assert s["obligations_present_rate"] == pytest.approx(0.5)
    assert s["alert_class_distribution"] == {"A11": 2}


# ── recovering a label from a format-locked answer ──────────────────────────


def test_a_label_is_recovered_from_inside_a_triage_record() -> None:
    """The model answers every prompt with triage JSON; the label may be in it."""
    raw = (
        '<think>\n\n</think>\n\n{"alert_class":"A3","summary":'
        '"This clause concerns Arbitration of disputes.","priority":"P2"}'
    )
    assert bench.pick_one(raw, TOS_LABELS) == "Arbitration"


def test_longer_labels_win_over_substrings() -> None:
    assert bench.pick_one("This is a Choice of law clause.", TOS_LABELS) == "Choice of law"


def test_no_label_mentioned_returns_none() -> None:
    assert bench.pick_one("I cannot determine the category.", TOS_LABELS) is None


def test_pick_many_returns_every_named_label() -> None:
    got = bench.pick_many("Arbitration and Jurisdiction both apply.", TOS_LABELS)
    assert set(got) == {"Arbitration", "Jurisdiction"}


# ── prompt building ─────────────────────────────────────────────────────────


def test_ledgar_prompt_carries_the_label_set_and_the_gold_name() -> None:
    labels = ["Adjustments", "Agreements", "Amendments"]
    system, user, gold = bench._ledgar({"text": "The parties agree.", "label": 2}, labels)
    assert gold == "Amendments"
    assert "Adjustments, Agreements, Amendments" in user
    assert "one label" in system


def test_unfair_tos_prompt_maps_indices_to_names() -> None:
    _, _, gold = bench._unfair_tos({"text": "x", "labels": [7, 5]}, TOS_LABELS)
    assert gold == ["Arbitration", "Choice of law"]


def test_obliqa_probe_uses_the_models_own_training_prompt() -> None:
    from scripts.triage.render import TRAIN_SYSTEM_PROMPT

    row = {
        "QuestionID": "q-1",
        "Question": "What must an Authorised Person do?",
        "Passages": [{"Passage": "A Person must maintain records.", "PassageID": "1",
                      "DocumentID": 1}],
    }
    system, user, gold = bench._obliqa(row, [])
    assert system == TRAIN_SYSTEM_PROMPT  # the probe measures its native behaviour
    assert "A Person must maintain records." in user
    assert gold == "q-1"


# ── sampling ────────────────────────────────────────────────────────────────


class _FakeDataset:
    def __init__(self, n: int):
        self._rows = [{"i": i} for i in range(n)]

    def __len__(self):
        return len(self._rows)

    def select(self, idx):
        d = _FakeDataset(0)
        d._rows = [self._rows[i] for i in idx]
        return d

    def __iter__(self):
        return iter(self._rows)


def test_sampling_is_seeded_and_reproducible() -> None:
    ds = _FakeDataset(1000)
    a, _ = bench.sample_rows(ds, 50, seed=42)
    b, _ = bench.sample_rows(ds, 50, seed=42)
    c, _ = bench.sample_rows(ds, 50, seed=7)
    assert a == b and a != c
    assert len(a) == 50 and len(set(a)) == 50
    assert a == sorted(a)


def test_a_limit_beyond_the_split_takes_everything() -> None:
    ds = _FakeDataset(10)
    idx, rows = bench.sample_rows(ds, 500, seed=42)
    assert len(idx) == 10 and len(rows) == 10


# ── re-scoring a finished run against different labels ──────────────────────


def _record(source_id: str, doc_hash: str, cls: str) -> dict:
    return {
        "metadata": {"source_id": source_id, "doc_hash": doc_hash},
        "classification": cls,
        "instruction": "i",
        "input_text": "t",
        "tool_use": None,
        "thought_trace": "r",
        "output": {
            "alert_class": cls,
            "alert_class_confidence": "high",
            "priority": "P2",
            "primary_functions": ["Compliance"],
            "secondary_functions": [],
            "lines_of_defence": [2],
            "summary": "s",
            "key_dates": [],
            "applicability": [],
            "obligations_present": True,
        },
    }


def test_predictions_are_reparsed_from_the_saved_generations(tmp_path: Path) -> None:
    records = [_record("a#1", "h1", "A3"), _record("b#2", "h2", "A11")]
    outputs = tmp_path / "out.jsonl"
    write_jsonl(
        outputs,
        [
            {"source_id": "a#1", "raw": '<think>\n\n</think>\n\n{"alert_class":"A3"}'},
            {"source_id": "b#2", "raw": "not json at all"},
        ],
    )
    preds, missing = rescore.predictions_for(records, outputs)
    assert missing == 0
    assert preds[0] == {"alert_class": "A3"}
    assert preds[1] is None


def test_a_record_with_no_saved_prediction_is_reported_not_silently_dropped(
    tmp_path: Path,
) -> None:
    records = [_record("a#1", "h1", "A3"), _record("b#2", "h2", "A11")]
    outputs = tmp_path / "out.jsonl"
    write_jsonl(outputs, [{"source_id": "a#1", "raw": '{"alert_class":"A3"}'}])
    preds, missing = rescore.predictions_for(records, outputs)
    assert missing == 1 and preds[1] is None


def test_relabelling_swaps_the_verdict_and_drops_unjudged_records() -> None:
    records = [_record("a#1", "h1", "A3"), _record("b#2", "h2", "A11")]
    judge = {"h1": {"alert_class": "A10", "priority": "P1", "obligations_present": False,
                    "primary_functions": ["Legal"], "rationale": "ignored"}}
    swapped = rescore.relabelled(records, judge)
    assert len(swapped) == 1
    assert swapped[0]["output"]["alert_class"] == "A10"
    assert "rationale" not in swapped[0]["output"]  # OUTPUT_FIELDS only
    assert records[0]["output"]["alert_class"] == "A3"  # original untouched


def test_reproduces_flags_a_drifting_metric() -> None:
    measured = {"alert_class_accuracy": 0.6442, "json_valid_rate": 1.0}
    assert rescore.reproduces(measured, {"alert_class_accuracy": 0.6442,
                                         "json_valid_rate": 1.0}) == []
    drift = rescore.reproduces(measured, {"alert_class_accuracy": 0.50,
                                          "json_valid_rate": 1.0})
    assert len(drift) == 1 and "alert_class_accuracy" in drift[0]
