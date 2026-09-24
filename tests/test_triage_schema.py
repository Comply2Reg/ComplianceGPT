"""The label space and the record schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triage.doc_input import DocInput, build_doc_input  # noqa: E402
from scripts.triage.labeller import validate_label  # noqa: E402
from scripts.triage.prompts import SYSTEM_PROMPT, build_messages  # noqa: E402
from scripts.triage.schema import OUTPUT_FIELDS, TriageRecord  # noqa: E402
from scripts.triage.taxonomy import (  # noqa: E402
    ALERT_CLASSES,
    CLASS_TO_FUNCTION,
    CLASS_TO_STRATUM,
    FUNCTIONS,
    STRATA,
    TARGET_MIX,
    functions_for,
)
from scripts.triage.weak_labels import heuristic_class  # noqa: E402


def test_every_class_has_a_stratum_and_the_mix_sums_to_one() -> None:
    assert set(ALERT_CLASSES) == set(CLASS_TO_STRATUM)
    assert set(CLASS_TO_STRATUM.values()) == set(STRATA)
    assert abs(sum(TARGET_MIX.values()) - 1.0) < 1e-9


def test_matrix_covers_every_class_with_valid_marks() -> None:
    for cls in ALERT_CLASSES:
        table = CLASS_TO_FUNCTION[cls]
        assert table, cls
        assert set(table) <= set(FUNCTIONS)
        assert set(table.values()) <= {"P", "S", "I"}
    # Intelligence is inform-only by design; every other class has an owner.
    assert [c for c in ALERT_CLASSES if not functions_for(c)] == ["A11"]
    assert functions_for("A6") == [
        "Compliance",
        "Financial crime",
        "Legal",
        "Internal audit",
    ]


def test_schema_is_strict_mode_friendly() -> None:
    js = TriageRecord.model_json_schema()
    assert set(js["required"]) == set(js["properties"])
    assert js.get("additionalProperties") is False
    assert "rationale" not in OUTPUT_FIELDS and "alert_class" in OUTPUT_FIELDS


def test_schema_round_trip() -> None:
    rec = TriageRecord(
        alert_class="A2",
        alert_class_confidence="high",
        priority="P1",
        primary_functions=["Regulatory change", "Compliance"],
        secondary_functions=["Risk"],
        lines_of_defence=[2],
        summary="Final rules on X apply to banks from 1 Jan 2027.",
        key_dates=[{"label": "in force", "date": "2027-01-01"}],
        applicability=["banks"],
        obligations_present=True,
        rationale="It is a PS.",
    )
    again = TriageRecord.model_validate_json(rec.model_dump_json())
    assert again == rec
    with pytest.raises(Exception):
        TriageRecord.model_validate({**rec.model_dump(), "alert_class": "A99"})


def test_validate_label_flags_off_matrix_and_disagreement() -> None:
    label = {
        "alert_class": "A6",
        "alert_class_confidence": "low",
        "primary_functions": ["Privacy"],
        "secondary_functions": [],
        "summary": "x",
    }
    flags = validate_label(label, "A2")
    assert {
        "function_off_matrix",
        "class_disagrees_with_registry",
        "low_confidence",
    } <= set(flags)
    ok = {
        "alert_class": "A2",
        "alert_class_confidence": "high",
        "primary_functions": ["Compliance"],
        "secondary_functions": [],
        "summary": "x",
    }
    assert validate_label(ok, "A2") == []
    assert "class_disagrees_with_registry" not in validate_label(ok, "MIXED")


def _row(**kw):
    base = {
        "id": 7,
        "regulator": "FCA",
        "document_type": "policy_statements",
        "title": "PS26/1 – Something",
        "release_date": "2026-03-01",
        "alert_class": "A2",
        "tier": "AMBER",
        "sha256": "ab" * 32,
    }
    base.update(kw)
    return base


def test_doc_input_uses_headings_and_scope_chunks() -> None:
    chunks = [
        {"section": "PS26/1 – Something", "text": "intro " * 50},
        {
            "section": "Who this applies to",
            "text": "This applies to banks and building societies.",
        },
        {"section": "PS26/1 – Something (part 2)", "text": "more"},
        {"section": "Next steps", "text": "Rules come into force on 1 July 2026."},
    ]
    doc = build_doc_input(_row(), "canonical text " * 10, chunks)
    assert isinstance(doc, DocInput)
    assert doc.headings == ["PS26/1 – Something", "Who this applies to", "Next steps"]
    assert len(doc.applicability) == 2
    text = doc.as_text()
    assert "Registry prior for alert class: A2" in text
    assert "Who this applies to" in text
    messages = build_messages(doc)
    assert (
        messages[0]["content"] == SYSTEM_PROMPT and "PS26/1" in messages[1]["content"]
    )
    assert "A14" in SYSTEM_PROMPT and "Internal audit" in SYSTEM_PROMPT


def test_mixed_prior_is_not_shown_as_a_hint() -> None:
    doc = build_doc_input(_row(alert_class="MIXED"), "text", [])
    assert "Registry prior" not in doc.as_text()


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Final Notice 2026: JS Motors", "A6"),
        ("CP26/3 – Reforms to securitisation", "A3"),
        ("PS12/26 – Review of the SM&CR", "A2"),
        ("SS1/26 – Operational resilience", "A4"),
        ("Dear CEO letter: wholesale brokers", "A5"),
        ("Speech by Nikhil Rathi at the City Dinner", "A11"),
        ("The Financial Services Act 2021 (Commencement No. 3) Regulations 2022", "A1"),
        ("Something without a signal", None),
    ],
)
def test_title_heuristics(title, expected) -> None:
    assert heuristic_class(title) == expected


def test_prompt_is_json_safe() -> None:
    json.dumps(build_messages(build_doc_input(_row(), "x", [])))
