"""Seed gold -> obligation_span tasks: licence from our corpus, silver batches skipped."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regnano.seed import loose, seed_tasks  # noqa: E402
from scripts.regnano.sources import Doc  # noqa: E402
from scripts.regnano.verifiers import verify  # noqa: E402

SECTION = (
    "56B. -(1) An authorised firm must inform the FCA within 28 days.\r\n\r\n"
    "(2) The Treasury may by regulations exempt a firm.\r\n\r\n"
    "(3) A firm must not act as a custodian without permission."
)
GREEN = Doc("alert_corpus:1", "HMT", "draft_si", "Draft SI", "2026-01-01", "GREEN",
            "uk_ogl_v3", "https://example.gov.uk", "abc", "Preamble.\n" + SECTION.replace("\r\n", "\n"))


def _record(example_id: str, batch: str, items: list, text: str = SECTION) -> dict:
    return {"example_id": example_id, "task": "rule_extraction",
            "input": {"section_heading": None, "section_text": text},
            "expected_output": {"regulatory_items": items},
            "provenance": {"gold_batch": batch, "annotation_status": "approved"}}


def _item(kind: str, text: str) -> dict:
    return {"item_type": kind, "source_text": text, "summary": "s"}


def test_seed_tasks_group_by_type_and_take_licence_from_corpus() -> None:
    must = "An authorised firm must inform the FCA within 28 days."
    may = "The Treasury may by regulations exempt a firm."
    rec = _record("hmt-1", "layer2_hmt_batch1", [_item("obligation", must), _item("permission", may),
                                                  _item("amendment", "56B.")])
    tasks, skipped = seed_tasks([rec], [(GREEN, loose(GREEN.text))], "sha")
    by_type = {t["gold"]["item_type"]: t for t in tasks}
    assert set(by_type) == {"obligation", "permission"} and not skipped
    t = by_type["obligation"]
    assert t["tier"] == "GREEN" and t["licence"] == "uk_ogl_v3" and t["gold"]["modality"] == "must"
    assert "\r" not in t["context"]
    assert verify(t, json.dumps({"spans": [must], "modality": "must"})).passed
    assert not verify(t, json.dumps({"spans": [must], "modality": "may"})).passed
    assert not verify(t, json.dumps({"spans": [must, may], "modality": "must"})).passed


def test_records_outside_green_corpus_and_silver_batches_are_skipped() -> None:
    green = [(GREEN, loose(GREEN.text))]
    elsewhere = _record("sebi-1", "layer1_batch1_tier1", [_item("obligation", "Brokers shall file a report.")],
                        text="Brokers shall file a report.")
    silver = _record("hmt-2", "layer2_remaining8", [_item("obligation", "An authorised firm must inform the FCA within 28 days.")])
    tasks, skipped = seed_tasks([elsewhere, silver], green, "sha")
    assert tasks == []
    assert skipped == {"not in a GREEN corpus document": 1, "silver batch": 1}
