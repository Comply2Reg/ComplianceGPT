"""RegNano task mining and calibration bookkeeping — fixture docs, no model."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regnano.calibrate import band, summarise  # noqa: E402
from scripts.regnano.mine import mine_defined_terms, mine_effective_date  # noqa: E402
from scripts.regnano.prompt import to_messages  # noqa: E402
from scripts.regnano.sources import Doc  # noqa: E402
from scripts.regnano.verifiers import verify  # noqa: E402


def _doc(text: str, regulator: str = "OCC") -> Doc:
    return Doc("us_corpus:1", regulator, "final_rules", "A rule", "2026-09-01", "GREEN",
               "us_public_domain", "https://example.gov/x", "abc", text)


FINAL = (
    "AGENCY:\nOffice of the Comptroller of the Currency.\nACTION:\nFinal rule.\n"
    "SUMMARY:\nThe OCC is amending its rules.\n"
    "DATES:\nThe final rule is effective November 2, 2026.\n"
    "FOR FURTHER INFORMATION CONTACT:\nJane Doe.\n"
)
PROPOSED = (
    "ACTION:\nProposed rule.\nSUMMARY:\nThe OCC proposes to amend its rules.\n"
    "DATES:\nComments must be received on or before December 1, 2026.\n"
    "ADDRESSES:\nSubmit comments online.\n"
)
STAGGERED = (
    "DATES:\nThis rule is effective January 1, 2027, except section 3 which is effective "
    "July 1, 2027.\nADDRESSES:\nNone.\n"
)
DEFS = (
    "2.—(1) In this Order—\n"
    "“the Authority” means the Financial Conduct Authority;\n"
    "“the commencement date” means 1st April 2013;\n"
    "“the Act” means the Financial Services Act 2012.\n"
)


def test_effective_date_gold_from_dates_paragraph() -> None:
    t = mine_effective_date(_doc(FINAL))
    assert t and t["gold"] == {"date": "2026-11-02"} and t["tier"] == "GREEN"
    assert "Published: 2026-09-01" in t["context"]  # the distractor date is shown
    assert verify(t, json.dumps({"date": "2026-11-02"})).passed
    assert not verify(t, json.dumps({"date": "2026-09-01"})).passed


def test_comment_only_proposal_has_null_gold_and_staggered_is_skipped() -> None:
    t = mine_effective_date(_doc(PROPOSED))
    assert t and t["gold"] == {"date": None}
    assert not verify(t, json.dumps({"date": "2026-12-01"})).passed  # a comment deadline
    assert mine_effective_date(_doc(STAGGERED)) is None
    assert mine_effective_date(_doc("No dates here.")) is None


def test_defined_terms_need_a_unique_term_and_distractors() -> None:
    tasks = list(mine_defined_terms(_doc(DEFS, "UKLEG"), random.Random(0), per_doc=5))
    assert {t["gold"]["term"] for t in tasks} == {"the Authority", "the commencement date", "the Act"}
    for t in tasks:
        assert t["gold"]["distractor_definitions"] == 2
        assert verify(t, json.dumps({"definition": t["gold"]["definition"]})).passed
    lone = "In this Order “the Act” means the Financial Services Act 2012.\n"
    assert list(mine_defined_terms(_doc(lone), random.Random(0))) == []
    twice = DEFS + "“the Act” means something else entirely.\n"
    assert "the Act" not in {t["gold"]["term"] for t in mine_defined_terms(_doc(twice), random.Random(0), 5)}


def test_task_ids_are_stable_across_runs() -> None:
    a = mine_effective_date(_doc(FINAL))
    b = mine_effective_date(_doc(FINAL))
    assert a["task_id"] == b["task_id"]


def test_band_keeps_only_partially_solved_tasks() -> None:
    assert not band(0.0, 0.0, 1.0) and not band(1.0, 0.0, 1.0)
    assert band(0.5, 0.0, 1.0) and band(0.125, 0.0, 1.0)
    assert not band(0.75, 0.0, 0.5)


def test_summarise_counts_never_always_and_in_band() -> None:
    rows = [
        {"family": "f", "k": 8, "passes": 0, "pass_rate": 0.0},
        {"family": "f", "k": 8, "passes": 8, "pass_rate": 1.0},
        {"family": "f", "k": 8, "passes": 4, "pass_rate": 0.5},
    ]
    s = summarise(rows)["f"]
    assert (s["never"], s["always"], s["in_band"], s["mean_pass"]) == (1, 1, 1, 0.5)


def test_prompt_puts_passage_before_task_and_system_first() -> None:
    m = to_messages({"context": "CTX", "question": "Q?"})
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[1]["content"].index("CTX") < m[1]["content"].index("Q?")


def test_grpo_row_carries_the_task_the_reward_needs(tmp_path: Path) -> None:
    from scripts.regnano import rl
    from scripts.triage.corpus import read_jsonl, write_jsonl

    task = mine_effective_date(_doc(FINAL))
    row = rl.to_row(task)
    assert row["system"] and row["type"] == "effective_date"
    assert "DATES:" in row["prompt"] and task["question"] in row["prompt"]
    back = json.loads(row["answer"])
    assert verify(back, json.dumps({"date": "2026-11-02"})).passed
    write_jsonl(tmp_path / "t.jsonl", [task])
    assert rl.main(["export", "--tasks", str(tmp_path / "t.jsonl"), "--out", str(tmp_path / "g")]) == 0
    assert len(read_jsonl(tmp_path / "g" / "train.jsonl")) + len(read_jsonl(tmp_path / "g" / "valid.jsonl")) == 1


def test_verifiers_import_only_the_standard_library() -> None:
    # Loaded inside the RL venv as a GRPO reward, where pydantic etc. are absent.
    import ast

    tree = ast.parse((ROOT / "scripts" / "regnano" / "verifiers.py").read_text(encoding="utf-8"))
    mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


def test_ambiguous_definitions_are_not_mined() -> None:
    """Each case was a 0/8 calibration task where the model's answer was right."""
    two_on_a_line = DEFS + (
        "Proposed § 9 would provide that the term “fuel” means material used to produce "
        "electricity. Proposed § 10 would provide that the term “feedstock” means any raw material.\n"
    )
    aliased = DEFS + "The term “Board” or “Board of Directors” means the Board of Directors of the Company.\n"
    continued = DEFS + "“Covered contract” means a contract involving a crypto asset;\nprovided that it is not a security.\n"
    restated = (
        "The Board interprets the term “offense committed” to mean the last date of the misconduct. "
        "In instances with multiple offenses,\n“offense committed” means the last date of any of the offenses.\n"
    ) + DEFS
    wrapped = DEFS + "“regulators” means the PSR, the FCA and the Bank of\nEngland, and “regulator” means any one of the regulators;\n"
    ligature = DEFS + "“ICO” means the Information Commissioner’s O\x00ce;\n"
    for text, term in ((wrapped, "regulator"), (ligature, "ICO"), (two_on_a_line, "fuel"), (aliased, "Board of Directors"),
                       (continued, "Covered contract"), (restated, "offense committed")):
        terms = {t["gold"]["term"] for t in mine_defined_terms(_doc(text), random.Random(0), per_doc=9)}
        assert term not in terms, term
        assert "the Authority" in terms  # the clean definitions beside it still mine


def test_pool_holds_out_every_difficulty_and_trains_only_in_band() -> None:
    from scripts.regnano.pool import build, is_held_out

    tasks = [{"task_id": f"t{i}", "family": "defined_term"} for i in range(400)]
    rates = [0.0, 0.5, 1.0, 0.875]
    cal = [{"task_id": f"t{i}", "pass_rate": rates[i % 4]} for i in range(400)]
    cal.append({"task_id": "gone", "pass_rate": 0.5})  # rejected by the miner since
    pool = build(tasks, cal, 0.2, 0.0, 1.0)
    held = {t["task_id"] for t in pool["heldout"]}
    train = {t["task_id"] for t in pool["train"]}
    assert held and not held & train and "gone" not in held | train
    assert {cal[int(i[1:])]["pass_rate"] for i in held} == set(rates)  # all difficulties
    assert all(0 < cal[int(i[1:])]["pass_rate"] < 1 for i in train)
    assert 40 < len(held) < 120 and pool["stats"]["dropped_by_miner"] == 1
    assert is_held_out("t1", 0.2) == is_held_out("t1", 0.2)


def test_rft_keeps_distinct_verified_answers_and_refuses_held_out_tasks() -> None:
    import pytest
    from scripts.regnano.rft import build_rows

    task = mine_effective_date(_doc(FINAL))
    good = json.dumps({"date": "2026-11-02"})
    passing = [{"task_id": task["task_id"], "completion": c} for c in
               (good, good + "\n", json.dumps({"date": "November 2, 2026"}),
                json.dumps({"date": "2026-09-01"}),  # wrong: re-verified and dropped
                json.dumps({"date": "2 November 2026"}))]
    rows = build_rows([task], passing, set(), per_task=2)
    texts = [r["messages"][-1]["content"] for d in rows.values() for r in d]
    assert len(texts) == 2 and texts[0] == good  # duplicate collapsed, capped at 2
    assert all(r["messages"][0]["role"] == "system" for d in rows.values() for r in d)
    with pytest.raises(ValueError):
        build_rows([task], passing, {task["task_id"]}, per_task=2)
