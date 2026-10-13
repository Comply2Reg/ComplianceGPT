"""RegNano verifiers: binary, deterministic, and never fooled by paraphrase."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regnano.verifiers import norm_citation, span_f1, verify  # noqa: E402

CONTEXT = (
    "Section 3. Definitions. In this part, covered institution means an insured "
    "depository institution with total assets of $10 billion or more.\n"
    "Section 4. Reporting. A covered institution must submit the report to the OCC "
    "within 30 days of the end of each quarter. The OCC may extend the deadline. "
    "This rule is effective January 1, 2027."
)
OBLIGATION = "A covered institution must submit the report to the OCC within 30 days of the end of each quarter."


def _ans(obj, think: bool = False) -> str:
    body = json.dumps(obj)
    return f"<think>\nreasoning\n</think>\n\n{body}" if think else body


def _task(family: str, gold: dict) -> dict:
    return {"task_id": "t", "family": family, "context": CONTEXT, "question": "q", "gold": gold}


def test_obligation_span_passes_verbatim_and_fails_paraphrase() -> None:
    t = _task("obligation_span", {"spans": [OBLIGATION], "modality": "must"})
    assert verify(t, _ans({"spans": [OBLIGATION], "modality": "must"})).passed
    # whitespace differences are not paraphrase
    assert verify(t, _ans({"spans": [OBLIGATION.replace(" the OCC", "\n the  OCC")], "modality": "must"}, think=True)).passed
    para = verify(t, _ans({"spans": ["Covered institutions shall file quarterly reports."], "modality": "must"}))
    assert not para.passed and "verbatim" in para.reason
    assert not verify(t, _ans({"spans": [OBLIGATION], "modality": "may"})).passed
    # a fragment is verbatim but too small to count as the obligation
    assert not verify(t, _ans({"spans": ["A covered institution must"], "modality": "must"})).passed
    # an extra unrelated sentence halves precision
    assert not verify(t, _ans({"spans": [OBLIGATION, "The OCC may extend the deadline."], "modality": "must"})).passed


def test_effective_date_accepts_formats_and_null_only_when_gold_is_null() -> None:
    t = _task("effective_date", {"date": "2027-01-01"})
    for d in ("2027-01-01", "January 1, 2027", "1 January 2027"):
        assert verify(t, _ans({"date": d})).passed, d
    assert not verify(t, _ans({"date": "2027-01-02"})).passed
    assert not verify(t, _ans({"date": None})).passed
    none = _task("effective_date", {"date": None})
    assert verify(none, _ans({"date": None})).passed
    assert not verify(none, _ans({"date": "2027-01-01"})).passed


@pytest.mark.parametrize(
    "a,b",
    [
        ("12 CFR 1026.19", "12 C.F.R. § 1026.19"),
        ("Article 28(2)", "Art. 28 (2)"),
        ("section 19", "s. 19"),
        ("regulation 8", "Reg 8."),
    ],
)
def test_citation_forms_normalise_together(a: str, b: str) -> None:
    assert norm_citation(a) == norm_citation(b)


def test_citation_distinguishes_different_provisions() -> None:
    t = _task("citation", {"citations": ["Section 4"]})
    assert verify(t, _ans({"citation": "s. 4"})).passed
    assert not verify(t, _ans({"citation": "Section 3"})).passed
    assert norm_citation("12 CFR 1026.19") != norm_citation("12 CFR 1026.1")


def test_defined_term_must_quote_the_definition() -> None:
    gold = "covered institution means an insured depository institution with total assets of $10 billion or more."
    t = _task("defined_term", {"term": "covered institution", "definition": gold})
    assert verify(t, _ans({"definition": gold})).passed
    assert not verify(t, _ans({"definition": "a bank with more than $10bn of assets"})).passed
    assert not verify(t, _ans({"definition": OBLIGATION})).passed


def test_triage_class_and_malformed_answers_never_pass() -> None:
    t = _task("triage_class", {"alert_class": "A2"})
    assert verify(t, _ans({"alert_class": "a2"})).passed
    assert not verify(t, _ans({"alert_class": "A3"})).passed
    for bad in ("", "no json here", "<think>{\"alert_class\": \"A2\"}</think>", "[1, 2]"):
        assert not verify(t, bad).passed, bad
    assert not verify({**t, "family": "nope"}, _ans({"alert_class": "A2"})).passed
    assert not verify({"family": "citation", "context": CONTEXT}, _ans({"citation": "s. 4"})).passed


def test_span_f1_is_none_when_any_span_is_invented() -> None:
    assert span_f1([OBLIGATION], [OBLIGATION], CONTEXT) == 1.0
    assert span_f1([OBLIGATION, "invented"], [OBLIGATION], CONTEXT) is None
    assert span_f1([], [], CONTEXT) == 1.0


def test_curly_and_straight_quotes_are_the_same_quote_but_words_are_not() -> None:
    ctx = "In this Order— “the Act” means the Financial Services Act 2012; “the Authority” means the FCA."
    t = {"family": "defined_term", "context": ctx, "question": "q",
         "gold": {"term": "the Act", "definition": "“the Act” means the Financial Services Act 2012;"}}
    assert verify(t, json.dumps({"definition": '"the Act" means the Financial Services Act 2012;'})).passed
    assert not verify(t, json.dumps({"definition": '"the Act" means the Financial Services Act 2000;'})).passed


def test_pdf_spacing_artefacts_do_not_make_a_correct_quote_wrong() -> None:
    ctx = "“financial year” means the period oftwelve months ending on 31st March;\n“the Act” means the Financial S ervices Act 2012;"
    t = {"family": "defined_term", "context": ctx, "question": "q",
         "gold": {"term": "financial year", "definition": "“financial year” means the period oftwelve months ending on 31st March;"}}
    assert verify(t, json.dumps({"definition": '"financial year" means the period of twelve months ending on 31st March;'})).passed
    assert not verify(t, json.dumps({"definition": '"financial year" means the period of twelve months ending on 30th April;'})).passed
