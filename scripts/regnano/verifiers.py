"""Deterministic verifiers: the reward for one task and one model answer.

Each verifier takes a task record and the model's raw answer text and
returns a Verdict with a reward in {0.0, 1.0} and a reason. Binary on
purpose: FrogNano's reward is "tests pass", and a graded reward on our tasks
would leak the very judgement noise this exists to avoid.

Task record (one JSONL line, written by author.py):

    {"task_id", "family", "source_id", "doc_hash", "tier", "licence",
     "context":  the regulatory passage shown to the model,
     "question": the instruction,
     "gold":     family-specific answer (never trained on)}

Answers are JSON, optionally after a <think>…</think> block. A family is
added by writing one function and registering it in VERIFIERS.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
WS_RE = re.compile(r"\s+")
# Typographic variants that are the same character to a reader: a model that
# quotes “covered person” as "covered person" has quoted it exactly. Each maps
# one char to one char, so offsets in the normalised text stay comparable.
TYPO = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                      "\u2013": "-", "\u2014": "-", "\u00a0": " "})

# A predicted span matches a gold span when their character overlap covers
# this share of the longer of the two. Strict enough that "the institution
# must" does not match a whole obligation sentence.
SPAN_MATCH = 0.8
# Obligation extraction passes at this span-set F1 (all spans must also be
# verbatim). 1.0 would punish splitting one sentence into two clauses.
SPAN_F1_PASS = 0.8
MODALITIES = ("must", "must_not", "should", "may")


@dataclass(frozen=True)
class Verdict:
    reward: float
    reason: str

    @property
    def passed(self) -> bool:
        return self.reward >= 1.0


def _fail(reason: str) -> Verdict:
    return Verdict(0.0, reason)


PASS = Verdict(1.0, "ok")


def norm_ws(text: str) -> str:
    return WS_RE.sub(" ", text.translate(TYPO)).strip()


def squash(text: str) -> str:
    return WS_RE.sub("", text.translate(TYPO))


def answer_json(text: str) -> Optional[Dict]:
    """The JSON object of an answer, after any think block.

    Same recovery as scripts.triage.render.parse_assistant_json, inlined so
    this module imports nothing outside the standard library: it is loaded
    inside the RL venv as a GRPO reward, where the triage stack is absent.
    """
    text = THINK_RE.sub("", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    longest = max(a[1] - a[0], b[1] - b[0])
    return inter / longest if longest else 0.0


def locate(span: str, context: str) -> Optional[tuple[int, int]]:
    """Character range of span in context, ignoring whitespace; None if absent.

    Whitespace is dropped, not just collapsed: PDF extraction both loses
    spaces ("period oftwelve months") and inserts them ("S ervices"), and a
    model that writes the words correctly has still quoted the passage.
    Ranges are in the whitespace-free text, which is all overlap needs.
    """
    needle = squash(span)
    if not needle:
        return None
    hay = squash(context)
    at = hay.find(needle)
    return (at, at + len(needle)) if at >= 0 else None


def span_f1(pred: List[str], gold: List[str], context: str) -> Optional[float]:
    """Span-set F1 by overlap, or None if any predicted span is not verbatim."""
    p = [locate(s, context) for s in pred]
    if any(r is None for r in p):
        return None
    g = [locate(s, context) for s in gold]
    g = [r for r in g if r is not None]
    if not p or not g:
        return 1.0 if not p and not g else 0.0
    hit_p = sum(any(_overlap(a, b) >= SPAN_MATCH for b in g) for a in p)
    hit_g = sum(any(_overlap(a, b) >= SPAN_MATCH for b in p) for a in g)
    precision, recall = hit_p / len(p), hit_g / len(g)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# ── families ────────────────────────────────────────────────────────────────


def verify_obligation_span(task: Dict, text: str) -> Verdict:
    """{"spans": [verbatim sentences], "modality": must|must_not|should|may}."""
    ans = answer_json(text)
    if not isinstance(ans, dict) or not isinstance(ans.get("spans"), list):
        return _fail("no spans list")
    spans = [s for s in ans["spans"] if isinstance(s, str)]
    f1 = span_f1(spans, task["gold"]["spans"], task["context"])
    if f1 is None:
        return _fail("span not verbatim in context")
    if f1 < SPAN_F1_PASS:
        return _fail(f"span F1 {f1:.2f} < {SPAN_F1_PASS}")
    if str(ans.get("modality", "")).lower() != task["gold"]["modality"]:
        return _fail("modality mismatch")
    return PASS


def parse_date(value) -> Optional[dt.date]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def verify_effective_date(task: Dict, text: str) -> Verdict:
    """{"date": "YYYY-MM-DD"} or {"date": null} when the passage states none."""
    ans = answer_json(text)
    if not isinstance(ans, dict) or "date" not in ans:
        return _fail("no date key")
    gold = task["gold"]["date"]
    if gold is None:
        return PASS if ans["date"] is None else _fail("date given where none is stated")
    got = parse_date(ans["date"])
    if got is None:
        return _fail("unparseable date")
    return PASS if got == parse_date(gold) else _fail("wrong date")


_CITE_SUBS = [
    (re.compile(r"§+"), " section "),
    (re.compile(r"\bart(?:icle)?\.?\s*", re.I), "article "),
    (re.compile(r"\b(?:sec(?:tion)?|s)\.?\s*(?=\d)", re.I), "section "),
    (re.compile(r"\breg(?:ulation)?\.?\s*", re.I), "regulation "),
    (re.compile(r"\bpara(?:graph)?\.?\s*", re.I), "paragraph "),
    (re.compile(r"\bc\.?\s*f\.?\s*r\.?\b", re.I), "cfr"),
]


def norm_citation(value: str) -> str:
    """Canonical form: lowercase, unified unit words, no punctuation noise."""
    out = value
    for pat, rep in _CITE_SUBS:
        out = pat.sub(rep, out)
    out = re.sub(r"[^\w().]+", " ", out.lower())
    out = re.sub(r"\.(?!\d)", " ", out)  # keep 1026.19, drop "cfr." and "8."
    out = re.sub(r"\s+\(", "(", out)  # "28 (2)" == "28(2)"
    out = re.sub(r"\bcfr section\b", "cfr", norm_ws(out))  # "12 CFR § 1026" == "12 CFR 1026"
    return norm_ws(out)


def verify_citation(task: Dict, text: str) -> Verdict:
    """{"citation": "..."} — equal to any accepted gold form after normalising."""
    ans = answer_json(text)
    if not isinstance(ans, dict) or not isinstance(ans.get("citation"), str):
        return _fail("no citation")
    accepted = {norm_citation(c) for c in task["gold"]["citations"]}
    return PASS if norm_citation(ans["citation"]) in accepted else _fail("wrong citation")


def verify_defined_term(task: Dict, text: str) -> Verdict:
    """{"definition": verbatim defining text} for the term in the question."""
    ans = answer_json(text)
    if not isinstance(ans, dict) or not isinstance(ans.get("definition"), str):
        return _fail("no definition")
    got = locate(ans["definition"], task["context"])
    if got is None:
        return _fail("definition not verbatim in context")
    gold = locate(task["gold"]["definition"], task["context"])
    if gold is None:
        return _fail("gold not verbatim (invalid task)")
    return PASS if _overlap(got, gold) >= SPAN_MATCH else _fail("different passage")


def verify_triage_class(task: Dict, text: str) -> Verdict:
    """{"alert_class": "A1".."A14"} — only for tasks whose gold is agreed or human."""
    ans = answer_json(text)
    if not isinstance(ans, dict) or not isinstance(ans.get("alert_class"), str):
        return _fail("no alert_class")
    return PASS if ans["alert_class"].strip().upper() == task["gold"]["alert_class"] else _fail(
        "wrong class"
    )


VERIFIERS: Dict[str, Callable[[Dict, str], Verdict]] = {
    "obligation_span": verify_obligation_span,
    "effective_date": verify_effective_date,
    "citation": verify_citation,
    "defined_term": verify_defined_term,
    "triage_class": verify_triage_class,
}


def verify(task: Dict, text: str) -> Verdict:
    fn = VERIFIERS.get(task.get("family"))
    if fn is None:
        return _fail(f"unknown family {task.get('family')!r}")
    try:
        return fn(task, text)
    except (KeyError, TypeError) as exc:  # malformed task, never a pass
        return _fail(f"invalid task: {exc}")
