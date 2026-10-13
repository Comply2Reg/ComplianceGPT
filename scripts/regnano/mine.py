#!/usr/bin/env python3
"""Mine verifiable tasks whose gold answer is in the document's own structure.

No author model: for these two families the regulator's text states the
answer in a fixed form, so the gold is extracted, not judged.

  effective_date  Federal Register "DATES:" paragraphs. Exactly one stated
                  effective date -> that date; comment deadlines only and
                  no "effective" anywhere -> null (teaches abstention).
                  Release date and comment deadline are in the context as
                  distractors.
  defined_term    `"X" means ...` definitions (UK SIs, CFR text). The context
                  is a window holding several neighbouring definitions, and
                  the term must be defined exactly once in it.

    python -m scripts.regnano.mine --out data/regnano/tasks/mined_v1.jsonl \\
        [--per-family 1000] [--seed 7]

Every task records source_id, doc_hash, tier and licence; only GREEN
documents are read (sources.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regnano.sources import Doc, iter_docs  # noqa: E402
from scripts.regnano.verifiers import locate, verify  # noqa: E402

MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
US_DATE = rf"{MONTH} \d{{1,2}}, \d{{4}}"
DATES_PARA = re.compile(r"^DATES:\n(.+?)(?=\n[A-Z][A-Z ,/]+:\n|\Z)", re.S | re.M)
EFFECTIVE = re.compile(rf"\beffective(?: on| as of| date is)? ({US_DATE})", re.I)
COMMENTS = re.compile(r"\bcomments?\b.{0,80}\b(?:received|submitted|due)\b", re.I | re.S)
# “term” means …;  /  "term" means ….   The definition must run to the end
# of its line: ending at the first ";"/"." + space cut "4:00 a.m. and …" at
# "a.m." and multi-sentence definitions at their first sentence, so the model
# was marked wrong for quoting the whole definition. Wrapped (multi-line)
# definitions are skipped rather than guessed.
DEFINITION = re.compile(r"[“\"]([^”\"\n]{2,60})[”\"] means ([^\n]{10,600}[;.])[ \t]*(?=\n|$)")

CONTEXT_CHARS = 2400

DATE_QUESTION = (
    "On what date does this rule or action take effect? Answer only with JSON "
    '{"date": "YYYY-MM-DD"}, or {"date": null} if the passage does not state an '
    "effective date. A comment deadline or publication date is not an effective date."
)
TERM_QUESTION = (
    "Quote, exactly as written in the passage, the definition of “{term}”: the text "
    'from the quoted term to the end of its definition. Answer only with JSON {{"definition": "..."}}.'
)


def task_id(family: str, source_id: str, key: str) -> str:
    return hashlib.sha256(f"{family}|{source_id}|{key}".encode()).hexdigest()[:16]


def _base(doc: Doc, family: str, key: str, context: str, question: str, gold: Dict) -> Dict:
    return {
        "task_id": task_id(family, doc.source_id, key),
        "family": family,
        "source_id": doc.source_id,
        "doc_hash": doc.doc_hash,
        "regulator": doc.regulator,
        "document_type": doc.document_type,
        "title": doc.title,
        "url": doc.url,
        "tier": doc.tier,
        "licence": doc.licence,
        "context": context,
        "question": question,
        "gold": gold,
        "origin": "mined",
    }


def _window(text: str, start: int, end: int, size: int = CONTEXT_CHARS) -> str:
    """size chars around [start, end), snapped to line breaks."""
    pad = max(0, size - (end - start)) // 2
    lo, hi = max(0, start - pad), min(len(text), end + pad)
    lo = text.rfind("\n", 0, lo) + 1 if lo > 0 else 0
    nl = text.find("\n", hi)
    hi = nl if nl != -1 else len(text)
    return text[lo:hi].strip()


def mine_effective_date(doc: Doc) -> Optional[Dict]:
    m = DATES_PARA.search(doc.text)
    if not m:
        return None
    para = m.group(1)[:1000]
    found = {dt.datetime.strptime(d, "%B %d, %Y").date() for d in EFFECTIVE.findall(para)}
    if len(found) == 1:
        gold = found.pop().isoformat()
    elif not found and "effective" not in doc.text.lower() and COMMENTS.search(para):
        gold = None
    else:
        return None  # staggered dates or an effective date stated elsewhere: ambiguous
    context = f"Published: {doc.release_date}\nTitle: {doc.title}\n\n" + _window(
        doc.text, m.start(), m.end()
    )
    return _base(doc, "effective_date", "dates", context, DATE_QUESTION, {"date": gold})


def mine_defined_terms(doc: Doc, rng: random.Random, per_doc: int = 2) -> Iterator[Dict]:
    matches = list(DEFINITION.finditer(doc.text))
    rng.shuffle(matches)
    emitted = 0
    for m in matches:
        if emitted >= per_doc:
            return
        term = m.group(1).strip()
        context = _window(doc.text, m.start(), m.end())
        # The term must be defined once in the window, and the gold must be
        # locatable — otherwise the task has two right answers or none.
        defined_here = [d for d in DEFINITION.finditer(context) if d.group(1).strip() == term]
        if len(defined_here) != 1:
            continue
        gold = m.group(0)
        if locate(gold, context) is None:
            continue
        others = len(DEFINITION.findall(context)) - 1
        if others < 1:
            continue  # a lone definition is a copy task, not a reading task
        yield _base(
            doc,
            "defined_term",
            term.lower(),
            context,
            TERM_QUESTION.format(term=term),
            {"term": term, "definition": gold, "distractor_definitions": others},
        )
        emitted += 1


def mine(per_family: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    dates: List[Dict] = []
    terms: List[Dict] = []
    for doc in iter_docs():
        t = mine_effective_date(doc)
        if t:
            dates.append(t)
        terms.extend(mine_defined_terms(doc, rng))
    out: List[Dict] = []
    # Balance dated vs null-dated so "null" is not the majority answer.
    dated = [t for t in dates if t["gold"]["date"]]
    null = [t for t in dates if not t["gold"]["date"]]
    rng.shuffle(dated)
    rng.shuffle(null)
    n_null = min(len(null), per_family // 4)
    out += dated[: per_family - n_null] + null[:n_null]
    rng.shuffle(terms)
    out += terms[:per_family]
    # Self-check: every gold answer must pass its own verifier.
    for t in out:
        verdict = verify(t, json.dumps(_gold_answer(t)))
        if not verdict.passed:
            raise AssertionError(f"{t['task_id']}: gold fails its verifier ({verdict.reason})")
    return out


def _gold_answer(task: Dict) -> Dict:
    if task["family"] == "effective_date":
        return {"date": task["gold"]["date"]}
    return {"definition": task["gold"]["definition"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-family", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    tasks = mine(args.per_family, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    by_family: Dict[str, int] = {}
    for t in tasks:
        by_family[t["family"]] = by_family.get(t["family"], 0) + 1
    regulators: Dict[str, int] = {}
    for t in tasks:
        regulators[t["regulator"]] = regulators.get(t["regulator"], 0) + 1
    print(json.dumps({"out": str(args.out), "by_family": by_family, "by_regulator": regulators}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
