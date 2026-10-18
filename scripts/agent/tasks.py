#!/usr/bin/env python3
"""Turn RegNano's verified tasks into tool-use tasks for the agent playground.

    python -m scripts.agent.tasks --out data/agent/tasks/v1.jsonl

No new gold is written: every task reuses a RegNano task's gold (mined from the
document's own structure, or human-reviewed seed) and its verifier. What changes
is that the passage is no longer handed over, so the model has to find it.

| family | from | the model gets |
|---|---|---|
| lookup_definition | defined_term | regulator, title, date, term |
| lookup_effective_date | effective_date | regulator, title, date |
| cited_obligations | seed obligation_span | title and the passage's opening words |
| no_tool_needed | defined_term / effective_date | the passage itself (tools still offered) |
| not_found | a defined_term's term, asked of another document that never mentions it | regulator, title, date, term |

Split by **document**: a document's tasks are all train or all held out
(sha256 of doc_hash), so held-out scores are on documents never searched in
training. Only documents present in the GREEN index are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent.index import DEFAULT_DB, Corpus  # noqa: E402
from scripts.regnano.verifiers import squash  # noqa: E402

TASKS_DIR = ROOT / "data" / "regnano" / "tasks"
DEFAULT_SOURCES = [TASKS_DIR / "mined_v2.jsonl", TASKS_DIR / "seed_obligation_v1.jsonl"]
HELD_OUT_SHARE = 0.2
NO_TOOL_SHARE = 0.15
NOT_FOUND_SHARE = 0.15
MODALITY_WORDS = {"must": "must", "must_not": "must_not", "should": "should", "may": "may"}


def held_out(doc_hash: str, share: float = HELD_OUT_SHARE) -> bool:
    return int(hashlib.sha256(doc_hash.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def doc_ref(doc: Dict, with_id: bool = False) -> str:
    """How the prompt names the document. With the id, as the app would when a regulation is open;
    without it, the model has to find the document by title first."""
    bits = [f"published {doc['release_date']}"] if doc.get("release_date") else []
    if with_id:
        bits.append(f"document_id {doc['document_id']}")
    extra = f" ({', '.join(bits)})" if bits else ""
    return f'the {doc["regulator"]} document "{doc["title"]}"{extra}'


def names_id(task_id: str) -> bool:
    """Half the lookup tasks give the document_id, chosen by hash so it is stable."""
    return int(hashlib.sha256(f"id:{task_id}".encode()).hexdigest()[:8], 16) % 2 == 0


def _tid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _base(t: Dict, family: str, prompt: str, doc: Dict) -> Dict:
    return {
        "task_id": _tid(family, t["task_id"]),
        "family": family,
        "base_family": t["family"],
        "base_task_id": t["task_id"],
        "source_id": t["source_id"],
        "doc_hash": t["doc_hash"],
        "regulator": doc["regulator"],
        "title": doc["title"],
        "tier": t["tier"],
        "licence": t["licence"],
        "split": "heldout" if held_out(t["doc_hash"]) else "train",
        "prompt": prompt,
        "context": t["context"],
        "gold": t["gold"],
    }


def lookup_definition(t: Dict, doc: Dict) -> Dict:
    term = t["gold"]["term"]
    prompt = (
        f"In {doc_ref(doc, names_id(t['task_id']))}, how is “{term}” defined? Quote the definition exactly as written, from the "
        "quoted term to the end of its definition, and cite the provision it is in. Answer only with JSON "
        '{"definition": "...", "provision_id": "..."}.'
    )
    return _base(t, "lookup_definition", prompt, doc)


def lookup_effective_date(t: Dict, doc: Dict) -> Dict:
    prompt = (
        f"When does {doc_ref(doc, names_id(t['task_id']))} take effect? A comment deadline or publication date is not an effective "
        'date. Answer only with JSON {"date": "YYYY-MM-DD", "provision_id": "..."}, citing the provision that '
        'states it, or {"date": null, "provision_id": "..."} if the document states no effective date.'
    )
    return _base(t, "lookup_effective_date", prompt, doc)


def cited_obligations(t: Dict, doc: Dict) -> Dict:
    opening = " ".join(t["context"].split()[:14])
    ask = t["question"].split(" Answer only with JSON")[0].replace("in the passage", "in that passage")
    modality = MODALITY_WORDS[t["gold"]["modality"]]
    prompt = (
        f'In {doc_ref(doc, names_id(t["task_id"]))}, find the passage that begins "{opening}". {ask} Cite the provisions you quote. '
        f'Answer only with JSON {{"spans": ["...", ...], "modality": "{modality}", "provision_ids": ["...", ...]}}.'
    )
    return _base(t, "cited_obligations", prompt, doc)


def no_tool_needed(t: Dict, doc: Dict) -> Dict:
    prompt = f"Passage:\n{t['context']}\n\nTask: {t['question']}"
    return _base(t, "no_tool_needed", prompt, doc)


def not_found(t: Dict, other: Dict) -> Dict:
    """The term of task t, asked of a different document whose text never contains it."""
    term = t["gold"]["term"]
    prompt = (
        f"In {doc_ref(other, names_id(t['task_id'] + other['document_id']))}, how is “{term}” defined? Quote the definition exactly as written, from the "
        "quoted term to the end of its definition, and cite the provision it is in. Answer only with JSON "
        '{"definition": "...", "provision_id": "..."}.'
    )
    row = _base(t, "not_found", prompt, other)
    row.update({"task_id": _tid("not_found", t["task_id"], other["document_id"]), "source_id": other["document_id"],
                "doc_hash": other["doc_hash"], "split": "heldout" if held_out(other["doc_hash"]) else "train",
                "gold": {"status": "not_found", "term": term}, "context": ""})
    return row


def _absent(term: str, text: str) -> bool:
    return squash(term).lower() not in squash(text).lower()


def build(sources: List[Path], corpus: Corpus, seed: int = 7) -> List[Dict]:
    rng = random.Random(seed)
    base: List[Dict] = []
    for path in sources:
        with path.open(encoding="utf-8") as fh:
            base.extend(json.loads(line) for line in fh if line.strip())
    docs = {d["document_id"]: d for d in corpus.documents()}
    base = [t for t in base if t["source_id"] in docs and t.get("tier") == "GREEN"]

    out: List[Dict] = []
    for t in base:
        doc = docs[t["source_id"]]
        if t["family"] == "obligation_span":
            out.append(cited_obligations(t, doc))
        elif rng.random() < NO_TOOL_SHARE:
            out.append(no_tool_needed(t, doc))
        elif t["family"] == "defined_term":
            out.append(lookup_definition(t, doc))
        elif t["family"] == "effective_date":
            out.append(lookup_effective_date(t, doc))

    # not_found: pair a term with another document of the same regulator, same split side,
    # whose full text never contains the term.
    by_reg: Dict[str, List[Dict]] = {}
    for d in docs.values():
        by_reg.setdefault(d["regulator"], []).append(d)
    terms = [t for t in base if t["family"] == "defined_term"]
    want = int(len(out) * NOT_FOUND_SHARE)
    rng.shuffle(terms)
    made = 0
    for t in terms:
        if made >= want:
            break
        side = held_out(t["doc_hash"])
        pool = [d for d in by_reg.get(t["regulator"], []) if d["document_id"] != t["source_id"]
                and held_out(d["doc_hash"]) == side]
        for other in rng.sample(pool, min(5, len(pool))):
            if _absent(t["gold"]["term"], corpus.document_text(other["document_id"])):
                out.append(not_found(t, other))
                made += 1
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; write a new version instead of overwriting")
    rows = build(args.sources, Corpus(args.db), args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": len(rows), "by_family_split": Counter(f"{r['family']}/{r['split']}" for r in rows)},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
