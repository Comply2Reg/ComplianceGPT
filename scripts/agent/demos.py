#!/usr/bin/env python3
"""Scripted demonstrations: code plays a correct episode for a task, from its gold answer.

    python -m scripts.agent.demos --tasks data/agent/tasks/v2.jsonl --out data/agent/rollouts/demos_v2.jsonl

The warm start for a base model that solves too few tool episodes to learn from
its own successes (docs/agent.md). Each demo:

- calls only real tools through the real harness, so every tool result is what
  the model would have seen;
- fetches a provision only if an earlier result in the same episode returned
  its id, so the path is one a model could have taken from what it was shown;
- answers with the gold, quoted from the fetched text;
- is kept only if verify_episode passes it, the same check every other
  trajectory gets.

Held-out tasks are refused. The search strategy is deliberately plain (find the
document by title, search inside it, read the provision); teacher trajectories
(teacher.py) and the model's own successes add variety.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent.env import Episode, start, step, transcript  # noqa: E402
from scripts.agent.index import DEFAULT_DB, Corpus  # noqa: E402
from scripts.agent.verify import verify_episode  # noqa: E402
from scripts.regnano.verifiers import locate, parse_date  # noqa: E402
from scripts.triage.corpus import read_jsonl  # noqa: E402


def call(name: str, **arguments) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def answer(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _last_ids(ep: Episode) -> List[str]:
    return ep.calls[-1].returned_ids if ep.calls and ep.calls[-1].ok else []


def _same_doc(corpus: Corpus, doc_id: str, task: Dict) -> bool:
    if doc_id == task["source_id"]:
        return True
    a, b = corpus.document(doc_id), corpus.document(task["source_id"])
    return bool(a and b and (a["title"], a["release_date"]) == (b["title"], b["release_date"]))


def find_doc(ep: Episode, task: Dict, corpus: Corpus) -> Optional[str]:
    """The document id: from the prompt if it names one, else by a title search (one call)."""
    if f"document_id {task['source_id']}" in task["prompt"]:
        return task["source_id"]
    step(ep, call("search_regulation", query=task["title"], regulator=task["regulator"]), corpus)
    for pid in _last_ids(ep):
        doc = pid.split("#", 1)[0]
        if _same_doc(corpus, doc, task):
            return doc
    return None


def holder(corpus: Corpus, doc_id: str, text: str) -> Optional[str]:
    """The provision of doc_id whose text contains text verbatim."""
    for p in corpus.provisions_of(doc_id):
        if locate(text, p["text"]) is not None:
            return p["provision_id"]
    return None


def search_then_read(ep: Episode, corpus: Corpus, query: str, doc_id: str, target: str) -> bool:
    """Search inside the document unless an earlier result already returned target; then read it."""
    if target not in ep.returned_ids:
        step(ep, call("search_regulation", query=query, document_id=doc_id), corpus)
    if target not in ep.returned_ids:
        return False
    step(ep, call("get_provision", provision_id=target), corpus)
    return True


def _date_text(iso: str) -> List[str]:
    d = dt.date.fromisoformat(iso)
    return [f"{d:%B} {d.day}, {d.year}", f"{d.day} {d:%B} {d.year}", iso]


def demo(task: Dict, corpus: Corpus) -> Optional[Episode]:
    ep = start(task)
    fam, gold = task["family"], task["gold"]

    if fam == "no_tool_needed":
        if task["base_family"] == "defined_term":
            step(ep, answer({"definition": gold["definition"]}), corpus)
        else:
            step(ep, answer({"date": gold["date"]}), corpus)
        return ep

    doc_id = find_doc(ep, task, corpus)
    if doc_id is None:
        return None

    if fam == "lookup_definition":
        target = holder(corpus, doc_id, gold["definition"])
        if not target or not search_then_read(ep, corpus, f"{gold['term']} means", doc_id, target):
            return None
        step(ep, answer({"definition": gold["definition"], "provision_id": target}), corpus)

    elif fam == "lookup_effective_date":
        target = None
        if gold["date"]:
            target = next((t for t in (holder(corpus, doc_id, s) for s in _date_text(gold["date"])) if t), None)
        target = target or holder(corpus, doc_id, "DATES:")
        if not target or not search_then_read(ep, corpus, "DATES effective date", doc_id, target):
            return None
        got = parse_date(gold["date"]) if gold["date"] else None
        step(ep, answer({"date": got.isoformat() if got else None, "provision_id": target}), corpus)

    elif fam == "cited_obligations":
        targets = []
        for s in gold["spans"]:
            t = holder(corpus, doc_id, s)
            if t is None:
                return None
            if t not in targets:
                targets.append(t)
        opening = task["prompt"].split('begins "', 1)[1].split('"', 1)[0]
        for t in targets:
            if not search_then_read(ep, corpus, opening, doc_id, t):
                return None
        step(ep, answer({"spans": gold["spans"], "modality": gold["modality"], "provision_ids": targets}), corpus)

    elif fam == "not_found":
        step(ep, call("search_regulation", query=f"{gold['term']} means", document_id=doc_id), corpus)
        step(ep, answer({"status": "not_found"}), corpus)

    else:
        return None
    return ep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; write a new version instead of overwriting")
    corpus = Corpus(args.db)
    stats: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for task in read_jsonl(args.tasks):
            if task["split"] != "train":
                continue
            ep = demo(task, corpus)
            if ep is None:
                stats[f"{task['family']}/unreachable"] += 1
                continue
            v = verify_episode(ep, corpus)
            stats[f"{task['family']}/{'kept' if v.passed else 'failed: ' + v.reason}"] += 1
            if v.passed:
                row = transcript(ep)
                row.update({"sample": 0, "split": "train", "passed": True, "reason": v.reason, "source": "scripted"})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
