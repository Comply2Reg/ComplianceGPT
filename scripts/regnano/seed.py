#!/usr/bin/env python3
"""Turn a human-reviewed seed gold file into obligation_span tasks.

    python -m scripts.regnano.seed --seed final_golden_dataset.jsonl \\
        --out data/regnano/tasks/seed_obligation_v1.jsonl

The seed records (rule_extraction / alert_understanding, schema rule-1 /
vnext-1) list regulatory items, each with a verbatim `source_text` and an
`item_type`. One task per (record, item type): "quote every provision of
this passage that imposes an obligation" -> the items of that type, checked
by verifiers.verify_obligation_span (every span verbatim, span-set F1 >= 0.8,
modality match).

Licence is decided from our corpus, never from the seed file: a record is
admitted only if all its spans are found in a GREEN document from
sources.iter_docs, and the task carries that document's tier and licence.
Batches marked --silver-batch (script-filled, not reviewed) are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regnano.mine import task_id  # noqa: E402
from scripts.regnano.sources import Doc, iter_docs  # noqa: E402
from scripts.regnano.verifiers import locate, verify  # noqa: E402

MODALITY = {"obligation": "must", "prohibition": "must_not", "permission": "may", "recommendation": "should"}
DESCRIBE = {
    "obligation": "imposes an obligation (something that must be done)",
    "prohibition": "imposes a prohibition (something that must not be done)",
    "permission": "grants a permission (something that may be done)",
    "recommendation": "makes a recommendation (something that should be done)",
}
QUESTION = (
    "Quote, exactly as written in the passage, every provision that {what}. "
    'Answer only with JSON {{"spans": ["...", ...], "modality": "{modality}"}}.'
)
# Script-filled batches in the 2026-10 seed file: summaries copy the source
# sentence and applies_to is always empty, so they are not reviewed gold.
SILVER_BATCHES = ("layer2_remaining8",)
MAX_CONTEXT_CHARS = 6000  # ~1,500 tokens: prompt + completion must fit 2,048


def loose(text: str) -> str:
    """Alphanumerics only, lower-cased — for finding a seed passage in our corpus."""
    return re.sub(r"\W+", "", text or "").lower()


def context_of(record: Dict) -> str:
    inp = record["input"]
    return (inp.get("section_text") or inp.get("text") or "").replace("\r\n", "\n")


def find_green(spans: List[str], docs: Iterable[tuple[Doc, str]]) -> Optional[Doc]:
    keys = [loose(s)[:200] for s in spans]
    for doc, text in docs:
        if all(k in text for k in keys):
            return doc
    return None


def seed_tasks(records: List[Dict], green: List[tuple[Doc, str]], seed_sha: str,
               silver: Iterable[str] = SILVER_BATCHES) -> tuple[List[Dict], Counter]:
    tasks: List[Dict] = []
    skipped: Counter = Counter()
    for r in records:
        prov = r.get("provenance") or {}
        if prov.get("gold_batch") in silver:
            skipped["silver batch"] += 1
            continue
        items = [i for i in (r.get("expected_output") or {}).get("regulatory_items") or []
                 if i.get("item_type") in MODALITY and i.get("source_text")]
        if not items:
            skipped["no typed items"] += 1
            continue
        context = context_of(r)
        if len(context) > MAX_CONTEXT_CHARS:
            skipped["context too long"] += 1
            continue
        doc = find_green([i["source_text"] for i in items], green)
        if doc is None:
            skipped["not in a GREEN corpus document"] += 1
            continue
        for item_type in MODALITY:
            spans = [i["source_text"].replace("\r\n", "\n") for i in items if i["item_type"] == item_type]
            if not spans or any(locate(s, context) is None for s in spans):
                continue
            key = f"{r['example_id']}|{item_type}"
            tasks.append({
                "task_id": task_id("obligation_span", doc.source_id, key),
                "family": "obligation_span",
                "source_id": doc.source_id,
                "doc_hash": doc.doc_hash,
                "regulator": doc.regulator,
                "document_type": doc.document_type,
                "title": doc.title,
                "url": doc.url,
                "tier": doc.tier,
                "licence": doc.licence,
                "context": context,
                "question": QUESTION.format(what=DESCRIBE[item_type], modality=MODALITY[item_type]),
                "gold": {"spans": spans, "modality": MODALITY[item_type], "item_type": item_type},
                "origin": "seed",
                "seed": {"example_id": r["example_id"], "gold_batch": prov.get("gold_batch"),
                         "annotation_status": prov.get("annotation_status"), "file_sha256": seed_sha},
            })
    for t in tasks:
        verdict = verify(t, json.dumps({"spans": t["gold"]["spans"], "modality": t["gold"]["modality"]}))
        if not verdict.passed:
            raise AssertionError(f"{t['task_id']}: gold fails its verifier ({verdict.reason})")
    return tasks, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    raw = args.seed.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    green = [(d, loose(d.text)) for d in iter_docs()]
    tasks, skipped = seed_tasks(records, green, hashlib.sha256(raw).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(args.out), "records": len(records), "tasks": len(tasks),
        "by_type": Counter(t["gold"]["item_type"] for t in tasks),
        "by_regulator": Counter(t["regulator"] for t in tasks),
        "skipped_records": skipped,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
