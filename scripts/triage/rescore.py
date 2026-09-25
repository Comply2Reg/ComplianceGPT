#!/usr/bin/env python3
"""Score saved predictions against a second, independent set of labels.

The test split's labels were written by the same model that wrote the training
labels, so the headline accuracy measures agreement with that labeller rather
than correctness. Re-labelling the same documents with a stronger model and
scoring the *same* predictions against both gives an error bar instead of a
single number: if the two labellers disagree more than the model does, the
headline figure is largely labeller noise.

Nothing is regenerated. Predictions come from the eval run's `_outputs.jsonl`,
so this is cheap and cannot drift from what was measured.

    python -m scripts.triage.rescore \\
        --dataset-dir data/uk/datasets/green --version v1 \\
        --eval-outputs data/uk/datasets/green/eval_v1_..._outputs.jsonl \\
        --judge-labels data/uk/labels/triage_judge1.jsonl \\
        [--cache-dir data/uk/labels/cache] [--out data/uk/datasets/green/rescore_v1.json]

Exits non-zero if scoring the original labels fails to reproduce the published
evaluation: that is the check that the re-scoring path itself is sound before
any conclusion is drawn from the judge's numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.triage.corpus import read_jsonl  # noqa: E402
from scripts.triage.metrics import score  # noqa: E402
from scripts.triage.render import parse_assistant_json  # noqa: E402
from scripts.triage.report import compare_pair, cross_model  # noqa: E402
from scripts.triage.schema import OUTPUT_FIELDS  # noqa: E402

log = logging.getLogger("triage.rescore")

SCORED = (
    "json_valid_rate",
    "alert_class_accuracy",
    "alert_class_macro_f1",
    "priority_accuracy",
    "obligations_present_accuracy",
    "primary_function_jaccard",
)


def predictions_for(
    records: List[Dict], outputs_path: Path
) -> Tuple[List[Optional[Dict]], int]:
    """Re-parse the saved generations, aligned to `records` by source_id."""
    by_id = {
        row["source_id"]: row
        for row in read_jsonl(outputs_path)
        if row.get("source_id")
    }
    preds: List[Optional[Dict]] = []
    missing = 0
    for rec in records:
        row = by_id.get(rec["metadata"]["source_id"])
        if row is None:
            missing += 1
            preds.append(None)
            continue
        preds.append(parse_assistant_json(row.get("raw") or ""))
    return preds, missing


def relabelled(records: List[Dict], labels: Dict[str, Dict]) -> List[Dict]:
    """The same records with `output` replaced by another labeller's verdict.

    Records with no judge label are dropped rather than scored against a stale
    one, and the caller is told how many went.
    """
    out = []
    for rec in records:
        label = labels.get(rec["metadata"]["doc_hash"])
        if not label:
            continue
        swapped = dict(rec)
        swapped["output"] = {f: label[f] for f in OUTPUT_FIELDS if f in label}
        out.append(swapped)
    return out


def load_labels(path: Path) -> Dict[str, Dict]:
    return {
        row["doc_hash"]: row["label"]
        for row in read_jsonl(path)
        if row.get("doc_hash") and row.get("label")
    }


def reproduces(measured: Dict, published: Dict, tol: float = 5e-4) -> List[str]:
    """Which metrics fail to match the published evaluation."""
    drift = []
    for key in SCORED:
        a, b = measured.get(key), published.get(key)
        if a is None or b is None:
            continue
        if abs(float(a) - float(b)) > tol:
            drift.append(f"{key}: rescored {a} vs published {b}")
    return drift


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--eval-outputs", type=Path, required=True)
    ap.add_argument(
        "--judge-labels",
        type=Path,
        default=None,
        help="a second labeller's output; omit to only self-check",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/uk/labels/cache"),
        help="label cache, for labeller-vs-labeller agreement",
    )
    ap.add_argument("--out", type=Path, default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = read_jsonl(args.dataset_dir / f"test_{args.version}.jsonl")
    if not records:
        log.error("no test split under %s", args.dataset_dir)
        return 1
    preds, missing = predictions_for(records, args.eval_outputs)
    if missing:
        log.warning("%d record(s) had no saved prediction", missing)

    original = score(records, preds)
    result = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "version": args.version,
        "eval_outputs": str(args.eval_outputs),
        "records": len(records),
        "against_original_labels": {k: original.get(k) for k in SCORED},
    }

    # Self-check: the same predictions scored against the same labels must give
    # back the published numbers, or nothing below can be trusted.
    published_path = Path(str(args.eval_outputs).replace("_outputs.jsonl", ".json"))
    if published_path.exists():
        published = json.loads(published_path.read_text(encoding="utf-8"))
        drift = reproduces(original, published)
        result["reproduces_published_eval"] = not drift
        if drift:
            for d in drift:
                log.error("%s", d)
            log.error("re-scoring does not reproduce the published evaluation")
            return 1
        log.info("self-check passed: reproduces %s", published_path.name)

    if args.judge_labels:
        labels = load_labels(args.judge_labels)
        judged = relabelled(records, labels)
        log.info("judge labels cover %d of %d test records", len(judged), len(records))
        if judged:
            judged_ids = {r["metadata"]["source_id"] for r in judged}
            judged_preds = [
                p
                for r, p in zip(records, preds)
                if r["metadata"]["source_id"] in judged_ids
            ]
            against_judge = score(judged, judged_preds)
            result["against_judge_labels"] = {
                k: against_judge.get(k) for k in SCORED
            }
            result["judge_coverage"] = len(judged)
            result["delta"] = {
                k: round(
                    (against_judge.get(k) or 0) - (original.get(k) or 0), 4
                )
                for k in SCORED
            }

        # Which two labellers actually bear on this comparison: the one that
        # wrote the test labels, and the judge. Other models may sit in the
        # cache from earlier passes, and their agreement says nothing about
        # this test set.
        from collections import Counter

        test_labeller = Counter(
            r["metadata"].get("model") for r in records if r["metadata"].get("model")
        ).most_common(1)
        judge_model = Counter(
            row.get("model")
            for row in read_jsonl(args.judge_labels)
            if row.get("model")
        ).most_common(1)
        if test_labeller and judge_model:
            a, b = sorted({test_labeller[0][0], judge_model[0][0]})
            result["primary_pair"] = f"{a} vs {b}"
            result["test_labeller"] = test_labeller[0][0]
            result["judge_model"] = judge_model[0][0]

        # How much the two labellers themselves agree, on these documents only.
        hashes = {r["metadata"]["doc_hash"] for r in records}
        overlap = {
            h: m for h, m in cross_model(args.cache_dir).items() if h in hashes
        }
        models = sorted({m for v in overlap.values() for m in v})
        result["labellers_compared"] = models
        if len(models) >= 2:
            pairs = {}
            for i, a in enumerate(models):
                for b in models[i + 1 :]:
                    cmp = compare_pair(overlap, a, b)
                    if cmp:
                        cmp.pop("disputed_hashes", None)
                        pairs[f"{a} vs {b}"] = cmp
            result["labeller_agreement"] = pairs

    out = args.out or args.dataset_dir / f"rescore_{args.version}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    log.info("rescore -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
