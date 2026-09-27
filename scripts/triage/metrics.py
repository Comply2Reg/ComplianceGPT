"""Scoring a triage model's predictions against gold records.

Shared by train_triage.py (CUDA/unsloth) and train_mlx.py (Apple Silicon) so
both produce the same eval_<v>_<model>.json.

`alert_class_macro_f1` averages every class the test set touches, so a class
with one or two gold items weighs exactly as much as one with 239. On the v2
test split A1 has 1 gold item, A4 has 2 and A13 has 3: a single prediction
moves the headline by ~7 points, which is how a run can look like a
regression without the model changing in any way that matters. The
`macro_f1_by_min_support` thresholds and `weighted_f1` are the figures to read
when comparing two checkpoints; the unthresholded one stays first so existing
cards and reports keep their meaning.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from scripts.triage.taxonomy import ALERT_CLASSES


def score(records: List[Dict], predictions: List[Optional[Dict]]) -> Dict:
    """`predictions[i]` is the parsed JSON for `records[i]`, or None."""
    n = len(records)
    hits: Counter = Counter()
    tp: Counter = Counter()
    pred_count: Counter = Counter()
    gold_count: Counter = Counter()
    jaccards: List[float] = []
    n_valid = 0
    for rec, pred in zip(records, predictions):
        gold = rec["output"]
        gold_count[gold["alert_class"]] += 1
        if not pred:
            continue
        n_valid += 1
        pc = pred.get("alert_class")
        pred_count[pc] += 1
        if pc == gold["alert_class"]:
            hits["alert_class"] += 1
            tp[pc] += 1
        if pred.get("priority") == gold.get("priority"):
            hits["priority"] += 1
        if pred.get("obligations_present") == gold.get("obligations_present"):
            hits["obligations_present"] += 1
        a = set(pred.get("primary_functions") or [])
        b = set(gold.get("primary_functions") or [])
        jaccards.append(len(a & b) / len(a | b) if (a | b) else 1.0)
    f1s = []
    per_class: Dict[str, Dict] = {}
    for c in ALERT_CLASSES:
        p, g = pred_count[c], gold_count[c]
        if p == 0 and g == 0:
            continue
        prec = tp[c] / p if p else 0.0
        rec_ = tp[c] / g if g else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
        f1s.append(f1)
        per_class[c] = {
            "gold": g,
            "pred": p,
            "tp": tp[c],
            "precision": round(prec, 4),
            "recall": round(rec_, 4),
            "f1": round(f1, 4),
        }
    # Thresholds are on gold support: a class the test set cannot measure is
    # excluded rather than scored 0, so the number reflects the model instead
    # of the split's tail.
    macro_by_support = {}
    for floor in (1, 5, 10, 20):
        sel = [v["f1"] for v in per_class.values() if v["gold"] >= floor]
        macro_by_support[str(floor)] = {
            "macro_f1": round(sum(sel) / len(sel), 4) if sel else None,
            "classes": len(sel),
        }
    total_gold = sum(v["gold"] for v in per_class.values())
    weighted = (
        round(
            sum(v["f1"] * v["gold"] for v in per_class.values()) / total_gold, 4
        )
        if total_gold
        else None
    )
    return {
        "records": n,
        "json_valid_rate": round(n_valid / n, 4) if n else None,
        "alert_class_accuracy": round(hits["alert_class"] / n, 4) if n else None,
        "alert_class_macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "priority_accuracy": round(hits["priority"] / n, 4) if n else None,
        "obligations_present_accuracy": (
            round(hits["obligations_present"] / n, 4) if n else None
        ),
        "primary_function_jaccard": (
            round(sum(jaccards) / len(jaccards), 4) if jaccards else None
        ),
        "per_class_gold": dict(gold_count),
        "per_class_tp": dict(tp),
        "macro_f1_by_min_support": macro_by_support,
        "weighted_f1": weighted,
        "per_class": per_class,
        "classes_scored": sum(1 for v in per_class.values() if v["tp"]),
        "classes_present": sum(1 for v in per_class.values() if v["gold"]),
    }
