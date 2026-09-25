#!/usr/bin/env python3
"""Run public benchmarks against a local MLX model, base and fine-tuned.

No public benchmark scores UK regulatory alert triage, so nothing here measures
the product metric. What these measure is whether fine-tuning on 1,683 narrow
JSON examples preserved the legal and financial ability the base model had, or
spent it. The answer is the **delta between the two models**, never the absolute
score: a 4B model's standalone number on an out-of-domain legal benchmark is not
interesting, and ours is additionally format-locked — trained to answer every
prompt with a triage record, which costs it marks on any task wanting a
different shape. That cost is itself a finding, so it is measured rather than
prompted away.

    # both sides of the comparison
    python -m scripts.triage.bench run --task ledgar --limit 500 \\
        --model-path models/publish/regulatory-alert-triage-qwen3-4b-v1
    python -m scripts.triage.bench run --task ledgar --limit 500      # base

    python -m scripts.triage.bench compare --task ledgar
    python -m scripts.triage.bench run --task ledgar --limit 5 --dry-run

Needs `datasets` (arrives with lm-eval). Tasks lm-eval already ships, such as
LegalBench, are run through `python -m mlx_lm evaluate` instead of here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import write_jsonl  # noqa: E402
from scripts.triage.render import parse_assistant_json  # noqa: E402

log = logging.getLogger("triage.bench")

BENCH_DIR = Path("data/uk/bench")


# ── scorers ─────────────────────────────────────────────────────────────────


def score_single_label(golds: List, preds: List) -> Dict:
    n = len(golds)
    hits = sum(1 for g, p in zip(golds, preds) if p is not None and p == g)
    answered = sum(1 for p in preds if p is not None)
    return {
        "n": n,
        "accuracy": round(hits / n, 4) if n else None,
        "answered_rate": round(answered / n, 4) if n else None,
        # Accuracy among the items it actually answered, which separates
        # "wrong" from "did not produce a usable label at all".
        "accuracy_when_answered": (
            round(hits / answered, 4) if answered else None
        ),
    }


def score_multi_label(golds: List, preds: List) -> Dict:
    tp = fp = fn = 0
    exact = 0
    answered = 0
    for g, p in zip(golds, preds):
        gold = set(g)
        if p is None:
            fn += len(gold)
            continue
        answered += 1
        pred = set(p)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += gold == pred
    n = len(golds)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "micro_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "micro_precision": round(prec, 4),
        "micro_recall": round(rec, 4),
        "exact_set_match": round(exact / n, 4) if n else None,
        "answered_rate": round(answered / n, 4) if n else None,
    }


def score_schema(golds: List, preds: List) -> Dict:
    """JSON validity, and conformance to the supplied schema where checkable."""
    try:
        import jsonschema  # optional
    except ImportError:
        jsonschema = None
    n = len(golds)
    valid_json = conforms = 0
    checkable = 0
    for schema, pred in zip(golds, preds):
        if pred is None:
            continue
        valid_json += 1
        if jsonschema is None:
            continue
        checkable += 1
        try:
            jsonschema.validate(pred, schema)
            conforms += 1
        except Exception:
            pass
    return {
        "n": n,
        "json_valid_rate": round(valid_json / n, 4) if n else None,
        "schema_conformance": (
            round(conforms / checkable, 4) if checkable else None
        ),
        "schema_checked": checkable,
        "note": None if jsonschema else "install jsonschema to check conformance",
    }


def score_probe(golds: List, preds: List) -> Dict:
    """Descriptive only: a probe reports behaviour, it does not grade it."""
    n = len(golds)
    produced = [p for p in preds if isinstance(p, dict)]
    flagged = sum(1 for p in produced if p.get("obligations_present") is True)
    classes: Dict[str, int] = {}
    for p in produced:
        c = p.get("alert_class")
        if c:
            classes[c] = classes.get(c, 0) + 1
    return {
        "n": n,
        "produced_triage_json": round(len(produced) / n, 4) if n else None,
        "obligations_present_rate": (
            round(flagged / len(produced), 4) if produced else None
        ),
        "alert_class_distribution": dict(
            sorted(classes.items(), key=lambda kv: -kv[1])
        ),
        "descriptive_only": True,
    }


# ── label recovery from free text ───────────────────────────────────────────


def _match_labels(text: str, labels: Sequence[str]) -> List[str]:
    """Labels named in the output, longest first so 'Choice of law' beats 'law'."""
    hay = text.lower()
    found = []
    for label in sorted(labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label.lower())}\b", hay):
            if not any(label.lower() in f.lower() for f in found):
                found.append(label)
    return found


def pick_one(text: str, labels: Sequence[str]) -> Optional[str]:
    # A triage-locked model answers with JSON; look inside it before giving up.
    obj = parse_assistant_json(text)
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, str):
                hit = _match_labels(value, labels)
                if hit:
                    return hit[0]
    hits = _match_labels(text, labels)
    return hits[0] if hits else None


def pick_many(text: str, labels: Sequence[str]) -> Optional[List[str]]:
    obj = parse_assistant_json(text)
    if isinstance(obj, dict):
        pooled = " ".join(str(v) for v in obj.values())
        hits = _match_labels(pooled, labels)
        if hits:
            return hits
    hits = _match_labels(text, labels)
    return hits if hits else ([] if text.strip() else None)


# ── the benchmark registry ──────────────────────────────────────────────────
#
# Verified against the Hugging Face datasets server on 2026-09-25. Two datasets
# were considered and rejected, recorded here so they are not re-proposed:
#
#   LexGLUE/eurlex  — labels are bare EuroVoc concept ids ("100163"), so a
#                     zero-shot generative model cannot name them and the score
#                     would measure nothing. unfair_tos covers multi-label with
#                     readable class names instead.
#   FinBen          — no individually addressable dataset on the Hub, and its
#                     tasks are QA, sentiment and numeric reasoning rather than
#                     document classification. ObliQA carries the
#                     financial-regulatory slot.


def _ledgar(row, labels):
    system = (
        "You classify contract provisions. Reply with exactly one label from "
        "the list and nothing else."
    )
    user = (
        "Labels:\n"
        + ", ".join(labels)
        + "\n\nProvision:\n"
        + row["text"][:4000]
        + "\n\nLabel:"
    )
    return system, user, labels[row["label"]]


def _unfair_tos(row, labels):
    system = (
        "You review consumer terms of service for potentially unfair clauses. "
        "Reply with the applicable labels as a comma-separated list, or the "
        "word None."
    )
    user = (
        "Labels:\n"
        + ", ".join(labels)
        + "\n\nClause:\n"
        + row["text"][:4000]
        + "\n\nApplicable labels:"
    )
    return system, user, [labels[i] for i in row["labels"]]


def _jsonschema(row, _labels):
    schema = json.loads(row["json_schema"])
    system = (
        "You produce JSON. Reply with a single JSON object that satisfies the "
        "given JSON Schema. Output only the JSON."
    )
    user = "JSON Schema:\n" + json.dumps(schema)[:4000] + "\n\nJSON:"
    return system, user, schema


def _obliqa(row, _labels):
    """Financial-regulatory text through our own triage prompt, as a probe.

    ObliQA carries no triage labels, so this cannot be graded. What it shows is
    how the model behaves on financial regulation from a jurisdiction it never
    saw: whether it still emits a well-formed record, and what it says about
    obligations in text that is, by construction, obligation-bearing.
    """
    from scripts.triage.render import TRAIN_SYSTEM_PROMPT

    passages = "\n\n".join(p["Passage"] for p in row["Passages"])[:4000]
    user = (
        "Instruction: You are the regulatory-change desk of a UK bank. Read the "
        "publication and return a triage record.\nContext: Regulator: ADGM\n"
        "Publication type: rulebook\nTitle: "
        + row["Question"][:200]
        + "\n\n"
        + passages
    )
    return TRAIN_SYSTEM_PROMPT, user, row["QuestionID"]


BENCHMARKS: Dict[str, Dict] = {
    "ledgar": {
        "dataset": "coastalcph/lex_glue",
        "config": "ledgar",
        "split": "test",
        "labels_from": "label",
        "build": _ledgar,
        "parse": pick_one,
        "score": score_single_label,
        "max_new_tokens": 32,
        "about": "100-class contract provision classification (LexGLUE)",
    },
    "unfair_tos": {
        "dataset": "coastalcph/lex_glue",
        "config": "unfair_tos",
        "split": "test",
        "labels_from": "labels",
        "build": _unfair_tos,
        "parse": pick_many,
        "score": score_multi_label,
        "max_new_tokens": 64,
        "about": "8-class multi-label unfair terms-of-service clauses (LexGLUE)",
    },
    "jsonschema": {
        "dataset": "epfl-dlab/JSONSchemaBench",
        "config": "Github_easy",
        "split": "test",
        "labels_from": None,
        "build": _jsonschema,
        "parse": lambda text, _l: parse_assistant_json(text),
        "score": score_schema,
        "max_new_tokens": 400,
        "about": "does the JSON skill generalise beyond the trained schema",
    },
    "obliqa": {
        "dataset": "RegNLP/ObliQA",
        "config": "default",
        "split": "test",
        "labels_from": None,
        "build": _obliqa,
        "parse": lambda text, _l: parse_assistant_json(text),
        "score": score_probe,
        "max_new_tokens": 400,
        "about": "ADGM financial regulation, behavioural probe (not graded)",
    },
}


def label_names(dataset, key: Optional[str]) -> List[str]:
    if not key:
        return []
    feature = dataset.features[key]
    names = getattr(feature, "names", None)
    if names is None:  # Sequence(ClassLabel) for the multi-label sets
        names = getattr(getattr(feature, "feature", None), "names", None)
    if names is None:
        raise ValueError(f"no class names on feature {key!r}")
    return list(names)


def sample_rows(dataset, limit: Optional[int], seed: int):
    """A seeded sample, so a run is reproducible and not just the first N rows."""
    n = len(dataset)
    if not limit or limit >= n:
        return list(range(n)), dataset
    idx = sorted(random.Random(seed).sample(range(n), limit))
    return idx, dataset.select(idx)


# ── run ─────────────────────────────────────────────────────────────────────


def model_tag(model_path: Optional[Path], mlx_id: str) -> str:
    return model_path.name if model_path else "base"


def run(args) -> int:
    from datasets import load_dataset

    spec = BENCHMARKS[args.task]
    log.info("%s: %s", args.task, spec["about"])
    ds = load_dataset(spec["dataset"], spec["config"], split=spec["split"])
    labels = label_names(ds, spec["labels_from"])
    indices, rows = sample_rows(ds, args.limit, args.seed)
    log.info("%d of %d rows (seed %d)", len(rows), len(ds), args.seed)

    built = [spec["build"](row, labels) for row in rows]
    if args.dry_run:
        for (system, user, gold), i in list(zip(built, indices))[:3]:
            print(f"\n--- row {i} ---\nSYSTEM: {system}\nUSER: {user[:600]}")
            print(f"GOLD: {str(gold)[:200]}")
        log.info("dry run: %d prompts built, no model loaded", len(built))
        return 0

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    mlx_id = cfg.MODELS[args.model]["mlx_id"]
    target = str(args.model_path) if args.model_path else mlx_id
    log.info("loading %s", target)
    model, tokenizer = load(target)
    sampler = make_sampler(temp=0.0)

    preds, outputs = [], []
    started = time.monotonic()
    for i, (system, user, gold) in enumerate(built, 1):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=spec["max_new_tokens"],
            sampler=sampler,
            verbose=False,
        )
        pred = spec["parse"](text, labels)
        preds.append(pred)
        outputs.append({"gold": gold, "pred": pred, "raw": text})
        if i % 25 == 0:
            log.info("%d/%d", i, len(built))
    elapsed = time.monotonic() - started

    golds = [g for _, _, g in built]
    result = {
        "task": args.task,
        "about": spec["about"],
        "dataset": spec["dataset"],
        "config": spec["config"],
        "split": spec["split"],
        "model": target,
        "is_base": args.model_path is None,
        "n_sampled": len(built),
        "n_available": len(ds),
        "seed": args.seed,
        "max_new_tokens": spec["max_new_tokens"],
        "seconds": round(elapsed),
        "evaluated_at": dt.datetime.now(dt.UTC).isoformat(),
        "metrics": spec["score"](golds, preds),
    }
    tag = model_tag(args.model_path, mlx_id)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"bench_{args.task}_{tag}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_jsonl(out_dir / f"bench_{args.task}_{tag}_outputs.jsonl", outputs)
    print(json.dumps(result, indent=2))
    log.info("-> %s", out_dir / f"bench_{args.task}_{tag}.json")
    return 0


def compare(args) -> int:
    """Base versus fine-tuned on one task. The delta is the result."""
    out_dir = args.out_dir
    base_path = out_dir / f"bench_{args.task}_base.json"
    tuned_path = out_dir / f"bench_{args.task}_{args.tuned_tag}.json"
    for p in (base_path, tuned_path):
        if not p.exists():
            log.error("missing %s — run both sides first", p)
            return 1
    base = json.loads(base_path.read_text(encoding="utf-8"))
    tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
    if base["seed"] != tuned["seed"] or base["n_sampled"] != tuned["n_sampled"]:
        log.error("the two runs used different samples; not comparable")
        return 1
    deltas = {}
    for key, tv in tuned["metrics"].items():
        bv = base["metrics"].get(key)
        if isinstance(tv, (int, float)) and isinstance(bv, (int, float)):
            deltas[key] = round(tv - bv, 4)
    result = {
        "task": args.task,
        "about": tuned["about"],
        "n_sampled": tuned["n_sampled"],
        "seed": tuned["seed"],
        "base": base["metrics"],
        "tuned": tuned["metrics"],
        "delta": deltas,
        "compared_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    (out_dir / f"compare_{args.task}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=("run", "compare", "list"))
    ap.add_argument("--task", choices=sorted(BENCHMARKS), default=None)
    ap.add_argument("--model", default=cfg.FOCUS_MODEL, choices=sorted(cfg.MODELS))
    ap.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="a local MLX directory; omit to run the untuned base model",
    )
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=BENCH_DIR)
    ap.add_argument(
        "--tuned-tag",
        default="regulatory-alert-triage-qwen3-4b-v1",
        help="compare: the directory name the tuned run was tagged with",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build and print prompts; load no model",
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # A dataset download logs one line per HTTP request, which buries the run.
    for noisy in ("httpx", "urllib3", "filelock", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.command == "list":
        for name, spec in sorted(BENCHMARKS.items()):
            print(f"{name:12} {spec['about']}")
            print(f"{'':12} {spec['dataset']} [{spec['config']}] {spec['split']}")
        return 0
    if not args.task:
        log.error("--task is required for %s", args.command)
        return 1
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
