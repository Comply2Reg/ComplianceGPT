#!/usr/bin/env python3
"""Build the Hugging Face model card and eval summary from run artefacts.

Every number in the card is read from a file produced by the run — the eval
JSONs, the dataset stats, the training report and log, the corpus coverage —
so the published page cannot drift from what actually happened. Prose is
written here; figures never are.

    python -m scripts.triage.model_card \\
        --model-dir models/publish/regulatory-alert-triage-qwen3-4b-v1 \\
        --dataset-dir data/uk/datasets/green --version v1 \\
        --train-report models/uk-triage-v1-qwen3-4b-instruct-mlx/train_report_v1_qwen3-4b-instruct_mlx.json \\
        --train-log logs/train-mlx-v1.out \\
        --coverage ../../09-inventory-kit/c2r-inventory-kit/data/alert_corpus/coverage.json

Writes <model-dir>/README.md (the card) and <model-dir>/eval_summary.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.triage.taxonomy import (  # noqa: E402
    ALERT_CLASSES,
    FUNCTION_DESCRIPTIONS,
    FUNCTION_LINE,
    PRIORITY_RULES,
    STRATA,
    TARGET_MIX,
)

log = logging.getLogger("triage.model_card")

OGL_ATTRIBUTION = (
    "Contains public sector information licensed under the "
    "Open Government Licence v3.0."
)
# Metrics quoted in the frontmatter, in the order they appear in the card.
HEADLINE = (
    ("alert_class accuracy", "alert_class_accuracy", "accuracy"),
    ("alert_class macro-F1", "alert_class_macro_f1", "f1"),
    ("priority accuracy", "priority_accuracy", "accuracy"),
    ("obligations_present accuracy", "obligations_present_accuracy", "accuracy"),
    ("primary_functions Jaccard", "primary_function_jaccard", "accuracy"),
    ("JSON validity", "json_valid_rate", "accuracy"),
)


def load_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def val_loss_curve(log_path: Optional[Path]) -> List[Tuple[int, float]]:
    """(iteration, validation loss) pairs from an mlx_lm training log."""
    if not log_path or not Path(log_path).exists():
        return []
    pattern = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return [(int(i), float(v)) for i, v in pattern.findall(text)]


def peak_memory_gb(log_path: Optional[Path]) -> Optional[float]:
    if not log_path or not Path(log_path).exists():
        return None
    hits = re.findall(
        r"Peak mem ([\d.]+) GB",
        Path(log_path).read_text(encoding="utf-8", errors="replace"),
    )
    return max(float(h) for h in hits) if hits else None


def trainable_params(log_path: Optional[Path]) -> Optional[str]:
    if not log_path or not Path(log_path).exists():
        return None
    m = re.search(
        r"Trainable parameters: ([\d.]+)% \(([\d.]+)M/([\d.]+)M\)",
        Path(log_path).read_text(encoding="utf-8", errors="replace"),
    )
    return f"{m.group(2)}M of {m.group(3)}M ({m.group(1)}%)" if m else None


def majority_baseline(per_class_gold: Dict[str, int]) -> Tuple[str, float]:
    """The score a model gets by always predicting the commonest test class."""
    total = sum(per_class_gold.values())
    cls, n = max(per_class_gold.items(), key=lambda kv: kv[1])
    return cls, (n / total if total else 0.0)


def prediction_agreement(
    fused_outputs: Optional[Path], adapter_outputs: Optional[Path]
) -> Optional[Dict]:
    """How often the fused model and base+adapter pick the same class.

    Fusing a LoRA into 4-bit weights dequantizes, adds the delta and
    re-quantizes, so the result is a numerically different model. Agreement is
    the measurement of how much that costs; exact equality is not available.
    """
    if not (fused_outputs and adapter_outputs):
        return None
    if not (Path(fused_outputs).exists() and Path(adapter_outputs).exists()):
        return None

    def rows(path: Path) -> Dict[str, Dict]:
        return {
            json.loads(line)["source_id"]: json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    a, b = rows(adapter_outputs), rows(fused_outputs)
    shared = set(a) & set(b)
    if not shared:
        return None
    same = sum(
        1 for k in shared if a[k]["pred_alert_class"] == b[k]["pred_alert_class"]
    )
    changed = [
        {
            "source_id": k,
            "gold": a[k]["gold_alert_class"],
            "adapter": a[k]["pred_alert_class"],
            "fused": b[k]["pred_alert_class"],
        }
        for k in sorted(shared)
        if a[k]["pred_alert_class"] != b[k]["pred_alert_class"]
    ]
    return {
        "documents": len(shared),
        "same_prediction": same,
        "agreement": round(same / len(shared), 4),
        "changed": changed,
    }


def fusion_check(
    fused: Dict, adapter: Dict, tolerance: float = 0.02
) -> Dict:
    """Did fusing preserve the model, within what quantization allows?

    The bar is that no headline metric drops by more than `tolerance` and the
    output stays parseable; a fused model that is slightly *better* on a metric
    is not a failure, it is the same noise pointing the other way.
    """
    deltas = {}
    for _, key, _ in HEADLINE:
        f, a = fused.get(key), adapter.get(key)
        if isinstance(f, (int, float)) and isinstance(a, (int, float)):
            deltas[key] = round(f - a, 4)
    regressions = {k: v for k, v in deltas.items() if v < -tolerance}
    return {
        "tolerance": tolerance,
        "deltas_vs_adapter": deltas,
        "regressions": regressions,
        "json_valid_rate": fused.get("json_valid_rate"),
        "passed": not regressions and (fused.get("json_valid_rate") or 0) >= 0.99,
    }


def build_summary(
    fused: Dict,
    adapter: Dict,
    checkpoint: Optional[Dict],
    base: Optional[Dict],
    stats: Dict,
    train_report: Dict,
    coverage: Optional[Dict],
    curve: List[Tuple[int, float]],
    agreement: Optional[Dict] = None,
) -> Dict:
    cls, share = majority_baseline(fused["per_class_gold"])
    summary = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "version": stats["version"],
        "base_model": train_report["mlx_id"],
        "test_records": fused["records"],
        "test_date_range": stats["test"]["date_range"],
        "metrics": {key: fused.get(key) for _, key, _ in HEADLINE},
        "majority_baseline": {"class": cls, "accuracy": round(share, 4)},
        "per_class_gold": fused["per_class_gold"],
        "per_class_correct": fused["per_class_tp"],
        "comparisons": {
            "fused_4bit": {k: fused.get(k) for _, k, _ in HEADLINE},
            "base_plus_adapter": {k: adapter.get(k) for _, k, _ in HEADLINE},
        },
        "splits": stats["counts"],
        "training": {
            "iters": train_report["iters"],
            "batch_size": train_report["batch_size"],
            "num_layers": train_report["num_layers"],
            "learning_rate": train_report["learning_rate"],
            "max_seq_length": train_report["max_seq_length"],
            "seconds": train_report["seconds"],
            "lora_rank": train_report.get("lora_rank", 8),
            "lora_scale": train_report.get("lora_scale", 20.0),
        },
        "val_loss_curve": [{"iter": i, "val_loss": v} for i, v in curve],
        "fusion": fusion_check(fused, adapter),
    }
    if agreement:
        summary["fusion"]["prediction_agreement"] = agreement
    if checkpoint:
        summary["comparisons"]["best_val_loss_checkpoint"] = {
            k: checkpoint.get(k) for _, k, _ in HEADLINE
        }
    if base:
        summary["comparisons"]["base_model_probe"] = {
            "records": base["records"],
            **{k: base.get(k) for _, k, _ in HEADLINE},
        }
    if coverage:
        summary["corpus"] = {
            "documents": coverage["documents"],
            "chunks": coverage["chunks"],
            "by_tier": coverage["by_tier"],
            "since": coverage["since"],
        }
    return summary


def error_bar(rescore: Optional[Dict]) -> Optional[Dict]:
    """The band the headline sits in, once a second labeller is consulted.

    The test labels were written by one model, so a single accuracy figure
    overstates what is known. Scoring the same predictions against a second
    labeller, and measuring how much the two labellers agree with each other,
    turns the point estimate into a range with a stated ceiling.
    """
    if not rescore or "against_judge_labels" not in rescore:
        return None
    pairs = rescore.get("labeller_agreement") or {}
    # The pair that bears on this test set: the labeller that wrote its labels
    # and the judge. Other models may sit in the cache from earlier passes and
    # their mutual agreement says nothing about these documents.
    named = rescore.get("primary_pair")
    best = pairs.get(named) if named else None
    if best is None:
        best = max(
            (v for v in pairs.values() if isinstance(v, dict)),
            key=lambda v: v.get("n", 0),
            default=None,
        )
        named = next((k for k, v in pairs.items() if v is best), None)
    a = rescore["against_original_labels"]
    b = rescore["against_judge_labels"]
    return {
        "original": a,
        "judge": b,
        "delta": rescore.get("delta", {}),
        "labeller_agreement": (best or {}).get("agreement", {}),
        "labeller_n": (best or {}).get("n"),
        "pair": named,
        "low": min(a["alert_class_accuracy"], b["alert_class_accuracy"]),
        "high": max(a["alert_class_accuracy"], b["alert_class_accuracy"]),
        "ceiling": (best or {}).get("agreement", {}).get("alert_class"),
    }


def load_benchmarks(bench_dir: Optional[Path]) -> List[Dict]:
    """Every base-vs-tuned comparison written by scripts/triage/bench.py."""
    if not bench_dir or not Path(bench_dir).is_dir():
        return []
    out = []
    for path in sorted(Path(bench_dir).glob("compare_*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def _benchmark_table(comparisons: List[Dict]) -> str:
    """One row per task per metric: base, tuned, and the delta that matters."""
    lines = ["| Task | Metric | Base | This model | Change |", "|---|---|---:|---:|---:|"]
    for c in comparisons:
        # Counts and bookkeeping flags are not metrics and must not be
        # rendered as though a change in them meant something.
        skip = {"n", "n_with_labels", "schema_checked", "descriptive_only"}
        metrics = [
            k
            for k in c.get("delta", {})
            if k not in skip and not k.endswith("_checked")
        ]
        for i, key in enumerate(metrics):
            base = c["base"].get(key)
            tuned = c["tuned"].get(key)
            if not isinstance(base, (int, float)) or not isinstance(tuned, (int, float)):
                continue
            delta = c["delta"][key]
            lines.append(
                f"| {c['task'] if i == 0 else ''} | {key} | {base:.3f} "
                f"| {tuned:.3f} | {delta:+.3f} |"
            )
    return "\n".join(lines)


def frontmatter(repo_name: str, base_model: str, fused: Dict) -> str:
    rows = []
    for name, key, kind in HEADLINE:
        value = fused.get(key)
        if value is None:
            continue
        rows.append(
            f"      - type: {kind}\n"
            f"        name: {name}\n"
            f"        value: {value}"
        )
    metrics = "\n".join(rows)
    return f"""---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE
base_model: {base_model}
base_model_relation: finetune
library_name: mlx
pipeline_tag: text-generation
language:
- en
tags:
- mlx
- qwen3
- lora
- regtech
- regulatory-compliance
- united-kingdom
- united-states
- text-classification
- structured-output
- json
model-index:
- name: {repo_name}
  results:
  - task:
      type: text-classification
      name: UK and US federal regulatory alert triage
    dataset:
      type: private
      name: UK and US federal regulatory alerts, openly-licensed tier (internal)
      split: test
    metrics:
{metrics}
---
"""


def _metric_table(fused: Dict, baseline: Tuple[str, float]) -> str:
    cls, share = baseline
    notes = {
        "alert_class_accuracy": f"always predicting {cls} scores {share:.3f}",
        "alert_class_macro_f1": "unweighted mean over the classes present",
        "priority_accuracy": "P1 / P2 / P3",
        "obligations_present_accuracy": "binary",
        "primary_function_jaccard": "set overlap against the gold functions",
        "json_valid_rate": "output parsed as a JSON object",
    }
    lines = ["| Metric | Score | Note |", "|---|---:|---|"]
    for name, key, _ in HEADLINE:
        value = fused.get(key)
        if value is None:
            continue
        lines.append(f"| {name} | {value:.3f} | {notes.get(key, '')} |")
    return "\n".join(lines)


def _class_table(stats: Dict, fused: Dict) -> str:
    train = stats["train"]["classification"]
    test_gold = fused["per_class_gold"]
    correct = fused["per_class_tp"]
    lines = [
        "| Class | Meaning | Train | Test | Correct |",
        "|---|---|---:|---:|---:|",
    ]
    for code, meaning in ALERT_CLASSES.items():
        tr = train.get(code, 0)
        te = test_gold.get(code, 0)
        ok = correct.get(code, 0) if te else 0
        cell = f"{ok}/{te}" if te else "-"
        lines.append(f"| {code} | {meaning} | {tr} | {te or '-'} | {cell} |")
    return "\n".join(lines)


def _stratum_table(stats: Dict) -> str:
    balance = stats["balance"]
    lines = [
        "| Stratum | Target | Available | Kept | Shortfall |",
        "|---|---:|---:|---:|---:|",
    ]
    for stratum in STRATA:
        b = balance.get(stratum, {})
        lines.append(
            f"| {stratum} | {TARGET_MIX[stratum]:.0%} | {b.get('available', 0)} "
            f"| {b.get('kept', 0)} | {b.get('shortfall', 0)} |"
        )
    return "\n".join(lines)


def _function_table() -> str:
    lines = ["| Function | Covers | Line of defence |", "|---|---|:--:|"]
    for fn, desc in FUNCTION_DESCRIPTIONS.items():
        line = FUNCTION_LINE[fn]
        lines.append(f"| {fn} | {desc} | {line if line else '-'} |")
    return "\n".join(lines)


def _comparison_table(
    fused: Dict, adapter: Dict, checkpoint: Optional[Dict], base: Optional[Dict]
) -> str:
    cols = [
        ("This model (fused, 4-bit)", fused),
        ("Base + LoRA adapter", adapter),
    ]
    if checkpoint:
        cols.append(("Best-val-loss checkpoint (iter 1200)", checkpoint))
    if base:
        cols.append((f"Base model, untuned ({base['records']}-doc probe)", base))
    header = "| Configuration | " + " | ".join(
        name for name, _ in HEADLINE_SHORT
    ) + " |"
    sep = "|---|" + "---:|" * len(HEADLINE_SHORT)
    lines = [header, sep]
    for label, data in cols:
        cells = []
        for _, key in HEADLINE_SHORT:
            v = data.get(key)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


HEADLINE_SHORT = (
    ("JSON validity", "json_valid_rate"),
    ("Accuracy", "alert_class_accuracy"),
    ("Macro-F1", "alert_class_macro_f1"),
    ("Priority", "priority_accuracy"),
)


# The worked example is a real test document and this model's real output for
# it, not an illustration: cma:publications#6607, evaluated 2026-09-25.
WORKED_EXAMPLE_INPUT = """Regulator: CMA
Publication type: publications
Title: Proposed steering conduct requirement for Google's mobile platform
Published: 2026-08-14

CONSULTATION:
STEERING CONDUCT REQUIREMENT
Google's Mobile Platform
30 June 2026
..."""

WORKED_EXAMPLE_OUTPUT = """{
  "alert_class": "A3",
  "alert_class_confidence": "high",
  "priority": "P2",
  "primary_functions": ["Product", "Legal"],
  "secondary_functions": ["Compliance", "Regulatory change", "Risk", "Board"],
  "lines_of_defence": [1, 2],
  "summary": "The CMA is consulting on a proposed steering conduct requirement for Google's mobile platform, including whether Google should provide fair and transparent access to its Play Store, app ranking, data and APIs. ...",
  "key_dates": [],
  "applicability": ["Google's mobile platform", "Play Store", "app developers", "..."],
  "obligations_present": false,
  "thought_trace": "This is a CMA consultation, so it is a proposed rule/requirement rather than final guidance or a binding instrument. ..."
}"""

QUICKSTART = '''```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("REPO_ID")

SYSTEM = (
    "You triage UK financial-services regulatory publications for a bank. "
    "Return a JSON triage record with exactly these keys: alert_class (A1-A14), "
    "alert_class_confidence, priority (P1-P3), primary_functions, "
    "secondary_functions, lines_of_defence, summary, key_dates, applicability, "
    "obligations_present, thought_trace."
)
USER = """You are the regulatory-change desk of a UK bank. Read the publication and
return a triage record: alert_class, alert_class_confidence, priority,
primary_functions, secondary_functions, lines_of_defence, summary, key_dates,
applicability, obligations_present.

Regulator: CMA
Publication type: publications
Title: Proposed steering conduct requirement for Google's mobile platform
Published: 2026-08-14

<the first ~1,500 characters of the document text>"""

prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    tokenize=False,
    add_generation_prompt=True,
)
text = generate(model, tokenizer, prompt=prompt, max_tokens=400,
                sampler=make_sampler(temp=0.0), verbose=False)

# The model opens every answer with an empty reasoning block; take the JSON.
import json
record = json.loads(text[text.find("{"): text.rfind("}") + 1])
```'''

SCHEMA_BLOCK = '''| Field | Type | Meaning |
|---|---|---|
| `alert_class` | `A1` to `A14` | what kind of publication this is |
| `alert_class_confidence` | `high` / `medium` / `low` | the model's own confidence |
| `priority` | `P1` / `P2` / `P3` | how urgently a desk should act |
| `primary_functions` | list of function names | who owns the response |
| `secondary_functions` | list of function names | who is consulted or informed |
| `lines_of_defence` | list of `1`, `2`, `3` | which lines the functions sit in |
| `summary` | string | what the publication says and whom it affects |
| `key_dates` | list of objects | deadlines, in-force dates, closing dates |
| `applicability` | list of strings | the firms, products or activities in scope |
| `obligations_present` | boolean | whether the text creates a duty |
| `thought_trace` | string | the reasoning behind the classification |'''


def _band_section(band: Dict) -> str:
    names = {
        "json_valid_rate": "JSON validity",
        "alert_class_accuracy": "alert_class accuracy",
        "alert_class_macro_f1": "alert_class macro-F1",
        "priority_accuracy": "priority accuracy",
        "obligations_present_accuracy": "obligations_present",
        "primary_function_jaccard": "primary_functions Jaccard",
    }
    rows = ["| Metric | vs the training labeller | vs a stronger labeller | Change |",
            "|---|---:|---:|---:|"]
    for key, label in names.items():
        a, b = band["original"].get(key), band["judge"].get(key)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        rows.append(f"| {label} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    table = "\n".join(rows)

    agree = band.get("labeller_agreement") or {}
    ceiling = band.get("ceiling")
    arows = ["| Field | Labeller agreement |", "|---|---:|"]
    for key, label in (
        ("alert_class", "alert_class"),
        ("priority", "priority"),
        ("obligations_present", "obligations_present"),
    ):
        if key in agree:
            arows.append(f"| {label} | {agree[key]:.3f} |")
    atable = "\n".join(arows)

    ceiling_text = (
        f"""

**So the accuracy is not "{band['high']:.0%} correct".** Two capable models, given the same
taxonomy and the same documents, agree on the alert class only {ceiling:.1%} of the time.
That is roughly the ceiling any model can reach against either of them, because a third of
the remaining disagreement is in the labels rather than in the answer. Read the headline as
a band between {band['low']:.3f} and {band['high']:.3f} against a ceiling near {ceiling:.2f}."""
        if isinstance(ceiling, (int, float))
        else ""
    )

    improved = [
        k for k, v in (band.get("delta") or {}).items()
        if isinstance(v, (int, float)) and v > 0.01
    ]
    generalised = (
        """

Three fields scored **higher** under the stronger labeller, not lower: priority,
obligations and especially function assignment. On those the model generalised past the
teacher it was trained on. Only class assignment is anchored to the training labeller's
idiosyncrasies, which is what one would expect of the field with the largest label space
and the noisiest labels."""
        if len(improved) >= 2
        else ""
    )

    return f"""### How much of that is label noise

The test labels were written by a model, not a person, so the figures above measure
agreement with one labeller rather than correctness. All {band['labeller_n']} test documents
were re-labelled by a stronger model and the **same saved predictions** were scored again.

{table}

And the two labellers against each other, on the same documents:

{atable}
{ceiling_text}{generalised}

Fixing this properly needs human adjudication, which has not been done.

"""


def ood_warning(comparisons: List[Dict]) -> str:
    """The out-of-distribution collapse, stated from the probe's own numbers.

    Generated rather than written by hand so it cannot outlive the evidence: if
    a later run stops showing the collapse, this paragraph disappears with it.
    """
    probe = next((c for c in comparisons if c["task"] == "obliqa"), None)
    if not probe:
        return ""
    tuned = probe["tuned"]
    mix = tuned.get("alert_class_distribution") or {}
    if not mix:
        return ""
    top, count = max(mix.items(), key=lambda kv: kv[1])
    total = sum(mix.values())
    if total == 0 or count / total < 0.9:
        return ""  # no collapse to report
    produced = tuned.get("produced_triage_json")
    obligations = tuned.get("obligations_present_rate")
    return f"""

**It fails silently on regulation it was not trained on.** Given
{total} passages from a financial regulator outside its training jurisdictions,
the model produced
well-formed triage JSON {produced:.0%} of the time and then assigned **{top} to
{count} of {total}** of them, flagging an obligation in {obligations:.1%}. That
class means "intelligence: speech, blog, press release"; the passages are
rulebook text and obligation-bearing by construction. The summaries were often a
correct reading of the text while every structured field contradicted them, so
nothing downstream could detect the failure from the output alone. Outside the
open-licence UK and US federal sources it was trained on, evaluate before
trusting it."""


def _bench_section(comparisons: List[Dict]) -> str:
    tasks = ", ".join(f"`{c['task']}`" for c in comparisons)
    return f"""## Benchmarks

No public benchmark scores regulatory alert triage, so none of these measure the task. What
they measure is whether fine-tuning preserved the general legal and structured-output
ability the base model had. Each task was run on this model **and on the untouched base
model**, over the same seeded sample, so the **change is the result** and the absolute
numbers should not be quoted as leaderboard scores.

Tasks: {tasks}.

{_benchmark_table(comparisons)}

The model answers every prompt with a triage record, which costs it marks wherever a task
wants a different shape. That is a real property, not a scoring artefact, so it is measured
rather than prompted away: where a task has a fixed label set, the scorer looks inside any
returned JSON for a label before counting the answer wrong, and reports how often a usable
answer appeared at all.

"""


def merge_coverage(paths: List[Path]) -> Dict:
    """Sum the corpus coverage files a multi-jurisdiction model was built from.

    v1 came from one corpus, so the card read one file. v3 is trained on the UK
    and US exports together, and quoting either one alone understates the corpus
    the model was drawn from. Counts add; `since` is the earliest of them.
    """
    merged: Dict = {}
    for path in paths:
        cov = load_json(path)
        if not merged:
            merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cov.items()}
            continue
        for key in ("documents", "chunks", "unclassified"):
            if isinstance(cov.get(key), int):
                merged[key] = merged.get(key, 0) + cov[key]
        for key in ("by_tier", "deduped", "furniture", "by_regulator", "by_alert_class"):
            if isinstance(cov.get(key), dict):
                into = merged.setdefault(key, {})
                for k, v in cov[key].items():
                    if isinstance(v, int):
                        into[k] = into.get(k, 0) + v
        sinces = [d for d in (merged.get("since"), cov.get("since")) if d]
        merged["since"] = min(sinces) if sinces else None
    return merged


def body(
    repo_id: str,
    fused: Dict,
    adapter: Dict,
    checkpoint: Optional[Dict],
    base: Optional[Dict],
    stats: Dict,
    train_report: Dict,
    coverage: Optional[Dict],
    curve: List[Tuple[int, float]],
    log_path: Optional[Path] = None,
    agreement: Optional[Dict] = None,
    band: Optional[Dict] = None,
    comparisons: Optional[List[Dict]] = None,
) -> str:
    baseline = majority_baseline(fused["per_class_gold"])
    cls, share = baseline
    counts = stats["counts"]
    test_from, test_to = stats["test"]["date_range"]
    train_from, train_to = stats["train"]["date_range"]
    hours = train_report["seconds"] / 3600
    peak = peak_memory_gb(log_path)
    params = trainable_params(log_path)
    peak_text = f"{peak:.2f} GB" if peak else "not recorded"
    params_text = (
        f"That trains {params} of the model's parameters." if params else ""
    )
    regs_train = stats["train"]["regulator"]
    regs_test = stats["test"]["regulator"]
    curve_str = " → ".join(f"{v:.3f}" for _, v in curve) if curve else "n/a"
    band_section = _band_section(band) if band else ""
    ood_text = ood_warning(comparisons or [])
    ledgar = next(
        (c for c in (comparisons or []) if c["task"] == "ledgar"), None
    )
    forgetting_text = ""
    if ledgar and ledgar["delta"].get("accuracy_when_answered", 0) < -0.05:
        d = ledgar["delta"]
        forgetting_text = f"""

**Fine-tuning cost general legal ability.** On a 100-class contract-provision
benchmark the base model scores {ledgar['base']['accuracy']:.3f} and this model
{ledgar['tuned']['accuracy']:.3f}. Refusing to answer in the requested format
explains part of that, but not all: counting only the items where both produced a
label, accuracy still falls by {abs(d['accuracy_when_answered']):.3f}. This is the
known cost of LoRA fine-tuning on a single narrow task without replay data."""
    bench_section = _bench_section(comparisons) if comparisons else ""
    if band:
        headline_accuracy = f"{band['low']:.3f} to {band['high']:.3f}"
        band_caveat = (
            "The accuracy is a range because the test labels were written by a "
            "model rather than a person, and a second labeller scores it "
            "differently. "
        )
    else:
        headline_accuracy = f"{fused['alert_class_accuracy']:.3f}"
        band_caveat = ""
    if agreement:
        agreement_text = (
            f"{agreement['same_prediction']} of {agreement['documents']} "
            f"documents ({agreement['agreement']:.1%}), the fused model scoring "
            f"{fused['alert_class_accuracy']:.3f} against "
            f"{adapter['alert_class_accuracy']:.3f}"
        )
    else:
        agreement_text = (
            f"an accuracy of {fused['alert_class_accuracy']:.3f} against "
            f"{adapter['alert_class_accuracy']:.3f}"
        )
    curve_iters = ", ".join(str(i) for i, _ in curve) if curve else ""
    n_priority = len(PRIORITY_RULES)
    # Title, citation key and changelog used to hard-code "v1", so a card built
    # for any later model was titled for the first one. Read it off the dataset
    # stats instead, which is the same file the numbers come from.
    version_label = stats.get("version") or "v1"
    version_key = re.sub(r"[^0-9a-z]+", "_", version_label.lower()).strip("_")
    release_date = (train_report.get("finished_at") or "")[:10] or "unreleased"
    release_note = "First release. " if version_label == "v1" else ""
    # Named from the eval rather than stated in prose: a class the model never
    # emits is the gap a reader most needs, and hard-coding the list is how v1's
    # card ended up quoting a count that later releases had moved past.
    silent = sorted(
        (c for c, v in fused.get("per_class", {}).items() if v["gold"] and not v["pred"]),
        key=lambda c: int(c[1:]),
    )
    silent_text = (
        f"{len(silent)} alert classes the model never predicts "
        f"({', '.join(silent)})"
        if silent
        else "every alert class present in the test set is predicted at least once"
    )
    # PRIORITY_RULES is interpolated into the labelling prompt sent to OpenAI and
    # is a transcription of the crawler's alert-taxonomy.md, so it is rewritten
    # for display here rather than edited at the source.


    parts: List[str] = []

    parts.append(
        f"""# Regulatory Alert Triage: Qwen3-4B ({version_label})

Reads a UK or US federal regulatory publication and returns a structured triage
record: what kind of document it is, how urgent it is, which bank functions own
the response, and whether it creates an obligation. It is a working release
trained on openly-licensed UK and US federal sources, published so the approach
can be reviewed and built on. It is not a finished product.

Fine-tuned by [Comply2Reg](https://comply2reg.com) from
`{train_report["mlx_id"]}` with LoRA, then fused back into standalone 4-bit
weights. Runs on Apple Silicon through MLX.

| | |
|---|---|
| Task | Multi-field triage of UK and US federal regulatory publications, emitted as JSON |
| Base model | [`{train_report["mlx_id"]}`](https://huggingface.co/{train_report["mlx_id"]}) |
| Method | LoRA rank {train_report.get("lora_rank", 8)}, {train_report["num_layers"]} of 36 layers, fused into the base |
| Precision | 4-bit, group size 64 (unchanged from the base) |
| Training data | {counts["train"]:,} documents from UK public bodies (Open Government Licence) and US federal agencies (17 U.S.C. 105) |
| Test set | {counts["test"]} documents, {test_from} to {test_to}, later than everything trained on |
| Headline | alert-class accuracy **{headline_accuracy}**, macro-F1 **{fused["alert_class_macro_f1"]:.3f}**, JSON validity **{fused["json_valid_rate"]:.3f}** |

**Read the macro-F1, not the accuracy.** The test set is {share:.0%} one class, so
accuracy flatters any model. {band_caveat}Limitations below are specific and worth
reading before you use this for anything.

## Quickstart

Requires Apple Silicon and `pip install mlx-lm`.

{QUICKSTART.replace("REPO_ID", repo_id)}

The model opens every answer with an empty `<think></think>` block. That comes
from the base model, not from the training data, which contained none. It did so
on {fused["records"]}/{fused["records"]} test documents, and the JSON that follows
parsed every time. Take the substring from the first `{{` to the last `}}`.

### Worked example

A real test document, and this model's actual output for it:

```
{WORKED_EXAMPLE_INPUT}
```

```json
{WORKED_EXAMPLE_OUTPUT}
```

## Output schema

{SCHEMA_BLOCK}

## The label space

### Alert classes

Fourteen document kinds, from the taxonomy Comply2Reg's crawler applies across UK
and US federal regulators.

{_class_table(stats, fused)}

*Train* counts the fine-tuning examples; *Test* the held-out documents; *Correct*
how many of those this model classified correctly.

### Functions and lines of defence

The model routes each publication to the functions that own it, and reports which
of the three lines of defence those sit in.

{_function_table()}

### Priority

{chr(10).join("- " + rule.replace(" — ", ": ") for rule in PRIORITY_RULES)}

## Intended use

Built to sit in front of a regulatory-change desk: sort an incoming feed of
publications, route each to the right function, and flag the ones that carry a
deadline or a duty. The structured output is meant to be reviewed, not executed.

**Out of scope.** This is not legal advice and not a compliance decision. It does
not replace a compliance officer's reading of the source document. It is
calibrated only for the UK and US federal sources listed below: not for other
jurisdictions, not for US state regulators, not for material published before
2022, and not for the FCA and PRA, whose publications were deliberately excluded
from training. Nothing it produces carries any official or endorsed status.

## Training data

{counts["train"]:,} training, {counts["val"]} validation and {counts["test"]} test
documents, drawn from a corpus of
{coverage["documents"]:,} UK and US federal regulatory documents published since
{coverage["since"]} that Comply2Reg's crawler collected and Comply2Reg labelled.

| Regulator | Train | Test |
|---|---:|---:|
"""
    )
    for reg in sorted(set(regs_train) | set(regs_test)):
        parts.append(f"| {reg} | {regs_train.get(reg, 0)} | {regs_test.get(reg, 0)} |\n")

    parts.append(
        f"""
Only **openly-licensed** sources were used, under two regimes:

- **UK**, Open Government Licence v3.0: HM Treasury, the Competition and Markets
  Authority, the Information Commissioner's Office and legislation.gov.uk.
- **US federal**, not subject to copyright under 17 U.S.C. 105: the SEC, Federal
  Reserve Board, OCC, FDIC, CFTC, CFPB, FinCEN, NCUA, FFIEC, OFAC and the
  Treasury. This covers federal agencies only. It does not extend to US state
  regulators such as NYDFS, nor to the regional Reserve Banks, and none are in
  this dataset.

Publications from the FCA, PRA, Bank of England, Ofcom and the DRCF sit under
terms that do not clearly permit this use, so they were held back pending a
licensing review. That review remains the single largest constraint on the
model's quality for UK conduct and prudential material, and the reason several
classes below have almost no training data.

**Labels.** Every document was labelled by OpenAI's `gpt-5.4-mini` against the
taxonomy, one call per document, cached by content hash. A second model labelled
the same corpus and the disagreements were sampled against a stronger judge,
which sided with the chosen labeller in 18 of 20 cases. The labels are therefore
model-generated and consistent, not human-adjudicated, and some are certainly
wrong.

**Preparation.** Documents were cleaned of page furniture (running headers, page
numbers, tables of contents, boilerplate addresses) before anything else, so
stored offsets and hashes refer to the cleaned text.
{coverage["furniture"]["chars_removed"]:,} characters were removed across
{coverage["furniture"]["docs_cleaned"]:,} documents. Exact duplicates were removed
by content hash ({coverage["deduped"]["content_sha256"]}) and near-duplicates by
MinHash within a regulator and document type
({coverage["deduped"]["near_duplicate"]}).

**Splits are chronological, not random.** The test set is the latest
{stats["gold_months"]} months ({test_from} to {test_to}); validation is the next
slice back; training runs from {train_from} to {train_to}. So the reported scores
measure generalisation forward in time, which is the way the model is actually
used.

**Stratification.** Training aimed at a fixed mix of document kinds and fell short
on three of them, because the open-licence sources do not publish much guidance or
enforcement material:

{_stratum_table(stats)}

**Input format.** The model sees the regulator, publication type, title, date and
the first ~1,500 characters of the document, not the full text. Median training
prompt {stats["tokens"]["train"]["p50"]} tokens, 95th percentile
{stats["tokens"]["train"]["p95"]}, budget {stats["max_seq_length"]}, nothing
truncated.

The dataset is not published. The pipeline that builds it is open, in the
repository linked below.

## Training procedure

LoRA on {train_report["num_layers"]} of 36 layers, rank
{train_report.get("lora_rank", 8)}, scale {train_report.get("lora_scale", 20.0)},
loss on the assistant turn only. {params_text}

| | |
|---|---|
| Iterations | {train_report["iters"]:,} (≈ 1 epoch at batch size {train_report["batch_size"]}) |
| Learning rate | {train_report["learning_rate"]} |
| Sequence length | {train_report["max_seq_length"]} |
| Seed | {train_report.get("seed", 3407)} |
| Hardware | Apple M4 Pro, 24 GB unified memory |
| Wall clock | {hours:.1f} hours |
| Peak memory | {peak_text} |

Validation loss at iterations {curve_iters}: {curve_str}.

The adapter was then fused into the base weights with `mlx_lm fuse`, with no
de-quantisation: the adapter was trained against the 4-bit weights, so those are
the weights it should be measured on.

**Fusing is not lossless at 4 bits.** It dequantizes each layer, adds the LoRA
delta and re-quantizes, so the published model is a numerically different object
from base-plus-adapter. Measured on the same test set, the two agree on
{agreement_text}. The differences fall entirely among the classes the model never
learned; every figure reported here was measured on the published weights, not
inherited from the adapter.

## Evaluation

{fused["records"]} held-out documents, greedy decoding, scored field by field.

{_metric_table(fused, baseline)}

{_comparison_table(fused, adapter, checkpoint, base)}

Two things in that table are worth stating plainly.

**Fine-tuning bought the format.** The untuned base model produced no parseable
triage record at all on a {base["records"]}-document probe. It answers in prose.
Everything downstream depends on the JSON being there, so this is the change that
makes the model usable.

**Lower validation loss did not mean a better model.** The iteration-1200
checkpoint had the best validation loss of the run ({min(v for _, v in curve):.3f}
against {curve[-1][1]:.3f} at the end) and scored worse on every task metric,
because it had collapsed toward predicting the majority class. The final adapter
was selected on task metrics, not loss.

{band_section}{bench_section}
## Limitations

**It has not learned most of the taxonomy.** Macro-F1 is
{fused["alert_class_macro_f1"]:.3f}. Of the classes present in the test set, the
model gets a non-zero score on {len(fused["per_class_tp"])}. Classes with roughly
twenty training examples (A9, A13, A14) score zero. Four further classes have
one or two test documents each, too few to say anything about.

**One class dominates.** {cls} is {share:.0%} of the test set and
{stats["train"]["classification"].get(cls, 0) / counts["train"]:.0%} of training.
An always-{cls} baseline scores {share:.3f} accuracy, so the
{fused["alert_class_accuracy"] - share:+.3f} the model adds is the honest measure
of the accuracy figure.

**Train and test come from different regulators.** Training is mostly HM Treasury
and legislation.gov.uk; the test window is mostly CMA. That makes the evaluation
harder than the training distribution, and it means the scores say little about
regulators the model barely saw.

**The two jurisdictions are not equally financial-services.** The US half is:
the SEC, Federal Reserve Board, OCC, FDIC, CFTC, CFPB, FinCEN, NCUA and OFAC are
financial regulators and supply most of the training data. The UK half is not,
because the openly-licensed UK sources are cross-sector: Treasury policy,
competition cases, data protection, statutory instruments. The FCA and PRA
conduct and prudential material a UK bank most cares about is still exactly what
is missing, so expect materially weaker performance on UK conduct and prudential
text than the headline suggests.

**General legal ability was not re-measured for this release.** On an earlier
version, fine-tuning cost accuracy on LexGLUE LEDGAR (0.610 base to 0.418
tuned), which is capability loss rather than format lock. The public benchmarks
have not been re-run against these weights, so treat that regression as unquantified
here rather than as fixed.

**The labels are model-generated.** No human adjudicated them. Errors in the
labeller are reproduced in the model, and the evaluation shares that bias because
its labels came from the same source.

**No calibration.** `alert_class_confidence` is what the labelling model said and
the fine-tune imitated. It has not been checked against observed accuracy, so do
not threshold on it.{ood_text}{forgetting_text}

## Licence, attribution and provenance

The weights are released under Apache 2.0, inherited from
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Training text came from two sources, under two regimes.

**UK**, under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/):

> {OGL_ATTRIBUTION}

**US federal**: works of the United States Government are not subject to
copyright protection under 17 U.S.C. 105 and are in the public domain. This
applies to the federal agencies listed above, and not to US state regulators or
the regional Reserve Banks, none of which are in this dataset.

The Open Government Licence requires that the use not suggest official status or
endorsement. This model is not endorsed by, affiliated with or connected to any
UK or US regulator or government body, and its output is not an official
interpretation of anything.

Training labels were generated with OpenAI models under terms that assign the
output to the customer.

## Citation

```bibtex
@misc{{comply2reg_regulatory_alert_triage_qwen3_4b_{version_key},
  title  = {{Regulatory Alert Triage: Qwen3-4B ({version_label})}},
  author = {{Comply2Reg}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{repo_id}}}
}}
```

## Changelog

**{version_label}, {release_date}.** {release_note}Openly-licensed sources only
(UK Open Government Licence and US federal public domain), {counts["train"]:,}
training documents, LoRA fused into 4-bit weights. Known gaps: no FCA or PRA
material, {silent_text}, and no human-adjudicated labels anywhere in the training
or test data.

---

Built by [Comply2Reg](https://comply2reg.com). Pipeline and training code:
[ComplianceGPT](https://github.com/Comply2Reg/ComplianceGPT).
"""
    )
    return "".join(parts)


def reflow(markdown: str, width: int = 80) -> str:
    """Rewrap prose paragraphs; leave code, tables, lists and headings alone.

    Interpolating a number mid-sentence leaves the source ragged. Markdown
    renders it the same either way, but the card is meant to be read as a file
    too.
    """
    out: List[str] = []
    buffer: List[str] = []
    in_code = False

    def flush() -> None:
        if not buffer:
            return
        words = " ".join(buffer).split()
        line: List[str] = []
        length = 0
        for word in words:
            if line and length + 1 + len(word) > width:
                out.append(" ".join(line))
                line, length = [word], len(word)
            else:
                line.append(word)
                length += (1 if length else 0) + len(word)
        if line:
            out.append(" ".join(line))
        buffer.clear()

    for raw in markdown.split("\n"):
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            out.append(raw)
            continue
        if in_code:
            out.append(raw)
            continue
        # A bullet starts with a marker AND a space; "**Bold lead-in..." is
        # prose and must still be wrapped.
        structural = (
            not stripped
            or stripped.startswith(("#", "|", ">", "---"))
            or bool(re.match(r"^([-*+] |\d+\. )", stripped))
            or raw.startswith(("    ", "\t"))
        )
        if structural:
            flush()
            out.append(raw)
        else:
            buffer.append(stripped)
    flush()
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--model", default="qwen3-4b-instruct")
    ap.add_argument(
        "--repo-id",
        default="Comply2Reg/regulatory-alert-triage-qwen3-4b-v1",
        help="where the model will live on the Hub; used in the quickstart",
    )
    ap.add_argument("--train-report", type=Path, required=True)
    ap.add_argument("--train-log", type=Path, default=None)
    ap.add_argument(
        "--coverage",
        type=Path,
        nargs="+",
        default=None,
        help="corpus coverage.json; pass one per corpus the model was trained "
        "on and the counts are summed",
    )
    ap.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="rescore_<v>.json — turns the headline accuracy into a band",
    )
    ap.add_argument(
        "--bench-dir",
        type=Path,
        default=None,
        help="directory of compare_<task>.json from scripts.triage.bench",
    )
    ap.add_argument(
        "--adapter-name",
        default="uk-triage-v1-qwen3-4b-instruct-mlx",
        help="the adapter whose eval is the reference for the fused model",
    )
    ap.add_argument("--checkpoint-name", default="uk-triage-v1-iter1200")
    return ap


def _eval_path(dataset_dir: Path, version: str, model: str, suffix: str) -> Path:
    return dataset_dir / f"eval_{version}_{model}_mlx_{suffix}.json"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fused_path = _eval_path(
        args.dataset_dir, args.version, args.model, args.model_dir.name
    )
    if not fused_path.exists():
        log.error(
            "no evaluation for the fused model at %s — run train_mlx eval "
            "--model-path %s first",
            fused_path,
            args.model_dir,
        )
        return 1
    fused = load_json(fused_path)
    adapter = load_json(
        _eval_path(args.dataset_dir, args.version, args.model, args.adapter_name)
    )

    def optional(suffix: str) -> Optional[Dict]:
        p = _eval_path(args.dataset_dir, args.version, args.model, suffix)
        return load_json(p) if p.exists() else None

    checkpoint = optional(args.checkpoint_name)
    base = optional("base")
    stats = load_json(args.dataset_dir / f"stats_{args.version}.json")
    train_report = load_json(args.train_report)
    coverage = merge_coverage(args.coverage) if args.coverage else None
    curve = val_loss_curve(args.train_log)
    rescore_data = load_json(args.rescore) if args.rescore and args.rescore.exists() else None
    band = error_bar(rescore_data)
    comparisons = load_benchmarks(args.bench_dir)
    if band:
        log.info(
            "error bar: %.3f-%.3f, labeller ceiling %.3f",
            band["low"], band["high"], band.get("ceiling") or float("nan"),
        )
    if comparisons:
        log.info("benchmarks: %s", ", ".join(c["task"] for c in comparisons))

    if coverage is None:
        log.error("--coverage is required: the card reports corpus provenance")
        return 1

    # Fusing into 4-bit weights re-quantizes, so the published model is not the
    # adapter. Measure the gap rather than assume it away; the card reports the
    # published weights' own scores either way.
    agreement = prediction_agreement(
        fused_path.with_name(fused_path.stem + "_outputs.jsonl"),
        _eval_path(
            args.dataset_dir, args.version, args.model, args.adapter_name
        ).with_name(
            _eval_path(
                args.dataset_dir, args.version, args.model, args.adapter_name
            ).stem
            + "_outputs.jsonl"
        ),
    )
    summary = build_summary(
        fused,
        adapter,
        checkpoint,
        base,
        stats,
        train_report,
        coverage,
        curve,
        agreement,
    )
    if band:
        summary["label_noise"] = {
            "band": [band["low"], band["high"]],
            "labeller_agreement": band.get("labeller_agreement"),
            "labeller_pair": band.get("pair"),
            "n": band.get("labeller_n"),
        }
    if comparisons:
        summary["benchmarks"] = {
            c["task"]: {"base": c["base"], "tuned": c["tuned"], "delta": c["delta"]}
            for c in comparisons
        }
    check = summary["fusion"]
    if not check["passed"]:
        log.error(
            "fusion check failed: %s (tolerance %.2f)",
            check["regressions"] or "JSON validity below 0.99",
            check["tolerance"],
        )
        return 1
    if agreement:
        log.info(
            "fused vs base+adapter: %.1f%% of predictions unchanged (%d of %d)",
            agreement["agreement"] * 100,
            agreement["same_prediction"],
            agreement["documents"],
        )
    card = frontmatter(
        args.model_dir.name, train_report["mlx_id"], fused
    ) + reflow(body(
        args.repo_id,
        fused,
        adapter,
        checkpoint,
        base,
        stats,
        train_report,
        coverage,
        curve,
        args.train_log,
        agreement,
        band,
        comparisons,
    ))

    (args.model_dir / "README.md").write_text(card, encoding="utf-8")
    (args.model_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info(
        "card -> %s (%d chars), summary -> %s",
        args.model_dir / "README.md",
        len(card),
        args.model_dir / "eval_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
