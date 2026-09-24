#!/usr/bin/env python3
"""Dataset quality report — the checks to run before every training run.

    python -m scripts.triage.analyze_dataset --dataset-dir data/uk/datasets/green \\
        --version v1 [--model qwen3-4b-instruct] [--corpus-dir <corpus>] [--out …]

Per split: stratum / class / regulator counts, token lengths (from the
records' `n_tokens`, else counted now with the model tokenizer, else a
chars/4 proxy), rare classes, input diversity (unique 8-gram ratio per
class), repetition inside a record, date ranges, label-source mix; plus the
corpus-level dedupe and furniture figures when --corpus-dir is given.
Writes analysis_<v>.md next to the dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import read_jsonl  # noqa: E402
from scripts.triage.render import count_tokens, load_tokenizer, render, to_messages  # noqa: E402
from scripts.triage.taxonomy import ALERT_CLASSES, TARGET_MIX  # noqa: E402

RARE_CLASS_MIN = 30
NGRAM = 8
REPEAT_WINDOW = 10


def _q(values: List[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def _ngrams(text: str, n: int = NGRAM) -> List[str]:
    words = re.findall(r"\w+", text.lower())
    return [" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))]


def _has_repetition(text: str, window: int = REPEAT_WINDOW) -> bool:
    words = re.findall(r"\w+", text.lower())
    seen = set()
    for i in range(max(0, len(words) - window + 1)):
        w = " ".join(words[i : i + window])
        if w in seen:
            return True
        seen.add(w)
    return False


def table(headers: List[str], rows: List[List]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--model", default=cfg.FOCUS_MODEL, choices=sorted(cfg.MODELS))
    ap.add_argument("--corpus-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    splits: Dict[str, List[Dict]] = {
        name: read_jsonl(args.dataset_dir / f"{name}_{args.version}.jsonl")
        for name in ("train", "val", "test")
    }
    splits = {k: v for k, v in splits.items() if v}
    if not splits:
        print("no dataset files found")
        return 1
    spec = cfg.MODELS[args.model]
    budget = spec["max_seq_length"]
    tokenizer = None
    if any("n_tokens" not in r for rows in splits.values() for r in rows):
        tokenizer = load_tokenizer(args.model)
    counter_name = (
        f"{spec['hf_id']} tokenizer"
        if tokenizer
        else "records' n_tokens"
        if all("n_tokens" in r for rows in splits.values() for r in rows)
        else "chars/4 proxy"
    )

    lines = [
        f"# Dataset analysis — {args.dataset_dir.name} {args.version}",
        "",
        f"Target model `{args.model}` ({spec['hf_id']}), budget {budget} tokens; "
        f"token counter: {counter_name}.",
        "",
    ]

    # counts
    lines += ["## Records per split", ""]
    lines += table(
        ["split", "records", "strata", "classes", "regulators"],
        [
            [
                name,
                len(rows),
                dict(Counter(r["metadata"]["stratum"] for r in rows)),
                dict(Counter(r["classification"] for r in rows)),
                dict(Counter(r["metadata"]["regulator"] for r in rows)),
            ]
            for name, rows in splits.items()
        ],
    )

    # tokens
    lines += ["", "## Token lengths", ""]
    tok_rows = []
    over = 0
    for name, rows in splits.items():
        lengths = []
        for r in rows:
            if "n_tokens" in r:
                lengths.append(int(r["n_tokens"]))
            else:
                lengths.append(
                    count_tokens(
                        render(to_messages(r), tokenizer, args.model), tokenizer
                    )
                )
        over += sum(1 for n in lengths if n > budget)
        trimmed = sum(
            1 for r in rows if (r.get("metadata") or {}).get("context_trimmed")
        )
        tok_rows.append(
            [
                name,
                _q(lengths, 0.5),
                _q(lengths, 0.95),
                max(lengths, default=0),
                sum(1 for n in lengths if n > budget),
                trimmed,
            ]
        )
    lines += table(
        ["split", "p50", "p95", "max", f"over {budget}", "context trimmed"], tok_rows
    )
    full = [
        int((r.get("metadata") or {}).get("n_tokens_full_document") or 0)
        for rows in splits.values()
        for r in rows
    ]
    full = [n for n in full if n]
    if full:
        lines += [
            "",
            "Whole-document tokens (for a long-context variant): "
            f"p50 {_q(full, 0.5)}, p95 {_q(full, 0.95)}, max {max(full)} — "
            f"{sum(1 for n in full if n <= 8192)}/{len(full)} fit in 8k, "
            f"{sum(1 for n in full if n <= 16384)}/{len(full)} in 16k.",
        ]

    # classes
    train = splits.get("train", [])
    by_class = Counter(r["classification"] for r in train)
    rare = [c for c in ALERT_CLASSES if by_class.get(c, 0) < RARE_CLASS_MIN]
    lines += ["", "## Class coverage (train)", ""]
    lines += table(
        ["class", "label", "train", "val", "test"],
        [
            [
                c,
                ALERT_CLASSES[c][:48],
                by_class.get(c, 0),
                sum(1 for r in splits.get("val", []) if r["classification"] == c),
                sum(1 for r in splits.get("test", []) if r["classification"] == c),
            ]
            for c in ALERT_CLASSES
        ],
    )
    lines += [
        "",
        f"Classes with fewer than {RARE_CLASS_MIN} training examples: "
        f"{', '.join(rare) if rare else 'none'}. A model cannot learn a class it "
        "has not seen; these need more sources (see docs/uk-datasets.md) or must be "
        "excluded from the evaluation claims.",
    ]
    strata = Counter(r["metadata"]["stratum"] for r in train)
    n_train = len(train) or 1
    lines += ["", "## Train mix vs target", ""]
    lines += table(
        ["stratum", "records", "share", "target"],
        [
            [
                s,
                strata.get(s, 0),
                f"{100 * strata.get(s, 0) / n_train:.1f}%",
                f"{int(t * 100)}%",
            ]
            for s, t in TARGET_MIX.items()
        ],
    )

    # diversity and repetition
    lines += ["", "## Diversity and repetition (train inputs)", ""]
    div_rows = []
    for c in ALERT_CLASSES:
        texts = [r["input_text"] for r in train if r["classification"] == c]
        if not texts:
            continue
        grams = [g for t in texts for g in _ngrams(t)]
        ratio = len(set(grams)) / len(grams) if grams else 0.0
        rep = sum(1 for t in texts if _has_repetition(t))
        div_rows.append([c, len(texts), f"{ratio:.3f}", rep])
    lines += table(
        [
            "class",
            "records",
            f"unique {NGRAM}-gram ratio",
            f"records with an internal {REPEAT_WINDOW}-word repeat",
        ],
        div_rows,
    )
    lines += [
        "",
        "A ratio near 1.0 means inputs rarely share phrasing; a low ratio flags a "
        "templated class (expected for enforcement notices). Internal repeats usually "
        "mean leaked navigation or a running header the furniture cleaner missed.",
    ]

    # dates, sources
    lines += ["", "## Dates and label sources", ""]
    for name, rows in splits.items():
        dates = sorted(
            d for d in ((r["metadata"] or {}).get("release_date") for r in rows) if d
        )
        src = Counter((r["metadata"] or {}).get("label_source", "model") for r in rows)
        lines.append(
            f"- {name}: {dates[0] if dates else '—'} → {dates[-1] if dates else '—'}"
            f" ({len(rows) - len(dates)} undated); labels {dict(src)}"
        )

    # corpus-level
    if args.corpus_dir and (args.corpus_dir / "coverage.json").exists():
        cov = json.loads((args.corpus_dir / "coverage.json").read_text())
        lines += [
            "",
            "## Corpus (from coverage.json)",
            "",
            f"- documents {cov.get('documents')}, chunks {cov.get('chunks')}",
            f"- deduped {cov.get('deduped')}",
            f"- furniture {cov.get('furniture')}",
            f"- skipped {cov.get('skipped')}",
        ]

    out = args.out or args.dataset_dir / f"analysis_{args.version}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"analysis -> {out} | over-budget records: {over} | rare classes: {rare}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
