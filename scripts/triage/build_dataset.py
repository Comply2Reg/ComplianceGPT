#!/usr/bin/env python3
"""Turn triage labels into training records the Gemma notebook can train on.

    python -m scripts.triage.build_dataset --labels data/uk/labels/triage_v1.jsonl \\
        --corpus-dir <corpus> --out data/uk/datasets/green --version v1 --tier GREEN \\
        [--gold-months 2] [--val-frac 0.10] [--head-chars 2000] [--seed 42] \\
        [--apply-gold data/uk/gold/gold_labels_v1.csv]

Record envelope = the 7 keys of `main`'s ObligationRecord (metadata,
classification, instruction, input_text, tool_use, thought_trace, output) so
notebooks/ComplianceGPT_v1_Training.ipynb loads it unchanged.

Split is stratified by stratum AND date: the latest --gold-months of each
stratum become `test` (the gold candidates), the next --val-frac by date
`val`, the rest `train`. Only train is balanced towards TARGET_MIX (rule_final
taken whole, others capped, never oversampled). `all_<v>.jsonl` = train+val is
what the notebook's data_path points at; test is kept out of it on purpose.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import Corpus, read_jsonl, write_jsonl  # noqa: E402
from scripts.triage.prompts import INSTRUCTION_TEMPLATES  # noqa: E402
from scripts.triage.render import (  # noqa: E402
    clean_prose,
    count_tokens,
    fit_to_budget,
    load_tokenizer,
    to_messages,
)
from scripts.triage.schema import OUTPUT_FIELDS  # noqa: E402
from scripts.triage.taxonomy import CLASS_TO_STRATUM, STRATA, TARGET_MIX  # noqa: E402

log = logging.getLogger("triage.build_dataset")
GOLD_COLUMNS = (
    "doc_id",
    "document_id",
    "title",
    "regulator",
    "document_type",
    "url",
    "release_date",
    "split",
    "stratum",
    "prior_alert_class",
    "model_alert_class",
    "model_confidence",
    "model_priority",
    "primary_functions",
    "summary",
    "flags",
    "human_alert_class",
    "human_priority",
    "human_functions",
    "human_notes",
)


def month_index(date: Optional[str]) -> Optional[int]:
    if not date or len(date) < 7 or not date[:4].isdigit():
        return None
    return int(date[:4]) * 12 + int(date[5:7]) - 1


def apply_gold(labels: List[Dict], gold_csv: Path) -> int:
    """Human labels win over model labels; rows without a human class are ignored."""
    overrides = {}
    with gold_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("human_alert_class") or "").strip():
                overrides[int(row["doc_id"])] = row
    n = 0
    for rec in labels:
        g = overrides.get(rec["doc_id"])
        if not g:
            continue
        lab = rec["label"]
        lab["alert_class"] = g["human_alert_class"].strip()
        if (g.get("human_priority") or "").strip():
            lab["priority"] = g["human_priority"].strip()
        if (g.get("human_functions") or "").strip():
            lab["primary_functions"] = [
                f.strip() for f in g["human_functions"].split(";") if f.strip()
            ]
        lab["alert_class_confidence"] = "high"
        rec["label_source"] = "human"
        rec["flags"] = [
            f
            for f in rec.get("flags", [])
            if f not in ("class_disagrees_with_registry", "low_confidence")
        ]
        n += 1
    return n


def to_record(rec: Dict, head: str, stratum: str, stratum_source: str) -> Dict:
    label = rec["label"]
    output = {k: label[k] for k in OUTPUT_FIELDS if k in label}
    instruction = INSTRUCTION_TEMPLATES[rec["doc_id"] % len(INSTRUCTION_TEMPLATES)]
    input_text = (
        f"Regulator: {rec['regulator']}\nPublication type: {rec['document_type']}\n"
        f"Title: {rec.get('title') or ''}\n"
        f"Published: {rec.get('release_date') or 'unknown'}\n\n"
        f"{clean_prose(head)}"
    )
    return {
        "metadata": {
            "source_id": rec["document_id"],
            "regulation_name": rec.get("title") or "",
            "jurisdiction": "GB",
            "doc_type": rec["document_type"],
            "section": "document",
            "regulator": rec["regulator"],
            "release_date": rec.get("release_date"),
            "stratum": stratum,
            "stratum_source": stratum_source,
            "doc_hash": rec["doc_hash"],
            "tier": rec.get("tier"),
            "licence": rec.get("licence"),
            "label_source": rec.get("label_source", "model"),
            "model": rec.get("model"),
            "prompt_version": rec.get("prompt_version"),
        },
        "classification": label["alert_class"],
        "instruction": instruction,
        "input_text": input_text,
        "tool_use": None,
        "thought_trace": label.get("rationale", ""),
        "output": output,
    }


def split_by_date(
    items: List[Dict], gold_months: int, val_frac: float
) -> Dict[str, List[Dict]]:
    """Per stratum: latest `gold_months` → test, next `val_frac` by date → val."""
    by_stratum: Dict[str, List[Dict]] = defaultdict(list)
    for it in items:
        by_stratum[it["metadata"]["stratum"]].append(it)
    out = {"train": [], "val": [], "test": []}
    for stratum, rows in by_stratum.items():
        dated = [
            r for r in rows if month_index(r["metadata"]["release_date"]) is not None
        ]
        undated = [
            r for r in rows if month_index(r["metadata"]["release_date"]) is None
        ]
        out["train"].extend(undated)
        if not dated:
            continue
        dated.sort(key=lambda r: r["metadata"]["release_date"])
        latest = month_index(dated[-1]["metadata"]["release_date"])
        cutoff = latest - gold_months + 1
        test = [
            r for r in dated if month_index(r["metadata"]["release_date"]) >= cutoff
        ]
        rest = [r for r in dated if month_index(r["metadata"]["release_date"]) < cutoff]
        n_val = int(round(len(rest) * val_frac))
        out["val"].extend(rest[len(rest) - n_val :] if n_val else [])
        out["train"].extend(rest[: len(rest) - n_val] if n_val else rest)
        out["test"].extend(test)
    return out


def balance(train: List[Dict], seed: int) -> Tuple[List[Dict], Dict]:
    by_stratum: Dict[str, List[Dict]] = defaultdict(list)
    for r in train:
        by_stratum[r["metadata"]["stratum"]].append(r)
    n_rule = len(by_stratum.get("rule_final", []))
    total = n_rule / TARGET_MIX["rule_final"] if n_rule else len(train)
    rng = random.Random(seed)
    kept: List[Dict] = []
    report = {}
    for stratum in STRATA:
        rows = by_stratum.get(stratum, [])
        cap = (
            len(rows)
            if stratum == "rule_final"
            else int(round(TARGET_MIX[stratum] * total))
        )
        if len(rows) > cap:
            rows = sorted(rows, key=lambda r: r["metadata"]["doc_hash"])
            rng.shuffle(rows)
            rows = rows[:cap]
        report[stratum] = {
            "available": len(by_stratum.get(stratum, [])),
            "cap": cap,
            "kept": len(rows),
            "shortfall": max(0, cap - len(rows)),
        }
        kept.extend(rows)
    for stratum, rows in by_stratum.items():
        if stratum not in STRATA:  # unexpected label; keep, report
            kept.extend(rows)
            report[stratum] = {
                "available": len(rows),
                "cap": None,
                "kept": len(rows),
                "shortfall": 0,
            }
    kept.sort(key=lambda r: r["metadata"]["doc_hash"])
    return kept, report


def write_gold_sheet(
    path: Path,
    records: List[Dict],
    labels_by_hash: Dict[str, Dict],
    split_of: Dict[str, str],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        h = r["metadata"]["doc_hash"]
        lab = labels_by_hash[h]
        flags = lab.get("flags", [])
        split = split_of[h]
        if split != "test" and not (
            {"class_disagrees_with_registry", "low_confidence"} & set(flags)
        ):
            continue
        rows.append(
            {
                "doc_id": lab["doc_id"],
                "document_id": lab["document_id"],
                "title": lab.get("title"),
                "regulator": lab["regulator"],
                "document_type": lab["document_type"],
                "url": lab.get("url"),
                "release_date": lab.get("release_date"),
                "split": split,
                "stratum": r["metadata"]["stratum"],
                "prior_alert_class": lab.get("prior_alert_class"),
                "model_alert_class": lab["label"]["alert_class"],
                "model_confidence": lab["label"]["alert_class_confidence"],
                "model_priority": lab["label"]["priority"],
                "primary_functions": "; ".join(lab["label"]["primary_functions"]),
                "summary": lab["label"]["summary"],
                "flags": "; ".join(flags),
                "human_alert_class": "",
                "human_priority": "",
                "human_functions": "",
                "human_notes": "",
            }
        )
    rows.sort(
        key=lambda x: (x["split"] != "test", x["stratum"], x["release_date"] or "")
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, default=Path(cfg.TRIAGE["corpus_dir"]))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--tier", nargs="+", default=["GREEN"])
    ap.add_argument("--gold-months", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--head-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--model",
        default=cfg.FOCUS_MODEL,
        choices=sorted(cfg.MODELS),
        help="target model: its tokenizer measures the budget and its "
        "max_seq_length trims the context",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override the registry max_seq_length",
    )
    ap.add_argument("--apply-gold", type=Path, default=None)
    ap.add_argument(
        "--gold-out",
        type=Path,
        default=None,
        help="gold candidate sheet (default data/uk/gold/gold_candidates_<v>.csv)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    tiers = {t.upper() for t in args.tier}
    labels = [
        lab for lab in read_jsonl(args.labels) if str(lab.get("tier")).upper() in tiers
    ]
    if args.apply_gold:
        log.info(
            "applied %d human labels from %s",
            apply_gold(labels, args.apply_gold),
            args.apply_gold,
        )
    corpus = Corpus(args.corpus_dir)
    corpus.check()
    tokenizer = load_tokenizer(args.model)
    budget = args.max_tokens or cfg.MODELS[args.model]["max_seq_length"]
    token_counter = cfg.MODELS[args.model]["hf_id"] if tokenizer else "chars/4 proxy"
    if tokenizer is None:
        log.warning("target tokenizer unavailable; token counts are a chars/4 proxy")
    trimmed_count = 0

    seen = set()
    records: List[Dict] = []
    labels_by_hash: Dict[str, Dict] = {}
    for lab in labels:
        h = lab["doc_hash"]
        if h in seen:
            continue
        seen.add(h)
        stratum = lab.get("stratum")
        source = "registry"
        if stratum not in STRATA:
            stratum = CLASS_TO_STRATUM.get(lab["label"]["alert_class"], "intelligence")
            source = "model"
        try:
            canonical = corpus.canonical(lab["doc_id"])
        except OSError:
            log.warning("no canonical text for doc %s; skipped", lab["doc_id"])
            continue
        rec = to_record(lab, canonical[: args.head_chars], stratum, source)
        rec, n_tokens, trimmed = fit_to_budget(rec, tokenizer, args.model, budget)
        trimmed_count += int(trimmed)
        rec["messages"] = to_messages(rec)
        rec["n_tokens"] = n_tokens
        rec["metadata"]["context_trimmed"] = trimmed
        rec["metadata"]["n_tokens_full_document"] = count_tokens(canonical, tokenizer)
        rec["metadata"]["target_model"] = args.model
        records.append(rec)
        labels_by_hash[h] = lab

    splits = split_by_date(records, args.gold_months, args.val_frac)
    train, balance_report = balance(splits["train"], args.seed)
    split_of = {
        r["metadata"]["doc_hash"]: name for name, rows in splits.items() for r in rows
    }
    hashes = [
        r["metadata"]["doc_hash"]
        for rows in (train, splits["val"], splits["test"])
        for r in rows
    ]
    assert len(hashes) == len(set(hashes)), "a document landed in two splits"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in (
        ("train", train),
        ("val", splits["val"]),
        ("test", splits["test"]),
    ):
        counts[name] = write_jsonl(out / f"{name}_{args.version}.jsonl", rows)
    counts["all"] = write_jsonl(
        out / f"all_{args.version}.jsonl", train + splits["val"]
    )

    def breakdown(rows: List[Dict]) -> Dict:
        return {
            "stratum": dict(Counter(r["metadata"]["stratum"] for r in rows)),
            "classification": dict(Counter(r["classification"] for r in rows)),
            "regulator": dict(Counter(r["metadata"]["regulator"] for r in rows)),
            "tier": dict(Counter(r["metadata"]["tier"] for r in rows)),
            "date_range": [
                min(
                    (r["metadata"]["release_date"] or "9999" for r in rows),
                    default=None,
                ),
                max((r["metadata"]["release_date"] or "" for r in rows), default=None),
            ],
        }

    stats = {
        "version": args.version,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "labels_file": str(args.labels),
        "tiers": sorted(tiers),
        "gold_months": args.gold_months,
        "val_frac": args.val_frac,
        "head_chars": args.head_chars,
        "seed": args.seed,
        "target_model": args.model,
        "token_counter": token_counter,
        "max_seq_length": budget,
        "context_trimmed": trimmed_count,
        "tokens": {
            name: {
                "p50": sorted(r["n_tokens"] for r in rows)[len(rows) // 2]
                if rows
                else 0,
                "p95": sorted(r["n_tokens"] for r in rows)[int(len(rows) * 0.95)]
                if rows
                else 0,
                "max": max((r["n_tokens"] for r in rows), default=0),
                "over_budget": sum(1 for r in rows if r["n_tokens"] > budget),
            }
            for name, rows in (
                ("train", train),
                ("val", splits["val"]),
                ("test", splits["test"]),
            )
        },
        "counts": counts,
        "target_mix": TARGET_MIX,
        "balance": balance_report,
        "train": breakdown(train),
        "val": breakdown(splits["val"]),
        "test": breakdown(splits["test"]),
        "unbalanced_train": len(splits["train"]),
    }
    (out / f"stats_{args.version}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    gold_path = (
        args.gold_out
        or Path(cfg.TRIAGE["gold_dir"]) / f"gold_candidates_{args.version}.csv"
    )
    n_gold = write_gold_sheet(
        gold_path, train + splits["val"] + splits["test"], labels_by_hash, split_of
    )

    card = [
        f"# UK alert-triage dataset {args.version}",
        "",
        f"Generated {stats['generated_at']} from `{args.labels}` "
        f"(tiers {', '.join(sorted(tiers))}).",
        "",
        "| split | records |",
        "|---|---:|",
        *(f"| {k} | {v} |" for k, v in counts.items()),
        "",
        "## Train balance vs target",
        "",
        "| stratum | available | cap | kept | shortfall | target |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            f"| {s} | {b['available']} | {b['cap']} | {b['kept']} | {b['shortfall']} | "
            f"{int(TARGET_MIX.get(s, 0) * 100)}% |"
            for s, b in balance_report.items()
        ),
        "",
        f"Test = latest {args.gold_months} months per stratum; val = next "
        f"{int(args.val_frac * 100)}% by date; undated rows train. Gold candidates: "
        f"{n_gold} rows in `{gold_path}`.",
        "",
        "Record envelope: metadata, classification (alert class A1–A14), instruction, "
        "input_text (title + first head-chars of canonical text), tool_use (null), "
        "thought_trace (model rationale), output (TriageRecord without rationale), "
        f"plus `messages` (system/user/assistant) and `n_tokens` measured with "
        f"{token_counter} against a {budget}-token budget ({trimmed_count} contexts "
        "trimmed). AMBER-tier records are internal use only and never leave "
        "`datasets/internal/`.",
    ]
    (out / f"dataset_card_{args.version}.md").write_text(
        "\n".join(card) + "\n", encoding="utf-8"
    )
    log.info(
        "wrote %s | balance %s | gold candidates %d -> %s",
        counts,
        {k: v["kept"] for k, v in balance_report.items()},
        n_gold,
        gold_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
