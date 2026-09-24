#!/usr/bin/env python3
"""Gate a built dataset before anyone trains on it.

    python -m scripts.triage.validate_dataset --dataset-dir data/uk/datasets/green \\
        --version v1 [--task triage|obligation] [--strict] [--corpus-dir <corpus>] \\
        [--skip-corpus-check]

Checks: the 7-key envelope with the right types; classification in the task's
label set; no `status` key in a triage output (the notebook rewrites those);
input_text within --head-chars; no doc_hash in two splits; no AMBER record in
a directory named green; stats file counts match the files; for obligations,
records produced by `main`'s regex fallback ("Extracted Entity") are flagged.
Exit 1 on any failure (warnings only fail under --strict).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import read_jsonl  # noqa: E402
from scripts.triage.schema import OUTPUT_FIELDS  # noqa: E402
from scripts.triage.taxonomy import ALERT_CLASSES  # noqa: E402

ENVELOPE = (
    "metadata",
    "classification",
    "instruction",
    "input_text",
    "tool_use",
    "thought_trace",
    "output",
)
OBLIGATION_CLASSES = ("obligation", "non_obligation", "neutral")
FALLBACK_MARKERS = ("Extracted Entity", "Extracted Action")


def check_record(
    rec: Dict, task: str, head_chars: int, idx: int, split: str
) -> List[str]:
    errs: List[str] = []
    where = f"{split}[{idx}]"
    if tuple(rec.keys()) != ENVELOPE and set(rec.keys()) != set(ENVELOPE):
        errs.append(f"{where}: keys {sorted(rec.keys())} != envelope")
        return errs
    if not isinstance(rec["metadata"], dict) or "source_id" not in rec["metadata"]:
        errs.append(f"{where}: metadata missing source_id")
    if not isinstance(rec["instruction"], str) or not rec["instruction"].strip():
        errs.append(f"{where}: empty instruction")
    if not isinstance(rec["input_text"], str) or not rec["input_text"].strip():
        errs.append(f"{where}: empty input_text")
    if not isinstance(rec["thought_trace"], str):
        errs.append(f"{where}: thought_trace not a string")
    if task == "triage":
        if rec["classification"] not in ALERT_CLASSES:
            errs.append(f"{where}: classification {rec['classification']!r} not A1-A14")
        out = rec["output"]
        if not isinstance(out, dict):
            errs.append(f"{where}: output must be a dict")
        else:
            if "status" in out:
                errs.append(
                    f"{where}: output has a 'status' key (notebook rewrites it)"
                )
            missing = [k for k in OUTPUT_FIELDS if k not in out]
            if missing:
                errs.append(f"{where}: output missing {missing}")
            if out.get("alert_class") != rec["classification"]:
                errs.append(f"{where}: output.alert_class != classification")
        # head + the 4 header lines; allow a little slack for the header
        if len(rec["input_text"]) > head_chars + 400:
            errs.append(f"{where}: input_text {len(rec['input_text'])} > head-chars")
        if "doc_hash" not in rec["metadata"]:
            errs.append(f"{where}: metadata.doc_hash missing")
    else:
        if rec["classification"] not in OBLIGATION_CLASSES:
            errs.append(f"{where}: classification {rec['classification']!r}")
        blob = json.dumps(rec["output"])
        if any(m in blob for m in FALLBACK_MARKERS):
            errs.append(
                f"{where}: regex-fallback record (contains {FALLBACK_MARKERS[0]!r})"
            )
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--task", choices=("triage", "obligation"), default="triage")
    ap.add_argument("--head-chars", type=int, default=2000)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--corpus-dir", type=Path, default=Path(cfg.TRIAGE["corpus_dir"]))
    ap.add_argument("--skip-corpus-check", action="store_true")
    args = ap.parse_args(argv)

    errors: List[str] = []
    warnings: List[str] = []

    if not args.skip_corpus_check and args.task == "triage":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "validate_data.py"),
            "--source",
            "uk_alerts",
            "--corpus-dir",
            str(args.corpus_dir),
            "--strict",
            "--skip-raw",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            errors.append(
                "corpus fails validate_data.py --strict: "
                + (proc.stdout.strip().splitlines() or ["?"])[-1]
            )

    splits: Dict[str, List[Dict]] = {}
    for name in ("train", "val", "test"):
        path = args.dataset_dir / f"{name}_{args.version}.jsonl"
        if path.exists():
            splits[name] = read_jsonl(path)
    if not splits:
        errors.append(
            f"no {{train,val,test}}_{args.version}.jsonl under {args.dataset_dir}"
        )

    for name, rows in splits.items():
        for i, rec in enumerate(rows):
            errors.extend(check_record(rec, args.task, args.head_chars, i, name))

    seen: Dict[str, str] = {}
    for name, rows in splits.items():
        for rec in rows:
            h = (rec.get("metadata") or {}).get("doc_hash")
            if h and h in seen and seen[h] != name:
                errors.append(f"doc_hash {h[:12]} in both {seen[h]} and {name}")
            if h:
                seen[h] = name

    if "green" in args.dataset_dir.name.lower():
        for name, rows in splits.items():
            amber = sum(
                1 for r in rows if (r.get("metadata") or {}).get("tier") == "AMBER"
            )
            if amber:
                errors.append(
                    f"{amber} AMBER records in {name} under a green dataset dir"
                )

    stats_path = args.dataset_dir / f"stats_{args.version}.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for name, rows in splits.items():
            if stats.get("counts", {}).get(name) != len(rows):
                errors.append(
                    f"stats count for {name} ({stats.get('counts', {}).get(name)}) "
                    f"!= {len(rows)} records"
                )
    else:
        warnings.append("no stats file")

    all_path = args.dataset_dir / f"all_{args.version}.jsonl"
    if all_path.exists() and "test" in splits:
        all_hashes = {
            (r.get("metadata") or {}).get("doc_hash") for r in read_jsonl(all_path)
        }
        leaked = sum(
            1
            for r in splits["test"]
            if (r.get("metadata") or {}).get("doc_hash") in all_hashes
        )
        if leaked:
            errors.append(f"{leaked} test records also present in {all_path.name}")

    if args.task == "triage" and splits:
        dist = Counter(r["classification"] for rows in splits.values() for r in rows)
        thin = [c for c, n in dist.items() if n < 5]
        if thin:
            warnings.append(f"classes with < 5 records: {thin}")

    for w in warnings:
        print(f"WARN {w}")
    for e in errors[:50]:
        print(f"FAIL {e}")
    if len(errors) > 50:
        print(f"... {len(errors) - 50} more")
    total = sum(len(v) for v in splits.values())
    if errors or (args.strict and warnings):
        print(
            f"FAILED: {len(errors)} errors, {len(warnings)} warnings "
            f"over {total} records"
        )
        return 1
    print(
        f"OK: {total} records across {list(splits)} pass the {args.task} dataset gates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
