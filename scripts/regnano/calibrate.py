#!/usr/bin/env python3
"""Calibrate tasks against the current policy — FrogNano's difficulty filter.

Each task is attempted k times by the current model with sampling on; the
verifier scores every attempt. A task the policy always solves teaches
nothing, one it never solves gives no gradient (every group-relative
advantage is zero), so only tasks with 0 < pass rate < 1 are kept — FrogNano
targeted ~50% in iterations 1-4.

    python -m scripts.regnano.calibrate --tasks data/regnano/tasks/mined_v1.jsonl \\
        --out data/regnano/calibration/mined_v1_qwen3.5-4b_base.jsonl \\
        [--model qwen3.5-4b] [--model-path DIR] [--adapter-path DIR] \\
        [--k 8] [--limit 50] [--keep-out data/regnano/tasks/iter1.jsonl] \\
        [--passing-out data/regnano/rft/iter1_passing.jsonl]

--passing-out appends every verified-correct completion: the training data
for rejection-sampling fine-tuning (rft.py), the on-policy fallback to GRPO.

Resumable: task_ids already in --out are skipped. Iteration 1 runs with
thinking OFF (short JSON answers; thinking multiplies rollout time ~10x on
the laptop) — a recorded deviation from FrogNano.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.regnano.prompt import to_messages  # noqa: E402
from scripts.regnano.verifiers import verify  # noqa: E402
from scripts.triage.corpus import read_jsonl  # noqa: E402

log = logging.getLogger("regnano.calibrate")

# Qwen3.5 recommended non-thinking sampling.
SAMPLING = {"temp": 0.7, "top_p": 0.8, "top_k": 20}


def band(pass_rate: float, lo: float, hi: float) -> bool:
    return lo < pass_rate < hi


def summarise(rows: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for r in rows:
        fam = out.setdefault(r["family"], {"tasks": 0, "never": 0, "always": 0, "in_band": 0, "mean_pass": 0.0})
        fam["tasks"] += 1
        fam["mean_pass"] += r["pass_rate"]
        if r["passes"] == 0:
            fam["never"] += 1
        elif r["passes"] == r["k"]:
            fam["always"] += 1
        else:
            fam["in_band"] += 1
    for fam in out.values():
        fam["mean_pass"] = round(fam["mean_pass"] / fam["tasks"], 3)
    return out


def calibrate(args) -> int:
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    tasks = read_jsonl(args.tasks)
    done = {r["task_id"] for r in read_jsonl(args.out)} if args.out.exists() else set()
    todo = [t for t in tasks if t["task_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    log.info("%d tasks, %d already calibrated, %d to run", len(tasks), len(done), len(todo))

    spec = cfg.MODELS[args.model]
    model_id = str(args.model_path) if args.model_path else spec["mlx_id"]
    model, tokenizer = load(model_id, adapter_path=str(args.adapter_path) if args.adapter_path else None)
    tmpl = {"enable_thinking": False} if spec.get("thinking_switch") else {}
    sampler = make_sampler(**SAMPLING)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    passing_fh = None
    if args.passing_out:
        args.passing_out.parent.mkdir(parents=True, exist_ok=True)
        passing_fh = args.passing_out.open("a", encoding="utf-8")
    with args.out.open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), args.tasks_per_batch):
            chunk = todo[i : i + args.tasks_per_batch]
            prompts = []
            for t in chunk:
                ids = tokenizer.apply_chat_template(to_messages(t), add_generation_prompt=True, **tmpl)
                prompts.extend([ids] * args.k)
            resp = batch_generate(model, tokenizer, prompts, max_tokens=args.max_new_tokens, sampler=sampler)
            for j, t in enumerate(chunk):
                texts = resp.texts[j * args.k : (j + 1) * args.k]
                verdicts = [verify(t, x) for x in texts]
                passes = sum(v.passed for v in verdicts)
                fails = [v.reason for v in verdicts if not v.passed]
                if passing_fh:
                    for x, v in zip(texts, verdicts):
                        if v.passed:
                            passing_fh.write(json.dumps({"task_id": t["task_id"], "family": t["family"],
                                                         "completion": x}, ensure_ascii=False) + "\n")
                fh.write(
                    json.dumps(
                        {
                            "task_id": t["task_id"],
                            "family": t["family"],
                            "k": args.k,
                            "passes": passes,
                            "pass_rate": passes / args.k,
                            "fail_reasons": sorted(set(fails)),
                            "sample": texts[0][:400],
                            "model": model_id,
                            "adapter": str(args.adapter_path) if args.adapter_path else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fh.flush()
            if passing_fh:
                passing_fh.flush()
            n = i + len(chunk)
            rate = (time.monotonic() - started) / n
            log.info("%d/%d tasks, %.1fs/task, ETA %.0f min", n, len(todo), rate, rate * (len(todo) - n) / 60)

    if passing_fh:
        passing_fh.close()
    rows = read_jsonl(args.out)
    summary = summarise(rows)
    summary["_meta"] = {
        "model": model_id,
        "k": args.k,
        "sampling": SAMPLING,
        "thinking": False,
        "seconds_this_run": round(time.monotonic() - started),
        "at": dt.datetime.now(dt.UTC).isoformat(),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.keep_out:
        rates = {r["task_id"]: r["pass_rate"] for r in rows}
        kept = [t for t in tasks if t["task_id"] in rates and band(rates[t["task_id"]], args.lo, args.hi)]
        args.keep_out.parent.mkdir(parents=True, exist_ok=True)
        with args.keep_out.open("w", encoding="utf-8") as fh:
            for t in kept:
                fh.write(json.dumps({**t, "calibrated_pass_rate": rates[t["task_id"]]}, ensure_ascii=False) + "\n")
        log.info("kept %d of %d tasks in band (%s, %s) -> %s", len(kept), len(rows), args.lo, args.hi, args.keep_out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen3.5-4b", choices=sorted(cfg.MODELS))
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--adapter-path", type=Path, default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tasks-per-batch", type=int, default=2, help="prompts per batch = this x k")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-out", type=Path, default=None)
    ap.add_argument("--passing-out", type=Path, default=None, help="append verified-correct completions")
    ap.add_argument("--lo", type=float, default=0.0, help="keep pass rate > lo")
    ap.add_argument("--hi", type=float, default=1.0, help="keep pass rate < hi")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
