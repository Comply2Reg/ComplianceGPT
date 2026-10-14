#!/usr/bin/env python3
"""Rejection-sampling fine-tuning: the on-policy fallback when GRPO will not fit.

GRPO on Qwen3.5-4B took 265 s for one step at group size 4 and then pushed
the 24 GB M4 Pro into memory pressure (2026-10-13). RFT keeps FrogNano's
rule — learn only from what the verifier accepts, never from a gold answer —
but splits the work: sample k answers per task with calibrate.py
--passing-out (inference only), keep the verified-correct ones, then SFT on
them (one forward/backward per example, no group).

    # 1. sample + verify (writes every passing completion)
    python -m scripts.regnano.calibrate --tasks data/regnano/tasks/iter1/train.jsonl \\
        --out data/regnano/rft/iter1_calibration.jsonl --k 8 \\
        --passing-out data/regnano/rft/iter1_passing.jsonl
    # 2. chat SFT data, at most --per-task distinct answers per task
    python -m scripts.regnano.rft build --tasks data/regnano/tasks/iter1/train.jsonl \\
        --passing data/regnano/rft/iter1_passing.jsonl \\
        --heldout data/regnano/tasks/iter1/heldout.jsonl --out data/regnano/rft/iter1
    # 3. LoRA SFT in .venv-rl (mlx-lm-lora applies its Qwen3.5 recurrent patch;
    #    sft_eager.py runs its trainer uncompiled, which is what fits in memory)
    python -m scripts.regnano.rft train --data data/regnano/rft/iter1 --adapter-path models/regnano-rft-iter1

Distinct answers are capped per task so a task solved 8/8 does not outweigh
one solved 1/8. Held-out tasks are refused at build time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.regnano.prompt import to_messages  # noqa: E402
from scripts.regnano.rl import RL_PYTHON, is_valid  # noqa: E402
from scripts.regnano.verifiers import verify  # noqa: E402
from scripts.triage.corpus import read_jsonl, write_jsonl  # noqa: E402

log = logging.getLogger("regnano.rft")


def build_rows(tasks: List[Dict], passing: List[Dict], heldout_ids: set, per_task: int) -> Dict[str, List[Dict]]:
    by_id = {t["task_id"]: t for t in tasks}
    leaked = heldout_ids & set(by_id)
    if leaked:
        raise ValueError(f"{len(leaked)} held-out tasks in the training tasks")
    answers: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)
    for p in passing:
        tid = p["task_id"]
        if tid not in by_id or len(answers[tid]) >= per_task:
            continue
        text = p["completion"].strip()
        key = re.sub(r"\s+", "", text)
        if key in seen[tid] or not verify(by_id[tid], text).passed:  # re-verify: never trust the file
            continue
        seen[tid].add(key)
        answers[tid].append(text)
    out: Dict[str, List[Dict]] = {"train": [], "valid": []}
    for tid, texts in answers.items():
        split = "valid" if is_valid(tid) else "train"
        for text in texts:
            out[split].append({"messages": to_messages(by_id[tid]) + [{"role": "assistant", "content": text}],
                               "task_id": tid})
    return out


def drop_long(rows: Dict[str, List[Dict]], n_tokens: Callable[[List[Dict]], int], max_tokens: int) -> int:
    """Drop rows whose rendered chat exceeds max_tokens; truncating would cut the answer off.

    Qwen3.5 training memory grows with length (10 GB at 800 tokens, 17 GB at
    1,320), and one long row leaves MLX holding the high-water mark.
    """
    dropped = 0
    for split, data in rows.items():
        kept = [r for r in data if n_tokens(r["messages"]) <= max_tokens]
        dropped += len(data) - len(kept)
        rows[split] = kept
    return dropped


def chat_tokens(model_path: str) -> Callable[[List[Dict]], int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    return lambda messages: len(tok.encode(tok.apply_chat_template(messages, tokenize=False)))


def build(args) -> int:
    tasks = read_jsonl(args.tasks)
    heldout = {t["task_id"] for t in read_jsonl(args.heldout)} if args.heldout else set()
    rows = build_rows(tasks, read_jsonl(args.passing), heldout, args.per_task)
    if args.max_tokens:
        model = str(args.model_path) if args.model_path else cfg.MODELS[args.model]["mlx_id"]
        log.info("dropped %d rows over %d tokens", drop_long(rows, chat_tokens(model), args.max_tokens), args.max_tokens)
    for split, data in rows.items():
        write_jsonl(args.out / f"{split}.jsonl", data)
    covered = {r["task_id"] for d in rows.values() for r in d}
    log.info("rft data -> %s train=%d valid=%d from %d of %d tasks",
             args.out, len(rows["train"]), len(rows["valid"]), len(covered), len(tasks))
    return 0


def train(args) -> int:
    if not RL_PYTHON.exists():
        log.error("RL venv missing: python3.12 -m venv .venv-rl && .venv-rl/bin/pip install mlx-lm-lora")
        return 1
    n_train = len(read_jsonl(args.data / "train.jsonl"))
    model = str(args.model_path) if args.model_path else cfg.MODELS[args.model]["mlx_id"]
    args.adapter_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(RL_PYTHON), "-m", "scripts.regnano.sft_eager",  # stock CLI, step not compiled
        "--model", model,
        "--train", "--train-mode", "sft", "--train-type", "lora", "--mask-prompt",
        "--data", str(args.data),
        "--iters", str(args.iters or n_train * args.epochs),
        "--batch-size", "1",
        "--learning-rate", str(args.lr),
        "--num-layers", str(args.num_layers),
        "--max-seq-length", str(args.max_seq_length),
        "--adapter-path", str(args.adapter_path),
        "--steps-per-eval", str(args.steps_per_eval),
        "--save-every", str(args.save_every),
        "--seed", str(args.seed),
        "--grad-checkpoint",
    ]
    log.info("running: %s", " ".join(cmd))
    started = time.monotonic()
    rc = subprocess.call(cmd, cwd=ROOT)
    report = {"model": model, "data": str(args.data), "train_examples": n_train, "command": cmd,
              "returncode": rc, "seconds": round(time.monotonic() - started),
              "finished_at": dt.datetime.now(dt.UTC).isoformat()}
    (args.adapter_path / "rft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "train"))
    ap.add_argument("--tasks", type=Path)
    ap.add_argument("--passing", type=Path)
    ap.add_argument("--heldout", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--per-task", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024, help="build: drop longer rows (0 = keep all)")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--model", default="qwen3.5-4b", choices=sorted(cfg.MODELS))
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--adapter-path", type=Path, default=Path("models/regnano-rft-iter1"))
    ap.add_argument("--iters", type=int, default=None, help="default: epochs x train examples")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-layers", type=int, default=8)  # same peak memory as 16, half the step time
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=cfg.TRAIN["seed"])
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return build(args) if args.command == "build" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
