#!/usr/bin/env python3
"""Verified trajectories → per-turn chat examples → LoRA SFT on Qwen3-4B-Instruct-2507.

    python -m scripts.agent.sft build --rollouts data/agent/rollouts/demos_v2.jsonl data/agent/rollouts/teacher_v2.jsonl \\
        --out data/agent/sft/warm_v1
    python -m scripts.agent.sft train --data data/agent/sft/warm_v1 --adapter-path models/agent-qwen3-4b-warm-v1

**build.** Only rows with passed=true and split=train are read; a held-out row is
an error, not a skip. Each trajectory becomes one example per assistant turn:
messages up to and including that turn, plus the `tools` key. mlx-lm's ChatDataset
renders it with the tokenizer's own template and, with --mask-prompt, trains
only that last turn (verified: the generation prompt is an exact token prefix of
every cut; docs/agent.md). Tool results and prompts are never trained on.

Caps keep one source or family from dominating: at most --per-task trajectories
per task, and at most --scripted-per-family scripted ones per family (scripted
demos all search the same way; teacher and on-policy rows add variety). Rows
over --max-tokens are dropped, never truncated. 5% of tasks, by hash, are the
validation split.

**train.** `mlx_lm lora --mask-prompt` with LoRA settings written to a YAML in the
adapter folder, so the run records what it used. Runs as a child process that
is stopped if this wrapper is stopped (two trainers at once exhaust memory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.agent.env import TOOLS  # noqa: E402
from scripts.triage.corpus import read_jsonl  # noqa: E402

log = logging.getLogger("agent.sft")


def per_turn(messages: List[Dict]) -> List[List[Dict]]:
    """One example per assistant turn: the conversation up to and including it."""
    return [messages[: i + 1] for i, m in enumerate(messages) if m["role"] == "assistant"]


def is_valid(task_id: str, share: float = 0.05) -> bool:
    return int(hashlib.sha256(f"valid:{task_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def build(args) -> int:
    from transformers import AutoTokenizer

    if (args.out / "train.jsonl").exists():
        raise SystemExit(f"{args.out} already has data; write a new version instead of overwriting")
    rows: List[Dict] = []
    for path in args.rollouts:
        for r in read_jsonl(path):
            if not r.get("passed"):
                continue
            if r.get("split") != "train":
                raise SystemExit(f"{path}: row {r['task_id']} is split={r.get('split')!r}; held-out data never trains")
            r.setdefault("source", "onpolicy")
            rows.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    # Distinct trajectories only (k samples often repeat), then the caps.
    seen, kept = set(), []
    by_task: Counter = Counter()
    scripted: Counter = Counter()
    for r in sorted(rows, key=lambda r: r["source"] == "scripted"):  # teacher/on-policy first
        key = (r["task_id"], json.dumps(r["messages"], sort_keys=True, ensure_ascii=False))
        if key in seen or by_task[r["task_id"]] >= args.per_task:
            continue
        if r["source"] == "scripted":
            if scripted[r["family"]] >= args.scripted_per_family:
                continue
            scripted[r["family"]] += 1
        seen.add(key)
        by_task[r["task_id"]] += 1
        kept.append(r)

    # The HF tokenizer, not mlx_lm.load: same template, without loading 2 GB of weights.
    # tokenize=False + encode, because apply_chat_template(tokenize=True) returns a BatchEncoding.
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODELS[args.model]["hf_id"])
    out: Dict[str, List[Dict]] = defaultdict(list)
    stats: Counter = Counter()
    lengths: List[int] = []
    for r in kept:
        split = "valid" if is_valid(r["task_id"]) else "train"
        stats[f"trajectories/{r['source']}/{r['family']}"] += 1
        for msgs in per_turn(r["messages"]):
            n = len(tokenizer.encode(tokenizer.apply_chat_template(msgs, tools=TOOLS, tokenize=False),
                                     add_special_tokens=False))
            if n > args.max_tokens:
                stats["examples/dropped_too_long"] += 1
                continue
            lengths.append(n)
            kind = "tool_call" if msgs[-1].get("tool_calls") else "answer"
            stats[f"examples/{split}/{kind}"] += 1
            out[split].append({"messages": msgs, "tools": TOOLS})
    args.out.mkdir(parents=True, exist_ok=True)
    for split, exs in out.items():
        rng.shuffle(exs)
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for e in exs:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    lengths.sort()
    stats_out = {"counts": dict(sorted(stats.items())), "rollouts": [str(p) for p in args.rollouts],
                 "tokens": {"p50": lengths[len(lengths) // 2], "p95": lengths[int(len(lengths) * 0.95)],
                            "max": lengths[-1]} if lengths else {},
                 "caps": {"per_task": args.per_task, "scripted_per_family": args.scripted_per_family,
                          "max_tokens": args.max_tokens}}
    (args.out / "stats.json").write_text(json.dumps(stats_out, indent=1), encoding="utf-8")
    print(json.dumps(stats_out, indent=1))
    return 0


def train(args) -> int:
    spec = cfg.MODELS[args.model]
    args.adapter_path.mkdir(parents=True, exist_ok=True)
    lora_cfg = args.adapter_path / "mlx_lora_config.yaml"
    lora_cfg.write_text(f"lora_parameters:\n  rank: {args.lora_rank}\n  scale: {args.lora_scale}\n  dropout: 0.0\n",
                        encoding="utf-8")
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", str(args.model_path or spec["mlx_id"]), "--train",
           "--data", str(args.data), "--fine-tune-type", "lora", "--mask-prompt", "--iters", str(args.iters),
           "--batch-size", "1", "--num-layers", str(args.num_layers), "--learning-rate", str(args.lr),
           "--max-seq-length", str(args.max_tokens), "--adapter-path", str(args.adapter_path),
           "--steps-per-eval", str(args.steps_per_eval), "--save-every", str(args.steps_per_eval),
           "--val-batches", "40", "--seed", str(args.seed), "--grad-checkpoint", "-c", str(lora_cfg)]
    if args.resume_adapter_file:
        cmd += ["--resume-adapter-file", str(args.resume_adapter_file)]
    log.info("running: %s", " ".join(cmd))
    started = time.monotonic()
    child = subprocess.Popen(cmd, cwd=ROOT)

    def stop(signum, _frame):
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    rc = child.wait()
    log.info("trainer exited %s after %.1f min", rc, (time.monotonic() - started) / 60)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rollouts", type=Path, nargs="+", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--per-task", type=int, default=2)
    b.add_argument("--scripted-per-family", type=int, default=150)
    b.add_argument("--max-tokens", type=int, default=4096)
    b.add_argument("--model", default="qwen3-4b-instruct", choices=sorted(cfg.MODELS))
    b.add_argument("--seed", type=int, default=7)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--adapter-path", type=Path, required=True)
    t.add_argument("--model", default="qwen3-4b-instruct", choices=sorted(cfg.MODELS))
    t.add_argument("--model-path", type=Path)
    t.add_argument("--resume-adapter-file", type=Path)
    t.add_argument("--iters", type=int, default=1200)
    t.add_argument("--num-layers", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lora-rank", type=int, default=8)
    t.add_argument("--lora-scale", type=float, default=20.0)
    t.add_argument("--max-tokens", type=int, default=4096)
    t.add_argument("--steps-per-eval", type=int, default=200)
    t.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return build(args) if args.cmd == "build" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
