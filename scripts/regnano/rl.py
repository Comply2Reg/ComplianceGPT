#!/usr/bin/env python3
"""RegNano RL iteration on the laptop: GRPO via mlx-lm-lora, verifier reward.

    # 1. calibrated, in-band tasks -> GRPO dataset (train/valid by task_id hash)
    python -m scripts.regnano.rl export --tasks data/regnano/tasks/iter1.jsonl \\
        --out data/regnano/grpo/iter1
    # 2. train (runs in .venv-rl, which carries mlx-lm-lora)
    python -m scripts.regnano.rl train --data data/regnano/grpo/iter1 \\
        --model-path models/publish/<v4 fused> --adapter-path models/regnano-iter1 \\
        [--iters 200] [--group-size 4]

The dataset row is {"prompt", "system", "answer", "type"}; "answer" carries
the task JSON (context + gold) that grpo_rewards.py hands to the verifier.
The gold never becomes a training target — GRPO only sees the reward.

mlx-lm-lora is installed in a separate venv (.venv-rl) so its mlx-lm pin
cannot move the one the SFT path (train_mlx.py) and the evals use.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.regnano.prompt import SYSTEM_PROMPT, to_messages  # noqa: E402
from scripts.triage.corpus import read_jsonl, write_jsonl  # noqa: E402

log = logging.getLogger("regnano.rl")
RL_PYTHON = ROOT / ".venv-rl" / "bin" / "python"
REWARDS = ROOT / "scripts" / "regnano" / "grpo_rewards.py"
VALID_SHARE = 0.1


def is_valid(task_id: str) -> bool:
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 100 < VALID_SHARE * 100


def to_row(task: dict) -> dict:
    user = to_messages(task)[1]["content"]
    answer = {k: task[k] for k in ("task_id", "family", "context", "question", "gold")}
    return {"prompt": user, "system": SYSTEM_PROMPT, "answer": json.dumps(answer, ensure_ascii=False), "type": task["family"]}


def export(args) -> int:
    tasks = read_jsonl(args.tasks)
    if not tasks:
        log.error("no tasks in %s", args.tasks)
        return 1
    train = [to_row(t) for t in tasks if not is_valid(t["task_id"])]
    valid = [to_row(t) for t in tasks if is_valid(t["task_id"])]
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "valid.jsonl", valid)
    log.info("grpo data -> %s train=%d valid=%d", args.out, len(train), len(valid))
    return 0


def preflight() -> None:
    """Import the reward file in the RL venv exactly as the trainer will."""
    probe = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('custom_rewards',{str(REWARDS)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert m.REGNANO_REWARD_READY;"
        "import json;"
        "t={'task_id':'x','family':'effective_date','context':'c','question':'q','gold':{'date':'2027-01-01'}};"
        "r=m.regnano_verifier(['p','p'],['{\"date\": \"2027-01-01\"}','{\"date\": null}'],[json.dumps(t)]*2);"
        "assert r==[1.0,0.0],r;print('reward preflight ok')"
    )
    subprocess.run([str(RL_PYTHON), "-c", probe], cwd=ROOT, check=True)


def train(args) -> int:
    if not RL_PYTHON.exists():
        log.error("RL venv missing: python3.12 -m venv .venv-rl && .venv-rl/bin/pip install mlx-lm-lora")
        return 1
    preflight()
    model = str(args.model_path) if args.model_path else cfg.MODELS[args.model]["mlx_id"]
    args.adapter_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(RL_PYTHON), "-m", "mlx_lm_lora.train",
        "--model", model,
        "--train", "--train-mode", "grpo", "--train-type", "lora",
        "--data", str(args.data),
        "--reward-functions-file", str(REWARDS),
        "--reward-functions", "regnano_verifier",
        "--group-size", str(args.group_size),
        "--batch-size", str(args.batch_size),
        "--iters", str(args.iters),
        "--learning-rate", str(args.lr),
        "--num-layers", str(args.num_layers),
        "--max-completion-length", str(args.max_completion_length),
        "--max-seq-length", str(args.max_seq_length),
        "--temperature", str(args.temperature),
        "--beta", str(args.beta),
        "--adapter-path", str(args.adapter_path),
        "--steps-per-eval", str(args.steps_per_eval),
        "--save-every", str(args.save_every),
        "--seed", str(args.seed),
        "--grad-checkpoint",
    ]
    log.info("running: %s", " ".join(cmd))
    started = time.monotonic()
    rc = subprocess.call(cmd, cwd=ROOT)
    report = {
        "model": model, "data": str(args.data), "command": cmd, "returncode": rc,
        "seconds": round(time.monotonic() - started),
        "finished_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    (args.adapter_path / "rl_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("export", "train"))
    ap.add_argument("--tasks", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--model", default="qwen3.5-4b", choices=sorted(cfg.MODELS))
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--adapter-path", type=Path, default=Path("models/regnano-iter1"))
    ap.add_argument("--iters", type=int, default=200)  # FrogNano: 200-update climbs
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--max-completion-length", type=int, default=320)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.0, help="KL weight; FrogNano uses none")
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=cfg.TRAIN["seed"])
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "export":
        if not (args.tasks and args.out):
            log.error("export needs --tasks and --out")
            return 2
        return export(args)
    if not args.data:
        log.error("train needs --data")
        return 2
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
