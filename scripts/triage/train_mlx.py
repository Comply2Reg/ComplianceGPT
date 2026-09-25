#!/usr/bin/env python3
"""Train and evaluate the triage model locally on Apple Silicon with MLX.

unsloth/bitsandbytes are CUDA-only; on a Mac the same records train through
Apple's mlx-lm against the 4-bit MLX conversion of the focus model
(config.MODELS[...]["mlx_id"]). A 4B model at 4 bits with LoRA on 16 of 36
layers fits comfortably in 16-24 GB of unified memory at 2048 tokens.

    DS=data/uk/datasets/green
    python -m scripts.triage.train_mlx export --dataset-dir $DS --version v1
    python -m scripts.triage.train_mlx train  --dataset-dir $DS --version v1 \\
        [--iters 600] [--batch-size 1] [--num-layers 16] [--lr 1e-4]
    python -m scripts.triage.train_mlx eval   --dataset-dir $DS --version v1 \\
        --adapter-path models/uk-triage-v1-qwen3-4b-instruct-mlx

`export` writes data/uk/datasets/<tier>/mlx/{train,valid,test}.jsonl in
mlx-lm's chat format ({"messages": [...]}) straight from the records'
`messages`; `train` runs `mlx_lm lora --mask-prompt` (loss on the assistant
turn only); `eval` generates greedily on the test split and scores it with
scripts/triage/metrics.py — the same numbers train_triage.py --eval-only
produces on CUDA.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import read_jsonl, write_jsonl  # noqa: E402
from scripts.triage.metrics import score  # noqa: E402
from scripts.triage.render import parse_assistant_json, to_messages  # noqa: E402

log = logging.getLogger("triage.train_mlx")
SPLIT_FILES = {"train": "train", "val": "valid", "test": "test"}


def export(args) -> int:
    out = args.dataset_dir / "mlx"
    counts = {}
    for split, mlx_name in SPLIT_FILES.items():
        rows = read_jsonl(args.dataset_dir / f"{split}_{args.version}.jsonl")
        if not rows:
            continue
        records = [{"messages": r.get("messages") or to_messages(r)} for r in rows]
        counts[mlx_name] = write_jsonl(out / f"{mlx_name}.jsonl", records)
    if "train" not in counts:
        log.error("no train split under %s", args.dataset_dir)
        return 1
    log.info("mlx data -> %s %s", out, counts)
    return 0


def train(args) -> int:
    spec = cfg.MODELS[args.model]
    mlx_id = spec.get("mlx_id")
    if not mlx_id:
        log.error("no MLX conversion registered for %s", args.model)
        return 1
    data_dir = args.dataset_dir / "mlx"
    if not (data_dir / "train.jsonl").exists():
        rc = export(args)
        if rc:
            return rc
    adapter = (
        args.adapter_path
        or Path("models") / f"uk-triage-{args.version}-{args.model}-mlx"
    )
    adapter.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        mlx_id,
        "--train",
        "--data",
        str(data_dir),
        "--fine-tune-type",
        "lora",
        "--mask-prompt",
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        str(args.num_layers),
        "--learning-rate",
        str(args.lr),
        "--max-seq-length",
        str(args.max_seq_length),
        "--adapter-path",
        str(adapter),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--save-every",
        str(args.save_every),
        "--seed",
        str(args.seed),
        "--grad-checkpoint",
    ]
    if args.val_batches is not None:
        cmd += ["--val-batches", str(args.val_batches)]
    log.info("running: %s", " ".join(cmd))
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT)
    report = {
        "version": args.version,
        "model": args.model,
        "mlx_id": mlx_id,
        "backend": "mlx-lm",
        "command": cmd,
        "returncode": proc.returncode,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "seconds": round(time.monotonic() - started),
        "finished_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    (adapter / f"train_report_{args.version}_{args.model}_mlx.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log.info("adapters -> %s (rc=%s, %ss)", adapter, proc.returncode, report["seconds"])
    return proc.returncode


def evaluate(args) -> int:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    spec = cfg.MODELS[args.model]
    rows = read_jsonl(args.dataset_dir / f"test_{args.version}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        log.error("no test split")
        return 1
    model, tokenizer = load(
        spec["mlx_id"],
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
    )
    sampler = make_sampler(temp=0.0)
    preds: List[Dict] = []
    outputs: List[Dict] = []
    started = time.monotonic()
    gen_chars = 0
    for i, r in enumerate(rows, 1):
        messages = (r.get("messages") or to_messages(r))[:2]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        gen_chars += len(text)
        pred = parse_assistant_json(text)
        preds.append(pred)
        outputs.append(
            {
                "source_id": r["metadata"]["source_id"],
                "gold_alert_class": r["output"]["alert_class"],
                "pred_alert_class": (pred or {}).get("alert_class"),
                "raw": text[:1200],
            }
        )
        if i % 10 == 0:
            log.info("%d/%d generated", i, len(rows))
    elapsed = time.monotonic() - started
    result = score(rows, preds)
    result.update(
        {
            "version": args.version,
            "model": args.model,
            "backend": "mlx-lm",
            "mlx_id": spec["mlx_id"],
            "adapter": str(args.adapter_path) if args.adapter_path else None,
            "seconds": round(elapsed),
            "chars_per_second": round(gen_chars / elapsed, 1) if elapsed else None,
            "evaluated_at": dt.datetime.now(dt.UTC).isoformat(),
        }
    )
    suffix = "" if args.adapter_path else "_base"
    out = args.dataset_dir / f"eval_{args.version}_{args.model}_mlx{suffix}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_jsonl(
        args.dataset_dir
        / f"eval_{args.version}_{args.model}_mlx{suffix}_outputs.jsonl",
        outputs,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("per_class_gold", "per_class_tp")
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=("export", "train", "eval"))
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--model", default=cfg.FOCUS_MODEL, choices=sorted(cfg.MODELS))
    ap.add_argument("--adapter-path", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=cfg.TRAIN_MLX["iters"])
    ap.add_argument("--batch-size", type=int, default=cfg.TRAIN_MLX["batch_size"])
    ap.add_argument("--num-layers", type=int, default=cfg.TRAIN_MLX["num_layers"])
    ap.add_argument("--lr", type=float, default=cfg.TRAIN_MLX["learning_rate"])
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument(
        "--steps-per-eval", type=int, default=cfg.TRAIN_MLX["steps_per_eval"]
    )
    ap.add_argument("--save-every", type=int, default=cfg.TRAIN_MLX["save_every"])
    ap.add_argument("--val-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=cfg.TRAIN["seed"])
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None, help="eval: records")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg.load_env()
    if args.max_seq_length is None:
        args.max_seq_length = cfg.MODELS[args.model]["max_seq_length"]
    if args.command == "export":
        return export(args)
    if args.command == "train":
        return train(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
