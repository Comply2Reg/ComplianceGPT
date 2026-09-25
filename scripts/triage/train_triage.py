#!/usr/bin/env python3
"""Fine-tune the focus model on the triage dataset, and evaluate it.

    # CPU, no GPU libraries: render, tokenize, print lengths and one example
    python -m scripts.triage.train_triage --dataset-dir data/uk/datasets/green \\
        --version v1 --dry-run

    # GPU (Colab T4/L4 or better; pip install unsloth trl)
    python -m scripts.triage.train_triage --dataset-dir data/uk/datasets/green \\
        --version v1 --out models/uk-triage-v1-qwen3-4b-instruct
    python -m scripts.triage.train_triage --dataset-dir data/uk/datasets/green \\
        --version v1 --eval-only models/uk-triage-v1-qwen3-4b-instruct

Defaults come from scripts/config.py MODELS[FOCUS_MODEL] (Qwen3-4B-Instruct-2507:
ChatML template, 2048-token budget, response-only loss on the assistant turn).
`--model gemma-4-e4b-it` swaps everything model-specific from the registry.
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
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import read_jsonl  # noqa: E402
from scripts.triage.render import (  # noqa: E402
    count_tokens,
    load_tokenizer,
    parse_assistant_json,
    render,
    to_messages,
)
from scripts.triage.metrics import score  # noqa: E402

log = logging.getLogger("triage.train")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--model", default=cfg.FOCUS_MODEL, choices=sorted(cfg.MODELS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=float, default=cfg.TRAIN["epochs"])
    ap.add_argument("--lr", type=float, default=cfg.TRAIN["learning_rate"])
    ap.add_argument("--seed", type=int, default=cfg.TRAIN["seed"])
    ap.add_argument("--batch", type=int, default=cfg.TRAIN["per_device_batch"])
    ap.add_argument("--grad-accum", type=int, default=cfg.TRAIN["grad_accum"])
    ap.add_argument("--lora-r", type=int, default=cfg.TRAIN["lora_r"])
    ap.add_argument("--lora-alpha", type=int, default=cfg.TRAIN["lora_alpha"])
    ap.add_argument(
        "--lora-mlp", action="store_true", help="also adapt gate/up/down projections"
    )
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="records per split (smoke)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval-only", type=Path, default=None, metavar="ADAPTER_DIR")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    return ap


def load_split(dataset_dir: Path, name: str, version: str, limit=None) -> List[Dict]:
    rows = read_jsonl(dataset_dir / f"{name}_{version}.jsonl")
    return rows[:limit] if limit else rows


def rendered_texts(rows: List[Dict], tokenizer, model_key: str) -> List[str]:
    return [
        render(r.get("messages") or to_messages(r), tokenizer, model_key) for r in rows
    ]


def dry_run(args, spec) -> int:
    tokenizer = load_tokenizer(args.model)
    train = load_split(args.dataset_dir, "train", args.version, args.limit)
    if not train:
        log.error("no train split under %s", args.dataset_dir)
        return 1
    budget = args.max_seq_length or spec["max_seq_length"]
    texts = rendered_texts(train, tokenizer, args.model)
    lengths = sorted(count_tokens(t, tokenizer) for t in texts)
    over = sum(1 for n in lengths if n > budget)
    print(
        f"model {args.model} ({spec['hf_id']}) | tokenizer "
        f"{'loaded' if tokenizer else 'unavailable — chars/4 proxy'} | "
        f"records {len(train)}"
    )
    print(
        f"tokens p50 {lengths[len(lengths) // 2]} "
        f"p95 {lengths[int(len(lengths) * 0.95)]} "
        f"max {lengths[-1]} | over budget {budget}: {over}"
    )
    print(
        f"response marker: {spec['response_part']!r} | instruction marker: "
        f"{spec['instruction_part']!r}"
    )
    print("---- first rendered example ----")
    print(texts[0][:2500])
    return 0 if over == 0 else 1


def train(args, spec) -> int:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    max_seq = args.max_seq_length or spec["max_seq_length"]
    out = args.out or Path("models") / f"uk-triage-{args.version}-{args.model}"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["hf_id"], max_seq_length=max_seq, load_in_4bit=True
    )
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.lora_mlp:
        targets += ["gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_rows = load_split(args.dataset_dir, "train", args.version, args.limit)
    val_rows = load_split(args.dataset_dir, "val", args.version, args.limit)
    train_ds = Dataset.from_list(
        [{"text": t} for t in rendered_texts(train_rows, tokenizer, args.model)]
    )
    val_ds = (
        Dataset.from_list(
            [{"text": t} for t in rendered_texts(val_rows, tokenizer, args.model)]
        )
        if val_rows
        else None
    )

    config = SFTConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        seed=args.seed,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_seq_length=max_seq,
        packing=False,
        dataset_text_field="text",
        report_to="none",
        bf16=False,
        fp16=True,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=config,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=spec["instruction_part"],
        response_part=spec["response_part"],
    )
    started = time.monotonic()
    result = trainer.train()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    report = {
        "version": args.version,
        "model": args.model,
        "hf_id": spec["hf_id"],
        "records": {"train": len(train_rows), "val": len(val_rows)},
        "max_seq_length": max_seq,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": targets},
        "train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
        "seconds": round(time.monotonic() - started),
        "finished_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    (out / f"train_report_{args.version}_{args.model}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    log.info("saved adapters and report -> %s", out)
    return 0


def evaluate(args, spec) -> int:
    from unsloth import FastLanguageModel

    max_seq = args.max_seq_length or spec["max_seq_length"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.eval_only), max_seq_length=max_seq, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    rows = load_split(args.dataset_dir, "test", args.version, args.limit)
    if not rows:
        log.error("no test split")
        return 1
    preds = []
    gen_tokens = 0
    started = time.monotonic()
    for r in rows:
        messages = (r.get("messages") or to_messages(r))[:2]  # system + user only
        prompt = render(messages, tokenizer, args.model, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        gen_tokens += int(new_tokens.shape[0])
        preds.append(
            parse_assistant_json(tokenizer.decode(new_tokens, skip_special_tokens=True))
        )
    elapsed = time.monotonic() - started
    result = score(rows, preds)
    result.update(
        {
            "version": args.version,
            "model": args.model,
            "backend": "unsloth",
            "adapter": str(args.eval_only),
            "tokens_per_second": round(gen_tokens / elapsed, 1) if elapsed else None,
            "evaluated_at": dt.datetime.now(dt.UTC).isoformat(),
        }
    )
    out = args.dataset_dir / f"eval_{args.version}_{args.model}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg.load_env()
    spec = cfg.MODELS[args.model]
    if args.dry_run:
        return dry_run(args, spec)
    if args.eval_only:
        return evaluate(args, spec)
    return train(args, spec)


if __name__ == "__main__":
    raise SystemExit(main())
