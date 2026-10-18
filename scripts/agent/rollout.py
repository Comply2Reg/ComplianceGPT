#!/usr/bin/env python3
"""Play many episodes with a local MLX model and score each one — probe, calibration, held-out eval.

    python -m scripts.agent.rollout --tasks data/agent/tasks/v1.jsonl --split train \\
        --per-family 4 --k 1 --out data/agent/rollouts/probe20.jsonl
    python -m scripts.agent.rollout --tasks data/agent/tasks/v1.jsonl --split train \\
        --per-family 60 --k 4 --out data/agent/rollouts/calib_v1_base.jsonl

Episodes run in lockstep: each round sends the next turn of every live episode
to batch_generate, applies the replies (tool calls run, answers recorded), and
drops finished episodes. Every prompt is rendered by the tokenizer's own chat
template with tools.json, exactly as mlx_lm.server would render it.

One row per episode (transcript + verdict). Resumable: (task_id, sample) pairs
already in --out are skipped. A summary per family is printed at the end and
written beside --out as <name>.summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.agent.env import TOOLS, start, step, transcript  # noqa: E402
from scripts.agent.index import DEFAULT_DB, Corpus  # noqa: E402
from scripts.agent.verify import summary, verify_episode  # noqa: E402
from scripts.triage.corpus import read_jsonl  # noqa: E402

log = logging.getLogger("agent.rollout")

# Qwen3-Instruct-2507 recommended sampling.
SAMPLING = {"temp": 0.7, "top_p": 0.8, "top_k": 20}


def select(tasks: List[Dict], split: str, families: List[str], per_family: int, seed: int) -> List[Dict]:
    rows = [t for t in tasks if (split == "all" or t["split"] == split) and (not families or t["family"] in families)]
    by_fam: Dict[str, List[Dict]] = defaultdict(list)
    for t in sorted(rows, key=lambda t: t["task_id"]):
        by_fam[t["family"]].append(t)
    out: List[Dict] = []
    for fam in sorted(by_fam):
        group = by_fam[fam]
        random.Random(f"{seed}:{fam}").shuffle(group)
        out.extend(group[:per_family] if per_family else group)
    return out


def run(args) -> int:
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    corpus = Corpus(args.db)
    tasks = select(read_jsonl(args.tasks), args.split, args.families, args.per_family, args.seed)
    done = {(r["task_id"], r["sample"]) for r in read_jsonl(args.out)} if args.out.exists() else set()
    todo = [(t, s) for t in tasks for s in range(args.k) if (t["task_id"], s) not in done]
    log.info("%d tasks x k=%d; %d episodes already done, %d to run", len(tasks), args.k, len(done), len(todo))

    spec = cfg.MODELS[args.model]
    model_id = str(args.model_path) if args.model_path else spec["mlx_id"]
    model, tokenizer = load(model_id, adapter_path=str(args.adapter_path) if args.adapter_path else None)
    sampler = make_sampler(**SAMPLING) if args.temp > 0 else make_sampler(temp=0.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    n_done = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            eps = [start(t) for t, _ in chunk]
            t0 = time.monotonic()
            while True:
                live = [e for e in eps if not e.done]
                if not live:
                    break
                prompts = [tokenizer.apply_chat_template(e.messages, tools=TOOLS, add_generation_prompt=True)
                           for e in live]
                resp = batch_generate(model, tokenizer, prompts, max_tokens=args.max_new_tokens, sampler=sampler)
                for e, text in zip(live, resp.texts):
                    step(e, text, corpus)
            per_ep = (time.monotonic() - t0) / len(eps)
            for (t, s), e in zip(chunk, eps):
                v = verify_episode(e, corpus)
                row = transcript(e)
                row.update({"sample": s, "split": t["split"], "passed": v.passed, "reason": v.reason,
                            "seconds": round(per_ep, 1), "model": model_id,
                            "adapter": str(args.adapter_path) if args.adapter_path else None})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += len(chunk)
            rate = (time.monotonic() - started) / n_done
            log.info("%d/%d episodes, %.1fs/episode, ETA %.0f min", n_done, len(todo), rate,
                     rate * (len(todo) - n_done) / 60)

    rows = read_jsonl(args.out)
    result = {"summary": summary(rows), "episodes": len(rows), "tasks_file": str(args.tasks),
              "split": args.split, "k": args.k, "model": model_id,
              "adapter": str(args.adapter_path) if args.adapter_path else None}
    args.out.with_suffix(".summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "heldout", "all"], default="train")
    ap.add_argument("--families", nargs="*", default=[])
    ap.add_argument("--per-family", type=int, default=0, help="tasks per family (0 = all)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="qwen3-4b-instruct", choices=sorted(cfg.MODELS))
    ap.add_argument("--model-path", type=Path)
    ap.add_argument("--adapter-path", type=Path)
    ap.add_argument("--temp", type=float, default=0.7, help="0 = greedy")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch", type=int, default=4, help="episodes in lockstep")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
