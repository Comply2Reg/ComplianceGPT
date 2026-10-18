#!/usr/bin/env python3
"""Teacher trajectories: an OpenAI model plays the same episodes, through the same harness and verifier.

    python -m scripts.agent.teacher --tasks data/agent/tasks/v2.jsonl --per-family 2 \\
        --out data/agent/rollouts/teacher_pilot.jsonl --max-usd 0.50

The teacher sees exactly what the student sees: the same system prompt, the same
tools.json, the same tool results from the local GREEN index. Its tool calls are
turned back into the student's own <tool_call> text, so a kept trajectory renders
through Qwen's chat template like any other. Only verified episodes are training
data; the rest are kept in --out (passed=false) for reading.

Train split by default. --split heldout scores the teacher on held-out tasks as a
reference point only: those rows carry split=heldout, which sft.py build refuses. --max-usd is a hard stop on
estimated spend (config.LABEL_MODELS prices); usage is recorded per episode.
Needs OPENAI_API_KEY in .env; the key is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.agent.env import TOOLS, from_openai, start, step, to_openai, transcript  # noqa: E402
from scripts.agent.index import DEFAULT_DB, Corpus  # noqa: E402
from scripts.agent.rollout import select  # noqa: E402
from scripts.agent.verify import summary, verify_episode  # noqa: E402
from scripts.triage.corpus import read_jsonl  # noqa: E402
from scripts.triage.labeller import completion_kwargs  # noqa: E402

log = logging.getLogger("agent.teacher")
DEFAULT_TEACHER = "gpt-5.4-mini"


def usd(spec: Dict, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = spec.get("price_per_1k") or (0.0, 0.0)
    return prompt_tokens / 1000 * p_in + completion_tokens / 1000 * p_out


async def play(client, model: str, spec: Dict, task: Dict, corpus: Corpus) -> Dict:
    ep = start(task)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.monotonic()
    kwargs = completion_kwargs(spec)
    while not ep.done:
        resp = await client.chat.completions.create(model=model, messages=to_openai(ep.messages), tools=TOOLS,
                                                    **kwargs)
        usage["prompt_tokens"] += resp.usage.prompt_tokens
        usage["completion_tokens"] += resp.usage.completion_tokens
        step(ep, from_openai(resp.choices[0].message), corpus)
    v = verify_episode(ep, corpus)
    row = transcript(ep)
    row.update({"sample": 0, "split": task["split"], "passed": v.passed, "reason": v.reason, "source": "teacher",
                "model": model, "usage": usage, "usd": round(usd(spec, **usage), 5),
                "seconds": round(time.monotonic() - t0, 1)})
    return row


async def run(args) -> int:
    from openai import AsyncOpenAI

    cfg.load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (.env)")
    spec = cfg.LABEL_MODELS.get(args.model)
    if not spec or not spec.get("price_per_1k"):
        raise SystemExit(f"{args.model} has no price in config.LABEL_MODELS; add one so --max-usd can hold")
    spec = dict(spec, max_output=args.max_output)

    tasks = select(read_jsonl(args.tasks), args.split, args.families, args.per_family, args.seed)
    done = {r["task_id"] for r in read_jsonl(args.out)} if args.out.exists() else set()
    spent = sum(r.get("usd", 0.0) for r in read_jsonl(args.out)) if args.out.exists() else 0.0
    todo = [t for t in tasks if t["task_id"] not in done]
    log.info("%d tasks, %d done, %d to run; spent so far $%.3f, cap $%.2f", len(tasks), len(done), len(todo),
             spent, args.max_usd)

    corpus = Corpus(args.db)
    client = AsyncOpenAI(max_retries=4, timeout=90.0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    async def one(task: Dict, fh) -> None:
        nonlocal spent, stop
        if stop:
            return
        async with sem:
            if stop:
                return
            try:
                row = await play(client, args.model, spec, task, corpus)
            except Exception as exc:  # an API failure is logged, never a training row
                log.warning("task %s failed: %s", task["task_id"], type(exc).__name__)
                return
        async with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            spent += row["usd"]
            if spent >= args.max_usd:
                stop = True
                log.warning("spend cap reached: $%.3f", spent)

    with args.out.open("a", encoding="utf-8") as fh:
        await asyncio.gather(*(one(t, fh) for t in todo))

    rows = read_jsonl(args.out)
    n = len(rows)
    tokens = {k: sum(r["usage"][k] for r in rows) for k in ("prompt_tokens", "completion_tokens")}
    result = {"summary": summary(rows), "episodes": n, "usd": round(sum(r["usd"] for r in rows), 4),
              "usd_per_episode": round(sum(r["usd"] for r in rows) / max(n, 1), 5),
              "tokens_per_episode": {k: round(v / max(n, 1)) for k, v in tokens.items()}, "model": args.model}
    args.out.with_suffix(".summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--families", nargs="*", default=[])
    ap.add_argument("--per-family", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=DEFAULT_TEACHER)
    ap.add_argument("--max-output", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-usd", type=float, required=True, help="hard stop on estimated spend")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
