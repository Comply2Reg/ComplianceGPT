#!/usr/bin/env python3
"""Agent playground: run one question through the tool-calling model and watch each step.

    python -m scripts.agent.play "How does the SEC define 'covered person' in Regulation S-P?"
    python -m scripts.agent.play --tasks data/agent/tasks/v1.jsonl --task-id 3f2c...   # scored
    python -m scripts.agent.play --server http://127.0.0.1:8080/v1 "..."               # via mlx_lm.server

Local mode loads the model with mlx-lm. Server mode talks to any OpenAI-compatible
endpoint (mlx_lm.server) with tools.json, the way the app would, and turns the
returned tool_calls back into the model's own <tool_call> text so the episode
runs through the same harness and verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.agent.env import TOOLS, from_openai, start, step, to_openai  # noqa: E402
from scripts.agent.index import DEFAULT_DB, Corpus  # noqa: E402
from scripts.agent.verify import verify_episode  # noqa: E402

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def local_generator(args):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    spec = cfg.MODELS[args.model]
    model, tokenizer = load(str(args.model_path) if args.model_path else spec["mlx_id"],
                            adapter_path=str(args.adapter_path) if args.adapter_path else None)
    sampler = make_sampler(temp=args.temp, top_p=0.8, top_k=20) if args.temp > 0 else make_sampler(temp=0.0)

    def gen(messages):
        prompt = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True)
        return generate(model, tokenizer, prompt, max_tokens=args.max_new_tokens, sampler=sampler)

    return gen


def server_generator(args):
    from openai import OpenAI

    client = OpenAI(base_url=args.server, api_key="not-needed")

    def gen(messages):
        msg = client.chat.completions.create(model=args.server_model, messages=to_openai(messages), tools=TOOLS,
                                             temperature=args.temp, max_tokens=args.max_new_tokens).choices[0].message
        return from_openai(msg)

    return gen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--tasks", type=Path)
    ap.add_argument("--task-id")
    ap.add_argument("--model", default="qwen3-4b-instruct", choices=sorted(cfg.MODELS))
    ap.add_argument("--model-path", type=Path)
    ap.add_argument("--adapter-path", type=Path)
    ap.add_argument("--server", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--server-model", default="default_model")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args(argv)

    if args.task_id:
        if not args.tasks:
            ap.error("--task-id needs --tasks")
        task = next((json.loads(line) for line in args.tasks.open(encoding="utf-8")
                     if json.loads(line)["task_id"] == args.task_id), None)
        if task is None:
            ap.error(f"task {args.task_id} not in {args.tasks}")
    elif args.question:
        task = {"task_id": "adhoc", "family": "adhoc", "prompt": args.question}
    else:
        ap.error("give a question or --tasks with --task-id")

    corpus = Corpus(args.db)
    gen = server_generator(args) if args.server else local_generator(args)
    ep = start(task)
    print(f"{BOLD}question{RESET} {task['prompt']}\n")
    while not ep.done:
        before = len(ep.calls)
        step(ep, gen(ep.messages), corpus)
        for c in ep.calls[before:]:
            print(f"{BOLD}call{RESET} {c.name}({json.dumps(c.arguments, ensure_ascii=False)})")
            shown = c.result_text if c.ok else f"error: {c.error}"
            print(f"{DIM}{shown[:800]}{'…' if len(shown) > 800 else ''}{RESET}\n")
        if ep.over_budget:
            print(f"{BOLD}refused{RESET} {ep.over_budget} call(s) past the budget\n")
    print(f"{BOLD}answer{RESET} {ep.final}")
    if task["family"] != "adhoc":
        v = verify_episode(ep, corpus)
        print(f"\n{BOLD}{'PASS' if v.passed else 'FAIL'}{RESET} {v.reason}  ({len(ep.calls)} calls, {ep.turns} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
