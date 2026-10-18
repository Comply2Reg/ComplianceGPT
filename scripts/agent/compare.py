#!/usr/bin/env python3
"""Compare two held-out rollouts of the same tasks: pass rate per family, paired by task.

    python -m scripts.agent.compare data/agent/rollouts/heldout_base.jsonl data/agent/rollouts/heldout_warm.jsonl

Each task's score is its mean pass over its k samples. The delta's 95% interval
is a bootstrap over tasks (10,000 resamples), as in docs/regnano.md. Also reports
tool behaviour: valid-call rate, calls per episode, invented-id rate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triage.corpus import read_jsonl  # noqa: E402


def per_task(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["family"]][r["task_id"]].append(int(r["passed"]))
    return {f: {t: sum(v) / len(v) for t, v in tasks.items()} for f, tasks in acc.items()}


def behaviour(rows: List[Dict]) -> Dict:
    calls = [c for r in rows for c in r["calls"]]
    bad = sum(not c["ok"] for c in calls) + sum(r["malformed"] for r in rows)
    invented = sum(1 for c in calls if not c["ok"] and c["error"] and c["error"].startswith("no "))
    n = max(len(rows), 1)
    return {"episodes": len(rows), "calls_per_episode": round(len(calls) / n, 2),
            "valid_call_rate": round(1 - bad / max(len(calls) + sum(r["malformed"] for r in rows), 1), 3),
            "invented_id_calls_per_episode": round(invented / n, 3),
            "over_budget_episodes": sum(r["over_budget"] > 0 for r in rows),
            "json_final_rate": round(sum(r["reason"] != "final answer is not JSON" for r in rows) / n, 3)}


def bootstrap(a: List[float], b: List[float], n: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    idx = range(len(a))
    deltas = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        deltas.append(sum(b[i] - a[i] for i in s) / len(s))
    deltas.sort()
    return deltas[int(0.025 * n)], deltas[int(0.975 * n)]


def compare(base_rows: List[Dict], new_rows: List[Dict]) -> Dict:
    a, b = per_task(base_rows), per_task(new_rows)
    out = {}
    all_a, all_b = [], []
    for fam in sorted(set(a) | set(b)):
        common = sorted(set(a.get(fam, {})) & set(b.get(fam, {})))
        if not common:
            continue
        xa = [a[fam][t] for t in common]
        xb = [b[fam][t] for t in common]
        all_a += xa
        all_b += xb
        lo, hi = bootstrap(xa, xb)
        out[fam] = {"tasks": len(common), "base": round(sum(xa) / len(xa), 3), "new": round(sum(xb) / len(xb), 3),
                    "delta": round((sum(xb) - sum(xa)) / len(xa), 3), "ci95": [round(lo, 3), round(hi, 3)],
                    "up": sum(y > x for x, y in zip(xa, xb)), "down": sum(y < x for x, y in zip(xa, xb))}
    lo, hi = bootstrap(all_a, all_b)
    out["ALL"] = {"tasks": len(all_a), "base": round(sum(all_a) / len(all_a), 3),
                  "new": round(sum(all_b) / len(all_b), 3), "delta": round((sum(all_b) - sum(all_a)) / len(all_a), 3),
                  "ci95": [round(lo, 3), round(hi, 3)]}
    return {"families": out, "behaviour": {"base": behaviour(base_rows), "new": behaviour(new_rows)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    result = compare(read_jsonl(args.base), read_jsonl(args.new))
    result.update({"base_file": str(args.base), "new_file": str(args.new)})
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
