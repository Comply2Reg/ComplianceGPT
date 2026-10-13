#!/usr/bin/env python3
"""Turn a calibration run into an RL iteration's task pool and its held-out eval.

    python -m scripts.regnano.pool --tasks data/regnano/tasks/mined_v2.jsonl \\
        --calibration data/regnano/calibration/pool_v1_qwen3.5-4b_base.jsonl \\
        --out-dir data/regnano/tasks/iter1

Only tasks still in --tasks count, so a calibrated task the miner has since
rejected (an ambiguous gold) is neither trained on nor scored.

  heldout.jsonl  --held-out-share of calibrated tasks by task_id hash, at
                 every difficulty (always / never / in band), never trained on.
                 Re-run calibrate.py on it with the RL adapter: same k and
                 sampling, so the base pass rate in pool.json is the baseline.
  train.jsonl    the rest, in band only (0 < pass < 1): tasks the policy
                 always or never solves give GRPO no advantage signal.
  pool.json      counts and the base held-out pass rate per family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regnano.calibrate import band  # noqa: E402


def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_held_out(task_id: str, share: float) -> bool:
    # Salted so the split is independent of rl.py's train/valid split.
    h = int(hashlib.sha256(f"heldout|{task_id}".encode()).hexdigest()[:8], 16)
    return h % 1000 < share * 1000


def build(tasks: List[Dict], calibration: List[Dict], share: float, lo: float, hi: float) -> Dict:
    by_id = {t["task_id"]: t for t in tasks}
    rates = {r["task_id"]: r["pass_rate"] for r in calibration if r["task_id"] in by_id}
    held = [by_id[i] for i in rates if is_held_out(i, share)]
    train = [by_id[i] for i in rates if not is_held_out(i, share) and band(rates[i], lo, hi)]
    stats: Dict[str, Dict] = {}
    for t in held:
        s = stats.setdefault(t["family"], {"held_out": 0, "base_pass_sum": 0.0, "train": 0})
        s["held_out"] += 1
        s["base_pass_sum"] += rates[t["task_id"]]
    for t in train:
        stats.setdefault(t["family"], {"held_out": 0, "base_pass_sum": 0.0, "train": 0})["train"] += 1
    for s in stats.values():
        s["base_held_out_pass"] = round(s.pop("base_pass_sum") / s["held_out"], 4) if s["held_out"] else None
    return {
        "train": train,
        "heldout": held,
        "stats": {
            "by_family": stats,
            "calibrated": len(rates),
            "dropped_by_miner": sum(r["task_id"] not in by_id for r in calibration),
            "held_out_share": share,
            "band": [lo, hi],
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--held-out-share", type=float, default=0.2)
    ap.add_argument("--lo", type=float, default=0.0)
    ap.add_argument("--hi", type=float, default=1.0)
    args = ap.parse_args(argv)
    calibration = [r for p in args.calibration for r in read_jsonl(p)]
    pool = build(read_jsonl(args.tasks), calibration, args.held_out_share, args.lo, args.hi)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", pool["train"])
    write_jsonl(args.out_dir / "heldout.jsonl", pool["heldout"])
    (args.out_dir / "pool.json").write_text(json.dumps(pool["stats"], indent=1), encoding="utf-8")
    print(json.dumps(pool["stats"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
