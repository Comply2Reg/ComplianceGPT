"""RegNano verifiers as an mlx-lm-lora GRPO reward (--reward-functions-file).

mlx-lm-lora loads this file by path and calls

    regnano_verifier(prompts, completions, answer, types) -> List[float]

with one entry per sampled completion. `answer` is the task JSON that
rl.py export put in the dataset's "answer" field (task_id, family,
context, gold), so the reward is exactly verifiers.verify — binary, no
judge model.

The loader swallows import errors and silently falls back to its default
math-format rewards, so rl.py imports this file in the RL venv before
launching and refuses to train if REGNANO_REWARD_READY is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function  # noqa: E402

from scripts.regnano.verifiers import verify  # noqa: E402


@register_reward_function("regnano_verifier")
def regnano_verifier(
    prompts: List[str], completions: List[str], answer: List[str], types: Optional[List] = None
) -> List[float]:
    rewards = []
    for completion, task_json in zip(completions, answer):
        try:
            task = json.loads(task_json)
        except (TypeError, json.JSONDecodeError):
            rewards.append(0.0)
            continue
        rewards.append(verify(task, completion).reward)
    return rewards


REGNANO_REWARD_READY = True
