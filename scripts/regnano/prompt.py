"""How a RegNano task is put to the model — shared by calibration, RL and eval."""

from __future__ import annotations

from typing import Dict, List

SYSTEM_PROMPT = (
    "You are a regulatory compliance analyst. Answer strictly from the passage you are "
    "given, never from memory. Quote text exactly when asked to quote. Reply with the "
    "requested JSON object only."
)


def to_messages(task: Dict) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage:\n{task['context']}\n\nTask: {task['question']}"},
    ]
