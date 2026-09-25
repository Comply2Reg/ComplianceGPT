"""Scoring and the MLX data export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triage import train_mlx  # noqa: E402
from scripts.triage.corpus import write_jsonl  # noqa: E402
from scripts.triage.metrics import score  # noqa: E402


def _rec(cls: str, prio: str = "P1", funcs=("Compliance",), obl=True) -> dict:
    return {
        "metadata": {"source_id": f"x:{cls}"},
        "classification": cls,
        "instruction": "i",
        "input_text": "t",
        "tool_use": None,
        "thought_trace": "r",
        "output": {
            "alert_class": cls,
            "alert_class_confidence": "high",
            "priority": prio,
            "primary_functions": list(funcs),
            "secondary_functions": [],
            "lines_of_defence": [2],
            "summary": "s",
            "key_dates": [],
            "applicability": [],
            "obligations_present": obl,
        },
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": json.dumps({"alert_class": cls})},
        ],
    }


def test_score_counts_validity_accuracy_and_jaccard() -> None:
    records = [_rec("A2"), _rec("A6", "P2", ("Compliance", "Legal")), _rec("A11", "P3")]
    preds = [
        {"alert_class": "A2", "priority": "P1", "primary_functions": ["Compliance"],
         "obligations_present": True},
        {"alert_class": "A3", "priority": "P2", "primary_functions": ["Legal"],
         "obligations_present": True},
        None,
    ]
    s = score(records, preds)
    assert s["records"] == 3
    assert abs(s["json_valid_rate"] - 2 / 3) < 1e-3
    assert abs(s["alert_class_accuracy"] - 1 / 3) < 1e-3
    assert abs(s["priority_accuracy"] - 2 / 3) < 1e-3
    assert abs(s["primary_function_jaccard"] - (1.0 + 0.5) / 2) < 1e-3
    assert s["per_class_gold"] == {"A2": 1, "A6": 1, "A11": 1}
    assert 0 < s["alert_class_macro_f1"] < 1


def test_mlx_export_writes_chat_format(tmp_path: Path) -> None:
    ds = tmp_path / "green"
    write_jsonl(ds / "train_t.jsonl", [_rec("A2"), _rec("A3")])
    write_jsonl(ds / "val_t.jsonl", [_rec("A6")])
    write_jsonl(ds / "test_t.jsonl", [_rec("A11")])
    rc = train_mlx.main(["export", "--dataset-dir", str(ds), "--version", "t"])
    assert rc == 0
    for name, n in (("train", 2), ("valid", 1), ("test", 1)):
        lines = [json.loads(line) for line in (ds / "mlx" / f"{name}.jsonl").open()]
        assert len(lines) == n
        assert list(lines[0]) == ["messages"]
        roles = [m["role"] for m in lines[0]["messages"]]
        assert roles == ["system", "user", "assistant"]
