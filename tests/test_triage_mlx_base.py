"""The non-thinking-by-default template patch for Qwen3.5 — no model download needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.mlx_base import PATCHED, STOCK, patch_dir, patch_template  # noqa: E402

# The generation-prompt tail of Qwen/Qwen3.5-4B's chat_template.jinja.
TAIL = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    f"    {STOCK}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
)


def _render(template: str, **kwargs) -> str:
    from jinja2 import Environment

    return Environment().from_string(template).render(add_generation_prompt=True, **kwargs)


def test_patch_makes_non_thinking_the_default_and_keeps_explicit_thinking() -> None:
    patched, changed = patch_template(TAIL)
    assert changed and PATCHED in patched and STOCK not in patched
    empty = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert _render(patched) == empty
    assert _render(patched, enable_thinking=False) == empty
    # mlx_lm injects enable_thinking=True behind every caller; it must not matter.
    assert _render(patched, enable_thinking=True) == empty
    assert _render(patched, regnano_thinking=True) == "<|im_start|>assistant\n<think>\n"
    # The stock template thinks unless told not to — the reason for the patch.
    assert _render(TAIL) == "<|im_start|>assistant\n<think>\n"


def test_patch_is_idempotent_and_refuses_an_unknown_template() -> None:
    once, _ = patch_template(TAIL)
    assert patch_template(once) == (once, False)
    with pytest.raises(ValueError):
        patch_template("{{ messages }}")


def test_patch_dir_rewrites_jinja_and_tokenizer_config(tmp_path: Path) -> None:
    (tmp_path / "chat_template.jinja").write_text(TAIL, encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": TAIL, "eos_token": "<|im_end|>"}), encoding="utf-8"
    )
    assert patch_dir(tmp_path) == ["chat_template.jinja", "tokenizer_config.json"]
    assert PATCHED in (tmp_path / "chat_template.jinja").read_text(encoding="utf-8")
    tok = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert PATCHED in tok["chat_template"] and tok["eos_token"] == "<|im_end|>"
    assert patch_dir(tmp_path) == []


def test_qwen35_registry_entry_trains_on_the_prefilled_empty_think_block() -> None:
    spec = cfg.MODELS["qwen3.5-4b"]
    assert spec["thinking_switch"] is True
    assert spec["response_part"].endswith("<think>\n\n</think>\n\n")
    assert spec["mlx_id"].startswith("models/base/")
