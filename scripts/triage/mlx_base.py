#!/usr/bin/env python3
"""Make a local MLX conversion of a thinking-by-default model default to non-thinking.

Qwen3.5's chat template opens a `<think>` block on the generation prompt unless
the caller passes enable_thinking=False. mlx_lm's trainer (tuner/datasets.py)
calls apply_chat_template with no template kwargs, so with the stock template
the --mask-prompt offset ends at `<think>\\n` and the loss covers the closing
`</think>` — training on a different prompt than inference uses. This rewrites
the generation-prompt branch so that thinking is ON only when explicitly
requested with regnano_thinking=True; enable_thinking (True or False) no
longer changes anything, since mlx_lm sets it to True behind every caller.
Training, eval, `mlx_lm.server` and the published card then all see the same
prompt; tasks that want reasoning pass regnano_thinking=True.

    python -m scripts.triage.mlx_base models/base/qwen3.5-4b-mlx-4bit

Idempotent. Rewrites chat_template.jinja and, if it carries one, the
chat_template field of tokenizer_config.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STOCK = "{%- if enable_thinking is defined and enable_thinking is false %}"
# Keyed on our own variable, not enable_thinking: mlx_lm's TokenizerWrapper
# injects enable_thinking=True whenever the vocab has <think> and the caller
# passes nothing (tokenizer_utils.apply_chat_template) — which is exactly what
# its trainer does — so an enable_thinking default can never take effect.
PATCHED = "{%- if not (regnano_thinking is defined and regnano_thinking is true) %}"


def patch_template(text: str) -> tuple[str, bool]:
    """Return (template, changed). Raises if the template has neither form."""
    if PATCHED in text:
        return text, False
    if STOCK not in text:
        raise ValueError("generation-prompt thinking switch not found in template")
    return text.replace(STOCK, PATCHED), True


def patch_dir(model_dir: Path) -> list[str]:
    changed = []
    jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        text, did = patch_template(jinja.read_text(encoding="utf-8"))
        if did:
            jinja.write_text(text, encoding="utf-8")
            changed.append(jinja.name)
    tok_cfg = model_dir / "tokenizer_config.json"
    if tok_cfg.exists():
        data = json.loads(tok_cfg.read_text(encoding="utf-8"))
        if isinstance(data.get("chat_template"), str):
            text, did = patch_template(data["chat_template"])
            if did:
                data["chat_template"] = text
                tok_cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")
                changed.append(tok_cfg.name)
    if not jinja.exists() and not changed:
        raise FileNotFoundError(f"no chat template under {model_dir}")
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    args = ap.parse_args(argv)
    changed = patch_dir(args.model_dir)
    print(f"{args.model_dir}: " + (", ".join(changed) if changed else "already patched"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
