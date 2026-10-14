#!/usr/bin/env python3
"""mlx-lm-lora's SFT trainer with its step left uncompiled. Runs in .venv-rl.

    .venv-rl/bin/python -m scripts.regnano.sft_eager --model ... --train --train-mode sft ...

train_sft wraps its step in mx.compile, and on Qwen3.5-4B the compiled graph
runs out of Metal memory where the eager one does not (measured 2026-10-14,
800 tokens, 16 LoRA layers, grad checkpointing: eager 10.3 GB / 10 s per step,
compiled kIOGPUCommandBufferCallbackErrorOutOfMemory on step 1). Only that
module's compile is disabled; everything else is the stock CLI.

The buffer cache is capped too: after one long row MLX otherwise keeps its
~17 GB high-water mark, and every later step swaps (~100 s/step vs ~7 s).
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm_lora import train
from mlx_lm_lora.trainer import sft_trainer


class _EagerMx:
    """mlx.core, except compile returns the function unchanged."""

    def __getattr__(self, name):
        return getattr(mx, name)

    @staticmethod
    def compile(fun, *args, **kwargs):
        return fun


sft_trainer.mx = _EagerMx()
mx.set_cache_limit(2 * 1024**3)

if __name__ == "__main__":
    train.main()
