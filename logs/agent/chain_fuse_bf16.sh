#!/bin/zsh
# Stage 6: a single-file model without the 4-bit re-quantization loss: fuse warm_v1 with --dequantize.
cd /Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT
source .venv/bin/activate
python -m mlx_lm fuse --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path models/agent-qwen3-4b-warm-v1 \
  --save-path models/agent-qwen3-4b-v1-fused-bf16 --dequantize > logs/agent/fuse_v1_bf16.log 2>&1 || { echo "$(date) FUSE FAILED"; exit 1; }
echo "$(date) fused bf16; eval start"
python -m scripts.agent.rollout --tasks data/agent/tasks/v2.jsonl --split heldout --per-family 40 --k 2 --batch 4 \
  --model-path models/agent-qwen3-4b-v1-fused-bf16 --out data/agent/rollouts/heldout_fused_bf16_v1.jsonl > logs/agent/heldout_fused_bf16_v1.log 2>&1 || { echo "$(date) EVAL FAILED"; exit 1; }
python -m scripts.agent.compare data/agent/rollouts/heldout_warm_v1.jsonl data/agent/rollouts/heldout_fused_bf16_v1.jsonl --out data/agent/rollouts/compare_warm_fused_bf16_v1.json > /dev/null
echo "$(date) CHAIN DONE"
