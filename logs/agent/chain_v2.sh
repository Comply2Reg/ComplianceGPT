#!/bin/zsh
# Stage 4: rft_v1 (on-policy only, from warm) regressed 0.704 -> 0.566: the model drifted to
# one search then not_found. Retry with replay: one SFT from base on demos + teacher + on-policy
# successes together. Kept only if it beats warm_v1 on held-out; v1 files are never touched.
cd /Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT
source .venv/bin/activate
until grep -q "CHAIN DONE" logs/agent/chain_final.out 2>/dev/null; do
  grep -qE "FAILED|not starting" logs/agent/chain_final.out 2>/dev/null && { echo "$(date) final chain failed; not starting"; exit 1; }
  sleep 30
done
python -m scripts.agent.sft build --rollouts data/agent/rollouts/demos_v2.jsonl data/agent/rollouts/teacher_v2.jsonl \
  data/agent/rollouts/onpolicy_warm_v1.jsonl --per-task 3 --out data/agent/sft/mix_v2 > logs/agent/build_mix_v2.log 2>&1 || { echo "$(date) BUILD FAILED"; exit 1; }
echo "$(date) mix_v2 train start"
python -m scripts.agent.sft train --data data/agent/sft/mix_v2 --adapter-path models/agent-qwen3-4b-mix-v2 \
  --iters 2400 --num-layers 8 --lr 1e-4 --steps-per-eval 600 > logs/agent/train_mix_v2.log 2>&1 || { echo "$(date) TRAIN FAILED"; exit 1; }
echo "$(date) mix_v2 eval start"
python -m scripts.agent.rollout --tasks data/agent/tasks/v2.jsonl --split heldout --per-family 40 --k 2 --batch 4 \
  --adapter-path models/agent-qwen3-4b-mix-v2 --out data/agent/rollouts/heldout_mix_v2.jsonl > logs/agent/heldout_mix_v2.log 2>&1 || { echo "$(date) EVAL FAILED"; exit 1; }
python -m scripts.agent.compare data/agent/rollouts/heldout_base.jsonl data/agent/rollouts/heldout_mix_v2.jsonl --out data/agent/rollouts/compare_base_mix_v2.json > /dev/null
python -m scripts.agent.compare data/agent/rollouts/heldout_warm_v1.jsonl data/agent/rollouts/heldout_mix_v2.jsonl --out data/agent/rollouts/compare_warm_mix_v2.json > /dev/null
WIN=$(python -c "
import json
c=json.load(open('data/agent/rollouts/compare_warm_mix_v2.json'))['families']['ALL']
print('yes' if c['new'] > c['base'] else 'no')")
echo "$(date) mix_v2 beats warm_v1: $WIN"
if [[ $WIN == yes ]]; then
  python -m mlx_lm fuse --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path models/agent-qwen3-4b-mix-v2 \
    --save-path models/agent-qwen3-4b-v2-fused > logs/agent/fuse_v2.log 2>&1 || { echo "$(date) FUSE FAILED"; exit 1; }
  python -m scripts.agent.rollout --tasks data/agent/tasks/v2.jsonl --split heldout --per-family 40 --k 2 --batch 4 \
    --model-path models/agent-qwen3-4b-v2-fused --out data/agent/rollouts/heldout_fused_v2.jsonl > logs/agent/heldout_fused_v2.log 2>&1 || { echo "$(date) EVAL FAILED"; exit 1; }
  python -m scripts.agent.compare data/agent/rollouts/heldout_base.jsonl data/agent/rollouts/heldout_fused_v2.jsonl --out data/agent/rollouts/compare_base_fused_v2.json > /dev/null
fi
echo "$(date) CHAIN DONE"
