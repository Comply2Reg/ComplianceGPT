#!/bin/zsh
cd /Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT
source .venv/bin/activate
until [ -f data/agent/rollouts/heldout_base.summary.json ]; do sleep 20; done
echo "$(date) train start"
python -m scripts.agent.sft train --data data/agent/sft/warm_v1 --adapter-path models/agent-qwen3-4b-warm-v1 \
  --iters 2000 --num-layers 8 --lr 1e-4 --steps-per-eval 500 > logs/agent/train_warm_v1.log 2>&1 || { echo "$(date) TRAIN FAILED"; exit 1; }
echo "$(date) train done; eval start"
python -m scripts.agent.rollout --tasks data/agent/tasks/v2.jsonl --split heldout --per-family 40 --k 2 --batch 4 \
  --adapter-path models/agent-qwen3-4b-warm-v1 --out data/agent/rollouts/heldout_warm_v1.jsonl > logs/agent/heldout_warm_v1.log 2>&1 || { echo "$(date) EVAL FAILED"; exit 1; }
python -m scripts.agent.compare data/agent/rollouts/heldout_base.jsonl data/agent/rollouts/heldout_warm_v1.jsonl --out data/agent/rollouts/compare_base_warm_v1.json > /dev/null
echo "$(date) CHAIN DONE"
