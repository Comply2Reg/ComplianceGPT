#!/bin/zsh
# Stage 2: on-policy rejection-sampling FT from the warm adapter (docs/agent.md).
cd /Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT
source .venv/bin/activate
until grep -q "CHAIN DONE" logs/agent/chain_warm.out 2>/dev/null; do
  grep -q "FAILED" logs/agent/chain_warm.out 2>/dev/null && { echo "$(date) warm chain failed; not starting"; exit 1; }
  sleep 30
done
echo "$(date) calibrate start"
python -m scripts.agent.rollout --tasks data/agent/tasks/v2.jsonl --split train --per-family 60 --k 4 --batch 4 \
  --adapter-path models/agent-qwen3-4b-warm-v1 --out data/agent/rollouts/onpolicy_warm_v1.jsonl > logs/agent/onpolicy_warm_v1.log 2>&1 || { echo "$(date) CALIB FAILED"; exit 1; }
python -m scripts.agent.sft build --rollouts data/agent/rollouts/onpolicy_warm_v1.jsonl --out data/agent/sft/rft_v1 > logs/agent/build_rft_v1.log 2>&1 || { echo "$(date) BUILD FAILED"; exit 1; }
ITERS=$(python -c "import json;c=json.load(open('data/agent/sft/rft_v1/stats.json'))['counts'];print(min(1500,max(200,c.get('examples/train/answer',0)+c.get('examples/train/tool_call',0))))")
echo "$(date) rft train start, iters=$ITERS"
python -m scripts.agent.sft train --data data/agent/sft/rft_v1 --adapter-path models/agent-qwen3-4b-rft-v1 \
  --resume-adapter-file models/agent-qwen3-4b-warm-v1/adapters.safetensors \
  --iters $ITERS --num-layers 8 --lr 5e-5 --steps-per-eval 500 > logs/agent/train_rft_v1.log 2>&1 || { echo "$(date) RFT TRAIN FAILED"; exit 1; }
echo "$(date) rft eval start"
python -m scripts.agent.rollout --tasks data/agent/tasks/v2.jsonl --split heldout --per-family 40 --k 2 --batch 4 \
  --adapter-path models/agent-qwen3-4b-rft-v1 --out data/agent/rollouts/heldout_rft_v1.jsonl > logs/agent/heldout_rft_v1.log 2>&1 || { echo "$(date) EVAL FAILED"; exit 1; }
python -m scripts.agent.compare data/agent/rollouts/heldout_base.jsonl data/agent/rollouts/heldout_rft_v1.jsonl --out data/agent/rollouts/compare_base_rft_v1.json > /dev/null
python -m scripts.agent.compare data/agent/rollouts/heldout_warm_v1.jsonl data/agent/rollouts/heldout_rft_v1.jsonl --out data/agent/rollouts/compare_warm_rft_v1.json > /dev/null
echo "$(date) CHAIN DONE"
