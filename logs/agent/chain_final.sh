#!/bin/zsh
# Stage 3: pick the better adapter on held-out, fuse, re-evaluate fused, serve and call it over the OpenAI API.
cd /Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT
source .venv/bin/activate
until grep -q "CHAIN DONE" logs/agent/chain_rft.out 2>/dev/null; do
  grep -qE "FAILED|not starting" logs/agent/chain_rft.out 2>/dev/null && { echo "$(date) rft chain failed; not starting"; exit 1; }
  sleep 30
done
BEST=$(python -c "
import json
c=json.load(open('data/agent/rollouts/compare_warm_rft_v1.json'))['families']['ALL']
print('models/agent-qwen3-4b-rft-v1' if c['new'] >= c['base'] else 'models/agent-qwen3-4b-warm-v1')")
echo "$(date) best adapter: $BEST"
echo "$BEST" > logs/agent/best_adapter.txt
python -m mlx_lm fuse --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path "$BEST" \
  --save-path models/agent-qwen3-4b-v1-fused > logs/agent/fuse_v1.log 2>&1 || { echo "$(date) FUSE FAILED"; exit 1; }
echo "$(date) fused; eval start"
python -m scripts.agent.rollout --tasks data/agent/tasks/v2.jsonl --split heldout --per-family 40 --k 2 --batch 4 \
  --model-path models/agent-qwen3-4b-v1-fused --out data/agent/rollouts/heldout_fused_v1.jsonl > logs/agent/heldout_fused_v1.log 2>&1 || { echo "$(date) EVAL FAILED"; exit 1; }
python -m scripts.agent.compare data/agent/rollouts/heldout_base.jsonl data/agent/rollouts/heldout_fused_v1.jsonl --out data/agent/rollouts/compare_base_fused_v1.json > /dev/null
echo "$(date) server check start"
python -m mlx_lm server --model models/agent-qwen3-4b-v1-fused --port 8088 > logs/agent/server_v1.log 2>&1 &
SERVER=$!
for i in {1..60}; do curl -s http://127.0.0.1:8088/v1/models > /dev/null && break; sleep 2; done
python - > logs/agent/server_tasks.txt <<'PY'
import json
seen, ids = set(), []
for line in open("data/agent/rollouts/heldout_fused_v1.jsonl"):
    r = json.loads(line)
    if r["family"] not in seen and r["sample"] == 0:
        seen.add(r["family"]); ids.append(r["task_id"])
print("\n".join(ids))
PY
: > logs/agent/server_check_v1.log
while read TID; do
  python -m scripts.agent.play --server http://127.0.0.1:8088/v1 --tasks data/agent/tasks/v2.jsonl --task-id "$TID" >> logs/agent/server_check_v1.log 2>&1
  echo "------" >> logs/agent/server_check_v1.log
done < logs/agent/server_tasks.txt
kill $SERVER
echo "$(date) CHAIN DONE"
