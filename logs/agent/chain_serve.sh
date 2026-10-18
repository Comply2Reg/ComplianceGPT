#!/bin/zsh
# Stage 5: serve the best adapter unfused (fusing into 4-bit cost 0.704 -> 0.638 on held-out)
# and call it over the OpenAI-compatible API: the playground on one task per family, plus a raw request.
cd /Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT
source .venv/bin/activate
until grep -q "CHAIN DONE" logs/agent/chain_v2.out 2>/dev/null; do
  grep -qE "FAILED|not starting" logs/agent/chain_v2.out 2>/dev/null && break
  sleep 30
done
BEST=$(python -c "
import json, os
p='data/agent/rollouts/compare_warm_mix_v2.json'
win = os.path.exists(p) and json.load(open(p))['families']['ALL']['new'] > json.load(open(p))['families']['ALL']['base']
print('models/agent-qwen3-4b-mix-v2' if win else 'models/agent-qwen3-4b-warm-v1')")
echo "$(date) serving $BEST"
echo "$BEST" > logs/agent/best_adapter_final.txt
python -m mlx_lm server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path "$BEST" --port 8088 > logs/agent/server_adapter.log 2>&1 &
SERVER=$!
for i in {1..90}; do curl -s http://127.0.0.1:8088/v1/models > /dev/null && break; sleep 2; done
: > logs/agent/server_check_adapter.log
while read TID; do
  python -m scripts.agent.play --server http://127.0.0.1:8088/v1 --tasks data/agent/tasks/v2.jsonl --task-id "$TID" >> logs/agent/server_check_adapter.log 2>&1
  echo "------" >> logs/agent/server_check_adapter.log
done < logs/agent/server_tasks.txt
python - > logs/agent/server_raw_adapter.json <<'PY'
import json, urllib.request
tools = json.load(open("scripts/agent/tools.json"))
body = {"model": "default_model", "temperature": 0, "max_tokens": 400, "tools": tools, "messages": [
  {"role": "user", "content": "Find how the CMA defines 'relevant merger situation'. Use the tools."}]}
req = urllib.request.Request("http://127.0.0.1:8088/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req, timeout=300)), indent=1))
PY
kill $SERVER
echo "$(date) CHAIN DONE"
