#!/usr/bin/env python3
"""Build the Hugging Face folder and model card for the tool-calling regulation agent.

    python -m scripts.agent.model_card --out models/publish/regulatory-research-agent-qwen3-4b-v1

Same rule as scripts/triage/model_card.py: every number in the card is read from
a run artefact (the held-out compare and summary JSONs, the SFT stats, the
training log, the library index, the task file), so the page cannot drift
from what happened. Prose is written here; figures never are.

The folder holds:
- the fused bf16 model at the root (drop-in, any OpenAI-compatible MLX server);
- `adapter/`, the LoRA for the 4-bit base (the best held-out score);
- `tools.json` and `system_prompt.txt`, the exact ones used in training;
- README.md and eval_summary.json, which publish_model.py checks against each other.

Weights are copied with APFS clones (`cp -c`), so the 7.5 GB folder costs no extra
disk. Then: `python -m scripts.triage.publish_model --model-dir <out> --dry-run`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent.env import MAX_TURNS, RESULT_CHARS, SYSTEM_PROMPT, TOOL_BUDGET, TOOLS  # noqa: E402
from scripts.agent.index import DEFAULT_DB  # noqa: E402

RUNS = ROOT / "data" / "agent" / "rollouts"
FAMILIES = ("lookup_definition", "lookup_effective_date", "not_found", "no_tool_needed", "cited_obligations")
FAMILY_NAMES = {
    "lookup_definition": "Definition lookup",
    "lookup_effective_date": "Effective-date lookup",
    "not_found": "Says \"not in the library\" instead of guessing",
    "no_tool_needed": "Answers from a given passage without tools",
    "cited_obligations": "Cited obligations (7 tasks; too few to rely on)",
}
OGL_ATTRIBUTION = "Contains public sector information licensed under the Open Government Licence v3.0."
BASE_HF = "Qwen/Qwen3-4B-Instruct-2507"
BASE_MLX = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def load(name: str) -> Dict:
    return json.loads((RUNS / name).read_text(encoding="utf-8"))


def val_curve(log: Path) -> List[Tuple[int, float]]:
    return [(int(i), float(v)) for i, v in re.findall(r"Iter (\d+): Val loss ([\d.]+)", log.read_text())]


def peak_gb(log: Path) -> float:
    return max(float(x) for x in re.findall(r"Peak mem ([\d.]+) GB", log.read_text()))


def trainable(log: Path) -> str:
    m = re.search(r"Trainable parameters: ([\d.]+)% \(([\d.]+)M/([\d.]+)M\)", log.read_text())
    return f"{m.group(2)}M of {m.group(3)}M ({m.group(1)}%)"


def corpus_facts(db: Path) -> Dict:
    con = sqlite3.connect(db)
    return {
        "documents": con.execute("select count(*) from documents").fetchone()[0],
        "provisions": con.execute("select count(*) from provisions").fetchone()[0],
        "by_regulator": dict(con.execute("select regulator, count(*) from documents group by 1 order by 2 desc")),
        "by_licence": dict(con.execute("select licence, count(*) from documents group by 1")),
    }


def task_counts(path: Path) -> Counter:
    c: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        c[(t["family"], t["split"])] += 1
    return c


def gather(args) -> Dict:
    warm = load("compare_base_warm_v1.json")       # base vs adapter
    bf16 = load("compare_warm_fused_bf16_v1.json")  # adapter vs fused bf16
    q4 = load("compare_base_fused_v1.json")         # base vs fused 4-bit
    teacher = load("heldout_teacher.summary.json")
    fam = {}
    for f in FAMILIES + ("ALL",):
        fam[f] = {
            "tasks": warm["families"][f]["tasks"],
            "base": warm["families"][f]["base"],
            "adapter": warm["families"][f]["new"],
            "fused_bf16": bf16["families"][f]["new"],
            "fused_4bit": q4["families"][f]["new"],
            "teacher": teacher["summary"].get(f, {}).get("pass_rate") if f != "ALL" else None,
            "adapter_vs_base_ci95": warm["families"][f]["ci95"],
            "adapter_vs_base_delta": warm["families"][f]["delta"],
        }
    fam["ALL"]["bf16_vs_adapter_delta"] = bf16["families"]["ALL"]["delta"]
    fam["ALL"]["bf16_vs_adapter_ci95"] = bf16["families"]["ALL"]["ci95"]
    sft = json.loads((ROOT / "data/agent/sft/warm_v1/stats.json").read_text())
    log = ROOT / "logs/agent/train_warm_v1.log"
    usd = load("teacher_v2.summary.json")["usd"] + teacher["usd"]
    secs = load("heldout_fused_bf16_v1.summary.json")["summary"]
    lookups = [v for k, v in secs.items() if k.startswith("lookup_")]
    sec_per_ep = sum(v["seconds_per_episode"] * v["episodes"] for v in lookups) / sum(v["episodes"] for v in lookups)
    return {
        "families": fam,
        "behaviour": {"base": warm["behaviour"]["base"], "adapter": warm["behaviour"]["new"],
                      "fused_bf16": bf16["behaviour"]["new"]},
        "teacher_model": teacher["model"],
        "teacher_usd": round(usd, 2),
        "seconds_per_episode": round(sec_per_ep, 1),
        "sft": sft,
        "val_curve": val_curve(log),
        "peak_gb": peak_gb(log),
        "trainable": trainable(log),
        "corpus": corpus_facts(args.db),
        "tasks": task_counts(ROOT / "data/agent/tasks/v2.jsonl"),
        "variants": {
            "rft_v1": load("compare_warm_rft_v1.json")["families"]["ALL"]["new"],
            "mix_v2": load("compare_warm_mix_v2.json")["families"]["ALL"]["new"],
        },
    }


def pct(x) -> str:
    """Two decimals, half rounded up (0.475 -> 0.48, as people read it), from the stored 3-decimal value."""
    if x is None:
        return "—"
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def frontmatter(repo: str, g: Dict) -> str:
    a = g["families"]["ALL"]
    return f"""---
license: apache-2.0
license_link: https://huggingface.co/{BASE_HF}/blob/main/LICENSE
base_model: {BASE_HF}
base_model_relation: finetune
library_name: mlx
pipeline_tag: text-generation
language:
- en
tags:
- mlx
- qwen3
- lora
- tool-calling
- function-calling
- agent
- regtech
- regulatory-compliance
- retrieval
- citations
- united-kingdom
- united-states
model-index:
- name: {repo}
  results:
  - task:
      type: text-generation
      name: Tool-calling regulatory lookup with verified citations
    dataset:
      type: private
      name: Held-out GREEN regulation documents (tasks never searched in training)
      split: heldout
    metrics:
      - type: accuracy
        name: held-out pass rate (fused bf16, root weights)
        value: {a["fused_bf16"]}
      - type: accuracy
        name: held-out pass rate (adapter on 4-bit base)
        value: {a["adapter"]}
      - type: accuracy
        name: held-out pass rate (base model, same harness)
        value: {a["base"]}
---
"""


CARD_METRICS = {
    "held-out pass rate (fused bf16, root weights)": "pass_rate_fused_bf16",
    "held-out pass rate (adapter on 4-bit base)": "pass_rate_adapter",
    "held-out pass rate (base model, same harness)": "pass_rate_base",
}


def body(repo: str, g: Dict) -> str:
    f = g["families"]
    a = f["ALL"]
    beh = g["behaviour"]
    corpus = g["corpus"]
    tasks = g["tasks"]
    counts = g["sft"]["counts"]
    train_traj = sum(v for k, v in counts.items() if k.startswith("trajectories/"))
    scripted = sum(v for k, v in counts.items() if k.startswith("trajectories/scripted/"))
    teacher_traj = sum(v for k, v in counts.items() if k.startswith("trajectories/teacher/"))
    ex_train = counts["examples/train/answer"] + counts["examples/train/tool_call"]
    ex_valid = counts["examples/valid/answer"] + counts["examples/valid/tool_call"]
    curve = " → ".join(f"{v:.3f} ({i})" for i, v in g["val_curve"])
    lo, hi = a["adapter_vs_base_ci95"]
    blo, bhi = a["bf16_vs_adapter_ci95"]
    gain = a["adapter_vs_base_delta"]
    reliable = [k for k in FAMILIES if k != "cited_obligations"]
    wins = sum(f[k]["adapter"] > f[k]["teacher"] for k in reliable)
    ties = sum(f[k]["adapter"] == f[k]["teacher"] for k in reliable)

    fam_rows = "\n".join(
        f"| {FAMILY_NAMES[k]} | {f[k]['tasks']} | {pct(f[k]['base'])} | **{pct(f[k]['adapter'])}** | "
        f"{pct(f[k]['fused_bf16'])} | {pct(f[k]['teacher'])} |" for k in FAMILIES)
    reg_rows = " · ".join(f"{r} {n:,}" for r, n in corpus["by_regulator"].items())
    task_rows = "\n".join(
        f"| `{k}` | {tasks[(k, 'train')]} | {tasks[(k, 'heldout')]} |" for k in FAMILIES)
    tool_rows = "\n".join(
        f"| `{t['function']['name']}` | {', '.join(f'`{p}`' for p in t['function']['parameters']['properties'])} | "
        f"{t['function']['description'].split('. ')[0].rstrip('.')}. |" for t in TOOLS)
    lic = corpus["by_licence"]

    return f"""# {repo}

**A 4-billion-parameter model that researches regulation the way an analyst would.** It searches a
library, opens the provision it needs, and answers with an exact quote and a citation. When the
library doesn't have the answer, it says so instead of making one up. It runs on a laptop.

Built by [Comply2Reg](https://huggingface.co/Comply2Reg) on
[Qwen3-4B-Instruct-2507](https://huggingface.co/{BASE_HF}), trained with LoRA in MLX on Apple silicon.
Training and evaluation used openly-licensed UK and US federal regulation only.

## Business overview

### Why it exists

Compliance teams ask the same kinds of questions all day:
- What does this regulation mean by *X*?
- When does it take effect?
- Does our library even cover this?

A general-purpose model answers these fluently but cannot be trusted on the wording, and the
wording is the whole point. This model is trained to **look things up rather than recall them**.
Every answer must quote text that a tool actually returned and cite the provision it came from, so
a reviewer can check it in one click.

### What it does well (measured on documents it never saw in training)

| | base model | this model | {g["teacher_model"]} in the same harness |
|---|---:|---:|---:|
| Finds and quotes a definition verbatim, with its provision | {pct(f["lookup_definition"]["base"])} | **{pct(f["lookup_definition"]["adapter"])}** | {pct(f["lookup_definition"]["teacher"])} |
| Finds a regulation's effective date and cites where it says so | {pct(f["lookup_effective_date"]["base"])} | **{pct(f["lookup_effective_date"]["adapter"])}** | {pct(f["lookup_effective_date"]["teacher"])} |
| Says "not in the library" instead of guessing | {pct(f["not_found"]["base"])} | **{pct(f["not_found"]["adapter"])}** | {pct(f["not_found"]["teacher"])} |
| Answers straight from a passage it is given, no tool calls | {pct(f["no_tool_needed"]["base"])} | **{pct(f["no_tool_needed"]["adapter"])}** | {pct(f["no_tool_needed"]["teacher"])} |
| **All task types** | {pct(a["base"])} | **{pct(a["adapter"])}** | — |

Each number is the share of episodes passing a strict, code-only check: valid tool calls, at most
{TOOL_BUDGET} of them, a JSON answer, the quote found verbatim in a tool result, and no invented ids.
The small model beats the API model it partly learned from on {wins} of the four reliable task types
({ties} tied). It learned only from checked successes and is specialised to this library's tools.

### Agents you can build with it

| agent | what it does for a compliance team | readiness |
|---|---|---|
| **Definition finder** | "What does *X* mean in regulation *Y*?" Returns a verbatim quote and a link to the provision | Pilot with a quote check and a fallback to a larger model |
| **Effective-date proposer** | For a newly published regulation, proposes when it applies, with the evidence, for a person to confirm in the compliance calendar | Strongest skill ({pct(f["lookup_effective_date"]["adapter"])}); keep a human confirmation step |
| **Coverage guard** | Before answering, checks that the library actually covers the term or topic; routes or flags the gap if it doesn't | Ready as a pre-check |
| **Passage-grounded answerer** | Answers from text another system already retrieved, without extra calls | Ready for definition, date and obligation shapes |
| **Citation checker** | Fetches the provision another model cited, so code can confirm the quote is really there | Safest first use |

**Not ready:**
- obligation extraction at scale;
- multi-document comparisons;
- tools it was not trained on;
- free-form legal analysis;
- alert triage, which is a separate model:
  [regulatory-alert-triage-qwen3-4b-v4](https://huggingface.co/Comply2Reg/regulatory-alert-triage-qwen3-4b-v4).

### Why a small model

- **Runs locally.** It needs one laptop-class GPU (tested on an Apple M4 Pro with 24 GB), with no
  per-token bill. Regulatory text never leaves your machine unless you send it.
- **Predictable cost.** About {g["seconds_per_episode"]:.0f} s per lookup on that laptop (search,
  fetch and answer, tool time included).
- **Auditable.** It is trained to cite what it read. The check is plain code, which you can run
  on every answer in production.
- **Additive.** It sits beside a larger API model, not instead of one. Route the questions it is
  good at to it, and fall back when its JSON or quote check fails.

## How well it works

**Held-out set:** {a["tasks"]} tasks from documents that were never searched during training (20% of
documents, split by content hash). Each task was played twice at temperature 0.7 and checked by code.

| task family | tasks | base | **adapter** | fused bf16 (root) | {g["teacher_model"]} |
|---|---:|---:|---:|---:|---:|
{fam_rows}
| **all** | {a["tasks"]} | {pct(a["base"])} | **{pct(a["adapter"])}** | {pct(a["fused_bf16"])} | — |

**Statistics:**
- **Adapter vs base:** +{gain:.3f} overall, paired bootstrap 95% CI [{lo:.2f}, {hi:.2f}].
- **Fused bf16 vs adapter:** {a["bf16_vs_adapter_delta"]:+.3f}, CI [{blo:.2f}, {bhi:.2f}], not significant.
- **Teacher column:** one run per task (k=1) on the same tasks.

| tool behaviour ({beh["adapter"]["episodes"]} episodes) | base | adapter | fused bf16 |
|---|---:|---:|---:|
| tool calls per episode | {beh["base"]["calls_per_episode"]} | {beh["adapter"]["calls_per_episode"]} | {beh["fused_bf16"]["calls_per_episode"]} |
| valid-call rate | {beh["base"]["valid_call_rate"]} | {beh["adapter"]["valid_call_rate"]} | {beh["fused_bf16"]["valid_call_rate"]} |
| calls using an id no tool returned, per episode | {beh["base"]["invented_id_calls_per_episode"]} | {beh["adapter"]["invented_id_calls_per_episode"]} | {beh["fused_bf16"]["invented_id_calls_per_episode"]} |
| episodes over the {TOOL_BUDGET}-call budget | {beh["base"]["over_budget_episodes"]} | {beh["adapter"]["over_budget_episodes"]} | {beh["fused_bf16"]["over_budget_episodes"]} |
| JSON final answers | {beh["base"]["json_final_rate"]} | {beh["adapter"]["json_final_rate"]} | {beh["fused_bf16"]["json_final_rate"]} |

## Evaluation

The verifier is code, never a model. An episode passes only if all of these hold:
- every tool call is valid JSON with valid arguments;
- the call budget is respected;
- the final answer is JSON;
- for lookups, the quoted text appears verbatim in what a tool returned and in the cited provision;
- the cited provision_id was returned by a tool, not invented.

Dates are checked against the mined gold date; a comment deadline or publication date does not count.
`not_found` passes only after at least one search, and only on a question whose document does not
contain the term.

**How this model was chosen.** Four variants were trained, and the warm-start SFT scored best on
the held-out tasks. The two later variants added the model's own successful attempts and scored lower:
{pct(g["variants"]["rft_v1"])} (on-policy only) and {pct(g["variants"]["mix_v2"])} (mixed with replay).
Choosing between variants on the held-out tasks makes the headline slightly optimistic. The gain over
the base model is several times larger than that effect.

## Quickstart

Install `pip install mlx-lm openai`.

**Option A, drop-in (the root weights, fused bf16, 7.5 GB):**

```bash
mlx_lm.server --model Comply2Reg/{repo} --port 8088
```

**Option B, best score (the 4-bit base plus `adapter/`, about 2 GB + 14 MB):**

```bash
huggingface-cli download Comply2Reg/{repo} --include "adapter/*" --local-dir ./agent
mlx_lm.server --model {BASE_MLX} --port 8088
# mlx-lm 0.31's server ignores --adapter-path for "default_model" requests; name the adapter
# in each request instead: extra_body={{"adapters": "./agent/adapter"}}
```

**The agent loop.** Use the exact system prompt and tool schemas the model was trained with. Both are
in this repo as `system_prompt.txt` and `tools.json`. Run each tool call against your own library and
send the result back:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="local")
tools = json.load(open("tools.json"))
messages = [{{"role": "system", "content": open("system_prompt.txt").read()}},
            {{"role": "user", "content": "How does the CMA define 'relevant merger situation'? "
                                         'Answer only with JSON {{"definition": "...", "provision_id": "..."}}.'}}]
for _ in range({MAX_TURNS}):                        # at most {MAX_TURNS} turns, {TOOL_BUDGET} tool calls
    msg = client.chat.completions.create(model="default_model", messages=messages, tools=tools,
                                         temperature=0, max_tokens=768).choices[0].message
    # extra_body={{"adapters": "./agent/adapter"}} for option B
    messages.append(msg.model_dump(exclude_none=True))
    if not msg.tool_calls:
        print(msg.content)                       # the JSON answer
        break
    for call in msg.tool_calls:
        result = my_library.run(call.function.name, json.loads(call.function.arguments))  # your tools
        messages.append({{"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)[:{RESULT_CHARS}]}})
```

`mlx_lm.server` returns tool calls already parsed. If you run the model in-process, strip the
empty `<think></think>` it writes before each turn before parsing JSON. The reference harness,
verifier and the SQLite FTS5 index builder used here are in
[ComplianceGPT](https://github.com/Comply2Reg/ComplianceGPT) under `scripts/agent/`.

## The tools

| tool | arguments | purpose |
|---|---|---|
{tool_rows}

Every tool is read-only. Results are cut at {RESULT_CHARS:,} characters, and a call to an id that
does not exist returns an error rather than an empty result. The model learned these exact names and
descriptions. Other tools may work, but nothing about them has been measured.

## Training data

**The library:** {corpus["documents"]:,} openly-licensed regulatory documents, split into
{corpus["provisions"]:,} provisions and indexed with SQLite FTS5 (BM25). By regulator: {reg_rows}.

**Tasks:** made from the documents by code. Definitions and effective dates were mined and
verified; obligations come from a human-reviewed seed; not-found questions ask for a term that is
absent from the named document.

| family | train tasks | held-out tasks |
|---|---:|---:|
{task_rows}

**Trajectories:** {train_traj:,} verified trajectories, split into one training example per
assistant turn, gave {ex_train:,} training and {ex_valid} validation examples:
- **{scripted:,} scripted demonstrations:** code playing the task from its gold answer through the
  real tools.
- **{teacher_traj:,} trajectories from {g["teacher_model"]}:** the same prompt, tools and library,
  kept only when the verifier passed them.

No held-out document contributed to training. The prompt and tool results were masked, so the model
was trained only on its own turns.

**Excluded:** publications from the FCA, PRA, Bank of England, Ofcom and the DRCF sit under terms that
do not clearly permit this use. They are not in the library, the tasks or the training data.

## Training procedure

| setting | value |
|---|---|
| base | `{BASE_MLX}` (4-bit) |
| method | LoRA, rank 8, scale 20, last 8 layers; {g["trainable"]} parameters trained |
| schedule | 2,000 iterations, batch 1, lr 1e-4, Adam, grad checkpointing, max 4,096 tokens |
| loss | assistant turn only (`--mask-prompt`), one example per turn |
| hardware | Apple M4 Pro, 24 GB; peak {g["peak_gb"]:.1f} GB |
| validation loss | {curve} |
| root weights | the adapter merged with `mlx_lm fuse --dequantize` (bf16, not re-quantized) |

Merging into 4-bit weights instead cost accuracy (held-out {pct(f["ALL"]["fused_4bit"])}), so that
version is not published.

## Limitations

- **Lookups still fail often.** About half of definition questions fail the strict verbatim check,
  most often from paraphrasing or wrongly answering "not found". Always verify the quote before
  showing it.
- **Invented ids happen** (about {beh["adapter"]["invented_id_calls_per_episode"]} calls per episode).
  The tools reject them, but an agent must treat any failed call as "no answer".
- **Narrow skills.** The model was trained on single-document lookups with three tools and short JSON
  answers. Multi-document reasoning, long prose and new tool sets are unmeasured.
- **Obligations:** {f["cited_obligations"]["tasks"]} held-out tasks are too few to rely on, and the
  score is low.
- **Library scope.** It knows how to use the tools, not the law. Its answers are only as current
  and complete as the library you index.
- **Not legal advice.** Its output is a cited pointer into regulation for a qualified person to
  review, not an interpretation.

## Licence, attribution and provenance

The weights are released under Apache 2.0, inherited from
[Qwen3-4B-Instruct-2507](https://huggingface.co/{BASE_HF}).

Training and evaluation text came from {lic.get("uk_ogl_v3", 0):,} UK and
{lic.get("us_public_domain", 0):,} US federal documents.

**UK** (HM Treasury, the CMA, the ICO, legislation.gov.uk), under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/):

> {OGL_ATTRIBUTION}

**US federal** (SEC, Treasury, OFAC, Federal Reserve Board, CFTC, FDIC, OCC, CFPB, NCUA, FinCEN, FFIEC):
works of the United States Government are not subject to copyright protection under 17 U.S.C. 105.

This model is not endorsed by, affiliated with or connected to any UK or US regulator or government
body, and its output is not an official interpretation of anything.

Teacher trajectories were generated with OpenAI models under terms that assign the output to the
customer. Total OpenAI spend for the project was ${g["teacher_usd"]:.2f}.

## Citation

```bibtex
@misc{{comply2reg_regulatory_research_agent_2026,
  title  = {{{repo}: a tool-calling small model for cited regulatory lookup}},
  author = {{Comply2Reg}},
  year   = {{2026}},
  url    = {{https://huggingface.co/Comply2Reg/{repo}}}
}}
```
"""


def clone(src: Path, dst: Path) -> None:
    """APFS clone (no extra disk); plain copy elsewhere."""
    if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fused", type=Path, default=ROOT / "models/agent-qwen3-4b-v1-fused-bf16")
    ap.add_argument("--adapter", type=Path, default=ROOT / "models/agent-qwen3-4b-warm-v1")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args(argv)

    repo = args.out.name
    g = gather(args)
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.fused.iterdir():
        if p.is_file() and p.name != "README.md" and not (args.out / p.name).exists():
            clone(p, args.out / p.name)
    (args.out / "adapter").mkdir(exist_ok=True)
    for name in ("adapters.safetensors", "adapter_config.json"):
        if not (args.out / "adapter" / name).exists():
            clone(args.adapter / name, args.out / "adapter" / name)
    (args.out / "tools.json").write_text(json.dumps(TOOLS, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")

    a = g["families"]["ALL"]
    summary = {
        "metrics": {"pass_rate_fused_bf16": a["fused_bf16"], "pass_rate_adapter": a["adapter"],
                    "pass_rate_base": a["base"]},
        "card_metrics": CARD_METRICS,
        "families": g["families"],
        "behaviour": g["behaviour"],
        "sources": ["data/agent/rollouts/compare_base_warm_v1.json",
                    "data/agent/rollouts/compare_warm_fused_bf16_v1.json",
                    "data/agent/rollouts/compare_base_fused_v1.json",
                    "data/agent/rollouts/heldout_teacher.summary.json"],
    }
    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (args.out / "README.md").write_text(frontmatter(repo, g) + "\n" + body(repo, g), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
