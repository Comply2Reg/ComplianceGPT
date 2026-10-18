# What the tool-calling SLM can do, and which agents to build with it

Model: **`agent-qwen3-4b`**, a LoRA adapter on `Qwen3-4B-Instruct-2507` (MLX 4-bit). It was
trained on 2026-10-18 to call three read-only tools over the GREEN regulation library and answer
in JSON with verbatim, cited quotes. Training, data and every number: `docs/agent.md`.

This document separates three things:
- what was **measured** on held-out documents;
- what should **work but is untested**;
- what the model **cannot do yet**.

The SLM is an extra provider beside OpenAI, not a replacement. In the app it is chosen per
feature by configuration, and OpenAI stays the default.

## 1. Measured capabilities

Held-out documents were never searched during training. Each task was run twice (k=2,
temperature 0.7) and checked by code (`scripts/agent/verify.py`).

| skill | what the agent does | base model | **agent-qwen3-4b** | gpt-5.4-mini, same harness |
|---|---|---:|---:|---:|
| **Definition lookup** | finds the document, fetches the provision, quotes the definition verbatim with its provision_id | 0.08 | **0.48** | 0.23 |
| **Effective-date lookup** | finds when a document takes effect (not the comment deadline or publication date) and cites where it says so | 0.25 | **0.80** | 0.75 |
| **"Not in the library"** | searches, and answers `{"status": "not_found"}` instead of making something up | 0.19 | **0.64** | 0.35 |
| **Knowing when not to call tools** | answers directly when the passage is already in the prompt | 0.88 | **0.98** | 0.98 |
| **Cited obligations** | quotes every "must" sentence in a passage, with provision ids | 0.14 | 0.29 | 0.14 |
| **All families** | | 0.34 | **0.70** | — |

**How it uses tools** (334 held-out episodes):
- **Valid tool calls:** 95%, up from 76% for the base model.
- **Invented ids:** 0.06 calls per episode use an id that no tool returned, down from 0.24.
- **Budget:** 2 of 334 episodes went past the 3-call limit.
- **Output:** 99% of final answers are valid JSON.
- **Calls per episode:** about 1.5 on average (no calls when the passage is given, 2 for a typical lookup).
- **Speed:** 10-14 s per episode on the M4 Pro (24 GB), batch of 4, including tool time.

**Caveats:**
- **Obligations** has only 7 held-out tasks; its score is too uncertain to rely on.
- **Selection:** picking between adapters used these held-out tasks, so the 0.70 is slightly optimistic. The gain over the base model (+0.37, CI 0.29-0.44) is not.
- **Packaging matters:**
  - adapter on the 4-bit base: 0.70;
  - single-file bf16 merge: 0.66 (difference not significant);
  - single-file 4-bit merge: 0.64 (avoid).
- **Four training variants were tried; the first warm start (warm_v1) won.** Two later rounds that
  added the model's own successful attempts both scored lower (0.57, 0.60), because the model learned
  to stop searching too early. Details are in `docs/agent.md`.

## 2. Agents that can be built now

These agents use only the skills measured above, with the same three tools.

| agent | where it fits in the product | skills it relies on | readiness |
|---|---|---|---|
| **Definition finder**: "what does *X* mean in regulation *Y*?" with a verbatim quote and a link to the provision | AskLia (grounded answer + citation); ReguLens document view (explain a term in the open document, `document_id` already known) | definition lookup, not-found | **Ready to pilot** behind a setting. About half of answers pass the strict verbatim check, so show the quote with its citation and fall back to OpenAI when the JSON or the quote check fails |
| **Effective-date extractor**: for a new or changed regulation, when does it apply? | RegPulse alerts and Calendar + Tasks: propose a deadline for a newly indexed document, with the provision as evidence | effective-date lookup, not-found (`date: null`) | **Strongest skill (0.80).** Use it to *propose* calendar entries a person confirms. Never create deadlines automatically |
| **"Is this covered?" guard** | Before AskLia or CGA answers from the library: check that the term or topic exists in the indexed GREEN corpus | not-found, search | **Ready as a pre-check.** A `not_found` result can route the question to OpenAI or flag the gap |
| **Passage-grounded answerer**: question plus passage in, cited JSON out, no tool calls | Any feature that has already retrieved the text (CGA, Lifecycle sections, AskLia after its own retrieval) | no-tool-needed (0.98) | **Ready**, if the answer is one of the trained shapes (a definition, a date, obligation spans) |
| **Citation checker** | After any model (OpenAI, Claude, the SLM) answers: fetch the cited provision and check that the quote is really there | `get_provision` plus the code verifiers | **Ready.** The checking is done by code; the model only needs to fetch. This is the safest first use |

**What every agent above should do:**
- **Keep the 3-call budget and the 5-turn limit** from `scripts/agent/env.py`. The model was trained under them.
- **Use the same system prompt and `scripts/agent/tools.json`.** The model has learned those exact tool names and descriptions.
- **Check before trusting.** Parse the JSON and confirm that quotes appear in what the tools returned (the `verify.py` checks). If that fails, fall back to OpenAI where the feature allows it.

## 3. Plausible but untested

These would need an eval set and probably training data before being used.

| idea | why it is not ready |
|---|---|
| **Obligation extraction for CGA v2 or the Lifecycle extractor** | 0.29 on only 7 tasks. Lifecycle wants a different schema (`requirement_code`, `risk_level`, …) the model has never seen. The CGA v2 extractor slot (`DeonticExtractor`) is the right home once there is training data |
| **New tools** (graph queries, the app's Milvus search, calendar writes) | Trained on three tools only. It may call a new schema correctly, but nothing was measured. Any tool that *writes* needs human confirmation in front of it |
| **Multi-document questions** ("compare the FCA and PRA definitions") | Tasks were single-document. Three calls rarely cover two documents |
| **Long reasoning or free-text answers** | Trained to give short JSON answers. Prose quality is unmeasured |
| **Alert triage (A1-A14, P1-P3)** | That is a different model: `regulatory-alert-triage-qwen3-4b-v4` |
| **AMBER or RED text, customer documents** | Trained and evaluated on GREEN text only. Customer data is out of scope until rights are confirmed |

## 4. How to run it

**Serve.** The server speaks the OpenAI API, and tool calls come back parsed as `tool_calls`.
There are two options:

```bash
cd 15-SLM/ComplianceGPT && source .venv/bin/activate
python -m scripts.agent.index build          # once: data/agent/corpus.sqlite, ~10 s

# A. best accuracy (0.70): 4-bit base + adapter. Clients must send "adapters" in each request (see below)
python -m mlx_lm server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8088

# B. drop-in (0.66): one 7.5 GB folder, no request changes; use this for c2r_cga_api
python -m mlx_lm server --model models/agent-qwen3-4b-v1-fused-bf16 --port 8088
```

For option A, `--adapter-path` on the command line is **not enough**. In mlx_lm 0.31 the server
ignores it for `default_model` requests and silently serves the bare base model. Name the adapter
in every request body instead: `extra_body={"adapters": "models/agent-qwen3-4b-warm-v1"}`, or
`play --server-adapter models/agent-qwen3-4b-warm-v1`. With that, the server check passed 4/5
(one task per family).

**Try it** in the playground. It prints every step, runs the tools locally, and scores the episode when given a task id:

```bash
python -m scripts.agent.play --server http://127.0.0.1:8088/v1 --server-adapter models/agent-qwen3-4b-warm-v1 \
    "How does the CMA define 'relevant merger situation'?"      # drop --server-adapter for option B
```

**Call it from any OpenAI client.** Pass `tools=json.load(open("scripts/agent/tools.json"))` and
`model="default_model"`. Then run each returned `tool_calls` entry against `scripts.agent.index.Corpus` and
send the result back as a `tool` message. `scripts/agent/play.py` is the reference loop, about 60 lines.
`mlx_lm.server` moves the model's empty `<think></think>` out of `content`. If you run the model in-process
instead, strip a leading `<think>…</think>` before parsing JSON.

**In `c2r_cga_api`** (branch `feat/lifecycle-extractor-provider`, local, not pushed):

```bash
SLM_BASE_URL=http://127.0.0.1:8088/v1      # serve option B (fused bf16) for the app
SLM_MODEL=default_model
LLM_FEATURE_PROVIDERS={"<feature key>": "slm,openai"}
LLM_FALLBACK_FEATURES=<feature key>          # optional: fall back to OpenAI on error
```

- **With these unset, every feature stays on OpenAI exactly as before.**
- **What the app can do today:** route a feature's chat call to the SLM. It does not yet run the tool loop or the three corpus tools.
- **What building an app-side agent needs:**
  - the loop from `play.py`;
  - tools backed by the app's own indexed corpus;
  - an eval of the model against those tools.

## 5. Limits to keep in mind

- **Context:** examples were at most 4,096 tokens, and the tools block alone is about 600. Tool results are cut at 6,000 characters.
- **Hardware:** one Mac (M4 Pro, 24 GB) serves one request at a time at about 10 s per lookup. Production hosting is not decided.
- **Library:** the GREEN snapshot indexed on 2026-10-18 (7,734 documents, 245,374 provisions). A new crawl means rebuilding the index; the model needs no retraining for that.
- **Ids:** it still occasionally invents a document id (about 6% of episodes). The tools reject it with an error, so it cannot pass a check, but the agent should treat any failed call as "no answer".

## 6. What exists on disk

All of this is local. Nothing is pushed or published.

| item | path |
|---|---|
| adapter (the model) | `models/agent-qwen3-4b-warm-v1/` (14 MB) |
| drop-in single-file model | `models/agent-qwen3-4b-v1-fused-bf16/` (7.5 GB) |
| tools, harness, verifier, playground | `scripts/agent/` (`tools.json`, `env.py`, `verify.py`, `play.py`) |
| library index | `data/agent/corpus.sqlite` (rebuild with `python -m scripts.agent.index build`) |
| held-out results | `data/agent/rollouts/compare_*.json` |
| how it was trained, every number | `docs/agent.md` |

**Cost:** $4.89 of OpenAI for teacher trajectories and the teacher reference. About 14 h of
Mac GPU time across four training variants and their evaluations.
