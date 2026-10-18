# Agent playground: a tool-calling SLM over GREEN regulation

Started 2026-10-18. Track B of `c2r_cga_api/docs/PLAN.md`. The goal is a small model that,
given a compliance question, decides whether to look something up, makes at most
three read-only tool calls with valid arguments, and answers with verbatim quotes
cited to provisions it actually fetched. Every reward is checked by code, never by
a model: the RegNano rule (`docs/regnano.md`).

Base model: `Qwen3-4B-Instruct-2507` (MLX 4-bit). Its own chat template has the tools
block and `<tool_call>` / `<tool_response>` turns, `mlx_lm.server` parses those into
OpenAI `tool_calls`, and it trains at sequence lengths Qwen3.5-4B cannot on this Mac.

## Pieces

| file | what |
|---|---|
| `scripts/agent/index.py` | GREEN corpus → provisions → SQLite FTS5 (BM25) at `data/agent/corpus.sqlite` (gitignored) |
| `scripts/agent/tools.json` | the three tool schemas: one file for training, eval, serving and, later, the app |
| `scripts/agent/env.py` | episode harness: system prompt, tool-call parsing, argument validation, budget |
| `scripts/agent/verify.py` | binary episode verifier |
| `scripts/agent/tasks.py` | RegNano tasks → tool tasks, split by document |
| `scripts/agent/rollout.py` | batched episodes with a local MLX model: probe, calibration, held-out eval |
| `scripts/agent/play.py` | the playground: one question, every step printed; local model or `mlx_lm.server` |

```bash
python -m scripts.agent.index build                                   # ~10 s, 7,734 docs, 245,374 provisions, 650 MB
python -m scripts.agent.tasks --out data/agent/tasks/v1.jsonl         # 1,520 tasks
python -m scripts.agent.play "How does the CMA define 'relevant merger situation'?"
python -m scripts.agent.play --tasks data/agent/tasks/v1.jsonl --task-id <id>          # scored
python -m scripts.agent.rollout --tasks data/agent/tasks/v1.jsonl --split train \
    --per-family 4 --k 1 --out data/agent/rollouts/probe20_base.jsonl
```

## The library

- **Only GREEN text**, through `scripts.regnano.sources.iter_docs`. There is no AMBER/RED override.
- **Provisions** are runs of whole lines up to 1,500 characters. Mined definitions run to
  the end of their line, so a line is never split unless it alone is over the limit.
  - Ids are `{source_id}#{n}` in document order.
  - Each provision's section is the nearest heading line at or before its start.
- **Search** is BM25 over title (weight 4), section (2) and text (1). Takes 40-200 ms.
  - Queries are OR-joined after dropping stopwords, so the number of matching words drives the ranking.
  - The table uses external content, so snippets come from the stored text.

## Tools and rules

| tool | arguments | returns |
|---|---|---|
| `search_regulation` | `query`, optional `regulator`, `document_id`, `limit` ≤ 5 | provision_id, document_id, title, regulator, date, section, snippet |
| `get_provision` | `provision_id` | the full provision plus its document's title and date |
| `list_sections` | `document_id` | first 60 provisions with their headings, plus the total |

- **Budget:** at most 3 calls per episode. A 4th call is refused with a tool message telling the model to answer, and the episode fails.
- **Bad calls:** an unknown tool, bad arguments or a missing id return an error as the tool result. The episode continues but fails.
- **Results** are cut at 6,000 characters, and the cut is marked.
- **Episode length:** at most 5 model turns.

## Task families (tasks v1, 1,520 tasks)

| family | train | held out | gold from | passes when |
|---|---:|---:|---|---|
| `lookup_definition` | 243 | 77 | RegNano `defined_term` | RegNano verifier, quote in a tool result and in the cited provision |
| `lookup_effective_date` | 619 | 140 | RegNano `effective_date` | RegNano verifier; cited provision from that document (or a same-title, same-day copy) |
| `cited_obligations` | 17 | 7 | human-reviewed seed (HMT) | span F1 ≥ 0.8 and modality; spans in tool results and in the cited provisions |
| `no_tool_needed` | 179 | 40 | either RegNano family, passage given | no tool call, and the RegNano verifier passes |
| `not_found` | 156 | 42 | a term asked of a same-regulator document whose text never contains it | at least one call, then `{"status": "not_found"}` |

Every family also requires valid tool JSON, no budget violation, a JSON final answer, and
no invented provision_id.

**Split:** 20% of documents are held out by `sha256(doc_hash)`, so held-out scores are on
documents never searched in training.

## Checks done

- `tests/test_agent_env.py`, 13 tests on a two-document fake corpus.
- **Template check:** with the Qwen3-4B-Instruct-2507 template and the tools block (about 600
  tokens), the generation prompt for every assistant turn is an exact token prefix of the full
  rendered trajectory. mlx-lm's `ChatDataset` (which masks everything before the last message,
  passing `tools`) therefore trains exactly one assistant turn per example, and never the tool
  results.

## Base-model probes (2026-10-18, Qwen3-4B-Instruct-2507 4-bit, temp 0.7, train split, k=1)

| probe | tasks | lookup_definition | lookup_effective_date | cited_obligations | not_found | no_tool_needed | s/episode |
|---|---|---:|---:|---:|---:|---:|---:|
| `probe20_base` | v1, 4/family | 0/4 | 0/4 | 0/4 | 3/4 | 3/4 | 2-4 |
| `probe40_base_v2` | v2, 8/family | 1/8 | 0/8 | 0/8 | 2/8 | 7/8 | 2-10 |
| `probe40_base_v2_tools2` | v2 + clearer tool descriptions | 0/8 | 2/8 | 0/8 | 1/8 | 7/8 | 2-10 |

Read by hand. What changed between probes, and why:

- **v1 → v2: the task design was at fault, not the model.**
  - The prompt named a document but never gave its id, so the model invented one (`HMT-2026-02-19`).
  - Search then quietly returned nothing for that unknown id, and the model answered `not_found`.
  - Fixes: an unknown `document_id` is now an error, and half the lookup tasks name the `document_id`, as the app would when a regulation is open.
- **tools v2:** the descriptions now say what a provision_id looks like, that a document_id is not one, and when to use `list_sections`. The model had been calling `get_provision("us_corpus:2229")`.

**What is left is real model error:**
- `spans` given as one string instead of a list;
- fetching a neighbouring provision instead of the one search returned;
- paraphrasing instead of quoting;
- answering a lookup without any tool call;
- saying `not_found` without looking.

**Result:** the base model solves about 5-10% of tool episodes, against 7/8 when the passage is given. Speed is 2-10 s per episode in batches of 4.

## Warm start: scripted demos + teacher (hybrid), 2026-10-18

Base model solves too few tool episodes to learn from its own successes, so the
first SFT mixes two sources, both kept only when `verify_episode` passes them:

**Scripted demos** (`demos.py`, $0): code plays each train task from its gold answer
through the real harness. It finds the document by title (or from the prompt's
document_id), searches inside it, reads the provision, and quotes the gold. It
fetches a provision only if an earlier result in the episode returned its id.

| family | kept | unreachable by the plain strategy |
|---|---:|---:|
| lookup_definition | 229 | 14 |
| lookup_effective_date | 504 | 115 |
| cited_obligations | 14 | 3 |
| no_tool_needed | 179 | — |
| not_found | 151 | 5 |

**Verifier bug found by the demos, fixed.** Quotes were checked against the
JSON-escaped tool output the model reads. A correct quote spanning a line break or
a quote mark could never match: 14 of 14 obligation demos failed on it. Quotes are
now checked against the unescaped result text (`Call.plain_text`), and a test
covers it.

**Teacher** (`teacher.py`, `gpt-5.4-mini`, reasoning none): the same prompt, tools
and index, with its calls turned back into Qwen `<tool_call>` text. Pilot: 10 episodes,
4 passed, $0.0378 ($0.0038/episode, about 4.3k prompt + 130 completion tokens).
Its failures match the base model's: invented document ids and going past the
3-call budget. Only verified episodes are used.

**Full teacher run** (train split, 1,214 episodes, $4.32): 714 passed.

| family | passed |
|---|---:|
| lookup_definition | 86/243 |
| lookup_effective_date | 414/619 |
| cited_obligations | 1/17 |
| no_tool_needed | 175/179 |
| not_found | 38/156 |

**Teacher held-out reference** (`--split heldout`, k=1, $0.57, never trained on):
definitions 9/40, dates 30/40, obligations 1/7, no_tool 39/40, not_found 14/40.
OpenAI spend for the whole track: **$4.89**.

### warm_v1 SFT

`sft.py build` on demos + teacher: 1,327 trajectories → 3,198 train / 120 valid
per-turn examples (p50 1,474 tokens, max 3,883; 3 over 4,096 dropped). LoRA rank 8,
scale 20, 8 layers, lr 1e-4, 2,000 iterations, batch 1, `--mask-prompt`,
grad checkpointing; peak 7.7 GB, 3 h 22 min. Val loss 3.327 → 0.119 (500) → 0.085
(1000) → 0.075 (1500) → 0.066 (2000).

### Held-out: base vs warm_v1 (k=2, temp 0.7, paired bootstrap per task)

| family | tasks | base | warm_v1 | Δ | 95% CI | teacher (k=1) |
|---|---:|---:|---:|---:|---|---:|
| lookup_definition | 40 | 0.075 | **0.475** | +0.400 | [0.24, 0.56] | 0.225 |
| lookup_effective_date | 40 | 0.250 | **0.800** | +0.550 | [0.38, 0.71] | 0.750 |
| cited_obligations | 7 | 0.143 | 0.286 | +0.143 | [0.00, 0.43] | 0.143 |
| no_tool_needed | 40 | 0.875 | 0.975 | +0.100 | [0.00, 0.23] | 0.975 |
| not_found | 40 | 0.188 | **0.637** | +0.450 | [0.30, 0.60] | 0.350 |
| **all** | 167 | 0.338 | **0.704** | +0.365 | [0.29, 0.44] | — |

| behaviour (334 episodes) | base | warm_v1 |
|---|---:|---:|
| tool calls / episode | 1.04 | 1.47 |
| valid-call rate | 0.758 | 0.946 |
| invented-id calls / episode | 0.243 | 0.063 |
| over-budget episodes | 0 | 2 |
| JSON final answer | 0.979 | 0.991 |

The 4B student beats its teacher on definitions and not_found: scripted demos teach
the plain search→fetch→quote path, the teacher adds variety, and only verified
episodes of either were kept. Obligations (7 tasks) is too small to read.

**Remaining failures (warm_v1, read by hand):**
- not_found: answers a term the document does not contain (16).
- definitions:
  - quote not verbatim (15);
  - `not_found` on an answerable question (12);
  - an invented document_id such as `UKLEG:1234` (about 10).
- dates: unparseable date (6), wrong date (5).
