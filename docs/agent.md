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
