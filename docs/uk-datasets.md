# UK alert corpus → fine-tuning datasets

How the UK regulatory corpus captured by `c2r-inventory-kit` becomes training
data for the domain SLM. Two tasks share one pipeline: **alert triage** (this
branch, `scripts/triage/`) and **obligation extraction** (`main`, run from a
worktree — the branches have no common ancestor and are never merged).

## Where the data is

| What | Where |
|---|---|
| Stage-1 corpus (input) | `../../09-inventory-kit/c2r-inventory-kit/data/alert_corpus/` — `canonical/{id}.txt`, `chunks.jsonl`, `manifest.jsonl`, `coverage.json` |
| Originals | `…/data/mirror/United Kingdom/{REG}/{type}/` (`manifest.local_original`) |
| Labels | `data/uk/labels/` — `triage_<v>.jsonl`, `weak_<v>.jsonl`, `failures_<v>.jsonl`, `cost_<v>.json`, `report_<v>.md`, `cache/` |
| Datasets | `data/uk/datasets/green/` (shareable, GREEN only) · `data/uk/datasets/internal/` (contains AMBER, never leaves) |
| Gold | `data/uk/gold/gold_candidates_<v>.csv` → hand-labelled `gold_labels_<v>.csv` |
| Obligation chunks | `data/uk/obligation/` |

The corpus is validated with `python scripts/validate_data.py --source uk_alerts
--corpus-dir <corpus> --strict --skip-raw` before anything below runs.

## Licence tiers — the one rule that matters

Every manifest row and chunk carries `tier`: **GREEN** (HM Treasury, CMA, ICO,
UK legislation — OGL) may be used for training with attribution; **AMBER**
(FCA, PRA, BoE, DRCF, Ofcom) is internal use only. GREEN has no guidance or
enforcement documents, so a GREEN-only dataset cannot reach the target mix.
The labeller and the adapter therefore refuse AMBER rows unless both
`--allow-amber` and `--counsel-signoff "<ref> <date>"` are given; the sign-off
is written into `cost_<v>.json` and onto every AMBER label. Build AMBER
datasets only under `data/uk/datasets/internal/` (gitignored).

## Run sequence

```bash
cp .env.example .env            # add OPENAI_API_KEY; never paste it into chat
CORPUS=../../09-inventory-kit/c2r-inventory-kit/data/alert_corpus

# 0. free baseline labels from the registry and titles
python -m scripts.triage.weak_labels --corpus-dir $CORPUS --version v1

# 1. smoke, then the GREEN pass (~2,300 docs); re-runs are free (cache)
python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN --dry-run
python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN --limit 50
python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN

# 2. datasets, gates, report, gold sheet
python -m scripts.triage.build_dataset --labels data/uk/labels/triage_v1.jsonl \
    --corpus-dir $CORPUS --out data/uk/datasets/green --version v1 --tier GREEN
python -m scripts.triage.validate_dataset --dataset-dir data/uk/datasets/green --version v1 --strict
python -m scripts.triage.report --labels data/uk/labels/triage_v1.jsonl \
    --weak data/uk/labels/weak_v1.jsonl --cost data/uk/labels/cost_v1.json

# 3. hand-label data/uk/gold/gold_candidates_v1.csv → gold_labels_v1.csv, then
python -m scripts.triage.build_dataset ... --apply-gold data/uk/gold/gold_labels_v1.csv

# 4. quality report, then train (GPU: Colab T4/L4 or better; pip install unsloth trl)
python -m scripts.triage.analyze_dataset --dataset-dir data/uk/datasets/green --version v1 --corpus-dir $CORPUS
python -m scripts.triage.train_triage --dataset-dir data/uk/datasets/green --version v1 --dry-run
python -m scripts.triage.train_triage --dataset-dir data/uk/datasets/green --version v1 \
    --out models/uk-triage-v1-qwen3-4b-instruct
python -m scripts.triage.train_triage --dataset-dir data/uk/datasets/green --version v1 \
    --eval-only models/uk-triage-v1-qwen3-4b-instruct        # eval_v1_qwen3-4b-instruct.json
#    (the main-branch notebook is NOT used for triage: it drops dict outputs and
#     truncates at 1,024 tokens; it stays as-is for the obligation task)

# 4b. the same on this Mac (Apple Silicon, mlx-lm; unsloth is CUDA-only)
python -m scripts.triage.train_mlx export --dataset-dir data/uk/datasets/green --version v1
python -m scripts.triage.train_mlx eval   --dataset-dir data/uk/datasets/green --version v1   # untrained baseline
python -m scripts.triage.train_mlx train  --dataset-dir data/uk/datasets/green --version v1 --iters 600
python -m scripts.triage.train_mlx eval   --dataset-dir data/uk/datasets/green --version v1 \
    --adapter-path models/uk-triage-v1-qwen3-4b-instruct-mlx
#    mlx-community/Qwen3-4B-Instruct-2507-4bit, LoRA on 16 layers, loss masked to the
#    assistant turn (--mask-prompt); ~2.5 GB of weights, fits 16-24 GB unified memory

# 5. after counsel sign-off only
python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN AMBER \
    --allow-amber --counsel-signoff "LEGAL-nnn YYYY-MM-DD"
python -m scripts.triage.build_dataset ... --out data/uk/datasets/internal --tier GREEN AMBER

# 6. obligations (second task)
python -m scripts.triage.obligation_adapter --corpus-dir $CORPUS \
    --out data/uk/obligation/chunks_uk_v1.jsonl --strata rule_final guidance \
    --regulator-include UKLEG --tier GREEN
git worktree add ../ComplianceGPT-main main
(cd ../ComplianceGPT-main/scripts/ObligationDataPipeline && USE_OPENAI=true \
    python run_extract.py --version uk-v1 --chunks $PWD/data/uk/obligation/chunks_uk_v1.jsonl)
python -m scripts.triage.validate_dataset --dataset-dir data/uk/obligation --version uk-v1 \
    --task obligation --skip-corpus-check
```

## Model choice (2026-09-25)

One focus model, chosen against the task (structured JSON, whole-document
inputs later, LoRA on a modest GPU, a licence a bank can ship under):

| | **Qwen3-4B-Instruct-2507** (focus) | Gemma 4 E4B-it (fallback) | Qwen3.5-4B | Phi-4-mini / Llama 3.1 8B |
|---|---|---|---|---|
| licence | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT / Llama licence |
| architecture | 4.0B dense, 36 layers, GQA 32/8 | 4.5B eff. (8B w/ embeddings), per-layer embeddings, hybrid attention | Gated DeltaNet + sparse MoE, VLM | dense |
| context | 262,144 native | 128k | 262k | 128k |
| thinking | non-thinking only (clean JSON) | non-thinking template | thinking by default | n/a |
| fine-tune | QLoRA ≈ 8–10 GB; mature in unsloth/TRL/vLLM | QLoRA 10 GB, LoRA 17 GB; known quirks | newest hybrid arch, tooling least mature | fine |

After fine-tuning the 4B candidates converge on structured-output accuracy, so
the choice is licence, context, tooling risk and serving cost — all favour
Qwen3-4B-Instruct-2507. What its architecture imposes on the data: ChatML
template rendered by `tokenizer.apply_chat_template` (never hand-built),
response-only loss on `<|im_start|>assistant\n`, budget measured with its
151k-vocab tokenizer (ungated), `max_seq_length` 2048 for head-of-document
records, compact JSON assistant turn with a short `thought_trace` last. The
`MODELS` registry in `scripts/config.py` carries these facts; `--model
gemma-4-e4b-it` swaps them (Gemma 4's template uses `<|turn>` markers — the v1
notebook's `<start_of_turn>` strings would be wrong for it).

Sources: Qwen3-4B-Instruct-2507 and Qwen3.5-4B model cards, the Gemma 4 model
card and Unsloth's Gemma 4 guide, and an on-device tool-calling comparison of
Phi-4-mini / Gemma 4 E4B / Qwen3-4B (September 2026).

## Record format

The 7-key envelope of `main`'s `ObligationRecord` (kept so the obligation tooling
can read the same files) plus the two keys the triage trainer actually uses:

| key | triage |
|---|---|
| `metadata` | `source_id` (`{regulator}:{doc_type}#{id}`), `regulation_name`, `jurisdiction` GB, `doc_type`, `section` "document", plus `regulator`, `release_date`, `stratum`, `stratum_source`, `doc_hash`, `tier`, `licence`, `label_source`, `model`, `prompt_version` |
| `classification` | alert class A1–A14 |
| `instruction` | one of three templates in `prompts.py` |
| `input_text` | regulator, type, title, date + first 1,500 chars of cleaned canonical text (trimmed further if the rendered record exceeds the token budget) |
| `tool_use` | `null` |
| `thought_trace` | the labeller's `rationale` |
| `output` | `TriageRecord` minus rationale — never a `status` key |
| `messages` | model-neutral `system/user/assistant` turns; the assistant turn is the compact JSON the model learns |
| `n_tokens` | rendered length under the target tokenizer; context trimmed to the budget, never the answer |

Splits are stratified by stratum **and** date: the latest two months of each
stratum are `test` (the gold candidates), the next 10 % by date `val`, the rest
`train`; only train is balanced towards 35/20/15/15/15 (rule_final taken whole,
nothing oversampled). `all_<v>.jsonl` = train + val; test never enters it.

## Data-quality checks (Chapter 4 checklist)

| item | where |
|---|---|
| cleaning | crawler `core/furniture.py` at export: sidebars, site chrome, running headers, page numbers, contents tables, addresses; `clean_prose` on the training input (URLs, e-mails, whitespace) |
| dedup | exporter: identical text → same attachment → same reference → **near-duplicate** (MinHash-LSH, Jaccard ≥ 0.9 on 8-word shingles, within one document type; dropped rows carry `duplicate_kind: near`) |
| format | 7-key envelope + `messages` + `n_tokens`; `validate_dataset.py --strict` |
| tokenizer / budget | Qwen3-4B-Instruct-2507 tokenizer, 2048 tokens; `stats_<v>.json` and `analyze_<v>.md` report p50/p95/max and whole-document sizes |
| split | stratum × date; test = latest two months; no hash in two splits; test never in `all_<v>.jsonl` |
| quality report | `analyze_dataset.py`: class coverage (rare classes < 30), diversity (unique 8-gram ratio), internal repetition, dates, label sources, corpus dedupe/furniture figures |

## Training on a Mac

`train_triage.py` needs CUDA (unsloth, bitsandbytes). On Apple Silicon use
`train_mlx.py`: it exports the records' `messages` to mlx-lm's chat JSONL,
runs `mlx_lm lora --mask-prompt` against `mlx-community/Qwen3-4B-Instruct-2507-4bit`
(the same weights, 4-bit), and evaluates with the shared `metrics.py` so
`eval_<v>_<model>_mlx.json` is comparable with the CUDA numbers. Defaults
(`config.TRAIN_MLX`): 600 iterations, batch 1, 16 layers, lr 1e-4, 2048 tokens,
gradient checkpointing — sized for a 24 GB M-series machine; close other apps.
Note: `Qwen3.5-4B` is a different (hybrid, thinking-by-default) model and has no
MLX 4-bit conversion; `-2507` belongs to Qwen3-4B-Instruct.

## Known caveats

- `main`'s `run_extract.py` falls back silently to a regex labeller on API
  failure; `validate_dataset.py --task obligation` flags those records
  (`Extracted Entity`). Drop them before training.
- HM Treasury rows are whole-of-Treasury; the labeller's class and functions
  are the only relevance filter today.
- 960 documents were unclassified at capture (MIXED sources); their stratum
  comes from the model class (`stratum_source: model`) and they are
  over-represented in the gold sheet on purpose.
