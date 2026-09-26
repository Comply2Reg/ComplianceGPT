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

## Labelling throughput — what actually limits it

OpenAI meters **per model**, so N processes on one model share one bucket and simply
collide: the first gpt-4o run spent its time retrying 429s (that tier is 30,000 TPM —
about 9 documents/minute) and hung on a stalled socket. Measured on this key,
2026-09-25:

| model | RPM | TPM | docs/min | $/doc | 2,284 docs |
|---|---:|---:|---:|---:|---:|
| gpt-4o | 500 | 30,000 | ~9 | $0.0093 | 4.5 h, $21 |
| gpt-4o-mini | 10,000 | 200,000 | ~50 | $0.0006 | 50 min, $1.31 |
| **gpt-5.4-mini** | 500 | 200,000 | ~55 | $0.0031 | ~40 min, $7 |

So the levers are, in order: pick a model whose bucket is big (`--probe-limits` prints
it); pace to it (`--tpm`, a sliding-window bucket that reserves an estimate and settles
to the real usage once the call returns — without settling it leaves a quarter of the
budget unspent); and split the work (`--shard i/N`) so one process's latency does not
idle the budget. Shards each write `triage_<v>.shard<i>.jsonl`; `--cache-only` merges
them and names anything still missing.

Different models *do* have separate buckets, so a second model can label the same
documents in parallel at no cost in wall-clock — which is how the second opinion in the
next section is free.

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

# 1. labelling: check the account's limits, smoke 5, then shard the bulk
python -m scripts.triage.labeller --probe-limits --model gpt-5.4-mini
python -m scripts.triage.labeller --corpus-dir $CORPUS --version smoke --tier GREEN \
    --model gpt-5.4-mini --sample 5                      # inspect these by hand
for i in 0 1 2; do
  nohup python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN \
      --model gpt-5.4-mini --shard $i/3 --concurrency 3 --tpm 60000 \
      > logs/label-green-v1-s$i.out 2>&1 &
done
python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN \
    --model gpt-5.4-mini --cache-only                    # merge the shards; reports gaps
python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN \
    --model gpt-5.4-mini                                 # fill any gap (cached = free)

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

## Choosing what the strong model labels

Two models disagreeing is a better gold-candidate signal than either disagreeing with
the registry: it needs no ground truth and it points at genuinely ambiguous documents
(the A11↔A3 and A4↔A2 boundaries, in practice).

```bash
# after two models have labelled the corpus (their caches are separate)
python -m scripts.triage.report --labels data/uk/labels/triage_v1.jsonl \
    --weak data/uk/labels/weak_v1.jsonl --cost data/uk/labels/cost_v1.json
#   -> report_v1.md gains a model-vs-model section
#   -> disputed_v1.txt lists the doc_ids where the classes differ

python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN \
    --model gpt-5.4 --ids-file data/uk/labels/disputed_v1.txt    # strong model, few docs
python -m scripts.triage.labeller --corpus-dir $CORPUS --version v1 --tier GREEN \
    --cache-only --prefer-models gpt-5.4,gpt-5.4-mini,gpt-4o-mini

python -m scripts.triage.build_dataset ... --disputed-file data/uk/labels/disputed_v1.txt
#   those rows join the gold sheet flagged 'models_disagree'
```

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

## First run, 2026-09-25 (v1, GREEN only)

Labels: 2,284 documents by `gpt-5.4-mini` in 3 shards, 0 failures, **$7.45**; a full
`gpt-4o-mini` pass ($1.32) in a separate rate-limit bucket served as the second opinion.
Which model to believe was settled with evidence, not preference: where the two minis
disagree, `gpt-4o` sides with gpt-5.4-mini 52% vs gpt-4o-mini 27%, and a 20-document
`gpt-5.4` probe ($0.43) agreed with gpt-5.4-mini 18/20 and gpt-4o-mini 0/20 — so the
full $20 strong pass over the 951 disputed rows was **not** run.

Dataset: train 1,683 / val 218 / test 104, p95 1,034 tokens against a 2,048 budget.

Training: `mlx-lm` LoRA on `mlx-community/Qwen3-4B-Instruct-2507-4bit`, 16 layers,
7.34M trainable parameters (0.18%), batch 1, 1,700 iterations (~1 epoch), 2 h 26 m,
**4.6 GB peak** on an M4 Pro. Validation loss 3.075 → 0.961 → 0.936 → 0.860 → 0.748
(iter 1200) → 0.873 → 0.805.

| metric (104 held-out documents) | base model | fine-tuned |
|---|---:|---:|
| JSON validity | 0.00 | **1.00** |
| alert_class accuracy | — | 0.635 |
| alert_class macro-F1 | — | 0.236 |
| priority accuracy | — | 0.750 |
| obligations_present accuracy | — | 0.913 |
| primary_function Jaccard | — | 0.594 |

The untrained model scores zero because it answers in verbose prose-valued JSON and
overruns the generation cap — teaching the compact schema is most of what the fine-tune
does. Per class it gets A11 50/56, A6 5/6, A3 4/7, A10 7/17 and **zero** on A1, A4, A8,
A9, A12, A13, A14: those have almost no training examples. Majority-class (always A11)
would score 0.54, so 0.635 is a real but modest gain, and macro-F1 is the honest number.

Two findings worth keeping:

- **Lower validation loss did not mean a better model.** The iteration-1200 checkpoint
  had the best loss (0.748) but scored worse on the task (0.587 accuracy, 0.962 JSON
  validity) than the final adapter (0.635, 1.000). Token cross-entropy on the answer is
  a poor proxy for classification accuracy; select checkpoints on `--eval-only`.
- **Batch 4 was slower than batch 1** on this machine (0.21 vs 0.35 examples/s): records
  pad to the longest in the batch, and the run is memory-bandwidth bound.

The ceiling here is class balance, not the recipe: A11 is 54% of the test split because
GREEN is dominated by HM Treasury news and CMA press releases. The fix is the AMBER
sign-off (FCA/PRA policy statements, supervisory guidance and enforcement notices), not
more labelling or more epochs.

## Publishing a trained model

Four steps, in order. The first three are local and reversible; only the last leaves
the machine.

```bash
# 1. fuse the adapter into standalone weights (no --dequantize: the adapter was
#    trained against the 4-bit base, so that is what it should ship as)
python -m mlx_lm fuse \
    --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --adapter-path models/uk-triage-v1-qwen3-4b-instruct-mlx \
    --save-path models/publish/regulatory-alert-triage-qwen3-4b-v1

# 2. evaluate the FUSED weights — not the adapter — on the test split
python -m scripts.triage.train_mlx eval --dataset-dir data/uk/datasets/green \
    --version v1 --model-path models/publish/regulatory-alert-triage-qwen3-4b-v1

# 3. generate the card and eval summary from the run artefacts
python -m scripts.triage.model_card \
    --model-dir models/publish/regulatory-alert-triage-qwen3-4b-v1 \
    --dataset-dir data/uk/datasets/green --version v1 \
    --train-report models/uk-triage-v1-qwen3-4b-instruct-mlx/train_report_v1_qwen3-4b-instruct_mlx.json \
    --train-log logs/train-mlx-v1.out \
    --coverage ../../09-inventory-kit/c2r-inventory-kit/data/alert_corpus/coverage.json

# 4. validate, then upload (HF_TOKEN in .env, write scope on the org)
python -m scripts.triage.publish_model \
    --model-dir models/publish/regulatory-alert-triage-qwen3-4b-v1 --dry-run
python -m scripts.triage.publish_model \
    --model-dir models/publish/regulatory-alert-triage-qwen3-4b-v1 \
    --repo-id Comply2Reg/regulatory-alert-triage-qwen3-4b-v1
```

Things that bite here:

- **`mlx_lm fuse --upload-repo` destroys the model card.** Its `upload_to_hub` keeps
  the YAML frontmatter but overwrites `card.text` with generic boilerplate. Always
  upload through `publish_model.py`, never through the fuse step.
- **`fuse` needs a complete snapshot of the base repo**, including `.gitattributes`
  and `README.md`, which `load()` never downloads. If it dies in `save()` with
  `IncompleteSnapshotError`, `hf_hub_download` those two files and re-run.
- **No GGUF for Qwen.** `mlx_lm`'s converter accepts only `llama`, `mixtral` and
  `mistral` model types and raises on anything else.
- **Fusing into 4-bit is lossy.** Each layer is dequantized, given the LoRA delta and
  re-quantized, so the fused model is not the adapter. v1 agreed with base+adapter on
  93 of 104 test documents (89.4%) and scored 0.644 against 0.635 — noise, but the card
  must quote the fused model's own numbers. `model_card.py` fails the build if any
  metric regresses by more than 0.02 or JSON validity drops below 0.99.
- **Repos are created private.** Going public needs `--public` *and*
  `--i-have-checked-licensing`, because the GREEN corpus carries OGL attribution
  obligations that a stray flag should not publish past.

v1 was published as `Comply2Reg/regulatory-alert-triage-qwen3-4b-v1`: 2.1 GB, 4-bit, fused,
private pending the licensing review.

## Benchmarking and the error bar

`docs/benchmarks.md` is the full write-up. The short form and the one fact that matters:

**The test labels are not ground truth.** They were written by `gpt-5.4-mini`, the same
model that wrote the training labels, so the headline accuracy measures agreement with one
labeller. Re-labelling the same 104 documents with `gpt-5.4` ($2.44, 0 failures) and
re-scoring the *same* predictions gives the band instead of a point:

| | vs `gpt-5.4-mini` | vs `gpt-5.4` |
|---|---:|---:|
| alert_class accuracy | 0.644 | 0.567 |
| macro-F1 | 0.245 | 0.155 |
| primary_functions Jaccard | 0.574 | 0.780 |

The two labellers agree with each other on **0.702** of alert classes, so that is roughly
the ceiling any model can reach against either of them. Quote the band, not 0.644.
Priority, obligations and functions all score *higher* against the stronger judge, so the
model generalised past its teacher on those fields; only class assignment is anchored to
the training labeller.

```bash
# re-label just the test split with a stronger model, then re-score
python -m scripts.triage.labeller --corpus-dir <corpus> --out data/uk/labels \
    --version judge1 --tier GREEN --model gpt-5.4 \
    --ids-file data/uk/labels/test_ids_v1.txt
python -m scripts.triage.rescore --dataset-dir data/uk/datasets/green --version v1 \
    --eval-outputs data/uk/datasets/green/eval_v1_..._outputs.jsonl \
    --judge-labels data/uk/labels/triage_judge1.jsonl

# public benchmarks: always both sides, the delta is the result
python -m scripts.triage.bench list
python -m scripts.triage.bench run --task ledgar --limit 200                # base
python -m scripts.triage.bench run --task ledgar --limit 200 --model-path <fused>
python -m scripts.triage.bench compare --task ledgar
```

- `rescore.py` re-parses the saved generations, so it costs nothing and cannot drift from
  what was measured. It **exits non-zero if it fails to reproduce the published eval**,
  which is the check that the re-scoring path is sound before the judge numbers are read.
- `train_mlx.py` now stores the whole generation, not the first 1,200 characters; a clipped
  record would silently re-parse as unanswerable.
- Public benchmarks measure transferable ability, never the product metric. Expect low
  absolute scores: the model answers every prompt with a triage record, so `bench.py`
  reports `answered_rate` and `accuracy_when_answered` separately and looks inside returned
  JSON for a label before scoring it wrong.
- Rejected after checking: **LexGLUE EUR-LEX** (labels are bare EuroVoc ids, unnameable
  zero-shot) and **FinBen** (not individually addressable on the Hub; QA and numeric tasks,
  not classification). `main`'s notebook cites `rcraigfieldwork/ObliQA`, which does not
  exist — the real id is `RegNLP/ObliQA`.

## Multi-jurisdiction corpus, 2026-09-26

The corpus is no longer UK-only. 2,727 US federal documents were crawled from twelve
regulators, exported as 2,362 documents / 60,153 chunks, and labelled; the UK set was
relabelled under the same prompt so the two agree.

| | UK v1 | UK+US v2 |
|---|---:|---:|
| documents | 2,284 | 4,646 |
| majority class share | 49% (A11) | 34% (A11) |
| classes with ≥50 examples | 5 | 11 |
| classes with <10 examples | 3 | 0 |

The classes the v1 model scored zero on are the ones that moved most:

| class | UK v1 | now |
|---|---:|---:|
| A2 final rules | 22 | 463 |
| A3 consultations | 228 | 825 |
| A7 sanctions | 4 | 228 |
| A8 reporting changes | 4 | 293 |
| A14 perimeter changes | 38 | 200 |
| A5 supervisory letters | 0 | 27 |

**Relabelling was conservative.** 86.9% of UK alert classes and 90.9% of priorities are
unchanged from v1. Where a class moved it was mostly A11 giving way to something more
specific — A10, A4, A6 — which is the right direction, since A11 over-use was the v1
model's central weakness.

**Cost:** $8.17 for the US set, $8.08 for the UK relabel, 0 failures either side.

**Frameworks.** 1,048 US and 327 UK documents name an international standard, led by
BSA/AML, Dodd-Frank, FATF, Basel III and MiFID. That field is the seed for the
cross-jurisdiction equivalence catalogue.

**Jurisdiction.** GB 2,212 / US 2,361 / ZZ 66 / EU 7. The ZZ rows are standard-setter
material binding no one directly; the EU ones are UK pages republishing EU instruments,
which is the field working as intended.

### Long runs: use caffeinate, and pace to the TPM limit

**A labelling run that appears to stall is almost certainly the Mac sleeping.** This cost
most of a day to find, so it is worth stating plainly.

The symptom is a process alive at 0% CPU, no network, no log output, silent gaps of
twenty minutes, and — the tell — **zero failures afterwards**. It looks exactly like a
hung socket or a deadlock, and it is neither.

The evidence that settles it is the two clocks disagreeing. The labeller's own
`time.monotonic()` counter said 25 documents in 91 seconds while the wall clock said 32
minutes. `monotonic()` does not advance while macOS is asleep, so the difference is
exactly the time the process spent suspended. `pmset -g log` showed 443 sleep/wake cycles.

```bash
caffeinate -i python -m scripts.triage.labeller ... --tpm 200000
```

Same 474 documents, uninterrupted: **366 seconds, 0.77s each**. Against roughly 80s each
when the machine was free to sleep.

`--tpm` matters once the run is actually fast. Unpaced, 474 documents produced 29 rate-limit
failures on the 200,000 TPM ceiling; paced, the same work finished with none.

Two things chased before the real cause, both dead ends, recorded so nobody repeats them:
the request timeout (raising it made matters worse, since the call sits inside the
concurrency semaphore) and prompt size (p50 is 6,300 characters, which is fine). The
schema, the model and the API were all healthy throughout — isolated calls measured 2-3
seconds the whole time.

### Running it

```bash
# US: crawl, export, label
python run_scraper.py --regulator occ --type final_rules --max-items 120   # per source
python scripts/export_alert_corpus.py --jurisdiction US --tier GREEN --out data/us_corpus
python -m scripts.triage.labeller --corpus-dir <us_corpus> --out data/uk/labels \
    --version us_v1 --tier GREEN --model gpt-5.4-mini
```

Labels cache per document and per prompt version, so an interrupted run resumes without
re-paying. That mattered: the UK relabel stalled twice and lost nothing.

## Known caveats

- `main`'s `run_extract.py` falls back silently to a regex labeller on API
  failure; `validate_dataset.py --task obligation` flags those records
  (`Extracted Entity`). Drop them before training.
- HM Treasury rows are whole-of-Treasury; the labeller's class and functions
  are the only relevance filter today.
- 960 documents were unclassified at capture (MIXED sources); their stratum
  comes from the model class (`stratum_source: model`) and they are
  over-represented in the gold sheet on purpose.
