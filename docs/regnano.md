# RegNano: verifiable-reward RL for a compliance SLM

Started 2026-10-13. Modelled on Microsoft FrogNano-4B-2609
([model card](https://huggingface.co/microsoft/FrogNano-4B-2609)): RL only, on
Qwen3.5-4B, over ~1,500 synthetic software tasks that an author model writes and
the current policy calibrates. Five iterations took SWE-bench Verified from 39.4%
to 61.5%.

The part we take is **reward only what can be verified**. v1-v3 learn from
GPT-written labels, and the two labellers agree on only 0.702 of alert classes
(`docs/benchmarks.md`). A RegNano task's gold answer is a *checker*, never a
training target, and every family has a deterministic verifier.

| FrogNano | RegNano |
|---|---|
| repository snapshot | a GREEN regulation (US federal register/CFR text, UK SIs, EUR-Lex) — no AMBER/RED under any sign-off |
| hidden tests | `scripts/regnano/verifiers.py` |
| TaskPilot author model | `mine.py` (gold read from the document's own structure); an LLM author for obligation/citation tasks is next |
| calibration rollouts, keep ~50% solve rate | `calibrate.py`, k=8, keep 0 < pass < 1 |
| DPPO, 8x B200 | GRPO via `mlx-lm-lora` on the M4 Pro (`rl.py`, `.venv-rl`) |

## Base model: Qwen3.5-4B

Registered as `qwen3.5-4b` in `scripts/config.py`. Dense, 32 layers (24 Gated
DeltaNet + 8 gated attention), vocab 248,320, 262K context, Apache-2.0. Local
4-bit conversion at 4.5 bits/weight, 2.2 GB:

```bash
python -m mlx_lm convert --hf-path Qwen/Qwen3.5-4B --mlx-path models/base/qwen3.5-4b-mlx-4bit -q --q-bits 4
python -m scripts.triage.mlx_base models/base/qwen3.5-4b-mlx-4bit   # required, see below
```

If `convert` fails with `IncompleteSnapshotError` on `.gitattributes`/`LICENSE`/
`README.md`, `hf_hub_download` those three and re-run; the weights are cached. Set
`HF_TOKEN` in the environment — unauthenticated downloads crawl.

**Thinking mode, and why the template is patched.** Qwen3.5 has no
instruct-only release; its template opens `<think>` unless the caller says
otherwise. mlx-lm's `TokenizerWrapper.apply_chat_template` injects
`enable_thinking=True` whenever the vocabulary has `<think>` and the caller passes
nothing, which is what its trainer does. Unpatched, the `--mask-prompt` prompt
ends `<think>\n` while the training sequence has `<think>\n\n</think>\n\n`, so
the prompt is not a token prefix of the sequence. `mlx_base.py` makes the
template think only when `regnano_thinking=True` is passed; anything else gets
the empty think block, during training and inference alike. Verified: the prompt
is an exact token prefix, and the loss covers only `{json}<|im_end|>`.

### Training cost on the M4 Pro: why triage stays on Qwen3

One LoRA step (rank 8, 16 layers, gradient checkpointing, batch 1), measured 2026-10-13:

| base | tokens | peak memory | s/step |
|---|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 1,536 | 4.4 GB | 5.8 |
| Qwen3.5-4B | 512 | 7.2 GB | 5.6 |
| Qwen3.5-4B | 768 | 10.0 GB | 10.0 |
| Qwen3.5-4B | 1,536 | 21.2 GB | 52-115 (swapping) |

Plain `mlx_lm lora` runs the DeltaNet layers through a per-token Python loop in
training mode (`use_kernel=not self.training`) and keeps every step's state:
18 GB at 256 tokens, out of memory on step 1 of the triage run. mlx-lm-lora's
`recurrent_patch` routes training through MLX's fused `mx.fast.gated_delta_update`
(verified: all 24 linear-attention calls take it), which is the table above; the
remaining cost is that kernel's backward in MLX 0.32.3.

Decision: **RegNano RL runs on Qwen3.5-4B** (task prompts p50 726 / p95 919
tokens), **triage v4 stays on Qwen3-4B-Instruct-2507** (records p50 1,050,
max 1,499 — ~22 h per run on Qwen3.5). Any Qwen3.5 training goes through
mlx-lm-lora, never plain `mlx_lm lora`.

## Task families

| family | gold from | verifier |
|---|---|---|
| `effective_date` | Federal Register `DATES:` paragraph: exactly one effective date, or `null` for comment-only proposals | date equality; `null` only when gold is null |
| `defined_term` | `“X” means …` running to end of line, unique in a window holding other definitions | verbatim span (typography- and whitespace-insensitive), overlap >= 0.8 |
| `obligation_span` | human-reviewed seed gold (`seed.py`); author model next | every span verbatim, span-set F1 >= 0.8, modality match |
| `citation` | author model / legislation.gov.uk changes feed (next) | normalised ID equality (`12 C.F.R. § 1026.19` == `12 CFR 1026.19`) |
| `triage_class` | agreed labels or human gold only | exact class |

```bash
python -m scripts.regnano.mine --out data/regnano/tasks/mined_v1.jsonl   # 909 date + 473 term tasks, all GREEN
python -m scripts.regnano.calibrate --tasks <tasks> --out <calibration.jsonl> --k 8 --keep-out <in-band.jsonl>
python -m scripts.regnano.pool --tasks <tasks> --calibration <calibration.jsonl> --out-dir data/regnano/tasks/iter1
python -m scripts.regnano.rl export --tasks data/regnano/tasks/iter1/train.jsonl --out data/regnano/grpo/iter1
python -m scripts.regnano.rl train --data data/regnano/grpo/iter1 --model-path <policy> --adapter-path models/regnano-iter1
```

`rl.py train` imports the reward file in `.venv-rl` first and refuses to start if
it fails: mlx-lm-lora swallows reward-file import errors and silently trains on
its default math-format rewards.

## Findings so far (2026-10-13, base Qwen3.5-4B, k=8, temp 0.7, thinking off)

Probe of 48 tasks, ~9.7 s/task:

- **`effective_date` is saturated**: 32/32 tasks solved 8/8, dated and null
  alike. FrogNano drops these. Needs harder variants (compliance vs effective
  date, staggered dates) or stays as a small easy share.
- **`defined_term` first scored 0.09, and most of that was the verifier.** The
  model wrote `"term"` where the text has `“term”`; quote/dash variants are now
  normalised. Then the mined gold was wrong: ending a definition at the first
  `;`/`.` + space cut `4:00 a.m. and …` at `a.m.` and multi-sentence definitions
  at their first sentence, so complete quotes were marked wrong. Definitions now
  must run to end of line. After the quote fix alone: mean 0.70, 10 always / 4
  never / 2 in band.

### Full pool, 2026-10-13 (573 tasks, k=8, 2 h 45 min)

| family | tasks | always 8/8 | never 0/8 | in band | mean pass |
|---|---:|---:|---:|---:|---:|
| `effective_date` | 100 | 100 | 0 | 0 | 1.00 |
| `defined_term` | 473 | 307 | 13 | 153 | 0.87 |

Reading the 13 never-solved tasks by hand: most were **gold errors, not model
errors**: two definitions on one line ("fuel" ran into "feedstock"), aliased
terms (`"Board" or "Board of Directors"`), definitions continued on the next
line (`…;\nprovided that`), a term quoted twice (`interprets "X" to mean …
"X" means …`), lost PDF ligatures (`O\x00ce`), and definitions wrapped
mid-sentence. `mine.py` (`ambiguous_definition`) now rejects each pattern.
Re-mined (`mined_v2.jsonl`, 389 terms), it drops 11 of the 13 never-solved
tasks but only 12% of the always-solved ones, so it is removing bad tasks,
not hard ones. The remaining failures are real: quoting past the definition
into a `See …` note, dropping the opening quote mark.

Iteration 1 pool (`pool.py`, 20% held out by task_id hash at every difficulty):

| | defined_term | effective_date |
|---|---:|---:|
| train (in band) | 82 (74 train / 8 valid) | 0 |
| held out | 71, base pass 0.935 | 21, base pass 1.00 |

The base model already solves 0.935 of held-out definitions, so iteration 1
has ~6 points of headroom on this family and is a test of the loop, not a
result. A meaningful RL gain needs harder families: obligation spans,
citations, staggered/compliance dates.

The lesson is FrogNano's own stated gap — it never checked task quality by hand.
Read the failures before trusting a pass rate: two of three "model errors" here
were task errors.

## Seed gold for obligation spans (2026-10-13)

`seed.py` turns a human-reviewed seed file (630 records, rule-1/vnext-1
schema: verbatim `source_text` + `item_type` per regulatory item) into one
`obligation_span` task per record and item type. Licence comes from **our**
corpus: a record is admitted only if its spans occur in a GREEN document from
`sources.iter_docs`. Of 630 records, 349 are a script-filled batch
(`layer2_remaining8`: summary = source sentence, `applies_to` always empty)
and are skipped as silver; 144 (SEBI/RBI/IRDAI, licence undetermined, and
FCA, AMBER) are not in a GREEN document. Result: **24 HMT tasks** (10
obligation, 11 permission, 3 prohibition).

Known risk: the seed lists items per type but may not list *every* provision
of that type (a passage's second "may" can be unlisted), and span-set F1
then penalises a model for quoting it. Calibrate, then read the 0/8 tasks
before training on them.

## GRPO did not fit; rejection-sampling FT instead (2026-10-13)

The GRPO smoke run (group size 4, Qwen3.5-4B, ~1,050 tokens per rollout) took
265 s for its first step and then drove the 24 GB M4 Pro into critical memory
pressure; the job was killed. At that pace 200 updates are ~15 h even if memory
held. Iteration 1 therefore uses **rejection-sampling fine-tuning** (`rft.py`):
`calibrate.py --passing-out` samples k=8 answers per in-band training task and
keeps every verified-correct one; `rft build` caps distinct answers per task
(default 4, so 8/8 tasks do not dominate), re-verifies each, and refuses
held-out tasks; `rft train` runs LoRA SFT in `.venv-rl` (mlx-lm-lora applies
the Qwen3.5 recurrent patch). Still on-policy and still never trained on a gold
answer. Verified: the generation prompt is an exact token prefix of the training
text; mlx-lm-lora masks at the end of the user turn, so the loss also covers the
constant `<|im_start|>assistant\n<think>\n\n</think>\n\n` header, which is harmless.

**mlx-lm-lora's compiled step does not fit; run it eager.** The first RFT
run (2026-10-14) died on step 1 with a Metal `Insufficient Memory` error, even
at ~780 tokens where the earlier probe measured 10 GB. The difference was
`train_sft` wrapping its step in `mx.compile`. Same model, 800 tokens, 16 LoRA
layers, gradient checkpointing: eager 10.3 GB at 10 s/step, compiled out of
memory. `sft_eager.py` runs the stock CLI with only that module's compile
disabled, and `rft train` calls it. RFT examples are p50 782 / max 1,321
tokens; the longest train at 17.6 GB, ~14 s/step.

Peak memory follows sequence length, not LoRA depth (10.2 GB at 800 tokens and
17.2 GB at 1,320, for 8 and 16 layers alike), and after one long row MLX's
buffer cache kept that high-water mark: the first eager run swapped at ~100
s/step. So `rft build` drops rows over 1,024 tokens (8 of 179; truncating
would cut the answer), `sft_eager.py` caps the cache at 2 GB, and `rft train`
defaults to 8 layers (half the step time). Result: ~9.5 s/step, no swapping.

## Iteration 1 result (2026-10-14): RFT on Qwen3.5-4B

82 in-band training tasks, k=8 at temp 0.7 (the base calibration's settings):
179 distinct verified answers from 79 tasks (at most 4 per task), 171 after the
1,024-token cap (152 train / 19 valid, 76 tasks). One epoch, LoRA rank 8 on the
top 8 layers, lr 2e-5: 152 steps, 24 min, peak 12.1 GB. Val loss 0.390 -> 0.046.

Held out (92 tasks never trained on, same k and sampling as the base run),
paired by task, 10,000-sample bootstrap over tasks:

| family | tasks | base | RFT iter 1 | delta (95% CI) | tasks up / down |
|---|---:|---:|---:|---:|---:|
| `defined_term` | 71 | 0.935 | **0.967** | +0.032 (+0.009, +0.058) | 12 / 3 |
| `effective_date` | 21 | 1.000 | 1.000 | 0 | 0 / 0 |

Tasks solved 8/8 went from 56 to 64 of 71. The failures that shrank most were
quoting a different definition of the same passage. This is a real but small
gain on a family the base already nearly solves; it shows the loop works end to
end on the laptop (sample -> verify -> SFT -> re-score), not that RegNano is a
better compliance model. That needs the harder families and the triage /
LEDGAR regression checks from the plan.

Adapter: `models/regnano-rft-iter1` (mlx-lm-lora also writes a full model copy,
~8 GB, beside the 16 MB adapter; mlx-lm loads the adapter correctly from that
directory: 62 non-zero `lora_b` tensors).

## Deviations from FrogNano, recorded

- Iteration 1 runs with thinking **off**: answers are short JSON, and thinking
  multiplies laptop rollout time roughly tenfold.
- GRPO with group size 4-8 and LoRA, not DPPO at 256 trajectories per update.
- Mined gold for two families instead of an author model.
