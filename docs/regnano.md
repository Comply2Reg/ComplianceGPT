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

## Task families

| family | gold from | verifier |
|---|---|---|
| `effective_date` | Federal Register `DATES:` paragraph: exactly one effective date, or `null` for comment-only proposals | date equality; `null` only when gold is null |
| `defined_term` | `“X” means …` running to end of line, unique in a window holding other definitions | verbatim span (typography- and whitespace-insensitive), overlap >= 0.8 |
| `obligation_span` | author model (next) | every span verbatim, span-set F1 >= 0.8, modality match |
| `citation` | author model / legislation.gov.uk changes feed (next) | normalised ID equality (`12 C.F.R. § 1026.19` == `12 CFR 1026.19`) |
| `triage_class` | agreed labels or human gold only | exact class |

```bash
python -m scripts.regnano.mine --out data/regnano/tasks/mined_v1.jsonl   # 909 date + 473 term tasks, all GREEN
python -m scripts.regnano.calibrate --tasks <tasks> --out <calibration.jsonl> --k 8 --keep-out <in-band.jsonl>
python -m scripts.regnano.rl export --tasks <in-band.jsonl> --out data/regnano/grpo/iter1
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

The lesson is FrogNano's own stated gap — it never checked task quality by hand.
Read the failures before trusting a pass rate: two of three "model errors" here
were task errors.

## Deviations from FrogNano, recorded

- Iteration 1 runs with thinking **off**: answers are short JSON, and thinking
  multiplies laptop rollout time roughly tenfold.
- GRPO with group size 4-8 and LoRA, not DPPO at 256 trajectories per update.
- Mined gold for two families instead of an author model.
