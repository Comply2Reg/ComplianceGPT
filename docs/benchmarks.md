# Benchmarking the regulatory alert triage model

Run 2026-09-25 against `Comply2Reg/regulatory-alert-triage-qwen3-4b-v1` (fused, 4-bit).
Artefacts live under `data/uk/bench/` and `data/uk/datasets/green/rescore_v1.json`;
every number below is read from one of them.

Two different questions are being answered here, and conflating them would be a mistake.

1. **Is the headline number real?** Answered internally, by re-labelling the test set with
   a stronger model. This is the part that matters.
2. **Did fine-tuning cost the base model its general ability?** Answered by public
   benchmarks, where only the *delta against the base model* means anything.

No public benchmark scores UK regulatory alert triage. Nothing here is a leaderboard
entry, and the absolute numbers on the public sets should not be quoted as if they were.

---

## 1. The error bar on 0.644

The test split's labels were written by `gpt-5.4-mini`, the same model that wrote the
training labels. So the published accuracy measured agreement with one labeller, not
correctness. All 104 test documents were therefore re-labelled by `gpt-5.4`
(104 documents, 0 failures, $2.44) and the **same saved predictions** were scored again.

| Metric | vs `gpt-5.4-mini` | vs `gpt-5.4` | Change |
|---|---:|---:|---:|
| JSON validity | 1.000 | 1.000 | 0.000 |
| alert_class accuracy | 0.644 | 0.567 | −0.077 |
| alert_class macro-F1 | 0.245 | 0.155 | −0.090 |
| priority accuracy | 0.760 | 0.798 | +0.039 |
| obligations_present | 0.913 | 0.942 | +0.029 |
| primary_functions Jaccard | 0.574 | 0.780 | +0.206 |

And the two labellers against each other, on those same 104 documents:

| Field | Labeller agreement |
|---|---:|
| alert_class | 0.702 |
| priority | 0.856 |
| obligations_present | 0.933 |
| primary_functions (Jaccard) | 0.617 |

### What this means

**The label-noise ceiling on alert_class is about 0.70.** Two capable models, given the
same taxonomy and the same documents, agree seven times in ten. A model cannot
meaningfully score above that against either of them, because a third of the disagreement
is in the labels rather than the answer.

**So 0.644 is not "64% correct".** It is 64% agreement with a labeller whose own
reliability against a stronger peer is 70%. Read as a fraction of the achievable ceiling,
the model recovers roughly nine tenths of it. The honest statement of the headline is a
band: **0.57 to 0.64 depending on which labeller is treated as truth**, against a ceiling
near 0.70.

**Three of the six fields got better under the stronger judge, not worse.** Priority,
obligations and especially function assignment (+0.206 Jaccard) agree *more* with
`gpt-5.4` than with the labeller the model was trained on. The model did not simply
memorise its teacher on those fields; it generalised past it. Only class assignment is
anchored to the training labeller's idiosyncrasies, which is what one would expect from
the field with the largest label space and the noisiest labels.

**The macro-F1 drop to 0.155 is the number to worry about.** It confirms that the
rare-class weakness is real and not an artefact of one labeller's preferences.

### Where the two labellers disagree

Top confusions, `gpt-5.4-mini` → `gpt-5.4`: A3→A10 (6), A11→A6 (6), A11→A10 (2),
A11→A14 (2), A9→A11 (2). These are exactly the boundaries the model gets wrong too:
consultation versus thematic review, intelligence versus enforcement. The taxonomy is
genuinely ambiguous at those edges, and no amount of training fixes a label definition
problem.

### What would actually fix this

Human adjudication. `data/uk/gold/gold_candidates_v1.csv` already holds 995 rows with
empty `human_alert_class` columns, and `build_dataset.py --apply-gold` already consumes
them. Note that path feeds gold labels into **training**; scoring against them needs
`rescore.py --judge-labels` pointed at a labels file built from the sheet.

---

## 2. Public benchmarks

Run on the fine-tuned model and on the untouched base
(`mlx-community/Qwen3-4B-Instruct-2507-4bit`), same seeded sample, same prompts.

| Task | Dataset | What it probes |
|---|---|---|
| `ledgar` | LexGLUE LEDGAR, 100 classes | legal document classification, single-label |
| `unfair_tos` | LexGLUE UNFAIR-ToS, 8 classes | legal classification, multi-label |
| `jsonschema` | JSONSchemaBench `Github_easy` | does the JSON skill generalise past one schema |
| `obliqa` | RegNLP ObliQA (ADGM) | financial-regulatory text, behavioural probe |
| `legalbench` | LegalBench via lm-eval | legal reasoning retention |

Results are in `data/uk/bench/compare_<task>.json`.

### Results

| Task | Metric | Base | Tuned | Change |
|---|---|---:|---:|---:|
| `ledgar` | accuracy | 0.610 | 0.418 | **−0.192** |
| | answered_rate | 0.980 | 0.846 | −0.134 |
| | accuracy_when_answered | 0.622 | 0.494 | −0.128 |
| `unfair_tos` | micro-F1 | 0.172 | 0.146 | −0.025 |
| | micro-precision | 0.095 | 0.080 | −0.016 |
| | micro-recall | 0.873 | 0.909 | +0.036 |
| `jsonschema` | JSON validity | 0.960 | 0.970 | +0.010 |
| | schema conformance | 0.891 | 0.830 | **−0.061** |
| `obliqa` | produced triage JSON | 0.015 | 0.990 | **+0.975** |

### Fine-tuning cost real legal ability, not just output format

LEDGAR is the clearest read. The base model classifies contract provisions into 100 classes
at 0.610 accuracy; ours manages 0.418. Format lock explains part of it, since the model
declines to give a usable label on 15% of items where the base model gave one. But it does
not explain all of it: **even counting only the items where both produced a label, accuracy
fell from 0.622 to 0.494**. That is capability loss, not formatting.

This is the documented failure mode of LoRA fine-tuning without replay data, and 1,700
iterations of a single narrow JSON task on 1,683 examples is exactly the recipe for it. If a
v2 is trained, mixing a few hundred general instruction-following examples into the training
set is the standard mitigation.

### It learned one schema, not schema-following

On JSONSchemaBench the model is marginally *better* at emitting syntactically valid JSON
(+0.010) and meaningfully *worse* at satisfying a schema it was given (−0.061). It did not
learn to follow schemas. It learned to emit one particular shape, and reaches for that shape
even when another is specified.

### The most serious finding: silent failure on out-of-distribution regulation

ObliQA passages are Abu Dhabi Global Market rulebook text. The model had never seen a non-UK
regulator. It produced well-formed triage JSON on 198 of 200 passages, against the base
model's 3 of 200, so the trained skill transfers as *format*.

The content does not transfer at all:

| | Result |
|---|---|
| `alert_class` | A11 on **198 of 198** |
| `priority` | P3 on **198 of 198** |
| `obligations_present` true | **1 of 198** |

A11 is "intelligence: speech, blog, press release"; P3 is "awareness only". Applied to a
financial regulator's rulebook, every one of those is wrong, and ADGM rulebook passages are
obligation-bearing by construction.

Worse, it is not visibly wrong. One passage got the summary *"ADGM sets out the requirements
for authorized persons to have adequate backup systems and arrangements to maintain essential
operations during a disaster"* — a correct reading of an obligation — alongside
`alert_class: A11`, `priority: P3`, `obligations_present: false`. The prose understood the
text; every structured field contradicted it.

**So the model collapses to its majority class on regulatory text outside its training
distribution, and does so silently.** Nothing downstream could detect this from the output
alone: the JSON is valid, the schema is satisfied, the summary is sensible. Any deployment
outside UK GREEN-tier sources needs its own evaluation before it is trusted, and a
distribution check in front of the model would be a reasonable safeguard.

### A caveat on unfair_tos

The delta there is weak evidence and should not be leaned on. 449 of the 500 sampled clauses
carry no label at all, so `exact_set_match` is mostly free credit for two empty sets and the
micro-F1 rests on just **51 informative rows**. Both models over-predict on those rows,
returning a mean of 2.9 labels against a gold mean of 1.1. `bench.py` now reports
`n_with_labels` so this is visible in the artefact rather than needing to be rediscovered.

### Expect low absolute scores, and read the delta

The model is trained to answer every prompt with a triage record. Asked to pick one of
LEDGAR's 100 contract-provision labels, it tends to answer with a triage record instead.
That is **format lock**, and it is a real property worth measuring rather than prompting
away. `bench.py` therefore reports `answered_rate` and `accuracy_when_answered` alongside
raw accuracy, so "wrong" is separated from "did not produce a usable label at all", and
it looks inside a returned JSON object for a label before giving up.

### Two datasets considered and rejected

- **LexGLUE EUR-LEX.** Its labels are bare EuroVoc concept identifiers (`100163`), which a
  zero-shot generative model cannot name. The score would measure nothing. UNFAIR-ToS
  covers the multi-label case with readable class names instead.
- **FinBen.** No individually addressable dataset on the Hub, and its tasks are QA,
  sentiment and numeric reasoning rather than document classification. ObliQA carries the
  financial-regulatory slot.

Note also that `main`'s training notebook references `rcraigfieldwork/ObliQA`, which does
not exist. That cell has been silently falling back to two synthetic samples. The real
dataset is `RegNLP/ObliQA`.

### MMLU and IFEval

Not run, by choice. `lm-eval` is installed for LegalBench, so adding
`--tasks ifeval mmlu` to the `mlx_lm evaluate` command would cover general-capability
regression if that becomes interesting.

---

## v2: retrained on the multi-jurisdiction corpus, 2026-09-27

4,114 training examples across GB and US against v1's 1,683, one epoch, 4,100
iterations, 5h48m on the M4 Pro. Validation loss 2.777 → 0.699 at its best.

**The headline is macro-F1, which nearly doubled.** That was the metric worth
quoting for v1 and it is the one that moved.

| Metric | v1 (UK, n=104) | v2 (UK+US, n=759) | Change |
|---|---:|---:|---:|
| alert_class macro-F1 | 0.244 | **0.431** | **+0.187** |
| alert_class accuracy | 0.644 | 0.656 | +0.012 |
| primary_functions | 0.574 | 0.584 | +0.009 |
| JSON validity | 1.000 | 0.976 | −0.024 |
| obligations_present | 0.913 | 0.851 | −0.062 |
| priority | 0.760 | 0.677 | −0.082 |

**The test sets are not the same**, so this is not a controlled comparison. v2
is measured on 759 documents spanning two jurisdictions; v1 on 104 UK ones.
The v2 split is broader and harder, which makes the macro-F1 gain more
convincing and the accuracy gain less so.

**Classes the model can actually do went from four to eight.** The new ones are
exactly those that had no training data before.

| class | v1 | v2 |
|---|---|---|
| A2 final rules | — | 32/67 |
| A7 sanctions | — | 29/33 |
| A8 reporting | 0/1 | 45/77 |
| A14 perimeter | 0/5 | 33/60 |
| A3 consultations | 5/7 | 174/239 |
| A6 enforcement | 5/6 | 36/45 |
| A11 intelligence | 52/56 | 148/184 |

A7 at 29/33 is the standout. Still zero: A4, A5, A9, A12, A13. A10 regressed
from 5/17 to 0/9, which is a real loss and the one result here that argues
against simply adding more data.

### The out-of-distribution probe: improved, not solved

This was the sharpest v1 failure — on 198 Abu Dhabi rulebook passages it
emitted perfect JSON and assigned A11 to every single one, flagging an
obligation in 1 of 198, on text that is obligation-bearing by construction.

| | base | v1 | v2 |
|---|---:|---:|---:|
| produced triage JSON | 0.015 | 0.990 | 1.000 |
| flagged an obligation | 0.0% | 0.5% | **36.5%** |
| distinct classes used | 1 | 1 | 3 |
| majority class share | — | 100% | 88% |

**The obligation collapse is substantially fixed** — 0.5% to 36.5%. **The class
collapse is only partly fixed**: still 88% A11, but the model now reaches for
A1 (legislation, 16) and A2 (final rule, 9), both of which are plausible
readings of rulebook text where A11 never was.

So more data and a jurisdiction field moved this a long way and did not finish
the job. Treating non-UK, non-US regulation as supported still needs either
training data from that jurisdiction or an explicit distribution check in front
of the model.

---

## Reproducing

```bash
# 1. the error bar
python -m scripts.triage.labeller --corpus-dir <corpus> --out data/uk/labels \
    --version judge1 --tier GREEN --model gpt-5.4 --ids-file data/uk/labels/test_ids_v1.txt
python -m scripts.triage.rescore --dataset-dir data/uk/datasets/green --version v1 \
    --eval-outputs data/uk/datasets/green/eval_v1_qwen3-4b-instruct_mlx_regulatory-alert-triage-qwen3-4b-v1_outputs.jsonl \
    --judge-labels data/uk/labels/triage_judge1.jsonl

# 2. public benchmarks, both models, then the delta
python -m scripts.triage.bench list
python -m scripts.triage.bench run --task ledgar --limit 500                    # base
python -m scripts.triage.bench run --task ledgar --limit 500 \
    --model-path models/publish/regulatory-alert-triage-qwen3-4b-v1             # tuned
python -m scripts.triage.bench compare --task ledgar

# 3. tasks lm-eval already ships
python -m mlx_lm evaluate --model models/publish/regulatory-alert-triage-qwen3-4b-v1 \
    --tasks legalbench --apply-chat-template --output-dir data/uk/bench/legalbench_tuned
```

`bench.py --dry-run` builds and prints prompts without loading a model, which is the
cheap way to check an adapter before a long run. Sampling is seeded (`--seed 42`), so the
two sides of a comparison see identical rows; `compare` refuses to run if they did not.
