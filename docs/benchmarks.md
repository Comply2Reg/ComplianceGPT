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

> **Superseded — do not read this table as a v1→v2 delta.** v1 has since been
> scored on v2's own 759-document split, and the three apparent regressions in
> priority, obligations and JSON validity are artefacts of the differing splits:
> priority is genuinely **+0.246**, not −0.082. See *The controlled v1-vs-v2
> comparison* below.

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

A7 at 29/33 is the standout. Still zero: A4, A5, A9, A12, A13.

> **The A10 claim below was wrong.** Measured on one split, v1 scores 2/9 and v2
> 0/9, and v1 got there with 40 A10 predictions (precision 0.05). It is a
> two-document difference between two models that both fail the class, not a
> capability v2 lost.

~~A10 regressed from 5/17 to 0/9, which is a real loss and the one result here
that argues against simply adding more data.~~

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

## The controlled v1-vs-v2 comparison, 2026-09-27

The v2 section above compares two models on two different test sets and says so.
This is the same comparison done properly: **v1's published fused weights scored
on v2's own 759-document test split**, which it has provably never seen
(`train_v1 ∩ test_v2` is empty — check it by `doc_hash`, not `document_id`,
because the UK and US exports share 5,989 ids). No retraining, one eval run,
1h02m.

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| alert-class accuracy | 0.302 | **0.656** | **+0.354** |
| macro-F1, all classes | 0.117 | **0.431** | **+0.314** |
| macro-F1, gold ≥ 20 | 0.223 | **0.720** | **+0.497** |
| weighted F1 | 0.225 | **0.650** | **+0.425** |
| priority accuracy | 0.431 | **0.677** | **+0.246** |
| obligations_present | 0.816 | **0.851** | **+0.036** |
| primary_functions Jaccard | 0.311 | **0.584** | **+0.273** |
| JSON validity | 0.991 | 0.976 | −0.015 |

**Three of the four "v2 regressions" were artefacts of the split, not results.**
Priority was reported as −0.082 and is actually **+0.246**; obligations_present as
−0.062 and is actually **+0.036**. Only JSON validity really fell, by 0.015 — two
malformed records. Comparing across test sets did not merely blur the picture, it
inverted the sign on two metrics. Do not quote the cross-split table above as a
v1-to-v2 delta.

**A10 was not a capability loss.** The ranked worry was "A10 regressed 5/17 → 0/9,
a class v1 could do and v2 cannot". On the same 759 documents v1 scores 2/9 and
v2 0/9. v1 reached that 2 by predicting A10 **40 times** — precision 0.05. It was
spraying the label, not recognising the class, and the difference between the two
models is two documents.

**What v2 actually fixed is the head-class collapse, halfway.** v1 answered A11
for **606 of 759 documents (80%)** and used 7 classes at all; v2 answers A11 322
times (42%) and uses 8. v1's headline 0.644 accuracy on its own 104-document UK
split is 0.302 here — the out-of-distribution weakness, quantified on documents
it was never asked about.

### Read macro-F1 with its support

`alert_class_macro_f1` averages every class the split touches, and this split has
A1 with 1 gold item, A4 with 2 and A13 with 3. One prediction on a 1-item class
moves the headline by ~7 points. `metrics.py` now also emits
`macro_f1_by_min_support` and `weighted_f1`:

| gold support floor | classes | v2 macro-F1 |
|---|---:|---:|
| ≥ 1 (the headline) | 14 | 0.431 |
| ≥ 5 | 11 | 0.458 |
| ≥ 10 | 9 | 0.560 |
| ≥ 20 | 7 | **0.720** |

Both figures are honest and they answer different questions: 0.431 is "how does
it do across the whole taxonomy including classes we barely have", 0.720 is "how
does it do where the test set can actually measure it". Quote the pair.

### Six classes are never emitted at all

A4, A5, A9, A10, A12 and A13 have `pred = 0` — not low precision, *no predictions
whatsoever*. For 35 of those 53 gold documents v2 answers A11 or A3, the two head
classes. Two causes, both fixable and neither of them corpus size:

1. **The train mix.** v2 balanced towards `TARGET_MIX`, which is stratum-shaped
   and orthogonal to class, and so trained on 30% A11 and 6 A5 examples. See the
   v3 section of `docs/uk-datasets.md`.
2. **The training prompt never defines the taxonomy.** `prompts.SYSTEM_PROMPT`
   (5,228 chars, all 14 classes described) is what the *labeller* sends to OpenAI.
   `render.TRAIN_SYSTEM_PROMPT`, the system turn of every training record, is 74
   tokens and names the classes only as the range "A1-A14". The model has to infer
   14 class meanings from examples alone, and has no definitions at inference
   either — which is also the most likely reason it collapses to A11 on
   out-of-distribution text. A compact one-line-per-class taxonomy costs 208
   tokens against a 2,048 budget where train p95 is 1,023.

## v3: class balance plus the taxonomy in the prompt, 2026-09-27

3,194 training examples (v2: 4,114), one epoch, 3,200 iterations, 4h23m on the
M4 Pro. Val loss 2.691 to 0.684, improving monotonically from iter 1000; v2's
best was 0.699 at iter 2000 and then worsened. Evaluated as the **adapter**, the
way v2 was, on the same 759 held-out documents.

| Metric | v1 | v2 | v3 | v3 - v2 |
|---|---:|---:|---:|---:|
| alert-class accuracy | 0.302 | 0.656 | **0.693** | +0.037 |
| priority accuracy | 0.431 | 0.677 | **0.715** | +0.038 |
| obligations_present | 0.816 | 0.851 | **0.864** | +0.013 |
| JSON validity | 0.991 | 0.976 | **0.995** | +0.018 |
| weighted F1 | 0.225 | 0.650 | **0.676** | +0.026 |
| macro-F1, gold >= 20 | 0.223 | 0.720 | 0.717 | -0.003 |
| primary_functions Jaccard | 0.311 | 0.584 | 0.572 | -0.012 |
| macro-F1, all classes | 0.117 | 0.431 | 0.411 | **-0.020** |

**The macro-F1 fall is one document.** A1 has a single gold item in this split.
v2 happened to get it right (F1 1.000), v3 got it wrong (F1 0.000), and one of
fourteen classes swinging the full range moves macro-F1 by 1/14 = 0.071.
Excluding A1, macro-F1 goes **0.388 to 0.443, up 0.055**. This is the exact
pathology `macro_f1_by_min_support` exists to expose, and it is worth noting the
metric misled in v2's favour this time and against it last time.

**The head-class collapse is fixed, and overshot.**

| | v1 | v2 | v3 | gold |
|---|---:|---:|---:|---:|
| distinct classes predicted | 7 | 8 | **12** | 14 |
| A11 predictions | 606 (80%) | 322 (42%) | **82 (11%)** | 184 (24%) |

A11 precision went 0.46 to 0.95 while recall fell 0.80 to 0.42. Capping A11 at
400 from 1,223 did not just remove the bias, it reversed it: the model now
under-predicts A11 by more than half.

**Four of the six silent classes now emit.**

| class | gold | v2 pred | v3 pred | v3 F1 |
|---|---:|---:|---:|---:|
| A10 | 9 | 0 | 13 | **0.545** |
| A4 | 2 | 0 | 19 | 0.191 |
| A12 | 15 | 0 | 14 | 0.000 |
| A13 | 3 | 0 | 1 | 0.000 |
| A5 | 9 | 0 | 0 | 0.000 |
| A9 | 15 | 0 | 0 | 0.000 |

A10 going 0.000 to 0.545 is the clearest single result here: it was the class
ranked as the reason not to add data, and rebalancing recovered it outright. A12
and A13 now produce predictions but hit nothing, so they have learned that the
class exists without learning what belongs in it. **A5 and A9 are still
completely silent** at 19 and 49 training examples, which is the floor below
which a class does not survive.

**The imbalance moved rather than went away.** Predictions over gold, v3:

| class | pred / gold |
|---|---:|
| A4 | 9.50 |
| A14 | 1.98 |
| A8 | 1.48 |
| A11 | 0.45 |
| A2 | 0.42 |

A14 went from 167 to 400 training examples and its precision fell 0.80 to 0.46;
A8 went 195 to 400 and fell 0.73 to 0.60. A single flat cap ignores a class's
true base rate, so it over-represents whatever was previously scarce. A v4 should
either set the cap per class against the observed frequency, or leave the data
alone and weight the loss instead, which changes the gradient without throwing
away 2,164 labelled documents.

**Net.** Five metrics up, one flat, one down by 0.012, and the headline down only
because of one document. The two changes did what they were aimed at. Neither is
finished: the taxonomy is in the prompt but A5 and A9 still have no data, and the
cap traded one skew for a smaller one.

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
