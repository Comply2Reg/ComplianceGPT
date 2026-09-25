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

Results are in `data/uk/bench/compare_<task>.json`. See the **Results** section appended
below once the runs complete.

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
