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

# 4. train: in notebooks/ComplianceGPT_v1_Training.ipynb (on main) set
#    cfg["data_path"] = ".../data/uk/datasets/green/all_v1.jsonl"

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

## Record format

The 7-key envelope of `main`'s `ObligationRecord`, so the Gemma notebook's
`load_and_map_jsonl` / `format_prompt` train it unchanged:

| key | triage |
|---|---|
| `metadata` | `source_id` (`{regulator}:{doc_type}#{id}`), `regulation_name`, `jurisdiction` GB, `doc_type`, `section` "document", plus `regulator`, `release_date`, `stratum`, `stratum_source`, `doc_hash`, `tier`, `licence`, `label_source`, `model`, `prompt_version` |
| `classification` | alert class A1–A14 |
| `instruction` | one of three templates in `prompts.py` |
| `input_text` | regulator, type, title, date + first 2,000 chars of canonical text |
| `tool_use` | `null` |
| `thought_trace` | the labeller's `rationale` |
| `output` | `TriageRecord` minus rationale — never a `status` key |

Splits are stratified by stratum **and** date: the latest two months of each
stratum are `test` (the gold candidates), the next 10 % by date `val`, the rest
`train`; only train is balanced towards 35/20/15/15/15 (rule_final taken whole,
nothing oversampled). `all_<v>.jsonl` = train + val; test never enters it.

## Known caveats

- `main`'s `run_extract.py` falls back silently to a regex labeller on API
  failure; `validate_dataset.py --task obligation` flags those records
  (`Extracted Entity`). Drop them before training.
- HM Treasury rows are whole-of-Treasury; the labeller's class and functions
  are the only relevance filter today.
- 960 documents were unclassified at capture (MIXED sources); their stratum
  comes from the model class (`stratum_source: model`) and they are
  over-represented in the gold sheet on purpose.
