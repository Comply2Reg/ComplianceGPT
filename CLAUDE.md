# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read this first: the branch decides everything

**Not on the live app request path.** `c2r_cga_api` still calls **OpenAI** for CGA, RegPulse
triage/classification, AskLia, etc. This repo is the replacement: parse/chunk regulatory text
and train a domain SLM for obligation extraction (and later those other tasks). Do not assume
the running UI/API already uses these models.

This repo holds **two unrelated codebases on two branches with no common ancestor**
(`git merge-base main feature/parser-v1` returns nothing). Before doing anything, run
`git rev-parse --abbrev-ref HEAD` and read the matching section below. Guidance for one branch is
actively wrong on the other.

| Branch | What it is | Layout |
|--------|-----------|--------|
| `feature/parser-v1` | **checked out in the local club** — Stage-1 document→chunk pipeline (Docling + UK CLML), with tests | flat `scripts/` + `tests/` |
| `feature/maitri/slm` | ancestor of `parser-v1`; earlier pypdf-based version of the same pipeline | flat `scripts/` |
| `main` | the SLM training half: notebooks, configs, dataset versions, and `scripts/ObligationDataPipeline/` | `notebooks/ configs/ data/ models/ scripts/ObligationDataPipeline/` |

`main` has **no `CLAUDE.md`** — if you check it out, this file disappears. It is committed on
`feature/parser-v1` only.

**Local club:** `~/Documents/17-Comply2Reg/15-SLM/ComplianceGPT`. Parent `17-Comply2Reg/` is not a
git repo; commit only here. Shared spine contract: `docs/platform-contract.md`
(canonical copy lives in `c2r_cga_api/docs/`). Cursor: `AGENTS.md` → `CLAUDE.md`.

---

# Branch `feature/parser-v1` (currently checked out)

Stage-1 of the data pipeline: take documents the crawler already put in MySQL
`regulation_documents` + S3, and turn them into **deterministic canonical text plus
offset-verifiable chunks**. It is strictly **read-only** against the shared spine — it never
writes the table or the bucket.

Active source: ESMA "Spotlight on Markets" newsletters (document ids 4424, 7673, 8162), plus a
standalone UK CLML (legislation.gov.uk XML) provision parser.

**UK alerts arrive pre-chunked.** `c2r-inventory-kit` captures the body text of every UK
regulatory alert and exports `canonical/{id}.txt` + `chunks.jsonl` in this repo's Stage-1 shape
(its `scripts/export_alert_corpus.py`). This repo validates that corpus rather than re-parsing it,
which is why `quality.py` now carries per-source profiles. The crawler's `normalize_whitespace`
is a byte-for-byte copy of `scripts/canonicalize.py` — if the two ever diverge, every exported
offset silently becomes wrong, and `test_content.py` over there asserts the parity.

## Commands

```bash
source .venv/bin/activate     # rebuild after a move: rm -rf .venv && python3 -m venv .venv && pip install -r requirements.txt
cp .env.example .env          # fill DB_* or MYSQL_*, AWS_*; never paste secrets into chat

pytest -q                                              # real test suite, fixture-backed, no DB/S3 needed
python scripts/run_pipeline.py --ids 4424 --dry-run    # inspect + ingest only
python scripts/run_pipeline.py --ids 4424              # inspect → ingest → chunk → validate
python scripts/run_pipeline.py --from-fixtures         # chunk + validate offline from tests/fixtures/docling/

python scripts/inspect_sources.py --ids 4424           # safe metadata only (values never printed)
python scripts/ingest_s3.py --ids 4424 [--dry-run] [--force]   # S3 → data/raw/
python scripts/chunk.py --ids 4424 [--append] [--allow-degraded]
python scripts/validate_data.py --ids 4424 --strict
python scripts/clml.py --xml tests/fixtures/legal/uk_clml_extract.xml
```

Run scripts as `python scripts/<x>.py` from the repo root. Modules import each other **flat**
(`from config import ...`, `from db import ...`), which works because each script inserts its own
directory on `sys.path` and `pytest.ini` sets `pythonpath = scripts`. `python -m scripts.chunk`
will not work.

## Pipeline

```
inspect_sources.py   read-only metadata from regulation_documents (db.py, never writes)
ingest_s3.py         read-only S3 download           → data/raw/{id}.{ext} + data/raw/manifest.json
chunk.py             Docling extraction + chunking   → data/canonical/{id}.txt, data/chunks.jsonl
validate_data.py     quality gates                   → pass/fail
```

`canonicalize.py` (pypdf/lxml) is the older path from `feature/maitri/slm` and is **not used for
ESMA PDFs** — `run_pipeline.py` skips it; Docling writes canonical text directly.

`scripts/fetch.py` is **quarantined** — importing or running it raises `SystemExit`, because it
used to overwrite `data/chunks.jsonl`. Use `scripts/experiments/fetch.py` to download XML and
`scripts/clml.py` to parse it.

## Invariants the gates enforce (`quality.py`)

Every chunk must carry: `chunk_id doc_id section text start end source_url doc_hash snapshot_id
content_sha extractor licence source_tier`.

- **G8 (provenance, always on):** `canonical[start:end] == text`, `doc_hash == sha256(canonical)`,
  `snapshot_id == doc_hash`, `content_sha == sha256(text)`. Offsets are the audit trail — any
  change to canonicalisation invalidates every stored chunk.
- **G1:** `extractor` must be one the source's profile allows. `--allow-degraded` relaxes this;
  don't use it to make a failing run pass.
- **G2 (`--strict`):** generic-section fraction ≤ 0.20 and ≥ 2 unique sections — catches an
  extractor that collapsed everything to `body`.
- **G4 (`--strict`):** no page-number-only lines, no promo lines, ≤ 1 running header.
- **Alert IDs are frozen and source-derived:** `esma:spotlight-{doc_id}#p{page}/{…}`. CLML IDs
  likewise (`ukpga:2000/8#s_19`, `uksi:2017/692#reg_8`) — walked from P1/P2/P3 structure, never
  from parse order. UK alert IDs are `{regulator}:{doc_type}#{doc_id}/{slug}`. Never renumber
  them; downstream references break silently.

### Source profiles

G1 and the chunk-id checks used to be hardcoded to ESMA (`esma:spotlight-` ids,
`extractor == "docling"`), which rejected every other source wholesale — including this repo's
own `clml.py` output. They are now per-source profiles in `quality.py`:

| profile | id shape | extractor | furniture gates |
|---|---|---|---|
| `esma_newsletter` (**default**) | `esma:spotlight-…#p…` | `docling` | running header, page numbers, promo |
| `uk_alerts` | `{regulator}:{doc_type}#{id}/…` | `c2r-inventory-kit/html`, `…/pdf` | none — HTML has no page numbers |
| `uk_legislation` | `ukpga:2000/8#s_19` | `clml` | none |

The default is unchanged, so existing callers and `test_docling_golden.py` behave exactly as
before. **G8, the required-field list and G2 apply to every profile** — a profile may only relax
publisher-specific furniture and the id/extractor shape, never the provenance gates.

```bash
python scripts/validate_data.py --source uk_alerts \
    --corpus-dir ../../09-inventory-kit/c2r-inventory-kit/data/alert_corpus \
    --strict --skip-raw
```

`--corpus-dir` points the canonical/chunks paths at a corpus outside `data/`; `--source` selects
the profile via `config.get_profile_name()`. A source with an empty `ids` list (`uk_alerts`)
validates whatever is present in `chunks.jsonl`.

## UK triage datasets (branch `feature/parser-v1`, `scripts/triage/`)

The UK corpus exported by `c2r-inventory-kit` (`data/alert_corpus/`: canonical
text, chunks, manifest with tier/class/stratum) is turned into fine-tuning data
here. `docs/uk-datasets.md` is the runbook; the short form:

```bash
python -m scripts.triage.weak_labels --corpus-dir <corpus>            # free baseline
python -m scripts.triage.labeller --corpus-dir <corpus> --tier GREEN  # OpenAI structured outputs, cached by doc_hash
python -m scripts.triage.build_dataset --labels data/uk/labels/triage_v1.jsonl --corpus-dir <corpus> --out data/uk/datasets/green --tier GREEN
python -m scripts.triage.validate_dataset --dataset-dir data/uk/datasets/green --strict
python -m scripts.triage.report --labels data/uk/labels/triage_v1.jsonl --weak data/uk/labels/weak_v1.jsonl
python -m scripts.triage.obligation_adapter --corpus-dir <corpus> --out data/uk/obligation/chunks_uk_v1.jsonl   # then main's run_extract.py from a worktree
```

- **Licence gate is code, not convention.** AMBER rows (FCA/PRA/BoE/DRCF/Ofcom,
  internal use only) are refused by the labeller and the adapter unless both
  `--allow-amber` and `--counsel-signoff` are given; AMBER datasets live only under
  `data/uk/datasets/internal/` (gitignored). GREEN alone has no guidance or
  enforcement documents, so the full 35/20/15/15/15 mix needs the sign-off.
- Training records keep the 7-key `ObligationRecord` envelope from `main` (so the
  obligation tooling reads them) plus `messages`/`n_tokens`; `classification` = alert
  class A1–A14, `output` = `TriageRecord` (never with a `status` key).
- Splits are stratified by stratum **and** date (latest two months = test, the
  gold candidates); `all_<v>.jsonl` = train + val and never contains test.
- `scripts/triage/taxonomy.py` transcribes `c2r-inventory-kit/docs/alert-taxonomy.md`
  and copies the exporter's `CLASS_TO_STRATUM`/`TARGET_MIX`; keep them identical.
- No regex fallback anywhere: a failed API call lands in `failures_<v>.jsonl`.
- **Labelling throughput is capped per model by the account's TPM, not by concurrency** —
  several processes on one model just collide. `--probe-limits` prints the ceiling,
  `--tpm` paces to it, `--shard i/N` splits the work, `--cache-only` merges the shards
  and names what is missing. Measured 2026-09-25: gpt-4o 30k TPM (~9 docs/min),
  gpt-4o-mini and gpt-5.4-mini 200k (~50).
- `config.LABEL_MODELS` holds the per-model call profile (these are the models that
  *produce labels*; `config.MODELS` is the model we *fine-tune*). The gpt-5 family needs
  `max_completion_tokens`, refuses `temperature`, and takes `reasoning_effort`
  (`none|low|medium|high|xhigh` — not the gpt-5.0 `minimal`); its reasoning tokens bill
  as output and count against the cap, so a truncated answer is failed, never cached.
- Two labelling models disagreeing is the gold-candidate signal: `report.py` writes
  `disputed_<v>.txt`, the strong model re-labels exactly those (`--ids-file`), and
  `--prefer-models` merges strongest-first.
- **Target model is `Qwen/Qwen3-4B-Instruct-2507`** (`config.MODELS`, fallback Gemma 4 E4B).
  Records carry `messages` and are rendered with the model's own `apply_chat_template`;
  `train_triage.py` trains response-only at 2048 tokens. The `main` notebook is not used
  for triage — it drops dict outputs and truncates at 1,024.
- On this Mac train with `train_mlx.py` (mlx-lm, 4-bit MLX weights, prompt-masked loss);
  `train_triage.py` is the CUDA path. Both score with `metrics.py`.
- `analyze_dataset.py` is the pre-training quality report (token lengths, rare classes,
  diversity, repetition); run it before every training run.
- `scripts/triage/` never imports `src.obligation_pipeline` from `main`
  (`tests/test_no_private_imports.py`); the branches have no common ancestor.

## Tests

`pytest -q` runs entirely off `tests/fixtures/` — no DB, no S3, no network.

- `test_docling_golden.py` pins chunk counts per fixture (4424→24, 7673→46, 8162→34) and asserts
  the gates. **If your change moves these numbers, that's a deliberate extractor change** — update
  the `GOLDEN` dict in the same commit and say why.
- `test_clml.py` — UK CLML provision parsing.
- `test_source_profiles.py` — the profile system: ESMA stays the default, UK chunks pass under
  `uk_alerts` and fail under `esma_newsletter`, and G8 holds for every profile.
- `test_no_private_imports.py` — no `hierarchical` imports may creep back into `scripts/`.

## Things that bite

- **`requirements.txt` is incomplete.** `db.py` needs `sqlalchemy` and `s3_client.py` needs
  `boto3`, neither of which is listed. `pytest` and `--from-fixtures` work without them; anything
  touching MySQL or S3 does not. Install them explicitly, or add them when you next touch the file.
- **`config.py` accepts three DB naming schemes** — `DATABASE_URL`, or a complete `MYSQL_*` set, or
  a complete `DB_*` set, checked in that order. A half-filled set produces a confusing "incomplete"
  error naming only the scheme it guessed. Same for buckets:
  `AWS_S3_BUCKET_REGU_LENS` → `AWS_S3_BUCKET` → `S3_BUCKET`.
- **Secrets discipline is deliberate.** `config.py` and `s3_uri.redact_s3_location` report env var
  *names* and redacted locations, never values. Keep it that way in new code.
- **Sources are hardcoded** in `config.SOURCES`. `esma_newsletter` (ids 4424/7673/8162) is the
  active one; `uk_alerts` and `uk_legislation` are registered with empty id lists because those
  documents arrive continuously. Adding a source means an entry there, not a CLI flag — and a
  matching profile in `quality.py` if its ids or extractor differ.
- **`chunk.py` only opens `{id}.pdf`.** `ingest_s3.py` will happily download `.html`/`.xml`/`.txt`
  from S3, but they dead-end there. UK alert text does not go through `chunk.py` at all — it is
  exported already chunked by the crawler and validated here.
- `data/raw/` is gitignored; `data/canonical/` and `data/chunks.jsonl` are committed as golden
  artifacts. `reference-parser/` is gitignored on purpose (company reference parser, do not publish).
- Docstrings citing "GraphRAG `GraphRag/src/ingestion/…`" are provenance notes — that repo is not
  in this club and is not a dependency.

---

# Branch `main` (the SLM / training half)

Domain-specialized SLM for **regulatory obligation extraction**: unstructured legal text →
structured JSON for GRC systems. Release v1 is a Gemma 4 E2B IT model fine-tuned with QLoRA,
published as `PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1` on Hugging Face.

Two loosely-coupled halves that meet only through a JSONL file format:

1. **Model side** (`notebooks/`, `configs/`, `data/`, `docs/`, `models/`) — training recipe,
   versioned configs, dataset artifacts. **No Python source at the repo root**; training runs
   entirely inside `notebooks/ComplianceGPT_v1_Training.ipynb` (Colab + GPU). `models/` is
   documentation only — weights live on Hugging Face.
2. **Data side** (`scripts/ObligationDataPipeline/`) — a self-contained package that mines
   regulatory PDFs and *produces* the fine-tuning dataset. The only runnable local code, with its
   own `requirements.txt`.

The contract between them is the `ObligationRecord` JSONL schema.

## Repository reality vs. README (branch `main`)

The root `README.md` is public-facing and describes files that do not exist:

- No root `requirements.txt`, `environment.yml`, `src/`, `benchmarks/`, or `LICENSE`, despite
  README instructions to `pip install -r requirements.txt` / `conda env create`.
- No test suite, linter, type-checker, or CI on this branch. Don't invent a test command; verify
  by running the relevant stage against a small `--limit`.
- `configs/v1/dataset_config.yaml` and `training_config.yaml` reference
  `data/v1/sample_dataset.jsonl`, which does not exist. The real training file is
  `data/v1/resampled_ft-v2.jsonl` (860 records); the samples present are the three
  `data/v1/sample_*.json` single-record files.
- The pipeline README advertises a 5-signal composite confidence score, tiers, and a
  `hallucination_flag`. **None of it is implemented.** `CONFIDENCE_W_*` in `config.py` is unused
  and `storage_vector.py` writes hardcoded `confidence_score: 1.0` / `confidence_tier: "unknown"`.

## Working in the ObligationDataPipeline

**Always run from `scripts/ObligationDataPipeline/`.** Modules import as
`from src.obligation_pipeline...` (absolute, rooted there) and paths like `data/output/` resolve
against the CWD. Running from the repo root fails on imports.

```bash
cd scripts/ObligationDataPipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill DB_*, AWS_*, OPENAI_API_KEY

python run_pipeline.py --version v1.0.0
python run_pipeline.py --version v1.0.0 --skip-ingest --skip-vector --skip-graph --limit 5

python run_ingestion.py                      # 1. MySQL metadata + S3 PDFs -> data/ingested/manifest.json
python run_parse.py --limit 10               # 2. PDFs -> legal hierarchy -> data/chunks/ + data/graph/
python run_extract.py --version v1.0.0       # 3. chunks -> OpenAI -> data/output/{raw,balanced,rejects}_<version>.jsonl
python run_ingest_vector.py --version v1.0.0 # 4. records -> ChromaDB
python run_ingest_graph.py --version v1.0.0  # 5. records -> Neo4j
```

To iterate without DB/S3, drop PDFs in `data/input_pdfs/`, run `python generate_local_manifest.py`,
then `run_parse.py --manifest data/output/local_test_manifest.json`.

Configuration is entirely `.env` via pydantic-settings (`src/obligation_pipeline/config.py`,
~60 settings with explicit `alias=`). Beyond `--version`, `--limit` and `--skip-*` there are no
CLI flags — to change chunk sizes, class ratios, models or DB targets, edit `.env`.

### Pipeline architecture

`pipeline.py` sequences five functions in `stages.py`; each is importable standalone. Heavy deps
(openai, chromadb, neo4j, sentence-transformers, PyMuPDF) are imported *inside* the stage functions
so partial runs work without the full dependency set.

- **Stage 2 holds the domain logic.** `parsing/hierarchy.py` (regex, `patterns.py`) detects Parts,
  Chapters, Sections, Subsections, Clauses, Provisos, Schedules. `ontology/graph_builder.py` turns
  that into typed nodes/edges from the closed vocabulary in `ontology/schema.py` (`EntityType`,
  `RelationType`, `SEMANTIC_ROLES`). `chunking/ontology_chunker.py` emits **one chunk per
  section/clause** — no token-window splitting; boundaries follow legal structure, so each chunk
  carries `section_number`, `clause_number`, `parent_path`, `semantic_role` and page provenance.
  `USE_LLM_STRUCTURE_PARSER=true` swaps the regex parser for an LLM pass
  (`parsing/llm_normalization.py` → `parsing/llm_bridge.py`) with automatic regex fallback.
- **Stage 3 always produces a record, even when the LLM fails.** `extraction.py` builds a
  rule-based record first (`_build_rule_record`, keyed on `shall|must` vs `may|can|should` vs
  neither), then tries OpenAI structured outputs; on exhausted retries it **silently returns the
  rule-based record**, whose obligations contain the literal placeholders `"Extracted Entity"` and
  `"Extracted Action"`. When auditing dataset quality, grep for those strings — they mark LLM
  failures, not real extractions. The LLM's `metadata` is overwritten post-parse to prevent
  hallucinated provenance.
- Stage 3 writes to a **hardcoded `data/output/`**, ignoring any settings-based output path.

## The record schema (branch `main`)

`src/obligation_pipeline/schema.py` (`ObligationRecord`) is the single source of truth, and its
central invariant is the **classification ↔ output-shape coupling**:

| classification | `output` type |
|---|---|
| `obligation` | **list** of `ObligationOutput` (obligation_id, subject, modality, modality_detected, action, conditions, deadline, reference_anchor) |
| `non_obligation`, `neutral` | **single** `NonObligationOutput` (status, message, reason) |

`validate.py` rejects any record that violates this. Every record also carries `metadata`,
`instruction`, `input_text`, `tool_use` (a *simulated* tool call — nothing is invoked; it trains
tool-augmented reasoning) and `thought_trace` (numbered reasoning steps).

The schema is duplicated in four places that must change together: `schema.py` (pydantic model),
`prompts.py` (worked examples in `SYSTEM_PROMPT`, which also encode the deontic-cue classification
rules), `validate.py` (shape check) and `data/v1/schema.md` (published spec). The notebook's own
`OBLIGATION_SCHEMA` jsonschema copy has already drifted — it uses `context`/`obligations` keys,
extra modalities (`MUST_NOT`, `ASPIRATIONAL_OBLIGATION`) and maps `input_text` → `context` on load.

## Versioning convention (branch `main`)

`configs/vN/`, `data/vN/` and a model release are pinned together: one dataset version ↔ one config
set ↔ one model. Starting v2 means copying the whole directory, not editing v1 — reproducibility of
published runs depends on v1 staying frozen. The six YAMLs in `configs/v1/` (training, lora,
dataset, generation, inference, mlflow) are **descriptive records, not loaded by any code** — the
notebook redefines the same values in its `cfg` dict, so editing a YAML alone changes nothing. Keep
them in sync by hand.

v1 facts: 2947 original samples resampled to 860 (50% obligation / 30% neutral / 20%
non_obligation, deliberately not the natural distribution), max_seq_length 1024, LoRA r=16
alpha=16 dropout=0.05, 4-bit NF4, 5 epochs, lr 1e-4, seed 3407, MLflow to local `./mlruns`. The
YAML lists LoRA targets as `q_proj/k_proj/v_proj` while the notebook also includes `o_proj`.

## Conventions (branch `main`)

- Model weights, `outputs/`, `mlruns/`, `**/data/input_pdfs/` and `**/data/output/` are gitignored.
  Committed pipeline data is limited to the small `data/chunks/chunks_ontology.jsonl` and
  `data/graph/` samples.
- `.gitignore` deliberately excludes three scratch scripts (`resample_ft_data.py`,
  `test_openai.py`, `test_schema.py`) — they may exist locally; don't add them back.
- Python uses `from __future__ import annotations`, PEP-604 unions, pydantic v2, module-level
  `logger = logging.getLogger(__name__)`, and section banners (`# ── Stage N: … ───`).
- Every directory (`data/`, `configs/`, `docs/`, `models/`, `notebooks/`) has its own README that
  is part of the published documentation — update it when the directory changes.

---

## Hard rules for agents (both branches)

- Commit only in this repo; `17-Comply2Reg/` is not a git repo.
- **Never write to `regulation_documents` or the S3 bucket from here** — this repo is a consumer.
  The crawler (`c2r-inventory-kit`) owns writes; see `docs/platform-contract.md`.
- Never merge `main` and `feature/parser-v1` — they share no history. Port code deliberately if
  it's needed on both sides.
- Never merge this remote into the API/UI/crawler remotes.
