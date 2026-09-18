# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

ComplianceGPT is a domain-specialized SLM project for **regulatory obligation extraction**: turning
unstructured legal/regulatory text into structured JSON for GRC systems. Release v1 is a
Gemma 4 E2B IT model fine-tuned with QLoRA, published at
`PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1` on Hugging Face.

The repo has **two loosely-coupled halves** that meet only through a JSONL file format:

1. **Model side** (`notebooks/`, `configs/`, `data/`, `docs/`, `models/`) — the training recipe,
   versioned configs, dataset artifacts, and documentation. There is **no Python source at the repo
   root**; training runs entirely inside `notebooks/ComplianceGPT_v1_Training.ipynb` (built for
   Google Colab + GPU). `models/` holds documentation only — weights live on Hugging Face.
2. **Data side** (`scripts/ObligationDataPipeline/`) — a self-contained Python package that mines
   regulatory PDFs and *produces* the fine-tuning dataset. It is the only part of the repo with
   runnable local code and its own `requirements.txt`.

The contract between them is the `ObligationRecord` JSONL schema (see "The record schema" below).

## Repository reality vs. README

The root `README.md` is a public-facing document and describes files that do not exist. Do not
assume they are missing by accident:

- There is **no root `requirements.txt`, `environment.yml`, `src/`, `benchmarks/`, or `LICENSE`**,
  despite README instructions to `pip install -r requirements.txt` / `conda env create`.
- There is **no test suite, linter config, type-checker config, or CI**. There is nothing to run
  for "build" or "lint". Don't invent a test command; if verification is needed, run the relevant
  stage script against a small `--limit`.
- `configs/v1/dataset_config.yaml` and `training_config.yaml` reference
  `data/v1/sample_dataset.jsonl`, which does not exist. The real training file is
  `data/v1/resampled_ft-v2.jsonl` (860 records); the samples present are the three
  `data/v1/sample_*.json` single-record files.
- The pipeline's own README advertises a 5-signal composite confidence score, tiers, and a
  `hallucination_flag`. **None of that is implemented.** `CONFIDENCE_W_*` settings in `config.py`
  are unused, and `storage_vector.py` writes hardcoded `confidence_score: 1.0` /
  `confidence_tier: "unknown"`.

## Branches

- `main` — everything described above.
- `origin/feature/maitri/slm` — an **unrelated history** (`git merge-base` fails). It is a separate
  Stage-1 data pipeline for ESMA newsletters built on Docling, with a flat `scripts/` layout
  (`inspect_sources.py` → `ingest_s3.py` → `canonicalize.py` → `chunk.py` → `validate_data.py`,
  orchestrated by `scripts/run_pipeline.py --ids 4424`). It shares the `regulation_documents` MySQL
  table and S3 conventions with `main`'s pipeline but shares no code or commits. A normal merge is
  not possible; treat it as a parallel experiment, not a feature to merge.

## Working in the ObligationDataPipeline

**Always run from `scripts/ObligationDataPipeline/`.** Modules import each other as
`from src.obligation_pipeline...` (absolute, rooted at that directory), and output paths like
`data/output/` are resolved relative to the CWD. Running from the repo root will fail on imports.

```bash
cd scripts/ObligationDataPipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in DB_*, AWS_*, OPENAI_API_KEY
```

Full 5-stage run, and the individual stage runners (each stage is independently re-runnable if its
inputs exist):

```bash
python run_pipeline.py --version v1.0.0
python run_pipeline.py --version v1.0.0 --skip-ingest --skip-vector --skip-graph --limit 5

python run_ingestion.py                      # 1. MySQL metadata + S3 PDFs -> data/ingested/manifest.json
python run_parse.py --limit 10               # 2. PDFs -> legal hierarchy -> data/chunks/chunks_ontology.jsonl + data/graph/{nodes,edges}.jsonl
python run_extract.py --version v1.0.0       # 3. chunks -> OpenAI -> data/output/{raw,balanced,rejects}_<version>.jsonl
python run_ingest_vector.py --version v1.0.0 # 4. records -> ChromaDB
python run_ingest_graph.py --version v1.0.0  # 5. records -> Neo4j
```

To iterate without DB/S3 access, drop PDFs in `data/input_pdfs/` and run
`python generate_local_manifest.py`, then pass the generated manifest to
`run_parse.py --manifest data/output/local_test_manifest.json`.

Everything is configured through `.env` via pydantic-settings (`src/obligation_pipeline/config.py`,
~60 settings with explicit `alias=` names). There are no CLI flags for configuration beyond
`--version`, `--limit`, and the `--skip-*` switches — to change chunk sizes, class ratios, models,
or DB targets, edit `.env`.

### Pipeline architecture

`pipeline.py` sequences five functions in `stages.py`; each stage is also importable standalone.
Heavy dependencies (openai, chromadb, neo4j, sentence-transformers, PyMuPDF) are imported *inside*
the stage functions so partial runs work without the full dependency set installed.

- **Stage 2 is where the domain logic lives.** `parsing/hierarchy.py` (regex, `patterns.py`) detects
  the legal hierarchy — Parts, Chapters, Sections, Subsections, Clauses, Provisos, Schedules.
  `ontology/graph_builder.py` turns that hierarchy into typed nodes/edges from the closed vocabulary
  in `ontology/schema.py` (`EntityType`, `RelationType`, `SEMANTIC_ROLES`). `chunking/
  ontology_chunker.py` then emits **one chunk per section/clause** — there is no token-window
  splitting; boundaries follow the legal structure so each chunk carries `section_number`,
  `clause_number`, `parent_path`, `semantic_role`, and page provenance. Setting
  `USE_LLM_STRUCTURE_PARSER=true` swaps the regex parser for an LLM pass
  (`parsing/llm_normalization.py` → `parsing/llm_bridge.py`) with automatic fallback to regex.
- **Stage 3 always produces a record, even when the LLM fails.** `extraction.py` builds a
  rule-based record first (`_build_rule_record`, keyed on `shall|must` vs `may|can|should` vs
  neither), then tries OpenAI structured outputs; on exhausted retries it **silently returns the
  rule-based record**, whose obligations contain the literal placeholders `"Extracted Entity"` and
  `"Extracted Action"`. When auditing dataset quality, grep for those strings — they indicate LLM
  failures, not real extractions. The LLM's `metadata` fields are also overwritten post-parse to
  prevent hallucinated provenance.
- Stage 3 writes to a **hardcoded `data/output/`**, ignoring any settings-based output path.

## The record schema

`src/obligation_pipeline/schema.py` (`ObligationRecord`) is the single source of truth, and its
central invariant is the **classification ↔ output-shape coupling**:

| classification | `output` type |
|---|---|
| `obligation` | **list** of `ObligationOutput` (obligation_id, subject, modality, modality_detected, action, conditions, deadline, reference_anchor) |
| `non_obligation`, `neutral` | **single** `NonObligationOutput` (status, message, reason) |

`validate.py` rejects any record that violates this. Every record also carries `metadata`,
`instruction`, `input_text`, `tool_use` (a *simulated* tool call — no tool is actually invoked;
it exists to train tool-augmented reasoning), and `thought_trace` (numbered reasoning steps).

This schema is duplicated in four places that must be changed together:
`schema.py` (pydantic model), `prompts.py` (the worked examples in `SYSTEM_PROMPT`, which also encode
the deontic-cue classification rules), `validate.py` (the shape check), and `data/v1/schema.md`
(the published spec). The notebook has its own `OBLIGATION_SCHEMA` jsonschema copy that has already
drifted (it uses `context`/`obligations` keys and extra modalities like `MUST_NOT`, `ASPIRATIONAL_OBLIGATION`,
and maps `input_text` → `context` on load).

## Versioning convention

`configs/vN/`, `data/vN/`, and a model release are pinned together: one dataset version ↔ one config
set ↔ one model. When starting v2, copy the whole directory rather than editing v1 in place —
reproducibility of published runs depends on v1 staying frozen. The six YAML files in `configs/v1/`
(training, lora, dataset, generation, inference, mlflow) are **descriptive records, not loaded by any
code** — the notebook redefines the same values in its `cfg` dict, so changing a YAML alone changes
nothing. Keep them in sync manually.

Key v1 facts: 2947 original samples resampled to 860 (50% obligation / 30% neutral / 20%
non_obligation, deliberately not preserving the natural distribution), max_seq_length 1024,
LoRA r=16 alpha=16 dropout=0.05, 4-bit NF4, 5 epochs, lr 1e-4, seed 3407, MLflow to local `./mlruns`.
Note the YAML lists LoRA target modules as `q_proj/k_proj/v_proj` while the notebook also includes
`o_proj`.

## Conventions

- Model weights, `outputs/`, `mlruns/`, input PDFs (`**/data/input_pdfs/`) and pipeline outputs
  (`**/data/output/`) are gitignored. Committed pipeline data is limited to the small
  `data/chunks/chunks_ontology.jsonl` and `data/graph/` samples.
- `.gitignore` deliberately excludes three scratch scripts from the pipeline directory
  (`resample_ft_data.py`, `test_openai.py`, `test_schema.py`) — they may exist locally but are not
  tracked; don't add them back.
- Python source uses `from __future__ import annotations`, PEP-604 unions, pydantic v2, module-level
  `logger = logging.getLogger(__name__)`, and section banners (`# ── Stage N: … ───`).
- Every directory (`data/`, `configs/`, `docs/`, `models/`, `notebooks/`) has its own README that is
  part of the published documentation — update it when the directory's contents change.
