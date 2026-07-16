# Obligation Data Pipeline

Enterprise-grade pipeline designed for generating strict obligation-extraction datasets and knowledge graphs from regulatory PDFs. 

The pipeline runs synchronously across 5 distinct stages, porting the robust structural parsing and ontology concepts from the `GraphRag` infrastructure into an independent, dataset-focused entity.

---

## 🏗️ 5-Stage Architecture

1. **Stage 1 (Ingestion):** Connects to a MySQL database to fetch regulatory document metadata, downloads corresponding PDF files from an AWS S3 bucket, and builds a processing `manifest.json`.
2. **Stage 2 (Parse & Chunk):** Subjects the downloaded PDFs to regex or LLM-based parsing to construct a strict legal hierarchy (Parts, Chapters, Sections, Clauses). It then clusters text into context-rich **ontology-aware chunks**, drastically improving extraction downstream.
3. **Stage 3 (Extraction):** Utilizes OpenAI structured outputs (with rule-based fallbacks) to ingest chunks and evaluate them into 3 distinct classifications: `obligation`, `non_obligation`, or `neutral`. Generates detailed schema-conforming records with **composite confidence scoring**.
4. **Stage 4 (Vector DB):** Embeds extracted chunks and relevant metadata into ChromaDB using OpenAI or local SentenceTransformers.
5. **Stage 5 (Graph DB):** Inserts structured records into Neo4j. Extends chunk representation with derived ontological structure and confidence scores.

---

## 🎯 Enhanced Confidence Scoring

The schema natively supports a robust 5-signal composite confidence mechanism defined in the `Evaluation` object. The overarching `confidence_score` is defined by tunable weights blending five characteristics:

1. **Classification Confidence:** Probability the label choice applies natively.
2. **Modality Confidence:** Calibrated strictly based on deontic cues detected. (`shall/must` -> 1.0, `may` -> 0.2)
3. **Extraction Completeness:** The fraction of obligatory fields correctly populated (Subjects, Deadlines, Conditions, etc.).
4. **Reasoning Consistency:** Verification of logical step-consistency leading to the ultimate conclusion.
5. **Grounding Score:** Real sequence matching validating textual overlap to halt hallucination.

The evaluation will assign a tier (`high`, `medium`, `low`) and flip a `hallucination_flag` boolean to restrict dirty data pollution.

---

## 🚀 Quick Start

### 1. Prerequisites & Environment
Ensure you have an active MySQL server and AWS S3 credentials if running standard ingestion.

```bash
# Set up a virtual environment
python -m venv .venv
source .venv/Scripts/activate # On Windows

# Install required packages
pip install -r requirements.txt

# Create your .env
cp .env.example .env
```

### 2. Configuration Settings
Make sure to configure `.env`. Important flags:
- `USE_OPENAI=true` + `OPENAI_API_KEY`: Required to use OpenAI structured extractions.
- `DB_*` and `AWS_*`: Database and bucket configuration bounds for initial DB mapping.
- `NEO4J_*` and `CHROMA_*`: If exporting to external stores.

### 3. Run Pipeline Orchestration

You can effortlessly run the entire lifecycle from ingestion to knowledge graph construction:

```bash
python run_pipeline.py --version v1.0.0
```

Optionally skip stages if attempting to re-use an existing manifest/chunks file or ignore exporting targets:
```bash
python run_pipeline.py --skip-ingest --skip-vector --skip-graph --limit 5
```

---

## 🛠️ CLI Stage Runners

Individual elements of the lifecycle are modular and can be run independently out of order (assuming source inputs are met).

```bash
# 1. Download PDFs and create manifest
python run_ingestion.py 

# 2. Extract into ontology chunks
python run_parse.py --limit 10 

# 3. Request OpenAI inferences
python run_extract.py --version v1.0.0 

# 4. Vectorize Text
python run_ingest_vector.py --version v1.0.0

# 5. Connect Entities in Graph
python run_ingest_graph.py --version v1.0.0
```

---

## 📂 Main Outputs

Post pipeline execution, your `data/` structure will be populated:

- `data/ingested/` - Locally preserved AWS PDFs matched to database identifiers
- `data/chunks/chunks_ontology.jsonl` - Spatially segmented clauses
- `data/graph/` - Nodes and edges exports ready to be merged
- `data/output/raw_<version>.jsonl` - Absolute generated outputs containing traceability, reasoning, and context.
- `data/output/balanced_<version>.jsonl` - Ratio balanced dataset tailored for fine-tuning
- `data/output/run_report_<version>.json` - Holistic pipeline run metrics
