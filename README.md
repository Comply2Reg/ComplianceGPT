<div align="center">

# ComplianceGPT

### An Open-Source AI Platform for Regulatory Obligation Extraction and Compliance Automation

*ComplianceGPT builds domain-specialized Small Language Models (SLMs) for regulatory work: extracting structured compliance obligations from legal text, and triaging incoming regulatory publications into the classifications a compliance function acts on.*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)
[![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-orange)](https://github.com/unslothai/unsloth)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-success.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1)

</div>

---

> **ComplianceGPT** is an open-source Legal AI platform focused on automating regulatory compliance workflows through specialized Small Language Models (SLMs). Rather than relying on a single general-purpose Large Language Model (LLM), ComplianceGPT is designed as a modular ecosystem where task-specific models collaborate to solve different compliance problems with higher efficiency, lower inference cost, and improved domain specialization.

The first public release introduces a **Gemma 4 E2B IT** model fine-tuned specifically for **Regulatory Obligation Extraction**, transforming unstructured legal and regulatory text into structured JSON suitable for downstream Governance, Risk, and Compliance (GRC) systems.

A second model, **Qwen3-4B – Regulatory Alert Triage**, extends the platform to the step that comes before extraction: deciding what an incoming publication *is*, how urgent it is, and which function owns the response. Both models emit structured JSON, and both are trained with parameter-efficient fine-tuning on modest hardware.

---

# Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Project Vision](#project-vision)
- [Current Release](#current-release)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Project Workflow](#project-workflow)
- [Evaluation](#evaluation)

---

# Overview

Regulatory documents contain thousands of statements that vary in purpose. Some statements impose legal obligations, while others provide recommendations, explanatory information, definitions, or contextual guidance.

Traditional language models are capable of reading these documents, but they are not optimized for extracting compliance obligations in a structured format that can be consumed by downstream applications.

ComplianceGPT addresses this challenge by combining domain-specific instruction tuning with parameter-efficient fine-tuning to create specialized models capable of understanding regulatory language and converting it into structured machine-readable outputs.

The first release focuses on identifying regulatory obligations and extracting key compliance entities such as:

- Subject
- Required action
- Modality
- Conditions
- Deadlines
- Supporting metadata

The resulting structured outputs can be integrated into compliance monitoring systems, Retrieval-Augmented Generation (RAG) pipelines, knowledge graphs, and regulatory intelligence platforms.

---

# Motivation

Regulatory compliance is a knowledge-intensive task that requires organizations to continuously interpret large volumes of legal documents, standards, and regulatory updates.

Examples include:

- RBI Guidelines
- MAS Regulations

These documents contain a mixture of:

- Mandatory obligations
- Recommendations
- Informational statements
- Definitions
- Exceptions
- References

In real-world compliance workflows, organizations are primarily interested in identifying actionable obligations.

However, regulatory corpora are naturally imbalanced, with informative and neutral statements significantly outnumbering mandatory obligations. This imbalance makes it difficult for general-purpose models to consistently identify obligation statements without additional domain adaptation.

ComplianceGPT was developed to address this problem by building specialized models that prioritize regulatory obligation understanding while preserving contextual reasoning over legal text.

---

# Project Vision

ComplianceGPT is not intended to be a single language model.

It is envisioned as a modular compliance intelligence platform composed of multiple task-specific Small Language Models (SLMs), where each model specializes in a particular stage of the compliance lifecycle.

Future versions will integrate:

- Regulatory Obligation Extraction
- Regulatory Entity Recognition
- Compliance Question Answering
- Regulatory Summarization
- Knowledge Graph Generation
- Graph Retrieval-Augmented Generation (GraphRAG)
- Regulatory Change Detection
- Cross-document Compliance Reasoning
- Policy-to-Regulation Mapping

Instead of relying on one monolithic LLM, ComplianceGPT adopts a multi-model architecture in which specialized components collaborate to solve complex compliance tasks efficiently.

---

# Current Release

The first public model available in this repository is:

## Gemma 4 E2B IT – Regulatory Obligation Extraction v1

This model has been instruction-tuned using QLoRA to extract structured regulatory obligations from legal and compliance documents.

The model classifies text into three categories:

| Class | Description |
|--------|-------------|
| Obligation | Mandatory regulatory requirement requiring action |
| Non-obligation | Statements that do not impose mandatory actions |
| Neutral | Informational or descriptive regulatory statements |

For obligation statements, the model generates structured JSON suitable for downstream automation pipelines.

The released model is available on Hugging Face:

**https://huggingface.co/PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1**

## Qwen3-4B – Regulatory Alert Triage v1

The name is deliberately jurisdiction-neutral because the task is not, but **v1 is trained and evaluated entirely on UK sources**. Treat it as a UK model until a release says otherwise.

The second model addresses a different stage of the compliance lifecycle. Where obligation extraction reads a document that has already been selected for attention, triage decides which documents deserve attention at all.

Given a UK regulatory publication, the model returns a single structured record describing the document:

| Field | Description |
|--------|-------------|
| Alert class | One of fourteen document kinds, from primary legislation to enforcement action to speeches |
| Priority | P1 binding or imminent, P2 action likely, P3 awareness only |
| Functions | Which of fourteen bank functions own the response, and which are consulted |
| Lines of defence | Which of the three lines those functions sit in |
| Summary | What the publication says and whom it affects |
| Key dates | Deadlines, in-force dates, consultation closing dates |
| Applicability | The firms, products or activities in scope |
| Obligations present | Whether the text creates a duty |

The model was adapted with LoRA from `Qwen3-4B-Instruct-2507` in its 4-bit MLX conversion, then fused back into standalone weights, and trained entirely on a laptop. It was measured on 104 held-out documents published *later* than everything it trained on.

| Property | Value |
|----------|-------|
| Base model | Qwen3-4B-Instruct-2507 (4-bit MLX) |
| Method | LoRA rank 8, 16 of 36 layers, fused into the base |
| Training data | 1,683 UK documents under the Open Government Licence |
| JSON validity | 1.000 |
| Alert-class accuracy | 0.644 (majority-class baseline 0.538) |
| Alert-class macro-F1 | 0.244 |

The weights are held in the Comply2Reg organisation on Hugging Face and are currently **private**, pending a review of the training-corpus licensing. Access is available to Comply2Reg members on request; a public release is planned.

The honest reading of those numbers is in the [Evaluation](#evaluation) section, which is worth reading before drawing conclusions from the accuracy figure.

---

# Key Features

## Domain-Specific Fine-Tuning

Instead of generic instruction tuning, the model has been trained specifically for regulatory obligation extraction using carefully curated compliance data.

---

## Parameter-Efficient Training

The model was fine-tuned using:

- QLoRA
- LoRA adapters
- 4-bit NF4 quantization
- PEFT
- Unsloth

This significantly reduces GPU memory requirements while maintaining competitive performance.

---

## Structured JSON Generation

Instead of producing free-form text, ComplianceGPT generates structured outputs suitable for machine processing.

Example:

```json
{
  "output": [
    {
      "subject": "Financial Institution",
      "action": "Maintain customer records",
      "modality": "MUST",
      "conditions": "",
      "deadline": "Five years"
    }
  ]
}
```

---

## Designed for Compliance Automation

The generated outputs are suitable for:

- Governance, Risk and Compliance (GRC)
- Regulatory monitoring
- Compliance dashboards
- Rule engines
- Knowledge Graph construction
- Retrieval-Augmented Generation (RAG)
- Internal compliance assistants

---

## Experiment Tracking

The complete fine-tuning workflow is tracked using MLflow.

Tracked information includes:

- Hyperparameters
- Training configuration
- Dataset versions
- Model artifacts
- Evaluation metrics
- Experiment history

This improves reproducibility and simplifies future experimentation.

---

# Repository Structure

```text
ComplianceGPT/

├── README.md
├── docs/
├── data/
├── notebooks/
├── src/
├── configs/
├── benchmarks/
├── images/
├── scripts/

```

The repository is organized to separate project documentation, datasets, notebooks, source code, configurations, benchmarks, and deployment resources, making it easier to reproduce experiments and extend the project.

---

# Quick Start

## Clone the repository

```bash
git clone https://github.com/Comply2Reg/ComplianceGPT.git

cd ComplianceGPT
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Load the published model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)
```

## Load the triage model

The triage model runs through MLX on Apple Silicon. While the repository is private, loading it requires a Hugging Face token with access to the Comply2Reg organisation.

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("Comply2Reg/regulatory-alert-triage-qwen3-4b-v1")

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": DOCUMENT},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

text = generate(
    model, tokenizer, prompt=prompt, max_tokens=400,
    sampler=make_sampler(temp=0.0), verbose=False,
)
```

The model opens every answer with an empty reasoning block, so take the substring from the first `{` to the last `}` before parsing. The full prompt format and output schema are documented on the model card.

Additional examples for inference, evaluation, and fine-tuning are provided in the `docs/` directory and accompanying notebooks.

---

# Project Workflow

The current implementation follows the workflow illustrated below.

```text
Regulatory Documents
        │
        ▼
Document Parsing
        │
        ▼
Instruction Dataset Creation
        │
        ▼
Dataset Validation
        │
        ▼
Class Distribution Analysis
        │
        ▼
Dataset Resampling
        │
        ▼
QLoRA Fine-Tuning
        │
        ▼
Evaluation
        │
        ├────────► MLflow Tracking
        │
        ▼
Model Export
        │
        ▼
Hugging Face Deployment
        │
        ▼
Compliance Applications
```

The following sections describe each stage of this workflow in detail, including dataset engineering, prompt design, model configuration, evaluation methodology, and deployment.

---

# Regulatory Sources

The instruction dataset was created from publicly available regulatory standards and compliance frameworks spanning multiple industries.

Representative sources include:

| Regulation / Standard | Domain |
|------------------------|--------|
| RBI Guidelines | Banking |
| MAS Regulations | Financial Services |


These sources contain a mixture of mandatory obligations, recommendations, informative guidance, definitions, and references. The objective of the training process was to enable the model to distinguish actionable compliance requirements from surrounding contextual information.

## UK Regulatory Corpus

The triage model draws on a separate corpus, collected continuously from UK regulator and government websites since January 2022 and currently standing at 6,642 documents.

Sources are separated by licence before any of them reaches a training run. Only material published under the Open Government Licence v3.0 was used:

| Source | Licence | Used for training |
|--------|---------|-------------------|
| HM Treasury | Open Government Licence v3.0 | Yes |
| Competition and Markets Authority | Open Government Licence v3.0 | Yes |
| Information Commissioner's Office | Open Government Licence v3.0 | Yes |
| legislation.gov.uk | Open Government Licence v3.0 | Yes |
| Financial Conduct Authority | Terms not clearly permissive | Held back |
| Prudential Regulation Authority | Terms not clearly permissive | Held back |
| Bank of England | Terms not clearly permissive | Held back |
| Ofcom, DRCF | Terms not clearly permissive | Held back |

Holding back the FCA and PRA is the single largest constraint on the triage model's quality, because those bodies publish most of the supervisory guidance and enforcement material a UK bank cares about. Resolving that licensing question is the main route to a better model, ahead of any change to the training recipe.

Training text carries the required attribution: *Contains public sector information licensed under the Open Government Licence v3.0.*

---

# Dataset Construction Pipeline

The overall data preparation workflow is illustrated below.

```text
Regulatory Documents
        │
        ▼
Document Parsing
        │
        ▼
Chunk Generation
        │
        ▼
Manual Validation
        │
        ▼
Instruction Generation
        │
        ▼
JSONL Dataset
        │
        ▼
Distribution Analysis
        │
        ▼
Dataset Resampling
        │
        ▼
Final Training Dataset
```

Each stage contributes to improving the quality, consistency, and suitability of the training corpus for supervised instruction tuning.

## Triage Dataset Pipeline

The triage dataset is built by a separate, automated pipeline that starts from the crawled UK corpus.

```text
UK Regulator Websites
        │
        ▼
Scheduled Crawl
        │
        ▼
Furniture Removal          (running headers, page numbers, tables of contents)
        │
        ▼
Deduplication              (exact by content hash, near-duplicate by MinHash)
        │
        ▼
Licence Tiering            (open-licence sources separated from the rest)
        │
        ▼
Structured Labelling       (one model call per document, cached by content hash)
        │
        ▼
Cross-Model Adjudication   (disagreements re-labelled by a stronger model)
        │
        ▼
Chronological Split        (test = the most recent months, never trained on)
        │
        ▼
Stratified Balancing
        │
        ▼
Final Training Dataset
```

Two properties of this pipeline are worth drawing out.

**Cleaning happens before anything else.** Page furniture is removed before text is canonicalised, so every stored character offset and content hash refers to the cleaned text. Across the corpus this removed 1.86 million characters from 5,812 documents.

**Splits are chronological, not random.** The test set is the most recent months of publications; training runs on everything earlier. This measures whether the model generalises forward in time, which is how it is actually used, and it makes the reported scores harder to achieve than a random split would.

---

# Dataset Annotation Strategy

Each regulatory text segment was categorized into one of three semantic classes based on its regulatory intent.

| Class | Description |
|--------|-------------|
| **Obligation** | Statements that impose mandatory actions or compliance requirements. |
| **Non-Obligation** | Statements that do not impose enforceable actions, such as recommendations or explanatory text. |
| **Neutral** | Informational, descriptive, or contextual statements without actionable requirements. |

This classification enables the model to learn not only what constitutes an obligation but also what should **not** be interpreted as one.

## Triage Annotation

The triage dataset uses a different label space and a different annotation method. Each document receives a fourteen-class label drawn from a published taxonomy, together with priority, owning functions, lines of defence and an obligation flag.

Labels were generated by a frontier model against that taxonomy, one call per document, cached by content hash so a run is reproducible and re-runs are free. A second model labelled the same corpus independently; where the two disagreed, a stronger model adjudicated a sample, siding with the chosen labeller in 18 of 20 cases.

These labels are **model-generated and not human-adjudicated**. They are consistent, and some of them are wrong. Errors in the labeller are reproduced in the fine-tuned model, and the evaluation shares that bias because its labels come from the same source. This is a known limitation rather than a solved problem.

---


# Fine-Tuning Strategy

ComplianceGPT uses **QLoRA (Quantized Low-Rank Adaptation)** to specialize the base model.

Instead of updating every parameter in the neural network, QLoRA freezes the original model weights and learns only a small number of trainable adapter parameters.

This approach offers several advantages:

- Significantly lower GPU memory consumption.
- Faster training.
- Smaller checkpoints.
- Easy deployment using PEFT adapters.
- Minimal degradation compared to full fine-tuning.

The original pretrained knowledge remains intact while the adapters learn regulatory-domain behavior.

## Adapters and Fused Weights

An adapter is convenient during development but awkward to distribute: it only works alongside the exact base model it was trained against. The triage model is therefore published as **fused weights**, with the adapter merged back into the base so the result loads as an ordinary model.

Fusing into a 4-bit model is not lossless. The process dequantizes each layer, adds the adapter delta and re-quantizes, so the published weights are a numerically different object from base-plus-adapter. Measured on the same test set, the two agree on 93 of 104 documents, with the fused model scoring marginally higher overall. Every figure published for the triage model is measured on the fused weights that are actually distributed, not inherited from the adapter.

---

# Parameter-Efficient Fine-Tuning (PEFT)

The project uses the Hugging Face **PEFT** library together with Unsloth.

The overall training process consists of:

```text
Gemma 4 Base Model
        │
        ▼
Freeze Base Parameters
        │
        ▼
Attach LoRA Adapters
        │
        ▼
Train Only Adapter Weights
        │
        ▼
Merge (Optional)
        │
        ▼
Inference
```

Only a very small fraction of the model parameters are updated during training, making experimentation substantially more efficient.

---


# Training Environment

The model was fine-tuned using a cloud-based GPU environment with modern open-source tooling.

| Component | Value |
|-----------|-------|
| Development Environment | Google Colab |
| GPU | NVIDIA L4 (22 GB) |
| Framework | Unsloth |
| Transformers | Hugging Face Transformers |
| Adapter Library | PEFT |
| Training Library | TRL |
| Quantization | BitsAndBytes |
| Experiment Tracking | MLflow |

The use of Unsloth significantly reduced memory overhead and accelerated training compared to conventional QLoRA implementations.

## Apple Silicon Training

The triage model was trained on a laptop rather than a cloud GPU, using Apple's MLX framework against a 4-bit conversion of the base model.

| Component | Value |
|-----------|-------|
| Development Environment | Local workstation |
| Hardware | Apple M4 Pro, 24 GB unified memory |
| Framework | MLX / mlx-lm |
| Quantization | 4-bit, group size 64 |
| Adapter | LoRA rank 8, scale 20, 16 of 36 layers |
| Trainable parameters | 7.34 M of 4,022 M (0.18%) |
| Loss | Masked to the assistant turn |
| Wall clock | 2.4 hours for 1,700 iterations |
| Peak memory | 4.56 GB |

A 4 B model at 4-bit precision with LoRA on half its layers trains comfortably inside 5 GB of unified memory, which puts a full fine-tuning cycle within reach of a laptop and removes the cloud GPU from the iteration loop entirely.

---

# Training Workflow

The end-to-end fine-tuning workflow is summarized below.

```text
Instruction Dataset
        │
        ▼
Tokenization
        │
        ▼
Gemma 4 E2B IT
        │
        ▼
Load in 4-bit
        │
        ▼
Attach LoRA Adapters
        │
        ▼
QLoRA Fine-Tuning
        │
        ▼
Checkpoint Saving
        │
        ▼
Evaluation
        │
        ▼
MLflow Logging
        │
        ▼
Model Export
        │
        ▼
Hugging Face
```

This workflow emphasizes reproducibility by combining structured data preparation, parameter-efficient fine-tuning, experiment tracking, and version-controlled model publishing.

---


# Evaluation

The triage model was evaluated on 104 held-out UK regulatory publications, all published later than any document in the training set, with greedy decoding and field-by-field scoring.

| Metric | Score | Note |
|--------|------:|------|
| JSON validity | 1.000 | Output parsed as a JSON object |
| Alert-class accuracy | 0.644 | Always predicting the commonest class scores 0.538 |
| Alert-class macro-F1 | 0.244 | Unweighted mean across the classes present |
| Priority accuracy | 0.760 | P1 / P2 / P3 |
| Obligations present | 0.913 | Binary |
| Primary functions | 0.574 | Set overlap against the gold functions |

## How to read these numbers

**Macro-F1 is the honest figure, not accuracy.** The test set is 54% a single class, so an always-guess-the-majority baseline scores 0.538 accuracy without knowing anything. The model adds 0.106 on top of that.

**Fine-tuning bought the output format.** The untuned base model produced no parseable triage record at all on a held-out probe; it answers in prose. Everything downstream depends on the JSON being present and well-formed, so moving validity from 0.000 to 1.000 is the change that makes the model usable at all.

**The model has not learned most of its label space.** It scores non-zero on four of the eleven classes present in the test set. Classes with roughly twenty training examples score zero. This is a data problem rather than a recipe problem, and it traces directly back to the licence tiering described under Regulatory Sources.

**Lower validation loss did not mean a better model.** The checkpoint with the best validation loss of the run scored worse on every task metric, having collapsed toward predicting the majority class. The released adapter was selected on task metrics, not on loss — a reminder that loss is a proxy, and on an imbalanced label space it is a poor one.

---

# Reproducibility

To facilitate reproducibility, this repository provides:

- Training notebooks.
- Configuration files.
- Sample datasets.
- Documentation.
- Hyperparameter settings.
- Prompt templates.
- Model checkpoints.
- Experiment tracking information.

The goal is to enable other researchers and practitioners to reproduce the fine-tuning process or extend the project for additional compliance-related tasks.

---

# Summary

The first ComplianceGPT model demonstrates how parameter-efficient fine-tuning can transform a general-purpose instruction-tuned language model into a domain-specialized regulatory assistant.

By combining **Gemma 4**, **QLoRA**, **Unsloth**, **PEFT**, and **MLflow**, the project establishes a reproducible pipeline for building efficient Legal AI systems capable of extracting structured regulatory obligations while remaining practical to train and deploy on modest hardware.

The second model extends that argument further. Trained with **MLX** on a laptop in under three hours, it shows that a specialized compliance model can be produced without cloud GPUs at all — and, just as importantly, that the limiting factor is the data rather than the hardware or the training recipe.

---

# Example Inference

Example regulatory statement:

```text
Every financial institution shall maintain customer records for five years.
```

Expected structured output:

```json
{
  "classification": "obligation",
  "output": [
    {
      "subject": "Financial Institution",
      "action": "Maintain customer records",
      "modality": "MUST",
      "conditions": "",
      "deadline": "Five years"
    }
  ]
}
```

The structured output is intentionally designed for integration with downstream compliance systems rather than conversational interaction.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Comply2Reg/ComplianceGPT.git

cd ComplianceGPT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, create a dedicated environment:

```bash
conda env create -f environment.yml

conda activate compliancegpt
```

---

# Running Inference

Load the published model from Hugging Face.

```python
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_id = "PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)

prompt = """
Every financial institution shall maintain customer records for five years.
"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.1
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

More detailed inference examples are provided in the accompanying notebooks.

---

# Applications

ComplianceGPT is intended for research and engineering applications including:

- Regulatory obligation extraction
- Governance, Risk and Compliance (GRC)
- Regulatory document understanding
- Internal compliance assistants
- Retrieval-Augmented Generation (RAG)
- Regulatory knowledge graph construction
- Compliance dashboards
- Regulatory monitoring systems
- AI-assisted policy analysis

The project is designed to serve as a building block for larger compliance intelligence platforms.

---

# Current Limitations

The current release represents the first specialized model within the ComplianceGPT project.

Known limitations of the obligation extraction model include:

- Focused on English-language regulatory documents.
- Optimized specifically for obligation extraction.
- Performance may decrease on unseen regulatory writing styles.
- Long documents require preprocessing and chunking.
- Outputs should always be reviewed by qualified compliance or legal professionals before use in production.

Known limitations of the UK triage model include:

- Seven of the fourteen alert classes have too little training data to learn, and the model scores zero on them.
- One class accounts for over half the test set, so accuracy overstates capability and macro-F1 is the figure to quote.
- Training sources are cross-sector by necessity: Treasury policy, competition cases, data protection and statutory instruments. The FCA and PRA conduct and prudential material a UK bank most cares about is exactly what is absent.
- Training labels are model-generated and have not been adjudicated by a human.
- The confidence field is imitated from the labelling model and has not been calibrated against observed accuracy. It should not be used as a threshold.
- The model reads the first ~1,500 characters of a document, not its full text.

Neither model is legal advice, and neither replaces a compliance professional's reading of a source document.

These limitations inform the future development roadmap.

---

# Project Roadmap

ComplianceGPT is intended to evolve into a modular compliance AI platform composed of multiple specialized models working together.

## Completed

- Instruction dataset creation.
- Domain-specific prompt engineering.
- Dataset validation.
- Dataset resampling.
- QLoRA fine-tuning.
- Structured JSON generation.
- MLflow experiment tracking.
- Hugging Face model publication.
- Comprehensive project documentation.
- Continuous UK regulatory corpus collection with licence tiering.
- Automated furniture removal and near-duplicate detection.
- Cross-model label adjudication.
- Chronologically split evaluation.
- Apple Silicon fine-tuning path with fused-weight publication.

## Planned

- Expanded evaluation across additional regulatory frameworks.
- Improved extraction of conditional and nested obligations.
- Support for multilingual regulatory documents.
- Larger and more diverse training corpus.
- Additional benchmark datasets.
- Interactive inference interface.
- Model quantization for optimized deployment.
- Automated evaluation pipeline.
- Licensing review to admit FCA and PRA material to the triage corpus.
- Human adjudication of a gold evaluation set.
- Class rebalancing for the under-represented alert classes.
- Confidence calibration.

## Long-Term Vision

The broader ComplianceGPT platform is envisioned as a collection of specialized AI components supporting end-to-end compliance workflows.

Planned capabilities include:

- Regulatory Entity Recognition
- Regulatory Obligation Extraction
- Regulatory Classification
- Compliance Question Answering
- Regulatory Summarization
- Regulatory Change Detection
- Policy-to-Regulation Mapping
- Knowledge Graph Generation
- Vector Database Integration
- Graph Database Integration
- GraphRAG for compliance retrieval
- Multi-SLM orchestration
- Explainable compliance reasoning

Rather than relying on a single monolithic language model, the long-term objective is to build a modular ecosystem where each task is handled by a domain-specialized model.

---

# Acknowledgements

This project was made possible through the open-source AI ecosystem.

Special thanks to the communities and organizations behind:

- Google Gemma
- Hugging Face
- Transformers
- PEFT
- Unsloth
- TRL
- BitsAndBytes
- MLflow

The project also builds upon publicly available regulatory standards and documentation used exclusively for research and educational purposes.

---

# Contact

For questions, suggestions, or collaboration opportunities:

- **GitHub:** https://github.com/Comply2Reg/ComplianceGPT
- **Hugging Face:** https://huggingface.co/PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1

---

<div align="center">

### ComplianceGPT

**Building domain-specialized AI systems for intelligent regulatory compliance.**

*Current Release: Gemma 4 E2B IT – Regulatory Obligation Extraction v1*

</div>