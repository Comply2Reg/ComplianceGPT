<div align="center">

# ComplianceGPT

### An Open-Source AI Platform for Regulatory Obligation Extraction and Compliance Automation

*The first release of ComplianceGPT introduces a domain-specialized Small Language Model (SLM) fine-tuned on regulatory text to extract structured compliance obligations from legal documents.*

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

---

# Dataset Annotation Strategy

Each regulatory text segment was categorized into one of three semantic classes based on its regulatory intent.

| Class | Description |
|--------|-------------|
| **Obligation** | Statements that impose mandatory actions or compliance requirements. |
| **Non-Obligation** | Statements that do not impose enforceable actions, such as recommendations or explanatory text. |
| **Neutral** | Informational, descriptive, or contextual statements without actionable requirements. |

This classification enables the model to learn not only what constitutes an obligation but also what should **not** be interpreted as one.

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

Known limitations include:

- Focused on English-language regulatory documents.
- Optimized specifically for obligation extraction.
- Performance may decrease on unseen regulatory writing styles.
- Long documents require preprocessing and chunking.
- Outputs should always be reviewed by qualified compliance or legal professionals before use in production.

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

## Planned

- Expanded evaluation across additional regulatory frameworks.
- Improved extraction of conditional and nested obligations.
- Support for multilingual regulatory documents.
- Larger and more diverse training corpus.
- Additional benchmark datasets.
- Interactive inference interface.
- Model quantization for optimized deployment.
- Automated evaluation pipeline.

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