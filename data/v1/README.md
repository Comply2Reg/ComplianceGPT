# ComplianceGPT Dataset v1

## Overview

**ComplianceGPT Dataset v1** is the first public instruction-tuning dataset developed for training domain-specialized Small Language Models (SLMs) on **regulatory obligation extraction**.

The dataset was specifically engineered to teach a language model how to distinguish actionable regulatory obligations from non-obligatory and informational regulatory text while generating structured JSON outputs suitable for downstream compliance applications.

This dataset was used to train:

> **ComplianceGPT v1 — Gemma 4 E2B IT Regulatory Obligation Extraction**

---

# Purpose

Most publicly available language-model datasets are designed for general reasoning, instruction following, or conversational AI.

ComplianceGPT Dataset v1 was created with a different objective:

- Regulatory obligation extraction
- Compliance automation
- Structured information extraction
- Legal NLP
- Governance, Risk & Compliance (GRC)

Instead of generating natural language answers, the model learns to produce structured machine-readable representations of regulatory requirements.

---

# Dataset Version

| Property | Value |
|-----------|-------|
| Dataset Version | v1 |
| Status | Stable |
| Release | Initial Public Release |
| Format | JSONL |
| Training Style | Instruction Tuning |
| Output Format | Structured JSON |

---

# Dataset Relationship

This dataset was used to fine-tune the following model:

| Model | Link |
|--------|------|
| Gemma 4 E2B IT Regulatory Obligation Extraction v1 | https://huggingface.co/PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1 |

---

# Dataset Contents

The Version 1 dataset directory contains:

| File | Description |
|------|-------------|
| schema.md | Complete JSON schema |
| data_dictionary.md | Explanation of every dataset field |
| dataset_statistics.md | Dataset statistics and class distributions |
| sample_dataset.jsonl | Representative training examples |
| resampled_ft-v2.jsonl | Final instruction-tuning dataset |

---

# Regulatory Sources

The dataset was created from publicly available regulatory and compliance documents.

Representative sources include:

- RBI Guidelines
- MAS Regulations

These regulations span multiple industries including banking, cybersecurity, healthcare, financial services, privacy, and information security.

---

# Dataset Characteristics

The dataset was designed to teach the model three distinct regulatory concepts.

| Class | Purpose |
|--------|----------|
| Obligation | Mandatory regulatory requirement |
| Non-Obligation | Statements without enforceable actions |
| Neutral | Informational or descriptive content |

This formulation enables the model to distinguish actionable compliance requirements from surrounding context.

---

# Original Dataset

The original dataset contained:

| Metric | Value |
|----------|------:|
| Total Samples | 2,947 |
| Obligation | 430 |
| Non-Obligation | 282 |
| Neutral | 2,235 |

This distribution accurately reflects real-world regulatory documents but exhibits a substantial class imbalance.

---

# Resampled Dataset

To prioritize the downstream task of obligation extraction, the dataset was resampled before fine-tuning.

Final dataset:

| Metric | Value |
|----------|------:|
| Total Samples | 860 |
| Obligation | 430 |
| Neutral | 258 |
| Non-Obligation | 172 |

This resampling strategy intentionally increases the relative representation of obligation examples while preserving exposure to the remaining semantic classes.

---

# Training Format

Each record follows a chat-style instruction tuning format consisting of:

- System Prompt
- User Prompt
- Assistant Response

This format is compatible with modern instruction-tuned language models including Gemma, Llama, Mistral, and Qwen.

---

# Output Schema

For obligation samples, the assistant produces structured JSON containing:

- Classification
- Subject
- Action
- Modality
- Conditions
- Deadline

The complete schema is documented in:

```
schema.md
```

---

# Dataset Quality

Multiple validation procedures were applied before training.

These include:

- JSON validation
- Label verification
- Prompt consistency
- Output schema validation
- Manual inspection
- Duplicate removal
- Invalid sample removal

These quality-control measures improve training consistency and reduce annotation noise.

---

# Intended Use

This dataset is intended for:

- Instruction tuning
- Legal NLP research
- Regulatory obligation extraction
- Compliance automation
- Information extraction
- Academic experimentation

---

# Not Intended For

This dataset should not be considered:

- A complete regulatory corpus
- Legal advice
- An authoritative interpretation of regulations
- A replacement for compliance professionals

---

# Version History

## v1

Initial public release.

Features:

- Custom instruction dataset
- Structured JSON outputs
- Three-class annotation strategy
- Domain-specific prompt engineering
- Resampled training distribution
- Fine-tuning support for Gemma 4

---

# Future Versions

Future dataset releases may include:

- Additional regulations
- Multilingual documents
- Expanded annotation schema
- Nested obligations
- Cross-document relationships
- Knowledge graph annotations
- Human-reviewed benchmark datasets

---

# Related Documentation

For additional information, refer to:

- `schema.md`
- `data_dictionary.md`
- `dataset_statistics.md`
- `../../docs/dataset.md`

---

# Summary

ComplianceGPT Dataset v1 provides a carefully engineered instruction-tuning corpus for regulatory obligation extraction.

It serves as the foundation for the first ComplianceGPT model and establishes a reproducible dataset version that can be extended through future releases while preserving compatibility with previous experiments.