# ComplianceGPT Architecture

## Overview

ComplianceGPT is an open-source Legal AI platform designed to automate regulatory compliance tasks through domain-specialized Small Language Models (SLMs).

Rather than relying on a single general-purpose language model, ComplianceGPT follows a modular architecture where individual models specialize in distinct compliance tasks. This design improves efficiency, maintainability, and extensibility while reducing computational overhead.

The first release focuses on **Regulatory Obligation Extraction**, transforming unstructured legal text into structured machine-readable outputs suitable for downstream compliance workflows.

---

# Design Principles

The architecture of ComplianceGPT is guided by the following principles:

- Domain specialization over general-purpose reasoning.
- Modular AI components.
- Reproducible training pipelines.
- Structured outputs.
- Efficient deployment.
- Open-source tooling.
- Scalable system design.

These principles allow new compliance capabilities to be added without redesigning the entire platform.

---

# Current System Architecture (Version 1)

The current implementation consists of a single specialized language model trained for regulatory obligation extraction.

```text
                  Regulatory Documents
                           │
                           ▼
                  Document Parsing
                           │
                           ▼
                   Text Chunk Creation
                           │
                           ▼
                  Instruction Dataset
                           │
                           ▼
                  Gemma 4 E2B IT
                     + QLoRA
                           │
                           ▼
             Regulatory Obligation SLM
                           │
                           ▼
                  Structured JSON Output
```

The workflow consists of four primary stages:

1. Data preparation.
2. Instruction fine-tuning.
3. Structured obligation extraction.
4. JSON generation.

---

# Current Components

## Regulatory Documents

The system begins with publicly available regulatory documents.

Examples include:

- GDPR
- ISO 27001
- RBI Guidelines
- HIPAA
- PCI DSS
- DORA
- NIS2
- SEC Regulations

These documents contain both actionable obligations and non-actionable explanatory text.

---

## Dataset Engineering

The documents are transformed into an instruction-tuning dataset through:

- Document chunking
- Manual validation
- Instruction formatting
- JSON schema generation
- Dataset balancing

The resulting JSONL dataset forms the foundation of model training.

---

## Fine-Tuned Language Model

The first ComplianceGPT model is based on:

- Google Gemma 4 E2B IT
- QLoRA
- Unsloth
- PEFT

The model specializes in:

- Obligation detection
- Regulatory classification
- Structured information extraction

---

## Structured Output Layer

Instead of producing conversational responses, the model generates structured JSON.

Example:

```json
{
  "classification": "obligation",
  "output": [
    {
      "subject": "...",
      "action": "...",
      "modality": "...",
      "deadline": "..."
    }
  ]
}
```

Structured outputs simplify integration with compliance applications and downstream analytics.

---

# Training Architecture

The training workflow follows a reproducible pipeline.

```text
Regulatory Documents
        │
        ▼
Instruction Dataset
        │
        ▼
Dataset Validation
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
        ▼
MLflow Tracking
        │
        ▼
Hugging Face Deployment
```

Each stage is version-controlled to improve reproducibility.

---

# Inference Architecture

The deployed model follows a straightforward inference workflow.

```text
User Input
      │
      ▼
Tokenizer
      │
      ▼
Gemma 4 Regulatory SLM
      │
      ▼
Generated JSON
      │
      ▼
Compliance Application
```

---

# Future Platform Architecture

The long-term vision for ComplianceGPT extends beyond a single SLM.

Instead of using one large model for every task, the platform is intended to orchestrate multiple task-specific SLMs, each optimized for a particular compliance capability.

The conceptual architecture is illustrated below.

```text
                         Regulatory Documents
                                  │
                                  ▼
                          Document Parser
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             Vector Database             Graph Database
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                          Retrieval Layer
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
          Obligation SLM   Entity SLM   Classification SLM
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                         Compliance Orchestrator
                                  │
                                  ▼
                         Compliance Applications
```

This architecture reflects the broader vision of ComplianceGPT as a modular compliance AI platform.

---

# Why Multiple SLMs?

Different compliance tasks require different capabilities.

For example:

| Task | Preferred Model |
|-------|-----------------|
| Obligation Extraction | Obligation SLM |
| Entity Recognition | Entity SLM |
| Document Classification | Classification SLM |
| Compliance QA | QA SLM |
| Summarization | Summarization SLM |

This modular design enables each model to be optimized independently while simplifying maintenance and future upgrades.

---

# Planned Knowledge Layer

Future releases aim to incorporate complementary retrieval mechanisms.

### Vector Database

A vector database will support semantic retrieval of regulatory passages, enabling retrieval-augmented workflows.

Potential use cases include:

- Similar regulation search.
- Semantic document retrieval.
- Context retrieval for generation.

### Graph Database

A graph database will represent relationships among regulations, entities, obligations, controls, and policies.

Potential use cases include:

- Regulatory dependency analysis.
- Impact assessment.
- Traceability between obligations and internal controls.
- Compliance knowledge graphs.

The specific database technologies will be selected as the platform evolves.

---

# Future GraphRAG Integration

A future objective is to combine vector retrieval with graph-based retrieval.

GraphRAG would enable the system to:

- Retrieve semantically relevant regulatory text.
- Traverse relationships between regulatory entities.
- Improve explainability.
- Provide richer compliance reasoning.

This capability is not part of the current implementation but represents a planned direction for the platform.

---

# Scalability

The modular architecture allows additional SLMs to be introduced without affecting existing components.

Future additions may include:

- Regulatory summarization.
- Risk extraction.
- Policy mapping.
- Regulatory change detection.
- Compliance recommendation.

---

# Summary

ComplianceGPT currently implements a reproducible pipeline for fine-tuning a specialized regulatory obligation extraction model.

Its broader architectural vision is to evolve into a modular compliance intelligence platform composed of multiple specialized SLMs, supported by vector retrieval, graph databases, and explainable compliance reasoning.