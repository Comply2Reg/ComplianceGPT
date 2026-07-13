# Model Documentation

## Overview

The first release of ComplianceGPT is powered by **Google Gemma 4 E2B IT**, a modern instruction-tuned Small Language Model (SLM) that has been adapted for the specialized task of **regulatory obligation extraction**.

Rather than functioning as a general-purpose conversational assistant, this model is optimized to identify actionable compliance obligations within legal and regulatory documents and transform them into structured JSON suitable for downstream automation.

The specialization was achieved through **parameter-efficient fine-tuning (PEFT)** using **Quantized Low-Rank Adaptation (QLoRA)**.

---

# Model Information

| Property | Value |
|-----------|-------|
| Model Name | Gemma 4 E2B IT – Regulatory Obligation Extraction v1 |
| Base Model | Google Gemma 4 E2B IT |
| Architecture | Transformer Decoder |
| Fine-Tuning Method | QLoRA |
| Adapter Method | LoRA |
| Quantization | 4-bit NF4 |
| Framework | Unsloth |
| Deployment | Hugging Face |
| Output Format | Structured JSON |

---

# Why Gemma 4?

Selecting the base model is one of the most important design decisions in any domain adaptation project.

Gemma 4 E2B IT was selected because it provides a strong balance between:

- Instruction-following capability.
- Efficient fine-tuning.
- Open-source accessibility.
- Resource-efficient deployment.
- Compatibility with the Hugging Face ecosystem.

Its architecture makes it well suited for adapting to domain-specific structured generation tasks without requiring full-model retraining.

---

# Why Parameter-Efficient Fine-Tuning?

Training every parameter of a modern language model requires substantial computational resources.

Instead, ComplianceGPT employs **Parameter-Efficient Fine-Tuning (PEFT)**, where the pretrained weights remain frozen and only lightweight adapter layers are trained.

Benefits include:

- Lower GPU memory usage.
- Faster experimentation.
- Smaller checkpoints.
- Reduced storage requirements.
- Easier model sharing.
- Retention of pretrained language understanding.

This approach enables rapid specialization while maintaining the capabilities of the original model.

---

# Model Architecture

The model follows the standard decoder-only Transformer architecture inherited from Gemma 4.

Conceptually, the adapted model consists of:

```text
                 User Prompt
                      │
                      ▼
                 Tokenization
                      │
                      ▼
          Gemma 4 Transformer Layers
                      │
              LoRA Adapter Layers
                      │
                      ▼
          Domain-Specific Knowledge
                      │
                      ▼
              Structured JSON Output
```

The LoRA adapters modify selected transformer layers during training while leaving the pretrained parameters unchanged.

---

# Fine-Tuning Workflow

The adaptation process can be summarized as:

```text
Gemma 4 Base Model
        │
        ▼
Load in 4-bit NF4
        │
        ▼
Attach LoRA Adapters
        │
        ▼
Instruction Fine-Tuning
        │
        ▼
Regulatory Obligation Model
```

This workflow minimizes computational overhead while enabling effective domain adaptation.

---

# Instruction-Tuned Behavior

The model was trained to follow a structured instruction format consisting of:

1. System prompt
2. User input
3. Assistant response

This teaches the model to:

- Interpret regulatory language.
- Classify regulatory statements.
- Extract structured obligation entities.
- Produce valid JSON.

Unlike open-ended conversational models, ComplianceGPT prioritizes deterministic, machine-readable outputs.

---

# Output Schema

For obligation statements, the model generates structured JSON containing key regulatory entities.

| Field | Description |
|--------|-------------|
| classification | Predicted semantic class |
| subject | Responsible entity |
| action | Required regulatory action |
| modality | Regulatory strength |
| conditions | Conditional requirements |
| deadline | Time constraints |

This schema is intentionally simple so that it can be integrated into downstream systems without additional parsing.

---

# Supported Tasks

The current model supports:

- Regulatory obligation extraction.
- Obligation classification.
- Structured JSON generation.
- Compliance information extraction.

Future versions will expand these capabilities.

---

# Current Capabilities

The model can:

- Identify mandatory regulatory statements.
- Distinguish obligations from neutral content.
- Extract actionable compliance information.
- Produce structured outputs.
- Support downstream compliance automation.

---

# Current Limitations

Although specialized, the model has several limitations.

- English-language regulations only.
- Optimized for obligation extraction.
- Long documents require chunking.
- Performance depends on input quality.
- Does not provide legal advice.
- Should not be used as the sole basis for regulatory decisions.

Human review remains essential in production compliance workflows.

---

# Memory Efficiency

The model has been optimized for efficient training and inference through:

- 4-bit NF4 quantization.
- LoRA adapters.
- Gradient checkpointing.
- Mixed precision.
- 8-bit optimizer.

These optimizations significantly reduce computational requirements compared to full fine-tuning.

---

# Deployment

The model is distributed through Hugging Face and can be loaded using the Transformers library.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)
```

The model may be used directly with PEFT adapters or merged into a standalone checkpoint.

---

# Intended Applications

The model is suitable for:

- Regulatory obligation extraction.
- Compliance automation.
- Governance, Risk and Compliance (GRC).
- Regulatory intelligence.
- Legal NLP research.
- Knowledge graph construction.
- Retrieval-Augmented Generation (RAG) pipelines.

---

# Not Intended For

This model should **not** be used for:

- Legal advice.
- Regulatory interpretation.
- Contract drafting.
- Court proceedings.
- High-risk compliance decisions without human oversight.

---

# Future Evolution

This model represents the first specialized component within the broader ComplianceGPT platform.

Future model releases are expected to include:

- Regulatory Entity Recognition SLM.
- Compliance Classification SLM.
- Regulatory Summarization SLM.
- Compliance Question Answering SLM.
- Policy Mapping SLM.
- Knowledge Graph Construction SLM.

These models will eventually work together through a modular orchestration layer.

---

# Summary

The first ComplianceGPT model demonstrates how a modern instruction-tuned SLM can be specialized for regulatory obligation extraction using parameter-efficient fine-tuning.

By combining Gemma 4, QLoRA, and structured instruction tuning, the model transforms complex regulatory language into machine-readable outputs that can serve as the foundation for intelligent compliance applications.