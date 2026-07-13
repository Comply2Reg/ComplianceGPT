# Dataset Documentation

## Overview

The effectiveness of a domain-specific language model depends not only on the underlying architecture but also on the quality, diversity, and structure of its training data. Since ComplianceGPT is designed for **regulatory obligation extraction**, a custom instruction-tuning dataset was created specifically for this task.

Unlike generic language model datasets, the objective was not to maximize linguistic diversity but to teach the model to distinguish **actionable regulatory obligations** from surrounding explanatory or contextual text.

The dataset was therefore engineered to support structured information extraction rather than free-form text generation.

---

# Objectives

The dataset was designed with the following objectives:

- Identify regulatory obligations.
- Distinguish obligations from informational text.
- Extract structured compliance entities.
- Produce consistent JSON outputs.
- Support supervised instruction tuning.
- Improve downstream compliance automation.

---

# Regulatory Sources

The training corpus was compiled from publicly available regulatory and compliance documents spanning multiple industries.

Representative sources include:

| Source | Domain |
|----------|---------|
| RBI Guidelines | Banking |
| MAS Regulations | Financial Services |

These regulations contain a wide variety of linguistic structures, ranging from explicit mandatory obligations to descriptive guidance and definitions.

---

# Dataset Creation Pipeline

The overall dataset preparation workflow is illustrated below.

```text
Regulatory Documents
        │
        ▼
Document Parsing
        │
        ▼
Text Chunking
        │
        ▼
Instruction Creation
        │
        ▼
JSON Formatting
        │
        ▼
Dataset Validation
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

Each stage contributes to improving data quality and consistency before model training.

---

# Annotation Strategy

Every text segment was assigned one of three semantic classes according to its regulatory intent.

| Class | Description |
|--------|-------------|
| Obligation | Mandatory regulatory requirement requiring action. |
| Non-Obligation | Statements that do not impose enforceable actions. |
| Neutral | Informational or descriptive text without actionable requirements. |

This three-class formulation teaches the model both **what constitutes an obligation** and **what should not be classified as one**.

---

# Original Dataset

After preprocessing and validation, the original corpus contained **2,947** instruction samples.

| Class | Count | Percentage |
|--------|------:|-----------:|
| Obligation | 430 | 14.59% |
| Non-Obligation | 282 | 9.57% |
| Neutral | 2,235 | 75.84% |
| **Total** | **2,947** | **100%** |

The original distribution reflects the structure of real regulatory documents, where explanatory and contextual text substantially outweigh mandatory requirements.

Insert the original class distribution figure here:

```markdown
<p align="center">
<img src="../images/original_distribution.png" width="700">
</p>
```

---

# Class Imbalance

The dominance of neutral statements presents a common challenge in legal NLP.

Training directly on this naturally imbalanced distribution would likely encourage the model to predict neutral statements more frequently, reducing its effectiveness on the primary task of obligation extraction.

Since ComplianceGPT is intended for compliance automation, maximizing performance on obligation detection was prioritized over reproducing the natural distribution of regulatory documents.

---

# Resampling Strategy

To align the dataset with the intended downstream task, a targeted resampling strategy was applied.

The resampling process:

- Preserved all obligation examples.
- Reduced the proportion of neutral statements.
- Reduced the number of non-obligation samples.
- Maintained representative examples from every class.
- Increased the relative importance of actionable obligations during training.

This resulted in a more balanced optimization objective while still exposing the model to all three semantic categories.

---

# Final Training Dataset

The final instruction dataset contains **860** training samples.

| Class | Count | Percentage |
|--------|------:|-----------:|
| Obligation | 430 | 50% |
| Neutral | 258 | 30% |
| Non-Obligation | 172 | 20% |
| **Total** | **860** | **100%** |

This distribution intentionally emphasizes obligation examples while preserving sufficient diversity for robust classification.

Insert the resampled distribution figure here:

```markdown
<p align="center">
<img src="../images/resampled_distribution.png" width="700">
</p>
```

---

# Data Format

Training data is stored in **JSON Lines (JSONL)** format.

Each line represents one complete supervised instruction sample consisting of:

- System prompt
- User instruction
- Assistant response

This structure is compatible with instruction-tuned chat models and modern fine-tuning frameworks.

---

# Example Training Sample

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a legal NLP expert..."
    },
    {
      "role": "user",
      "content": "Every financial institution shall maintain customer records for five years."
    },
    {
      "role": "assistant",
      "content": {
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
    }
  ]
}
```

---

# Output Schema

The model generates structured JSON containing regulatory entities.

| Field | Description |
|----------|-------------|
| classification | Predicted semantic class. |
| subject | Responsible entity. |
| action | Required regulatory action. |
| modality | Regulatory strength. |
| conditions | Conditional requirements. |
| deadline | Time constraints, if specified. |

This schema is intended for machine consumption and integration into compliance systems.

---

# Quality Assurance

Multiple quality-control measures were applied during dataset preparation.

These included:

- Removal of invalid records.
- JSON validation.
- Consistent prompt formatting.
- Standardized output schema.
- Label verification.
- Manual inspection of representative samples.

These steps improve dataset consistency and reduce annotation noise.

---

# Limitations

Although carefully engineered, the dataset has several limitations.

- English-language regulatory documents only.
- Limited number of regulatory domains.
- Focused on obligation extraction rather than complete legal understanding.
- Does not cover every possible writing style used by regulators.
- Manual annotation decisions may introduce minor subjectivity.

These limitations provide opportunities for future dataset expansion.

---

# Future Dataset Improvements

Planned enhancements include:

- Additional regulatory frameworks.
- Multilingual support.
- Larger instruction datasets.
- Expanded obligation types.
- More complex nested obligations.
- Cross-document obligation linking.
- Automatic dataset validation pipelines.

---

# Summary

The ComplianceGPT dataset represents a purpose-built instruction corpus for regulatory obligation extraction.

Rather than mirroring the natural distribution of regulatory text, it was intentionally engineered to prioritize actionable compliance requirements while preserving representative examples of non-obligation and neutral statements.

This design supports efficient parameter-efficient fine-tuning and enables the resulting model to generate structured outputs suitable for downstream Governance, Risk, and Compliance (GRC) workflows.