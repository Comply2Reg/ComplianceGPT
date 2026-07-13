# ComplianceGPT Dataset Statistics (v1)

## Overview

This document provides a quantitative summary of **ComplianceGPT Dataset v1**.

The objective is to describe the dataset from a statistical perspective, including:

- Dataset size
- Class distribution
- Resampling strategy
- Data characteristics
- Intended training distribution

These statistics provide transparency regarding the composition of the dataset and explain the rationale behind the final training corpus.

---

# Dataset Summary

| Property | Value |
|-----------|------:|
| Dataset Version | v1 |
| Training Format | JSONL |
| Number of Classes | 3 |
| Original Samples | 2,947 |
| Final Training Samples | 860 |
| Fine-Tuning Method | Instruction Tuning |
| Output Format | Structured JSON |

---

# Original Dataset Distribution

The original corpus consisted of **2,947** instruction samples extracted from publicly available regulatory documents.

| Class | Count | Percentage |
|--------|------:|-----------:|
| Obligation | 430 | 14.59% |
| Non-Obligation | 282 | 9.57% |
| Neutral | 2,235 | 75.84% |
| **Total** | **2,947** | **100%** |

The original distribution reflects the structure of real-world regulatory documents, where explanatory and contextual statements substantially outnumber mandatory obligations.

---

# Original Dataset Visualization

Place the original class distribution chart in:

```text
images/original_distribution.png
```

Display it using:

```markdown
<p align="center">
<img src="../../images/original_distribution.png" width="700">
</p>
```

---

# Class Imbalance Analysis

The original dataset exhibits a strong class imbalance.

Approximately **three-quarters** of all samples belong to the **Neutral** class, while obligation examples account for less than **15%** of the corpus.

Although this mirrors the composition of regulatory documents, such an imbalance can bias a supervised model toward predicting the majority class.

For ComplianceGPT, the primary objective is not to reproduce the natural document distribution but to maximize performance on **regulatory obligation extraction**.

---

# Resampling Strategy

To better align the training data with the downstream task, the dataset was resampled before fine-tuning.

The resampling strategy followed three principles:

1. Preserve every validated obligation sample.
2. Retain representative non-obligation and neutral examples.
3. Increase the relative importance of obligation samples during optimization.

Rather than balancing all classes equally, a **50–30–20** distribution was selected to reflect the priorities of compliance automation systems.

---

# Final Training Distribution

The final training dataset contains **860** instruction samples.

| Class | Count | Percentage |
|--------|------:|-----------:|
| Obligation | 430 | 50.00% |
| Neutral | 258 | 30.00% |
| Non-Obligation | 172 | 20.00% |
| **Total** | **860** | **100%** |

This distribution emphasizes obligation extraction while maintaining sufficient diversity to distinguish between all three semantic categories.

---

# Resampled Dataset Visualization

Place the resampled class distribution chart in:

```text
images/resampled_distribution.png
```

Display it using:

```markdown
<p align="center">
<img src="../../images/resampled_distribution.png" width="700">
</p>
```

---

# Distribution Comparison

| Metric | Original | Resampled |
|---------|---------:|----------:|
| Total Samples | 2,947 | 860 |
| Obligation | 430 (14.59%) | 430 (50.00%) |
| Neutral | 2,235 (75.84%) | 258 (30.00%) |
| Non-Obligation | 282 (9.57%) | 172 (20.00%) |

---

# Why the Dataset Was Resampled

ComplianceGPT is designed to support **regulatory obligation extraction**, not general document understanding.

Training on the original distribution would encourage the model to over-predict the dominant Neutral class because it accounts for most of the training examples.

The resampled dataset shifts the optimization objective toward recognizing actionable obligations while still exposing the model to representative examples of the remaining semantic classes.

This design reflects the practical needs of Governance, Risk, and Compliance (GRC) systems, where identifying mandatory requirements is significantly more valuable than classifying descriptive text.

---

# Dataset Characteristics

ComplianceGPT Dataset v1 has the following characteristics.

| Characteristic | Description |
|----------------|-------------|
| Domain | Regulatory Compliance |
| Language | English |
| Annotation Type | Instruction Tuning |
| Output Style | Structured JSON |
| Primary Task | Regulatory Obligation Extraction |
| Secondary Task | Obligation Classification |

---

# Annotation Coverage

The dataset includes examples from multiple regulatory domains.

Representative sources include:

- GDPR
- ISO 27001
- RBI Guidelines
- HIPAA
- PCI DSS
- DORA
- MAS Regulations
- SEC Regulations
- FINRA
- NIS2

These sources expose the model to diverse regulatory writing styles and obligation patterns.

---

# Dataset Strengths

ComplianceGPT Dataset v1 was designed to maximize utility for compliance-focused language models.

Key strengths include:

- Domain-specific instruction tuning.
- Structured JSON supervision.
- Multi-domain regulatory coverage.
- High-quality obligation annotations.
- Balanced optimization objective.
- Explainable output schema.
- Metadata-rich samples.

---

# Known Limitations

Although carefully engineered, the dataset has several limitations.

- English-language documents only.
- Limited regulatory coverage compared to the global regulatory landscape.
- Moderate dataset size.
- Manual annotation decisions may introduce minor subjectivity.
- Long documents require chunking before inclusion.

Future dataset versions are expected to expand both coverage and diversity.

---

# Planned Dataset Growth

Future releases may include:

- Additional jurisdictions.
- Financial regulations from more countries.
- Cybersecurity standards.
- ESG regulations.
- AI governance frameworks.
- Multilingual regulatory documents.
- Larger instruction datasets.
- Human-reviewed benchmark sets.

---

# Relationship to Other Documentation

| Document | Purpose |
|----------|---------|
| README.md | Dataset overview |
| schema.md | Structural specification |
| data_dictionary.md | Field definitions |
| ../../docs/dataset.md | Dataset engineering methodology |

---

# Summary

ComplianceGPT Dataset v1 provides a carefully engineered instruction-tuning corpus for regulatory obligation extraction.

Although the original dataset accurately reflected the composition of regulatory documents, a targeted resampling strategy was applied to better align the training objective with the downstream task of identifying actionable compliance obligations.

The resulting dataset provides a balanced and reproducible foundation for fine-tuning specialized Legal AI models while preserving representative examples from all semantic classes.