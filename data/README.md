# ComplianceGPT Datasets

## Overview

This directory contains the datasets, schemas, and supporting documentation used for training and evaluating ComplianceGPT.

The datasets are organized using **semantic versioning** to ensure reproducibility and to preserve the exact training data associated with each model release.

Each dataset version contains:

- Dataset documentation
- Dataset schema
- Data dictionary
- Dataset statistics
- Sample records
- (Optional) Training dataset

This organization enables researchers and developers to reproduce previous experiments while allowing future datasets to evolve independently.

---

# Directory Structure

```text
data/

├── README.md

└── v1/
    ├── README.md
    ├── schema.md
    ├── data_dictionary.md
    ├── dataset_statistics.md
    ├── sample_dataset.jsonl
    └── resampled_ft-v2.jsonl
```

---

# Dataset Versions

## Version 1

Current dataset used for:

- ComplianceGPT v1
- Gemma 4 E2B IT Regulatory Obligation Extraction
- QLoRA Fine-Tuning

Version 1 contains the instruction dataset used to train the first publicly released ComplianceGPT model.

Future releases will introduce additional version folders while preserving previous datasets for reproducibility.

Example:

```text
data/

v1/

v2/

v3/
```

---

# Dataset Contents

Each version contains the following files.

| File | Description |
|------|-------------|
| README.md | Version-specific dataset documentation |
| schema.md | JSON schema used by the model |
| data_dictionary.md | Description of every dataset field |
| dataset_statistics.md | Dataset statistics and distributions |
| sample_dataset.jsonl | Representative training samples |
| resampled_ft-v2.jsonl | Full resampled training dataset (optional) |

---

# Dataset Philosophy

ComplianceGPT datasets are designed specifically for **instruction tuning** rather than conventional document classification.

The datasets prioritize:

- Regulatory obligation extraction
- Structured JSON generation
- Domain-specific supervision
- Consistent annotation
- Machine-readable outputs

Unlike generic NLP datasets, these datasets are engineered to support downstream Governance, Risk, and Compliance (GRC) workflows.

---

# Data Sources

The datasets are derived from publicly available regulatory and compliance documents.

Representative sources include:

- RBI Guidelines
- MAS Regulations

The datasets do **not** include confidential or proprietary regulatory material.

---

# Versioning Policy

ComplianceGPT follows a versioned dataset strategy.

A new dataset version will be created whenever there is a significant change in:

- Annotation methodology
- Output schema
- Prompt format
- Regulatory coverage
- Data quality
- Labeling guidelines

Minor documentation updates do not require a new dataset version.

---

# Reproducibility

Each published ComplianceGPT model references a specific dataset version.

This ensures that:

- Training experiments can be reproduced.
- Previous model releases remain traceable.
- Future improvements do not overwrite historical datasets.
- Researchers can compare models trained on different dataset versions.

---

# Future Dataset Roadmap

Future dataset versions may include:

- Additional regulatory frameworks
- Multilingual regulatory documents
- Expanded obligation categories
- Cross-document annotations
- Regulatory entity linking
- Knowledge graph annotations
- Human-reviewed evaluation datasets

---

# Related Documentation

For detailed information, refer to:

- `v1/README.md`
- `v1/schema.md`
- `v1/data_dictionary.md`
- `v1/dataset_statistics.md`

---

# Summary

The `data/` directory provides a version-controlled home for all datasets used within ComplianceGPT.

By versioning datasets independently from models, the project supports reproducible experimentation while enabling future dataset evolution without disrupting previous releases.