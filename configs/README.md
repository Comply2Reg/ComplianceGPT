# ComplianceGPT Configuration Files

## Overview

This directory contains the configuration files used for training, inference, evaluation, and experiment tracking within ComplianceGPT.

Configuration files are versioned independently to ensure reproducibility and maintain compatibility between datasets, models, and training pipelines.

Each model release references a specific configuration version, allowing experiments to be recreated exactly as originally executed.

---

# Directory Structure

```text
configs/

├── README.md

└── v1/
    ├── training_config.yaml
    ├── lora_config.yaml
    ├── generation_config.yaml
    ├── inference_config.yaml
    ├── dataset_config.yaml
    ├── mlflow_config.yaml
    └── environment.md
```

---

# Configuration Categories

The configuration files are organized by purpose.

| File | Purpose |
|------|----------|
| training_config.yaml | Training hyperparameters |
| lora_config.yaml | LoRA and QLoRA settings |
| generation_config.yaml | Text generation parameters |
| inference_config.yaml | Model loading configuration |
| dataset_config.yaml | Dataset information |
| mlflow_config.yaml | Experiment tracking configuration |
| environment.md | Software and hardware environment |

---

# Versioning

Configuration files follow semantic versioning.

Example:

```text
configs/

v1/

v2/

v3/
```

Each model release should reference the exact configuration version used during training.

---

# Why Version Configurations?

Separating configuration versions provides several benefits:

- Reproducibility
- Experiment tracking
- Easier comparison between model releases
- Backward compatibility
- Simplified maintenance

---

# Relationship to Other Components

Each configuration version corresponds to:

- One dataset version
- One training pipeline
- One or more model releases

For example:

| Component | Version |
|-----------|---------|
| Dataset | v1 |
| Configurations | v1 |
| Model | ComplianceGPT v1 |

---

# Summary

The `configs/` directory provides a version-controlled record of every configuration used throughout the ComplianceGPT project, ensuring that experiments remain reproducible as the platform evolves.