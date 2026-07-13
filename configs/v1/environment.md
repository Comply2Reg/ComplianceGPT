# ComplianceGPT Environment

## Overview

This document describes the software and hardware environment used during the development and fine-tuning of **ComplianceGPT v1**.

The objective of this document is to maximize reproducibility by recording the execution environment, core libraries, hardware specifications, and development platform.

---

# Environment Summary

| Property | Value |
|----------|-------|
| Project | ComplianceGPT |
| Version | v1 |
| Primary Task | Regulatory Obligation Extraction |
| Fine-Tuning Method | QLoRA |
| Training Framework | Unsloth |
| Experiment Tracking | MLflow |

---

# Development Platform

The initial development and experimentation were performed using cloud-based notebook environments.

| Component | Value |
|----------|-------|
| Development Environment | Google Colab |
| Notebook Type | Jupyter Notebook |
| Runtime | Python 3 |
| Accelerator | GPU |

The project was designed so that future versions can also be reproduced in local environments or cloud GPU instances.

---

# Hardware

The published model was trained using a commodity GPU environment.

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 16 GB |
| CUDA Support | Yes |
| Mixed Precision | FP16 |
| Quantization | 4-bit NF4 |

The hardware requirements were intentionally kept modest to demonstrate that domain-specialized SLMs can be trained without enterprise-scale infrastructure.

---

# Software Stack

The project relies on the following open-source ecosystem.

| Library | Purpose |
|----------|---------|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Model loading and inference |
| Unsloth | Efficient QLoRA training |
| PEFT | Parameter-efficient fine-tuning |
| TRL | Supervised fine-tuning |
| BitsAndBytes | Quantization |
| MLflow | Experiment tracking |

---

# Python Environment

The following packages are required for reproducing the training workflow.

```text
torch
transformers
unsloth
peft
trl
accelerate
bitsandbytes
datasets
mlflow
numpy
pandas
scikit-learn
tqdm
```

These dependencies are documented separately in `requirements.txt`.

---

# CUDA

ComplianceGPT v1 supports CUDA-enabled GPU execution.

Recommended configuration:

| Setting | Value |
|----------|-------|
| CUDA | Enabled |
| Mixed Precision | FP16 |
| Quantization | 4-bit NF4 |

GPU acceleration is recommended for both fine-tuning and inference.

---

# Memory Optimizations

Several techniques were used to reduce GPU memory consumption.

- 4-bit NF4 quantization
- QLoRA
- LoRA adapters
- Gradient checkpointing
- FP16 mixed precision
- AdamW 8-bit optimizer

Together, these techniques enable fine-tuning on GPUs with approximately 16 GB of VRAM.

---

# Project Dependencies

ComplianceGPT depends on several major components.

```text
ComplianceGPT

│

├── Python

├── PyTorch

├── Transformers

├── Unsloth

├── PEFT

├── BitsAndBytes

├── TRL

└── MLflow
```

---

# Directory Expectations

The expected repository structure is:

```text
ComplianceGPT/

configs/

data/

docs/

images/

models/

notebooks/

src/
```

This organization separates documentation, datasets, source code, experiments, and configuration files.

---

# Reproducibility

To reproduce ComplianceGPT v1, ensure:

- The correct dataset version is used.
- Matching configuration files are loaded.
- The same base model is selected.
- LoRA settings remain unchanged.
- Quantization settings match the published configuration.
- Random seeds are fixed where applicable.

Reproducing the exact software environment further improves consistency across experiments.

---

# Environment Verification

Before beginning training, verify the following:

- Python is installed.
- CUDA is available (if using a GPU).
- Required libraries are installed.
- The Hugging Face model can be downloaded.
- MLflow is configured.
- Dataset files are accessible.

---

# Future Environments

Future releases may support additional environments, including:

- Local Linux workstations
- Docker containers
- Kubernetes deployments
- Vertex AI
- Azure Machine Learning
- AWS SageMaker

Configuration files will be versioned independently for each major release to maintain reproducibility.

---

# Summary

ComplianceGPT v1 was developed using an open-source software stack centered around Python, PyTorch, Hugging Face Transformers, Unsloth, PEFT, and MLflow.

The project demonstrates that modern parameter-efficient fine-tuning techniques can be applied to domain-specific Legal AI tasks using accessible cloud GPU resources while maintaining reproducibility through comprehensive environment documentation.