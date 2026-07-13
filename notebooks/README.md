# Notebooks

This directory contains the Jupyter notebook(s) used to develop and reproduce **ComplianceGPT**.

## Contents

| Notebook | Description |
|----------|-------------|
| `ComplianceGPT_v1_Training.ipynb` | End-to-end workflow covering dataset preparation, QLoRA fine-tuning, MLflow experiment tracking, inference, and Hugging Face model publishing. |
 
## Requirements

Before running the notebook, ensure:

- Python dependencies from `requirements.txt` are installed.
- The dataset is available in the expected location.
- A CUDA-enabled GPU is recommended for training.

## Related Resources

- `data/` – Dataset and schema documentation
- `configs/` – Training and inference configurations
- `docs/` – Technical documentation
- `models/` – Model information and Hugging Face links

---

**Note:** The notebook is intended as a reproducible reference implementation for ComplianceGPT v1. Future releases may split the workflow into multiple notebooks as the project expands.