# Models

This directory contains information about the models developed as part of the **ComplianceGPT** project.

Model weights are **not stored in this repository**. They are hosted on Hugging Face to simplify distribution and version management.

## Current Models

| Model | Task | Base | Availability |
|--------|------|------|--------------|
| **ComplianceGPT v1** | Regulatory Obligation Extraction | Gemma 4 E2B IT | Public |
| **UK Alert Triage v1** | UK Regulatory Alert Triage | Qwen3-4B-Instruct-2507 | Private, pending licensing review |

Detailed documentation for each model:

- [uk-alert-triage-qwen3-4b-v1.md](uk-alert-triage-qwen3-4b-v1.md)

## ComplianceGPT v1 — Regulatory Obligation Extraction

**Model Repository**

https://huggingface.co/PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1

| Property | Value |
|----------|-------|
| Base Model | Google Gemma 4 E2B IT |
| Fine-Tuning Method | QLoRA |
| Framework | Unsloth + PEFT |
| Task | Regulatory Obligation Extraction |
| Output | Structured JSON |

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)
```

## UK Alert Triage v1

**Model Repository**

`Comply2Reg/uk-alert-triage-qwen3-4b-v1` — private while the training-corpus licensing is reviewed. Access is available to Comply2Reg members on request.

| Property | Value |
|----------|-------|
| Base Model | Qwen3-4B-Instruct-2507 (4-bit MLX conversion) |
| Fine-Tuning Method | LoRA rank 8, fused into the base |
| Framework | MLX / mlx-lm |
| Task | UK regulatory alert triage |
| Output | Structured JSON |
| Size | 2.1 GB (4-bit) |

### Loading the Model

```python
from mlx_lm import load, generate

model, tokenizer = load("Comply2Reg/uk-alert-triage-qwen3-4b-v1")
```

Requires Apple Silicon. See [uk-alert-triage-qwen3-4b-v1.md](uk-alert-triage-qwen3-4b-v1.md) for the prompt format, output schema, evaluation and limitations.

## Future Models

As ComplianceGPT evolves, additional task-specific models will be released, including:

- Regulatory Entity Recognition
- Compliance Classification
- Regulatory Summarization
- Compliance Question Answering
- Policy-to-Regulation Mapping

Each model will be published on Hugging Face and documented within the main project README.

---

For detailed model architecture, training methodology, and evaluation, refer to the `docs/` directory.
