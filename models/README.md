# Models

This directory contains information about the models developed as part of the **ComplianceGPT** project.

Model weights are **not stored in this repository**. They are hosted on Hugging Face to simplify distribution and version management.

## Current Model

| Model | Description |
|--------|-------------|
| **ComplianceGPT v1** | Fine-tuned Gemma 4 E2B IT model for Regulatory Obligation Extraction using QLoRA. |

## Hugging Face

The published model can be downloaded from:

**Model Repository**

https://huggingface.co/PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1

## Model Summary

| Property | Value |
|----------|-------|
| Base Model | Google Gemma 4 E2B IT |
| Fine-Tuning Method | QLoRA |
| Framework | Unsloth + PEFT |
| Task | Regulatory Obligation Extraction |
| Output | Structured JSON |

## Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "PrinceRansom7/gemma4-e2b-it-regulatory-obligation-v1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)
```

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