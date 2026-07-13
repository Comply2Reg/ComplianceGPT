# Training Methodology

## Overview

ComplianceGPT's first model was developed using **parameter-efficient fine-tuning (PEFT)** to specialize a general-purpose instruction-tuned language model for regulatory obligation extraction.

Rather than updating every parameter of the base model, the project adopts **Quantized Low-Rank Adaptation (QLoRA)**, enabling efficient training while significantly reducing GPU memory requirements.

The training pipeline emphasizes:

- Domain specialization
- Computational efficiency
- Reproducibility
- Experiment tracking
- Structured output generation

---

# Training Pipeline

The complete training workflow is illustrated below.

```text
                Regulatory Documents
                         │
                         ▼
                Dataset Preparation
                         │
                         ▼
              Instruction Dataset (.jsonl)
                         │
                         ▼
              Prompt Formatting
                         │
                         ▼
             Gemma 4 E2B IT Base Model
                         │
                         ▼
              Load Model in 4-bit NF4
                         │
                         ▼
               Attach LoRA Adapters
                         │
                         ▼
                  QLoRA Fine-Tuning
                         │
                         ▼
                Evaluation & Validation
                         │
                         ▼
                 MLflow Experiment Log
                         │
                         ▼
                Hugging Face Deployment
```

Each stage is designed to be reproducible and modular, allowing future iterations to replace or improve individual components without redesigning the entire workflow.

---

# Base Model

The first ComplianceGPT model is built upon Google's **Gemma 4 E2B IT**, an instruction-tuned Small Language Model (SLM).

The base model was selected because it provides:

- Strong instruction-following performance
- Efficient parameter-efficient fine-tuning
- Compatibility with open-source tooling
- Low-memory deployment
- High-quality structured text generation

Rather than learning language from scratch, ComplianceGPT adapts this pretrained model to the legal and compliance domain.

---

# Why QLoRA?

Training an entire language model requires substantial computational resources.

QLoRA enables domain adaptation by freezing the original model parameters and learning only a small set of trainable adapter weights.

Advantages include:

- Significantly lower GPU memory usage
- Faster training
- Smaller checkpoints
- Efficient experimentation
- Easy adapter sharing
- Near full fine-tuning performance

This makes specialized Legal AI models practical even on modest hardware.

---

# Low-Rank Adaptation (LoRA)

LoRA injects trainable low-rank matrices into selected transformer layers while leaving the original model weights unchanged.

Instead of updating billions of parameters, only a small fraction of the network is optimized.

Conceptually:

```text
Original Transformer Layer
        │
        ▼
Frozen Parameters
        │
        ▼
LoRA Adapter
        │
        ▼
Task-Specific Knowledge
```

This approach allows the model to acquire regulatory knowledge without overwriting its pretrained linguistic capabilities.

---

# Quantization

The base model was loaded using **4-bit NormalFloat (NF4)** quantization.

Benefits include:

- Lower VRAM requirements
- Faster loading
- Reduced storage footprint
- Larger effective context on limited hardware
- Efficient fine-tuning with QLoRA

Quantization is particularly valuable when training on cloud GPUs with limited memory.

---

# Training Configuration

The following configuration was used during fine-tuning.

| Parameter | Value |
|-----------|------:|
| Base Model | Google Gemma 4 E2B IT |
| Fine-Tuning Method | QLoRA |
| Quantization | 4-bit NF4 |
| Context Length | 1024 |
| LoRA Rank | 16 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW 8-bit |
| Scheduler | Cosine |
| Epochs | 5 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Mixed Precision | FP16 |
| Gradient Checkpointing | Enabled (Unsloth) |

These values were selected to balance convergence, memory efficiency, and training stability.

---

# Training Environment

Training was conducted using a cloud-based GPU environment.

| Component | Specification |
|------------|---------------|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 (16 GB) |
| Framework | Unsloth |
| Transformers | Hugging Face Transformers |
| Adapter Library | PEFT |
| Training Library | TRL |
| Quantization | BitsAndBytes |
| Experiment Tracking | MLflow |

This configuration enables efficient fine-tuning without requiring enterprise-scale hardware.

---

# Instruction-Tuning Strategy

Unlike conventional language modeling, ComplianceGPT was trained using **instruction-response pairs**.

Each training sample consists of:

1. System prompt
2. User input
3. Expected assistant response

This format teaches the model to:

- Understand regulatory language
- Identify obligation statements
- Produce structured JSON
- Follow a predefined schema

Instruction tuning improves consistency during inference and reduces output variability.

---

# Training Objective

The primary training objective is not generic text generation.

Instead, the model is optimized to:

- Recognize regulatory obligations
- Ignore irrelevant informational text
- Produce structured machine-readable outputs
- Maintain consistent formatting
- Preserve semantic accuracy

This task-specific objective aligns the model with downstream compliance automation requirements.

---

# Memory Optimization Techniques

Several optimization techniques were employed to maximize hardware efficiency.

## 4-bit Quantization

Reduces model memory consumption.

---

## LoRA Adapters

Minimizes the number of trainable parameters.

---

## Gradient Checkpointing

Trades additional computation for lower memory usage during backpropagation.

---

## Mixed Precision (FP16)

Reduces memory usage while accelerating training.

---

## AdamW 8-bit Optimizer

Further decreases optimizer memory overhead.

Together, these optimizations enable effective fine-tuning on commodity GPUs.

---

# Experiment Tracking

Every training run was tracked using **MLflow**.

The following information was logged:

- Hyperparameters
- Dataset version
- Learning rate
- Optimizer settings
- LoRA configuration
- Evaluation metrics
- Model checkpoints
- Training artifacts

MLflow improves reproducibility by maintaining a complete history of experiments and simplifying comparison across training runs.

---

# Model Checkpoints

During training, checkpoints can be periodically saved to enable:

- Recovery from interruptions
- Performance comparison
- Adapter export
- Future model merging

Checkpointing also supports iterative experimentation without restarting from scratch.

---

# Model Export

After fine-tuning, the resulting adapters can be:

- Loaded with PEFT
- Merged into the base model
- Uploaded to Hugging Face
- Shared independently
- Integrated into downstream applications

This flexibility simplifies deployment across different inference environments.

---

# Reproducibility

ComplianceGPT emphasizes reproducible machine learning practices.

The repository includes:

- Training notebooks
- Configuration files
- Dataset documentation
- Hyperparameter settings
- Prompt templates
- Model card
- Experiment tracking
- Deployment instructions

Together, these resources allow researchers and developers to reproduce the fine-tuning process or extend it for additional compliance-related tasks.

---

# Lessons Learned

Several observations emerged during development:

- Dataset quality has a greater impact than simply increasing dataset size.
- Class imbalance significantly affects obligation extraction performance.
- Structured outputs improve downstream automation compared to free-form text.
- QLoRA enables efficient specialization without requiring extensive hardware.
- Experiment tracking is essential for comparing fine-tuning configurations and ensuring reproducibility.

These insights will guide future iterations of ComplianceGPT.

---

# Future Improvements

Future training pipelines may include:

- Multi-stage instruction tuning
- Curriculum learning
- Domain-adaptive pretraining
- Synthetic data generation
- Active learning
- Multi-task fine-tuning
- Reinforcement learning from domain feedback
- Automated hyperparameter optimization

These enhancements are intended to improve both model performance and training efficiency.

---

# Summary

ComplianceGPT demonstrates that modern parameter-efficient fine-tuning techniques can transform a general-purpose language model into a domain-specialized regulatory assistant.

By combining **Gemma 4**, **QLoRA**, **Unsloth**, **PEFT**, **BitsAndBytes**, and **MLflow**, the project establishes a reproducible and resource-efficient pipeline for developing specialized Legal AI systems capable of extracting structured regulatory obligations.