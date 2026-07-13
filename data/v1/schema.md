# ComplianceGPT Dataset Schema (v1)

## Overview

This document defines the official schema for **ComplianceGPT Dataset v1**.

Unlike conventional instruction-tuning datasets that contain only prompts and responses, ComplianceGPT Dataset v1 is designed to capture the complete reasoning process behind regulatory obligation extraction.

Each training sample contains:

- Source metadata
- Instruction
- Regulatory text
- Optional tool invocation
- Reasoning trace
- Structured output

This richer representation enables future experimentation with explainable Legal AI, tool-augmented reasoning, and retrieval-based compliance systems.

---

# Dataset Structure

Each record is a JSON object with the following top-level structure.

```text
Record

├── metadata
├── classification
├── instruction
├── input_text
├── tool_use
├── thought_trace
└── output
```

---

# Top-Level Fields

| Field | Type | Required | Description |
|---------|------|----------|-------------|
| metadata | Object | ✅ | Source document information |
| classification | String | ✅ | Sample class |
| instruction | String | ✅ | Task instruction |
| input_text | String | ✅ | Regulatory text |
| tool_use | Object | ✅ | Simulated external tool interaction |
| thought_trace | String | ✅ | Model reasoning trace |
| output | Object / Array | ✅ | Expected structured response |

---

# Metadata

The metadata object records the origin of the regulatory text.

```json
"metadata": {
  "source_id": "...",
  "regulation_name": "...",
  "jurisdiction": "...",
  "doc_type": "...",
  "section": "..."
}
```

## Metadata Fields

| Field | Type | Description |
|--------|------|-------------|
| source_id | String | Unique chunk identifier |
| regulation_name | String | Regulation or standard |
| jurisdiction | String | Regulatory jurisdiction |
| doc_type | String | Document type |
| section | String | Source section |

---

# Classification

Every record belongs to exactly one semantic class.

Allowed values:

| Value | Description |
|--------|-------------|
| obligation | Mandatory regulatory requirement |
| non_obligation | No enforceable obligation |
| neutral | Informational or descriptive content |

---

# Instruction

Example

```json
"instruction": "Analyze the text to extract compliance obligations."
```

The instruction defines the task expected from the language model.

---

# Input Text

```json
"input_text": "Every financial institution shall maintain customer records..."
```

This field contains the regulatory passage presented to the model.

---

# Tool Use

One distinctive feature of ComplianceGPT Dataset v1 is the inclusion of **tool interaction metadata**.

```json
"tool_use": {
    "tool_rationale": "...",
    "tool_call": {
        "function": "...",
        "parameters": "..."
    },
    "external_context": "..."
}
```

Although the tools are simulated during training, this design prepares the dataset for future tool-augmented compliance systems.

## Tool Use Fields

| Field | Description |
|---------|-------------|
| tool_rationale | Why the tool should be invoked |
| tool_call.function | Tool name |
| tool_call.parameters | Tool parameters |
| external_context | Retrieved information |

---

# Thought Trace

```json
"thought_trace": "1. Identify the modal verb..."
```

The thought trace captures the reasoning process used to determine the final classification.

It generally includes:

1. Modal verb identification
2. Context analysis
3. Obligation assessment
4. Final decision

This field is intended for research into explainable Legal AI and reasoning-aware instruction tuning.

---

# Output

The output schema depends on the value of `classification`.

```
classification

├── obligation

├── non_obligation

└── neutral
```

---

# Obligation Output

For obligation samples, `output` is an **array**.

Example

```json
[
  {
    "obligation_id": "OBL-<chunk_id>",
    "subject": "...",
    "modality": "MUST",
    "modality_detected": "shall",
    "action": "...",
    "conditions": "...",
    "deadline": null,
    "reference_anchor": "..."
  }
]
```

## Obligation Fields

| Field | Type | Required |
|--------|------|----------|
| obligation_id | String | ✅ |
| subject | String | ✅ |
| modality | String | ✅ |
| modality_detected | String | ✅ |
| action | String | ✅ |
| conditions | String / Null | Optional |
| deadline | String / Null | Optional |
| reference_anchor | String / Null | Optional |

---

# Non-Obligation Output

For non-obligation samples, `output` is an object.

Example

```json
{
    "status":"rejected",
    "message":"No obligation present",
    "reason":"The text uses permissive language."
}
```

| Field | Description |
|--------|-------------|
| status | Processing status |
| message | User-facing message |
| reason | Explanation |

---

# Neutral Output

Example

```json
{
    "status":"no_obligation",
    "message":"Neutral content",
    "reason":"Informational statement."
}
```

---

# Validation Rules

Each record must satisfy the following requirements.

## General

- All top-level fields must exist.
- `classification` must be valid.
- `instruction` must not be empty.
- `input_text` must not be empty.
- `thought_trace` must not be empty.

---

## Obligation Records

If

```text
classification = obligation
```

Then:

- `output` must be an array.
- At least one obligation object must exist.
- Every obligation must contain:
  - obligation_id
  - subject
  - modality
  - modality_detected
  - action

---

## Non-Obligation Records

If

```text
classification = non_obligation
```

Then:

- `output.status` must exist.
- `message` must exist.
- `reason` must exist.

---

## Neutral Records

If

```text
classification = neutral
```

Then:

- `status` must exist.
- `message` must exist.
- `reason` must exist.

---

# Dataset Statistics

| Property | Value |
|-----------|------:|
| Total Samples | 860 |
| Obligation | 430 |
| Neutral | 258 |
| Non-Obligation | 172 |
| Output Types | 3 |
| Instruction Format | Custom |
| Tool Metadata | Included |
| Reasoning Trace | Included |

---

# Design Rationale

ComplianceGPT Dataset v1 extends conventional instruction-tuning datasets by incorporating metadata, simulated tool interactions, and explicit reasoning traces.

This richer schema enables experimentation with:

- Explainable AI
- Tool-augmented language models
- Retrieval-Augmented Generation (RAG)
- GraphRAG
- Multi-agent compliance systems
- Reasoning-aware fine-tuning

The schema is intentionally designed to support future versions of ComplianceGPT while remaining compatible with instruction-tuning workflows.