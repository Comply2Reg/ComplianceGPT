# ComplianceGPT Dataset Data Dictionary (v1)

## Overview

This document describes every field used in **ComplianceGPT Dataset v1**.

The objective of this document is to provide a semantic description of each dataset field beyond its structural definition.

Whereas `schema.md` specifies **how** records are structured, the data dictionary explains **what every field means**, why it exists, and how it should be interpreted.

This document should be considered the authoritative reference for anyone:

- Creating new training examples
- Extending the dataset
- Building annotation tools
- Developing downstream applications
- Integrating ComplianceGPT into production systems

---

# Dataset Structure

Each dataset record contains the following top-level fields.

| Field | Type | Description |
|--------|------|-------------|
| metadata | Object | Source information for the regulatory passage |
| classification | String | Semantic class assigned to the regulatory text |
| instruction | String | Task instruction presented to the model |
| input_text | String | Regulatory text to be analyzed |
| tool_use | Object | Simulated external tool interaction |
| thought_trace | String | Reasoning process used to derive the final output |
| output | Object / Array | Expected structured response |

---

# metadata

## Description

The `metadata` object contains contextual information describing the origin of the regulatory text.

This information is not part of the regulatory content itself but provides traceability and provenance.

---

## Fields

| Field | Type | Description | Example |
|--------|------|-------------|---------|
| source_id | String | Unique identifier for the source text chunk | GDPR-ART5-001 |
| regulation_name | String | Name of the regulation or standard | GDPR |
| jurisdiction | String | Governing jurisdiction | European Union |
| doc_type | String | Type of regulatory document | Regulation |
| section | String | Section or article reference | Article 5 |

---

## Why Metadata Matters

Metadata enables:

- Traceability
- Regulatory source attribution
- Auditability
- Future knowledge graph generation
- Source-aware retrieval

---

# classification

## Description

The `classification` field identifies the semantic category assigned to the regulatory passage.

Every record belongs to **exactly one** class.

---

## Allowed Values

| Value | Meaning |
|--------|---------|
| obligation | Mandatory regulatory requirement |
| non_obligation | No enforceable regulatory action |
| neutral | Informational or descriptive statement |

---

## Purpose

This field determines:

- Which output schema should be used.
- Which evaluation metrics apply.
- Which downstream workflow consumes the result.

---

# instruction

## Description

The instruction tells the model **what task it should perform**.

Unlike conversational datasets, ComplianceGPT uses highly structured task instructions to encourage deterministic outputs.

---

## Example

```text
Analyze the regulatory text and extract all compliance obligations.
```

---

## Characteristics

A good instruction should:

- Clearly describe the task.
- Avoid ambiguity.
- Encourage structured output.
- Be independent of the regulatory content.

---

# input_text

## Description

The `input_text` field contains the regulatory passage presented to the model.

This is the primary information source used for reasoning.

---

## Characteristics

Input text may contain:

- Mandatory obligations
- Recommendations
- Definitions
- Informational statements
- References
- Exceptions

---

## Example

```text
Every financial institution shall maintain customer records for five years.
```

---

# tool_use

## Description

The `tool_use` object represents simulated external tool interactions.

Although no external tool is executed during training, this field captures how future ComplianceGPT systems may interact with external resources.

---

## Purpose

This field prepares the dataset for:

- Tool-augmented LLMs
- Retrieval-Augmented Generation (RAG)
- GraphRAG
- Compliance APIs
- Knowledge graph lookups

---

## Fields

| Field | Description |
|--------|-------------|
| tool_rationale | Why the tool should be invoked |
| tool_call.function | Tool name |
| tool_call.parameters | Input parameters |
| external_context | Retrieved supporting information |

---

## Example

```json
{
  "tool_rationale": "Retrieve regulatory metadata.",
  "tool_call": {
    "function": "lookup_regulation",
    "parameters": {
      "regulation": "GDPR",
      "article": "5"
    }
  },
  "external_context": "Article 5 establishes the principles relating to personal data processing."
}
```

---

# thought_trace

## Description

The `thought_trace` field records the reasoning process used to derive the expected output.

Unlike free-form explanations, this trace is intended to represent a structured analytical workflow.

---

## Purpose

Reasoning traces support research into:

- Explainable AI
- Legal reasoning
- Transparent decision making
- Multi-step inference
- Future reasoning-aware training

---

## Typical Reasoning Steps

A reasoning trace may include:

1. Identify modal verbs.
2. Determine the regulated subject.
3. Identify required actions.
4. Detect conditions.
5. Detect deadlines.
6. Assign classification.
7. Produce structured output.

---

# output

The interpretation of the `output` field depends on the value of `classification`.

---

# Obligation Output

When

```text
classification = obligation
```

the output contains one or more structured obligation objects.

---

## Fields

| Field | Description |
|--------|-------------|
| obligation_id | Unique obligation identifier |
| subject | Responsible entity |
| modality | Normalized regulatory strength |
| modality_detected | Original modal expression found in text |
| action | Required regulatory action |
| conditions | Preconditions associated with the obligation |
| deadline | Time constraint |
| reference_anchor | Source location within the regulation |

---

## Example

```json
{
  "obligation_id": "OBL-001",
  "subject": "Financial Institution",
  "modality": "MUST",
  "modality_detected": "shall",
  "action": "Maintain customer records",
  "conditions": null,
  "deadline": "Five years",
  "reference_anchor": "Article 12"
}
```

---

# Non-Obligation Output

For non-obligation samples, the model returns a structured explanation instead of extracted obligations.

---

## Fields

| Field | Description |
|--------|-------------|
| status | Processing outcome |
| message | Human-readable summary |
| reason | Explanation for the classification |

---

## Example

```json
{
  "status": "rejected",
  "message": "No regulatory obligation detected.",
  "reason": "The passage provides explanatory guidance rather than an enforceable requirement."
}
```

---

# Neutral Output

Neutral statements represent contextual information that does not require compliance actions.

---

## Example

```json
{
  "status": "no_obligation",
  "message": "Informational statement.",
  "reason": "The text provides background information without imposing mandatory requirements."
}
```

---

# Normalized Modality

The `modality` field standardizes different linguistic expressions into a consistent representation.

| Normalized Value | Typical Expressions |
|------------------|---------------------|
| MUST | shall, must, is required to |
| SHOULD | should, recommended to |
| MAY | may, can, permitted to |
| REQUIRED | required, mandatory |

This normalization simplifies downstream analytics while preserving the original wording through `modality_detected`.

---

# Null Values

Some fields may legitimately contain `null`.

Examples include:

- No deadline specified.
- No conditions present.
- No reference anchor available.

Null values indicate that the information was **not present**, not that extraction failed.

---

# Data Quality Guidelines

Dataset records should satisfy the following principles.

- Correct semantic classification.
- Accurate obligation extraction.
- Consistent JSON formatting.
- Traceable metadata.
- Deterministic reasoning.
- Valid output schema.

---

# Relationship to Other Documentation

| Document | Purpose |
|----------|---------|
| README.md | Dataset overview |
| schema.md | Structural specification |
| dataset_statistics.md | Quantitative analysis |
| ../../docs/dataset.md | Dataset engineering methodology |

---

# Summary

The ComplianceGPT Data Dictionary provides the semantic specification for Dataset v1.

Together with the accompanying schema and documentation, it establishes a consistent and reproducible foundation for regulatory obligation extraction, ensuring that future dataset versions remain interpretable, extensible, and compatible with downstream compliance systems.