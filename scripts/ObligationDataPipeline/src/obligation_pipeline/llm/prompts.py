"""Prompt builders for LLM-based structural parsing."""

from __future__ import annotations

from typing import Optional

from .structure_models import DocumentMetadata, NodeType

UNIVERSAL_TYPE_LIST = ", ".join(t.value for t in NodeType)


def build_structural_parse_prompt(
    text: str,
    metadata: Optional[DocumentMetadata] = None,
) -> str:
    """Build an instruction prompt for structure-aware parsing of a single document."""
    meta_hint_lines: list[str] = []
    if metadata:
        if metadata.jurisdiction:
            meta_hint_lines.append(f"- Jurisdiction: {metadata.jurisdiction}")
        if metadata.document_type:
            meta_hint_lines.append(f"- Document type: {metadata.document_type}")
        if metadata.language:
            meta_hint_lines.append(f"- Language: {metadata.language}")
        if metadata.title:
            meta_hint_lines.append(f"- Title: {metadata.title}")

    meta_hint = "\n".join(meta_hint_lines) if meta_hint_lines else "  (no extra hints provided)"

    instructions = f"""
You are a legal document structural parser.

Your task is to read the provided legal or regulatory text and return a STRICT JSON object
with this shape:

{{
  "document_metadata": {{ ... }},
  "nodes": [ {{ ... }}, ... ],
  "relationships": [ {{ ... }}, ... ]
}}

Use this universal structural node type set (enum NodeType):
{UNIVERSAL_TYPE_LIST}

Universal hierarchy (conceptual):

Document
 ├── Preamble
 ├── Recitals (optional)
 ├── Part (optional)
 │    ├── Chapter (optional)
 │    │    ├── Section (optional)
 │    │    │    ├── Article / Section (core unit)
 │    │    │    │    ├── Subsection
 │    │    │    │    │    ├── Paragraph
 │    │    │    │    │    │    ├── Subparagraph
 │    │    │    │    │    │    │    ├── Point (a), (b), (i)
 │    │    │    │    │    │    │    └── Subpoint
 ├── Schedule / Annex
 │    ├── Chapter
 │    ├── Article
 │    └── Paragraph

SPECIAL HANDLING (must follow):
1. "ARRANGEMENT OF SECTIONS" must NOT be treated as structural law. Ignore it as non-structural.
2. Recitals in EU laws must use type: "recital" and be direct children of the document (level 1).
3. Schedules and Annexes must be separate structural branches (type "schedule" or "annex").
4. Definitions sections must use type: "definition" for each defined term, not just a single blob.
5. Do NOT merge numbered clauses. Keep each numbered or lettered unit as its own node.
6. Preserve nested numbering such as (1)(a)(i) using the "numbering" array and in the id_path.

Node requirements:
- id_path: a deterministic hierarchical path string using '/' separators.
- parent_id_path: id_path of this node's parent, or null for the root document node.
- type: one of the NodeType values listed above.
- text: the full text for this structural unit (trimmed).
- numbering: list of numbering tokens, e.g. ["1", "a", "i"] for (1)(a)(i).
- order_index: integer index among siblings, starting at 0.
- page_start/page_end: approximate 1-based page numbers when available; null otherwise.

Relationship requirements:
- relationships must be a flat list of objects with:
    "source_id_path": parent node id_path
    "target_id_path": child node id_path
    "kind": "CONTAINS"
- Every node except the root document node must have exactly one CONTAINS relationship from its parent.

document_metadata:
- Fill what you can infer: jurisdiction, document_type, source, language, title, act_number, year, publication_date, effective_date.
- If a field is unknown, set it to null.

Hints about this document:
{meta_hint}

VERY IMPORTANT:
- Respond with a single JSON object only.
- Do NOT include any prose explanation before or after the JSON.
"""

    return instructions.strip() + "\n\nTEXT TO PARSE:\n\n" + text
