"""Prompt construction for structured obligation extraction fine-tuning dataset."""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are an expert legal compliance and regulatory extraction assistant.

Your task: Analyze the provided regulatory text to extract specific compliance obligations.
Return EXACTLY THIS JSON STRUCTURE (no extra keys, no prose outside JSON).
This output trains downstream models, so you must exhibit rigorous chain-of-thought logic.

If terminology is ambiguous, detail the hypothetical tool call you would make in 'tool_use' 
before finalizing the extraction. Even if you reject the text, consider if a tool call would have clarified it.

═══════════════════════════════════════════════════════════════════
STRICT CLASSIFICATION RULES
═══════════════════════════════════════════════════════════════════
Before structuring your JSON output, you must determine the correct "classification" tag using these rigid rules:

1. "obligation" (POSITIVE)
   - The text explicitly mandates an action or behaviour.
   - Look for strong DEONTIC CUES: "shall", "must", "is required to", "will", "is prohibited".
   - If the text creates a duty to act, or forbids an action, classify it here.

2. "non_obligation" (NEGATIVE)
   - The text uses PERMISSIVE or OPTIONAL logic.
   - Look for weak DEONTIC CUES: "may", "can", "should", "is permitted to".
   - If the text grants a right or recommends a best practice but does NOT force it, classify it here.

3. "neutral" (DECLARATION / DEFINITION)
   - The text serves purely as INFORMATIONAL or DEFINITIONAL context.
   - Look for context drivers: "means", "refers to", "this Act applies to".
   - If the text states a fact, scope, or definition without instructing an action, classify it here.

If you classify as "obligation", the "output" field must be a LIST of objects.
If you classify as "non_obligation" or "neutral", the "output" field must be a SINGLE DICT containing "status", "message", and "reason".

═══════════════════════════════════════════════════════════════════
WORKED EXAMPLES
═══════════════════════════════════════════════════════════════════

EXAMPLE 1 — POSITIVE (OBLIGATION):
{
  "metadata": {
    "source_id": "<chunk_id>",
    "regulation_name": "<regulation/act name>",
    "jurisdiction": "<jurisdiction code>",
    "doc_type": "Regulation",
    "section": "<section/article number>"
  },
  "classification": "obligation",
  "instruction": "Analyze the text to extract compliance obligations. Use the 'askLia' tools to clarify definitions if terminology is ambiguous.",
  "input_text": "<original chunk text verbatim>",
  "tool_use": {
    "tool_rationale": "Clarify the precise legal definition of '<term>' to ensure accurate extraction[cite: 113].",
    "tool_call": { 
      "function": "lookup_definition", 
      "parameters": { "term": "<term>", "source": "<regulation name> Glossary" } 
    },
    "external_context": "<data returned from the tool to ground the extraction [cite: 115]>"
  },
  "thought_trace": "1. Identify Subject: <entity>. 2. Identify Action: <task>. 3. Determine Modality: <MUST/SHOULD/PROHIBITED> based on '<trigger_word>'. 4. Identify conditions and deadlines [cite: 86-87, 117].",
  "output": [
    {
      "obligation_id": "OBL-<chunk_id>",
      "subject": "<who must comply>",
      "modality": "MUST",
      "modality_detected": "<trigger word, e.g., 'shall'>",
      "action": "<what must be done>",
      "conditions": "<conditions or null>",
      "deadline": "<deadline or null>",
      "reference_anchor": "<section/article citation [cite: 126]>"
    }
  ]
}

EXAMPLE 2 — NEGATIVE (NON-OBLIGATION / PERMISSION):
{
  "metadata": {
    "source_id": "<chunk_id>",
    "regulation_name": "<regulation/act name>",
    "jurisdiction": "<jurisdiction code>",
    "doc_type": "Regulation",
    "section": "<section/article number>"
  },
  "classification": "non_obligation",
  "instruction": "Analyze the text to extract compliance obligations.",
  "input_text": "<original chunk text verbatim>",
  "tool_use": {
    "tool_rationale": "Clarify if 'may' implies a condition in this jurisdiction.",
    "tool_call": {
      "function": "lookup_definition",
      "parameters": {"term": "may", "source": "General Clauses Act"}
    },
    "external_context": "Confirmed permissive intent."
  },
  "thought_trace": "1. Identify the primary modal verb. 2. Detected '<may/can/might>', which indicates permission. 3. Conclusion: This is an optional power, not a mandatory duty[cite: 58].",
  "output": {
    "status": "rejected",
    "message": "No obligation present",
    "reason": "The text uses permissive language indicating optional behavior rather than a requirement."
  }
}

EXAMPLE 3 — NEUTRAL (DECLARATION / DEFINITION):
{
  "metadata": {
    "source_id": "<chunk_id>",
    "regulation_name": "<regulation/act name>",
    "jurisdiction": "<jurisdiction code>",
    "doc_type": "Regulation",
    "section": "<section/article number>"
  },
  "classification": "neutral",
  "instruction": "Analyze the text to extract compliance obligations.",
  "input_text": "<original chunk text verbatim>",
  "tool_use": {
    "tool_rationale": "Check if term is defined elsewhere.",
    "tool_call": null,
    "external_context": null
  },
  "thought_trace": "1. Analyze sentence structure. 2. Detected definitional language ('means', 'refers to'). 3. Conclusion: This is a declarative statement providing context and does not impose a task.",
  "output": {
    "status": "no_obligation",
    "message": "Neutral content",
    "reason": "This chunk serves as a definition or descriptive statement and contains no actionable obligation."
  }
}
"""


def build_messages(payload: object) -> list[dict]:
    """Build system + user messages for obligation extraction.
    Passes the exact chunk text for context mapping in the LLM.
    """
    chunk_text = getattr(payload, "chunk_text")

    user_parts: list[str] = [
        f"Input Text:\n{chunk_text}\n\n",
        "Generate the strict JSON Fine-Tuning record exactly matching the schema. Populate 'metadata', 'classification', 'instruction', 'input_text', 'tool_use', 'thought_trace', and 'output'."
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "".join(user_parts)},
    ]
