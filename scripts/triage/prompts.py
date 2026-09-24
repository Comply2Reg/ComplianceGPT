"""Prompts for triage labelling. Bump PROMPT_VERSION when either changes —
it is part of the cache key, so old labels are never mixed with new ones."""

from __future__ import annotations

from typing import Dict, List

from .doc_input import DocInput
from .taxonomy import (
    ALERT_CLASSES,
    CLASS_TO_FUNCTION,
    FUNCTION_DESCRIPTIONS,
    FUNCTIONS,
    PRIORITY_RULES,
)

PROMPT_VERSION = "triage-v1"


def _class_lines() -> str:
    lines = []
    for cls, label in ALERT_CLASSES.items():
        owners = [f for f in FUNCTIONS if CLASS_TO_FUNCTION[cls].get(f) == "P"]
        lines.append(
            f"- {cls}: {label}. Usual primary owners: {', '.join(owners) or 'none'}."
        )
    return "\n".join(lines)


def _function_lines() -> str:
    return "\n".join(
        f"- {name}: {desc}" for name, desc in FUNCTION_DESCRIPTIONS.items()
    )


SYSTEM_PROMPT = f"""You triage UK financial-services regulatory publications
for a bank's compliance function. You read one publication and return a structured
triage record.

Alert classes (choose exactly one):
{_class_lines()}

Functions (use only these names):
{_function_lines()}

Priority:
- {PRIORITY_RULES[0]}
- {PRIORITY_RULES[1]}
- {PRIORITY_RULES[2]}

Rules:
- Classify what the publication IS, not what it talks about: a press release
  about a fine is A11, the notice itself is A6; a consultation on a rule is A3,
  the policy statement that makes it is A2.
- The registry prior is a hint about the source, not the document. Override it
  when the text says otherwise, and lower your confidence when you do.
- primary_functions: the functions that must act; secondary_functions:
  consulted, impacted or informed. Prefer the usual owners for the class unless
  the topic clearly belongs elsewhere (AML → Financial crime; operational
  resilience or outsourcing → Operational resilience; capital, liquidity,
  reporting → Prudential; data protection → Privacy).
- lines_of_defence: 1 for Product/Ops/Prudential/Operational resilience, 2 for
  Compliance/Risk/Financial crime/Legal/Privacy/HR/InfoSec/Regulatory change,
  3 for Internal audit. Include a line only if a primary function sits on it.
- key_dates: only dates the text states (closing dates, in-force dates,
  reporting deadlines); give the ISO date or null, never invent one.
- applicability: the firm types or activities the text names (e.g. "banks",
  "insurers", "payment institutions", "all FSMA-authorised firms"); empty if
  none are named.
- obligations_present: true only when this text itself creates or amends
  requirements (shall/must/is required to). Enforcement notices and speeches
  are false.
- summary: at most 80 words, plain English, what changed, for whom, by when.
- rationale: one to three sentences.
Return only the structured record."""

USER_TEMPLATE = """Triage the following publication.

{document}"""


def build_messages(doc: DocInput) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(document=doc.as_text())},
    ]


# Instruction templates for the training records (the SLM sees one of these).
INSTRUCTION_TEMPLATES = (
    "Triage this UK regulatory publication: assign its alert class (A1-A14), "
    "priority (P1-P3), the functions that must act and those to consult, the "
    "lines of defence engaged, key dates, applicability, and whether it "
    "creates obligations. Answer in JSON.",
    "You are the regulatory-change desk of a UK bank. Read the publication "
    "and return a triage record: alert_class, alert_class_confidence, "
    "priority, primary_functions, secondary_functions, lines_of_defence, "
    "summary, key_dates, applicability, obligations_present.",
    "Classify and route this publication for a financial institution's "
    "compliance, risk, legal, product, operations and audit teams. Return the "
    "structured triage JSON only.",
)
