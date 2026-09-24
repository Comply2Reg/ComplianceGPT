"""The label space: alert classes, functions, lines of defence, strata.

Transcribed from c2r-inventory-kit/docs/alert-taxonomy.md (§1 classes,
§2 function axis, §3 class→function matrix). CLASS_TO_STRATUM and
TARGET_MIX are the exporter's tables and must stay identical to
c2r-inventory-kit/scripts/export_alert_corpus.py.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

AlertClass = Literal[
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
]

ALERT_CLASSES: Dict[str, str] = {
    "A1": "Primary legislation / statutory instrument",
    "A2": "Final rule / policy statement",
    "A3": "Proposed rule / consultation / discussion paper / call for input",
    "A4": "Supervisory guidance (supervisory statement, statement of policy, "
    "finalised guidance)",
    "A5": "Supervisory communication (Dear CEO / portfolio letter, supervisory "
    "priorities)",
    "A6": "Enforcement action (final, decision or warning notice, penalty)",
    "A7": "Sanctions / watchlist delta",
    "A8": "Reporting, returns or taxonomy change",
    "A9": "Market or operational notice",
    "A10": "Thematic review, multi-firm review or market study",
    "A11": "Intelligence: speech, blog, press release, research, working paper",
    "A12": "Codified rulebook or handbook, point-in-time",
    "A13": "Court or tribunal determination",
    "A14": "Perimeter, register or authorisation change (warning list, waivers)",
}

Function = Literal[
    "Board",
    "Regulatory change",
    "Compliance",
    "Financial crime",
    "Risk",
    "Prudential",
    "Operational resilience",
    "InfoSec",
    "Legal",
    "Product",
    "Ops",
    "Privacy",
    "HR",
    "Internal audit",
]

FUNCTIONS: Tuple[str, ...] = (
    "Board",
    "Regulatory change",
    "Compliance",
    "Financial crime",
    "Risk",
    "Prudential",
    "Operational resilience",
    "InfoSec",
    "Legal",
    "Product",
    "Ops",
    "Privacy",
    "HR",
    "Internal audit",
)

FUNCTION_DESCRIPTIONS: Dict[str, str] = {
    "Board": "Board / ExCo (governing body, SMF1/SMF9)",
    "Regulatory change": "Regulatory change management (SMF16)",
    "Compliance": "Compliance monitoring (SMF16)",
    "Financial crime": "AML, sanctions, fraud, market abuse (SMF17 MLRO)",
    "Risk": "Enterprise, operational, credit and market risk (SMF4)",
    "Prudential": "Prudential / treasury / finance (SMF2 CFO)",
    "Operational resilience": "Operational resilience, ICT, third parties (SMF24)",
    "InfoSec": "Information security / cyber (CISO)",
    "Legal": "Legal / general counsel",
    "Product": "Product, business line, distribution (1st line)",
    "Ops": "Operations / client onboarding (1st line)",
    "Privacy": "Data protection / privacy (DPO)",
    "HR": "HR, remuneration, certification (SMF12/SMF18)",
    "Internal audit": "Internal audit (SMF5, 3rd line)",
}

# Line of defence per function; Board is the governing body (no line).
FUNCTION_LINE: Dict[str, int | None] = {
    "Board": None,
    "Regulatory change": 2,
    "Compliance": 2,
    "Financial crime": 2,
    "Risk": 2,
    "Prudential": 1,
    "Operational resilience": 1,
    "InfoSec": 2,
    "Legal": 2,
    "Product": 1,
    "Ops": 1,
    "Privacy": 2,
    "HR": 2,
    "Internal audit": 3,
}

# Class → function: P primary owner, S secondary / consulted, I informed.
# Column order == FUNCTIONS. "" = not involved.
_MATRIX_ROWS: Dict[str, str] = {
    #      Board RegC Comp FinC Risk Prud OpRe Info Legl Prod Ops  Priv HR   Audt
    "A1": "I    P    S    S    S    S    S    -    P    S    -    S    -    S",
    "A2": "I    P    P    S    S    S    S    -    S    S    S    -    -    S",
    "A3": "-    P    S    S    S    S    S    -    S    P    -    -    -    -",
    "A4": "-    S    P    S    P    S    S    S    -    S    S    -    -    S",
    "A5": "P    S    P    S    P    -    S    -    -    S    -    -    -    S",
    "A6": "I    -    P    P    S    -    -    -    P    -    -    -    S    P",
    "A7": "-    -    S    P    -    -    -    -    S    -    P    -    -    S",
    "A8": "-    S    S    -    S    P    -    -    -    -    P    -    -    S",
    "A9": "-    -    -    -    S    P    S    -    -    S    P    -    -    -",
    "A10": "S    -    P    S    P    -    S    -    -    S    -    -    -    P",
    "A11": "S    S    I    I    S    I    I    -    -    I    -    -    -    -",
    "A12": "-    P    P    S    -    S    -    -    P    -    -    -    -    S",
    "A13": "-    -    S    S    -    -    -    -    P    -    -    S    -    -",
    "A14": "-    -    S    P    -    -    -    -    S    S    P    -    -    -",
}

CLASS_TO_FUNCTION: Dict[str, Dict[str, str]] = {
    cls: {fn: mark for fn, mark in zip(FUNCTIONS, row.split()) if mark != "-"}
    for cls, row in _MATRIX_ROWS.items()
}


def functions_for(alert_class: str, marks: Tuple[str, ...] = ("P",)) -> List[str]:
    """Functions carrying any of `marks` for a class, in FUNCTIONS order."""
    table = CLASS_TO_FUNCTION.get(alert_class, {})
    return [fn for fn in FUNCTIONS if table.get(fn) in marks]


# Stratum per alert class — identical to the exporter's table.
CLASS_TO_STRATUM: Dict[str, str] = {
    "A1": "rule_final",
    "A2": "rule_final",
    "A12": "rule_final",
    "A7": "rule_final",
    "A3": "consultation",
    "A4": "guidance",
    "A5": "guidance",
    "A10": "guidance",
    "A8": "guidance",
    "A6": "enforcement",
    "A13": "enforcement",
    "A9": "intelligence",
    "A11": "intelligence",
    "A14": "intelligence",
}
STRATA: Tuple[str, ...] = (
    "rule_final",
    "consultation",
    "guidance",
    "enforcement",
    "intelligence",
)
TARGET_MIX: Dict[str, float] = {
    "rule_final": 0.35,
    "consultation": 0.20,
    "guidance": 0.15,
    "enforcement": 0.15,
    "intelligence": 0.15,
}

Priority = Literal["P1", "P2", "P3"]
PRIORITY_RULES = (
    "P1: binding or imminent — legislation or final rules with an in-force or "
    "compliance date, enforcement against a firm type the reader could be, "
    "sanctions changes, reporting deadlines.",
    "P2: action likely but not yet binding — consultations with a closing date, "
    "supervisory guidance or letters setting expectations, thematic reviews.",
    "P3: awareness only — speeches, research, press releases, market notices "
    "with no direct obligation.",
)
