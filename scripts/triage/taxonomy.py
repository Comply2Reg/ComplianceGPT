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
    "finalised guidance, bulletin, circular)",
    "A5": "Supervisory communication (letter to firms, portfolio or priorities "
    "letter, supervisory advisory)",
    "A6": "Enforcement action (notice, order, consent order, penalty, "
    "settlement)",
    "A7": "Sanctions / watchlist delta",
    "A8": "Reporting, returns or taxonomy change",
    "A9": "Market or operational notice",
    "A10": "Thematic review, multi-firm review or market study",
    "A11": "Intelligence: speech, blog, press release, research, working paper",
    "A12": "Codified rulebook, handbook or code, point-in-time",
    "A13": "Court or tribunal determination",
    "A14": "Perimeter, register or authorisation change (warning list, waiver, "
    "modification, licence grant or revocation)",
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
    "Board": "Board / ExCo (governing body)",
    "Regulatory change": "Regulatory change management",
    "Compliance": "Compliance monitoring",
    "Financial crime": "AML, sanctions, fraud, market abuse",
    "Risk": "Enterprise, operational, credit and market risk",
    "Prudential": "Prudential / treasury / finance",
    "Operational resilience": "Operational resilience, ICT, third parties",
    "InfoSec": "Information security / cyber",
    "Legal": "Legal / general counsel",
    "Product": "Product, business line, distribution (1st line)",
    "Ops": "Operations / client onboarding (1st line)",
    "Privacy": "Data protection / privacy",
    "HR": "HR, remuneration, certification",
    "Internal audit": "Internal audit",
}

# The senior-manager regime that names each function, per jurisdiction. These
# used to be inside FUNCTION_DESCRIPTIONS, which meant they were rendered into
# the labelling prompt and the model learned "Financial crime = SMF17 MLRO" —
# true in the UK and meaningless anywhere else. They are an overlay now, so a
# UK-specific prompt can still carry them and a global one need not.
FUNCTION_REGIME_OVERLAY: Dict[str, Dict[str, str]] = {
    "GB": {  # Senior Managers & Certification Regime
        "Board": "SMF1 Chief Executive, SMF9 Chair",
        "Regulatory change": "SMF16",
        "Compliance": "SMF16 Compliance Oversight",
        "Financial crime": "SMF17 MLRO",
        "Risk": "SMF4 Chief Risk",
        "Prudential": "SMF2 CFO",
        "Operational resilience": "SMF24 Chief Operations",
        "InfoSec": "CISO, under SMF24",
        "HR": "SMF12/SMF18",
        "Internal audit": "SMF5 Head of Internal Audit",
    },
    "US": {
        "Financial crime": "BSA/AML Officer (31 CFR 1020.210), OFAC officer",
        "InfoSec": "CISO (NYDFS Part 500.4)",
        "Internal audit": "Chief Audit Executive",
        "Board": "Board / CEO",
        "Prudential": "CFO",
    },
}


def regime_names(jurisdiction: str) -> Dict[str, str]:
    """Senior-manager equivalents for a jurisdiction, empty when unmapped."""
    return FUNCTION_REGIME_OVERLAY.get((jurisdiction or "").upper(), {})


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
