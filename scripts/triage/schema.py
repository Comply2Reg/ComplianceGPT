"""TriageRecord — what the labeller asks OpenAI for and what the SLM learns.

Every field is required and enum-typed where possible so OpenAI's strict
structured-output mode accepts the schema unchanged. `rationale` becomes the
training record's `thought_trace`; everything else is the `output`.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .taxonomy import AlertClass, Function, Priority


class KeyDate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        description="What the date is, e.g. 'consultation closes', "
        "'in force', 'first reporting date'"
    )
    date: Optional[str] = Field(
        description="ISO date YYYY-MM-DD, or null if the text gives no exact date"
    )


class TriageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_class: AlertClass
    alert_class_confidence: Literal["high", "medium", "low"]
    priority: Priority
    primary_functions: List[Function] = Field(
        description="Functions that must act on this publication"
    )
    secondary_functions: List[Function] = Field(
        description="Functions consulted, impacted or informed"
    )
    lines_of_defence: List[Literal[1, 2, 3]] = Field(
        description="Lines of defence engaged by the primary functions"
    )
    summary: str = Field(
        description="One paragraph of at most 80 words: what changed, for whom, by when"
    )
    key_dates: List[KeyDate]
    applicability: List[str] = Field(
        description="Firm types or activities the text names as in scope"
    )
    obligations_present: bool = Field(
        description="True if the text itself creates or changes obligations "
        "(shall/must), not merely reports on them"
    )
    # Added for multi-jurisdiction coverage. The v1 model had no way to say
    # which country a document binds, which is why it read ADGM rulebook text
    # as UK intelligence.
    jurisdiction: str = Field(
        description="ISO-style code for the jurisdiction this publication "
        "binds: GB, US, EU, IN, SG, AU, CA, HK, JP, CH, AE, ZZ if unclear. "
        "An EU instrument binds EU even when a national regulator republishes it."
    )
    frameworks: List[str] = Field(
        description="International standards or cross-border regimes this text "
        "implements, amends or responds to, from: Basel III, Basel IV, FATF, "
        "IOSCO, IAIS, FSB, DORA, NIS2, GDPR, PSD2, MiFID, EMIR, Solvency II, "
        "CRR/CRD, AIFMD, MiCA, CSRD, AI Act, SOX, Dodd-Frank, BSA/AML, "
        "ISO 27001, NIST CSF. Empty when the text stands alone. This is what "
        "links a rule in one country to its equivalent in another."
    )
    rationale: str = Field(
        description="One to three sentences on why this class and priority"
    )


OUTPUT_FIELDS = tuple(f for f in TriageRecord.model_fields if f != "rationale")
