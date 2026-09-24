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
    rationale: str = Field(
        description="One to three sentences on why this class and priority"
    )


OUTPUT_FIELDS = tuple(f for f in TriageRecord.model_fields if f != "rationale")
