"""Pydantic schema for obligation extraction fine-tuning dataset."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FineTuningMetadata(BaseModel):
    source_id: str
    regulation_name: str
    jurisdiction: str
    doc_type: str = "Regulation"
    section: str


class ToolCallDetails(BaseModel):
    function: str
    parameters: str | None = Field(default=None, description="JSON string of parameters")


class ToolUse(BaseModel):
    tool_rationale: str | None = None
    tool_call: ToolCallDetails | None = None
    external_context: str | None = None


class ObligationOutput(BaseModel):
    obligation_id: str
    subject: str
    modality: str
    modality_detected: str | None = None
    action: str
    conditions: str | None = None
    deadline: str | None = None
    reference_anchor: str | None = None


class NonObligationOutput(BaseModel):
    status: str
    message: str
    reason: str


class ObligationRecord(BaseModel):
    metadata: FineTuningMetadata
    classification: str
    instruction: str
    input_text: str
    tool_use: ToolUse | None = None
    thought_trace: str
    output: list[ObligationOutput] | NonObligationOutput
