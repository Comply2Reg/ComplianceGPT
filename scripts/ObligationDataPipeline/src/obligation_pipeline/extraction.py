"""Extraction module for obligation pipeline parsing."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .prompts import build_messages
from .schema import (
    FineTuningMetadata,
    NonObligationOutput,
    ObligationOutput,
    ObligationRecord,
    ToolUse,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from .config import Settings


logger = logging.getLogger(__name__)


@dataclass
class ExtractInput:
    """Input payload for obligation extraction."""

    dataset_version: str
    document_id: str
    document_title: str
    chunk_id: str
    chunk_text: str
    page_number: int
    char_start: int
    char_end: int
    section_number: str | None = None
    clause_number: str | None = None
    semantic_role: str | None = None
    hierarchy_path: str | None = None
    act_name: str | None = None
    chapter: str | None = None
    regulator: str | None = None
    jurisdiction: str | None = None


def _build_rule_record(payload: ExtractInput, settings: Settings) -> ObligationRecord:
    metadata = FineTuningMetadata(
        source_id=payload.chunk_id,
        regulation_name=payload.document_title,
        jurisdiction=payload.jurisdiction or "UNKNOWN",
        doc_type="Regulation",
        section=payload.section_number or "Unknown",
    )
    instruction = "Analyze the provided regulatory text to extract specific compliance obligations. If terminology is ambiguous, use the 'askLia' tools to clarify definitions before finalizing the extraction."
    tool_use = ToolUse()
    
    text_lower = payload.chunk_text.lower()
    
    # Simple regex modality check
    if re.search(r"\b(shall|must|is required to|are required to)\b", text_lower):
        classification = "obligation"
        thought_trace = "1. Identified mandatory deontic cue. 2. Modality is MUST. 3. Synthesize obligation output."
        output = [
            ObligationOutput(
                obligation_id=f"OBL-{payload.chunk_id}",
                subject="Extracted Entity",
                modality="MUST",
                action="Extracted Action",
                reference_anchor=payload.section_number,
            )
        ]
    elif re.search(r"\b(may|can|should)\b", text_lower):
        classification = "non_obligation"
        thought_trace = "1. Identified permissive or recommended deontic cue. 2. Modality does not mandate action. 3. Conclude non-obligation (permission)."
        output = NonObligationOutput(
            status="rejected",
            message="Permission or non-mandatory clause",
            reason="Non-mandatory modality detected (e.g., may/can/should).",
        )
    else:
        classification = "neutral"
        thought_trace = "1. Analyzed syntax tree. 2. No deontic markers found indicating a requirement. 3. Conclude declarative statement."
        output = NonObligationOutput(
            status="no_obligation",
            message="Neutral/definition clause",
            reason="No mandatory action detected.",
        )

    return ObligationRecord(
        metadata=metadata,
        classification=classification,
        instruction=instruction,
        input_text=payload.chunk_text,
        tool_use=tool_use,
        thought_trace=thought_trace,
        output=output,
    )


async def extract_record(payload: ExtractInput, settings: Settings, client: AsyncOpenAI | None = None) -> ObligationRecord:
    if not settings.use_openai or client is None:
        return _build_rule_record(payload, settings)
    
    base = _build_rule_record(payload, settings)
    
    for attempt in range(settings.max_retries):
        try:
            completion = await client.beta.chat.completions.parse(
                model=settings.openai_model,
                messages=build_messages(payload),
                max_tokens=settings.max_output_tokens,
                response_format=ObligationRecord,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("OpenAI parse returned null parsed object")
            
            # Ensure metadata consistency overrides any LLM hallucinations
            parsed.metadata.source_id = payload.chunk_id
            parsed.metadata.regulation_name = payload.document_title
            parsed.metadata.jurisdiction = payload.jurisdiction or "UNKNOWN"
            parsed.metadata.doc_type = "Regulation"
            if payload.section_number:
                parsed.metadata.section = payload.section_number
            parsed.input_text = payload.chunk_text

            return parsed
        except Exception as e:
            if attempt == settings.max_retries - 1:
                logger.warning(f"Final LLM extraction attempt failed for {payload.chunk_id}. Fallback activated: {e}")
                break
            await asyncio.sleep((2**attempt) + 1)
            
    return base


def document_id_from_name(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
