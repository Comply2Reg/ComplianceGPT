"""LLM client abstraction for structural parsing."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.llm.prompts import build_structural_parse_prompt

logger = logging.getLogger(__name__)

StructuralParserFn = Callable[[str, int], Dict[str, Any]]


def _openai_structural_parser(settings: Settings) -> StructuralParserFn:
    """Return a function that calls OpenAI's chat completions API for structural parsing."""
    api_key = settings.get_llm_api_key()
    if not api_key:
        raise ValueError(
            "LLM_PROVIDER=openai but no LLM_API_KEY or OPENAI_API_KEY is set."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required for LLM_PROVIDER=openai. Install with: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key)
    model = settings.llm_model

    def _call(prompt: str, max_tokens: int) -> Dict[str, Any]:
        logger.info("Calling OpenAI LLM for structural parsing with model=%s", model)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise legal document structural parser."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode LLM JSON output: %s", exc)
            raise

    return _call


def get_structural_parser(settings: Settings) -> StructuralParserFn:
    """Return a callable that runs LLM-based structural parsing."""
    provider = (settings.llm_provider or "openai").lower()
    if provider == "openai":
        return _openai_structural_parser(settings)
    if provider == "azure":
        raise NotImplementedError("LLM_PROVIDER=azure is not yet implemented.")
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def run_structural_parse(text: str, settings: Settings) -> Dict[str, Any]:
    """High-level helper: build prompt and run structural parsing in one call."""
    from src.obligation_pipeline.llm.structure_models import DocumentMetadata

    metadata = DocumentMetadata()
    prompt = build_structural_parse_prompt(text, metadata=metadata)
    parser = get_structural_parser(settings)
    max_tokens = settings.llm_max_tokens
    return parser(prompt, max_tokens)
