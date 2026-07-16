"""Validation helpers for generated obligation records."""

from __future__ import annotations

from pydantic import ValidationError

from .schema import ObligationRecord


def validate_record(record: ObligationRecord) -> tuple[bool, str | None]:
    try:
        ObligationRecord.model_validate(record.model_dump())
        if record.classification == "obligation" and not isinstance(record.output, list):
            return False, "obligation classification must have array output"
        if record.classification != "obligation" and isinstance(record.output, list):
            return False, "non-obligation/neutral must have object output"
        return True, None
    except ValidationError as exc:
        return False, str(exc)
