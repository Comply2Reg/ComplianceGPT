"""Class balancing utilities for obligation dataset generation."""

from __future__ import annotations

from collections import defaultdict

from .schema import ObligationRecord


def balance_records(records: list[ObligationRecord], r_obl: float, r_non: float, r_neu: float) -> list[ObligationRecord]:
    buckets: dict[str, list[ObligationRecord]] = defaultdict(list)
    for rec in records:
        buckets[rec.classification].append(rec)

    total = len(records)
    if total == 0:
        return []

    targets = {
        "obligation": int(total * r_obl),
        "non_obligation": int(total * r_non),
        "neutral": int(total * r_neu),
    }
    out: list[ObligationRecord] = []
    for cls in ("obligation", "non_obligation", "neutral"):
        out.extend(buckets.get(cls, [])[: targets[cls]])

    # fill shortfall from leftovers
    if len(out) < total:
        leftovers: list[ObligationRecord] = []
        for cls in ("obligation", "non_obligation", "neutral"):
            leftovers.extend(buckets.get(cls, [])[targets[cls] :])
        out.extend(leftovers[: total - len(out)])
    return out
