"""Deterministic structural node IDs shared by graph_builder and retrieval anchor resolution."""

from __future__ import annotations

import re


def slug(value: str) -> str:
    """Create a filesystem/URI-friendly slug from a free-form label."""
    value = (value or "").strip()
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def structural_id(act_id: str, *parts: str) -> str:
    """Build a deterministic hierarchical ID for a structural node."""
    slugged = [slug(p) for p in parts]
    return "/".join([act_id] + slugged)
