"""Parser quality gates for Stage-1 artifacts.

G1 extractor == docling
G2 section diversity (not uniformly "body")
G4 furniture / promo / page-number lines
G8 offset round-trip + doc_hash == sha256(canonical)
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

RUNNING_HEADER_RE = re.compile(
    r"^spotlight on markets(?:\s*[–—-]\s*.*)?$",
    re.IGNORECASE,
)
PAGE_NUM_ONLY_RE = re.compile(r"^\d{1,3}$")
PROMO_RES = (
    re.compile(r"esma is on instagram", re.I),
    re.compile(r"@esmacomms", re.I),
    re.compile(r"gettyimages", re.I),
)

REQUIRED_ALERT_FIELDS = (
    "chunk_id",
    "doc_id",
    "section",
    "text",
    "start",
    "end",
    "source_url",
    "doc_hash",
    "snapshot_id",
    "content_sha",
    "extractor",
    "licence",
    "source_tier",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def furniture_counts(text: str) -> dict[str, int]:
    running = 0
    page_only = 0
    promo = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if RUNNING_HEADER_RE.match(s):
            running += 1
        if PAGE_NUM_ONLY_RE.match(s):
            page_only += 1
        if any(p.search(s) for p in PROMO_RES):
            promo += 1
    return {"running_header": running, "page_number_lines": page_only, "promo": promo}


def compute_metrics(canonical: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    sections = [str(c.get("section") or "") for c in chunks]
    counts = Counter(sections)
    n = len(chunks) or 1
    generic = sum(1 for s in sections if s.strip().lower() in {"body", "document", "section", ""})
    extractors = {str(c.get("extractor") or "") for c in chunks}
    furniture = furniture_counts("\n".join(c.get("text") or "" for c in chunks))
    return {
        "n_chunks": len(chunks),
        "unique_sections": len(set(sections)),
        "generic_section_frac": generic / n,
        "extractors": sorted(extractors),
        "furniture": furniture,
        "canonical_sha": sha256_text(canonical) if canonical else "",
    }


def check_gates(
    canonical: str,
    chunks: list[dict[str, Any]],
    *,
    strict: bool,
    allow_degraded: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not chunks:
        return ["no chunks"]

    for c in chunks:
        for field in REQUIRED_ALERT_FIELDS:
            if field not in c:
                errors.append(f"{c.get('chunk_id')}: missing field {field}")
        start, end, text = c.get("start"), c.get("end"), c.get("text")
        if not (isinstance(start, int) and isinstance(end, int) and isinstance(text, str)):
            errors.append(f"{c.get('chunk_id')}: start/end/text types invalid")
            continue
        if not (0 <= start <= end <= len(canonical)):
            errors.append(
                f"{c.get('chunk_id')}: bad range start={start} end={end} len={len(canonical)}"
            )
            continue
        if canonical[start:end] != text:
            errors.append(f"{c.get('chunk_id')}: G8 offset mismatch")
        expected_hash = sha256_text(canonical)
        if c.get("doc_hash") != expected_hash:
            errors.append(f"{c.get('chunk_id')}: G8 doc_hash != sha256(canonical)")
        if c.get("snapshot_id") != c.get("doc_hash"):
            errors.append(f"{c.get('chunk_id')}: snapshot_id must equal doc_hash")
        if c.get("content_sha") != sha256_text(text):
            errors.append(f"{c.get('chunk_id')}: content_sha mismatch")
        cid = str(c.get("chunk_id") or "")
        if not cid.startswith("esma:spotlight-"):
            errors.append(f"{cid}: alert ID must start with esma:spotlight-")
        if "#p" not in cid:
            errors.append(f"{cid}: alert ID must include #p{{page}}/")

    metrics = compute_metrics(canonical, chunks)
    extractors = set(metrics["extractors"])
    if not allow_degraded and extractors - {"docling", ""}:
        errors.append(f"G1 extractor must be docling, got {sorted(extractors)}")
    if "docling" not in extractors and not allow_degraded:
        errors.append(f"G1 extractor field missing/not docling: {sorted(extractors)}")

    if not strict:
        return errors

    if metrics["generic_section_frac"] > 0.20:
        errors.append(
            f"G2 generic section fraction {metrics['generic_section_frac']:.2f} > 0.20"
        )
    if metrics["unique_sections"] < 2 and metrics["n_chunks"] > 1:
        errors.append(
            f"G2 unique sections={metrics['unique_sections']} (section not uniformly body)"
        )
    furn = metrics["furniture"]
    if furn["running_header"] > 1:
        errors.append(f"G4 running_header count {furn['running_header']} > 1")
    if furn["page_number_lines"] > 0:
        errors.append(f"G4 page_number_lines {furn['page_number_lines']} > 0")
    if furn["promo"] > 0:
        errors.append(f"G4 promo count {furn['promo']} > 0")
    return errors
