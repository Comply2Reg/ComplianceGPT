"""Parser quality gates for Stage-1 artifacts.

G1 extractor is one the source declares
G2 section diversity (not uniformly "body")
G4 furniture / promo / page-number lines
G8 offset round-trip + doc_hash == sha256(canonical)

G1 and the chunk-id checks used to be hardcoded to ESMA newsletters
(`esma:spotlight-` ids, `extractor == "docling"`), which rejected every other
source wholesale - including the UK alert corpus produced by c2r-inventory-kit
and this repo's own clml.py UK legislation parser. They are now per-source
profiles. The default profile is the ESMA one, so existing callers and the
golden tests are unchanged.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
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

@dataclass(frozen=True)
class SourceProfile:
    """What a given source's chunks must look like.

    id_prefix / id_requires are the chunk-id shape; allowed_extractors is G1;
    furniture_patterns is G4, which is publisher-specific by nature (ESMA's
    running header is meaningless for an FCA policy statement).
    """

    name: str
    id_prefix: str = ""
    id_requires: tuple[str, ...] = ()
    allowed_extractors: frozenset[str] = frozenset({"docling", ""})
    running_header_re: re.Pattern | None = None
    promo_res: tuple[re.Pattern, ...] = ()
    check_page_numbers: bool = True


ESMA_PROFILE = SourceProfile(
    name="esma_newsletter",
    id_prefix="esma:spotlight-",
    id_requires=("#p",),
    allowed_extractors=frozenset({"docling", ""}),
    running_header_re=RUNNING_HEADER_RE,
    promo_res=PROMO_RES,
    check_page_numbers=True,
)

# UK alerts from c2r-inventory-kit. Ids are `{regulator}:{doc_type}#{id}/{slug}`
# and the text comes from HTML, so there are no page numbers to strip and no
# ESMA furniture. G8, the field list and G2 still apply in full.
UK_ALERTS_PROFILE = SourceProfile(
    name="uk_alerts",
    id_prefix="",
    id_requires=("#",),
    # /html = landing-page text; /pdf = text pypdf pulled from the attachment
    # (every FCA enforcement notice, a third of FCA policy statements).
    allowed_extractors=frozenset({"c2r-inventory-kit/html", "c2r-inventory-kit/pdf", ""}),
    # The crawler's furniture cleaner removes BoE/PRA running headers before
    # export; G4 (strict) now catches a regression of that.
    running_header_re=re.compile(
        r"^Bank of England( \| Prudential Regulation Authority)? Page \d+$"),
    promo_res=(),
    check_page_numbers=False,
)

# legislation.gov.uk via clml.py: ids are `ukpga:2000/8#s_19`, text is derived
# from CLML XML rather than an extractor.
UK_LEGISLATION_PROFILE = SourceProfile(
    name="uk_legislation",
    id_prefix="",
    id_requires=("#",),
    allowed_extractors=frozenset({"clml", ""}),
    running_header_re=None,
    promo_res=(),
    check_page_numbers=False,
)

PROFILES: dict[str, SourceProfile] = {
    ESMA_PROFILE.name: ESMA_PROFILE,
    UK_ALERTS_PROFILE.name: UK_ALERTS_PROFILE,
    UK_LEGISLATION_PROFILE.name: UK_LEGISLATION_PROFILE,
}
DEFAULT_PROFILE = ESMA_PROFILE


def get_profile(name: str | None) -> SourceProfile:
    if not name:
        return DEFAULT_PROFILE
    if name not in PROFILES:
        raise ValueError(
            f"Unknown source profile {name!r}. Known: {', '.join(sorted(PROFILES))}"
        )
    return PROFILES[name]


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


def furniture_counts(text: str, profile: "SourceProfile | None" = None) -> dict[str, int]:
    """Count publisher furniture that survived extraction.

    Which patterns count is per-source: ESMA's running header and Instagram
    promo are meaningless for an FCA policy statement, and page-number lines
    only exist in text extracted from a PDF.
    """
    prof = profile or DEFAULT_PROFILE
    running = 0
    page_only = 0
    promo = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if prof.running_header_re is not None and prof.running_header_re.match(s):
            running += 1
        if prof.check_page_numbers and PAGE_NUM_ONLY_RE.match(s):
            page_only += 1
        if any(p.search(s) for p in prof.promo_res):
            promo += 1
    return {"running_header": running, "page_number_lines": page_only, "promo": promo}


def compute_metrics(canonical: str, chunks: list[dict[str, Any]],
                    profile: "SourceProfile | None" = None) -> dict[str, Any]:
    sections = [str(c.get("section") or "") for c in chunks]
    counts = Counter(sections)
    n = len(chunks) or 1
    generic = sum(1 for s in sections if s.strip().lower() in {"body", "document", "section", ""})
    extractors = {str(c.get("extractor") or "") for c in chunks}
    furniture = furniture_counts(
        "\n".join(c.get("text") or "" for c in chunks), profile=profile
    )
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
    profile: SourceProfile | str | None = None,
) -> list[str]:
    prof = profile if isinstance(profile, SourceProfile) else get_profile(profile)
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
        if prof.id_prefix and not cid.startswith(prof.id_prefix):
            errors.append(f"{cid}: alert ID must start with {prof.id_prefix}")
        for fragment in prof.id_requires:
            if fragment not in cid:
                errors.append(f"{cid}: alert ID must include {fragment}")

    metrics = compute_metrics(canonical, chunks, profile=prof)
    extractors = set(metrics["extractors"])
    expected = set(prof.allowed_extractors)
    if not allow_degraded and extractors - expected:
        errors.append(
            f"G1 extractor must be one of {sorted(expected - {''})}, got {sorted(extractors)}"
        )
    if not allow_degraded and not (extractors & (expected - {""})):
        errors.append(
            f"G1 extractor field missing/unexpected: {sorted(extractors)}"
        )

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
