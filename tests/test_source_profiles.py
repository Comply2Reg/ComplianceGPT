"""Per-source quality-gate profiles.

The gates used to hardcode ESMA's chunk-id shape and `extractor == "docling"`,
which rejected every other source wholesale — including this repo's own clml.py
UK legislation output and the UK alert corpus exported by c2r-inventory-kit.

The ESMA behaviour is the default, so these tests sit alongside
test_docling_golden.py rather than replacing anything it pins.
"""

import hashlib

import pytest

from quality import (
    DEFAULT_PROFILE,
    ESMA_PROFILE,
    PROFILES,
    UK_ALERTS_PROFILE,
    check_gates,
    furniture_counts,
    get_profile,
)

CANONICAL = (
    "PS24/1: A policy statement\n"
    "\n"
    "Payment institutions and electronic money institutions holding relevant "
    "funds above five million pounds must safeguard them in a statutory trust.\n"
)


def _chunk(**overrides):
    text = CANONICAL[0:26]
    base = {
        "chunk_id": "fca:policy_statements#4242/ps24-1-a-policy-statement",
        "doc_id": 4242,
        "section": "PS24/1: A policy statement",
        "text": text,
        "start": 0,
        "end": 26,
        "source_url": "https://www.fca.org.uk/publications/policy-statements/ps24-1",
        "doc_hash": hashlib.sha256(CANONICAL.encode()).hexdigest(),
        "snapshot_id": hashlib.sha256(CANONICAL.encode()).hexdigest(),
        "content_sha": hashlib.sha256(text.encode()).hexdigest(),
        "extractor": "c2r-inventory-kit/html",
        "licence": "fca_publications",
        "source_tier": "AMBER",
    }
    base.update(overrides)
    return base


def test_default_profile_is_esma() -> None:
    """Existing callers must not change behaviour."""
    assert get_profile(None) is ESMA_PROFILE
    assert DEFAULT_PROFILE is ESMA_PROFILE


def test_unknown_profile_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown source profile"):
        get_profile("not_a_source")


def test_every_registered_profile_is_retrievable() -> None:
    for name in PROFILES:
        assert get_profile(name).name == name


def test_uk_chunk_fails_under_the_esma_profile() -> None:
    """The old behaviour, made explicit: ESMA gates reject UK ids."""
    errors = check_gates(CANONICAL, [_chunk()], strict=False)
    assert any("must start with esma:spotlight-" in e for e in errors)
    assert any("G1 extractor" in e for e in errors)


def test_uk_chunk_passes_under_its_own_profile() -> None:
    assert check_gates(CANONICAL, [_chunk()], strict=True, profile="uk_alerts") == []


def test_g8_still_applies_to_every_profile() -> None:
    """Offsets are the audit trail — no profile may relax them."""
    bad = _chunk(text="not what the canonical says there")
    errors = check_gates(CANONICAL, [bad], strict=False, profile="uk_alerts")
    assert any("G8 offset mismatch" in e for e in errors)

    wrong_hash = _chunk(doc_hash="0" * 64, snapshot_id="0" * 64)
    errors = check_gates(CANONICAL, [wrong_hash], strict=False, profile="uk_alerts")
    assert any("doc_hash" in e for e in errors)


def test_required_fields_still_apply() -> None:
    incomplete = _chunk()
    del incomplete["licence"]
    errors = check_gates(CANONICAL, [incomplete], strict=False, profile="uk_alerts")
    assert any("missing field licence" in e for e in errors)


def test_wrong_extractor_is_caught_per_profile() -> None:
    errors = check_gates(
        CANONICAL, [_chunk(extractor="something-else")], strict=False, profile="uk_alerts"
    )
    assert any("G1 extractor" in e for e in errors)


def test_page_number_furniture_is_esma_only() -> None:
    """Page-number lines only exist in text extracted from a PDF.

    PAGE_NUM_ONLY_RE matches a line of 1-3 digits, which is a page number in a
    PDF and an ordinary list marker in an HTML alert.
    """
    text = "Some body text\n12\nmore text"
    assert furniture_counts(text, ESMA_PROFILE)["page_number_lines"] == 1
    assert furniture_counts(text, UK_ALERTS_PROFILE)["page_number_lines"] == 0


def test_esma_promo_patterns_do_not_apply_to_uk() -> None:
    text = "ESMA is on Instagram now"
    assert furniture_counts(text, ESMA_PROFILE)["promo"] == 1
    assert furniture_counts(text, UK_ALERTS_PROFILE)["promo"] == 0


def test_uk_profile_accepts_pdf_extracted_chunks() -> None:
    """Enforcement notices arrive as PDFs; their chunks carry the /pdf extractor."""
    errors = check_gates(
        CANONICAL, [_chunk(extractor="c2r-inventory-kit/pdf")], strict=True, profile="uk_alerts"
    )
    assert errors == []
