from pathlib import Path

from clml import lookup_citation, normalize_citation, parse_clml

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "legal" / "uk_clml_extract.xml"


def _parse():
    return parse_clml(FIXTURE.read_bytes())


def test_walks_p1_p2_p3() -> None:
    canonical, chunks = _parse()
    ids = [c["provision_id"] for c in chunks]
    assert "ukpga:2000/8#s_19" in ids
    assert "ukpga:2000/8#s_19-1" in ids
    assert "ukpga:2000/8#s_19-1-a" in ids
    assert "ukpga:2000/8#s_21" in ids
    assert "ukpga:2000/8#s_9" in ids
    assert "ukpga:2000/8#s_9-2" in ids


def test_p1_text_excludes_nested_p2() -> None:
    _, chunks = _parse()
    by_id = {c["provision_id"]: c for c in chunks}
    assert "authorised person or an exempt person" in by_id["ukpga:2000/8#s_19"]["text"]
    assert "acting as a principal" not in by_id["ukpga:2000/8#s_19"]["text"]
    assert "acting as a principal" in by_id["ukpga:2000/8#s_19-1-a"]["text"]


def test_section_9_not_section_1() -> None:
    _, chunks = _parse()
    clause = next(c for c in chunks if c["provision_id"] == "ukpga:2000/8#s_9-2")
    assert clause["section_number"] == "9"


def test_offset_round_trip() -> None:
    canonical, chunks = _parse()
    for c in chunks:
        assert canonical[c["start"] : c["end"]] == c["text"]


def test_ids_stable_across_reparse() -> None:
    _, a = _parse()
    _, b = _parse()
    assert [c["provision_id"] for c in a] == [c["provision_id"] for c in b]
    assert a[0]["snapshot_id"] == b[0]["snapshot_id"]


def test_lookup_fsma_citations() -> None:
    _, chunks = _parse()
    hit = lookup_citation("FSMA 2000 s.19", chunks)
    assert hit is not None
    assert hit["provision_id"] == "ukpga:2000/8#s_19"
    assert lookup_citation("FSMA 2000 section 21", chunks)["provision_id"] == "ukpga:2000/8#s_21"


def test_normalize_eu_citation() -> None:
    assert (
        normalize_citation("Regulation (EU) 2024/1624 Art. 20")
        == "celex:32024R1624#art_20"
    )


def test_licence_green() -> None:
    _, chunks = _parse()
    assert all(c["licence"] == "OGL-UK" for c in chunks)
    assert all(c["source_tier"] == "GREEN" for c in chunks)
