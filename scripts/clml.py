"""UK CLML provision parser.

Walks P1/P2/P3 (not P1-only). IDs are source-derived, never parse-order.

  ukpga:2000/8#s_19
  uksi:2017/692#reg_8

Usage:
  python scripts/clml.py --xml tests/fixtures/legal/uk_clml_extract.xml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from lxml import etree

LEG = "http://www.legislation.gov.uk/namespaces/legislation"
UKM = "http://www.legislation.gov.uk/namespaces/metadata"

PROVISION_TAGS = ("P1", "P2", "P3", "P4", "P5", "P6")
LICENCE = "OGL-UK"
SOURCE_TIER = "GREEN"

# "FSMA 2000 s.19" / "FSMA 2000 section 19"
_FSMA_RE = re.compile(
    r"\bFSMA\s*2000\s*(?:s\.|s\s+|sec(?:tion)?\s+)(\d+[A-Za-z]?)",
    re.IGNORECASE,
)
# "Regulation (EU) 2024/1624 Art. 20"
_EU_REG_RE = re.compile(
    r"Regulation\s*\(EU\)\s*(\d{4})/(\d+)\s*(?:Art\.?|Article)\s*(\d+[A-Za-z]?)",
    re.IGNORECASE,
)
_PROVISION_ID_RE = re.compile(r"^(ukpga|uksi|celex):", re.IGNORECASE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _local(tag: str) -> str:
    if isinstance(tag, str) and "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return str(tag)


def _direct_pnumber(el) -> str:
    for child in el:
        if _local(child.tag) == "Pnumber":
            return (child.text or "").strip()
    return ""


def _local_text(el) -> str:
    """Text of this provision excluding nested P1–P6 children."""
    parts: list[str] = []
    for child in el:
        if _local(child.tag) in PROVISION_TAGS:
            continue
        if _local(child.tag) == "Pnumber":
            continue
        for t in child.itertext():
            s = t.strip()
            if s:
                parts.append(s)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _kind_from_uri(uri: str) -> tuple[str, str, str]:
    """Return (kind, year, number) from a legislation.gov.uk URI."""
    path = urlparse(uri).path.strip("/")
    bits = path.split("/")
    # ukpga/2000/8 or uksi/2017/692
    if len(bits) >= 3:
        return bits[0], bits[1], bits[2]
    return "ukpga", "0000", "0"


def _document_uri(root) -> str:
    for el in root.iter():
        if _local(el.tag) in {"PrimaryMetadata", "SecondaryMetadata"}:
            uri = el.get("DocumentURI") or ""
            if uri:
                return uri
        href = el.get("DocumentURI") or el.get("URI") or ""
        if "legislation.gov.uk" in href:
            return href
    return ""


def provision_id(kind: str, year: str, number: str, parts: list[tuple[str, str]]) -> str:
    if not parts:
        raise ValueError("provision_id requires at least one P-level")
    first_num = parts[0][1].lower()
    rest = [p[1].lower() for p in parts[1:]]
    if kind == "uksi":
        frag = "reg_" + first_num
    else:
        frag = "s_" + first_num
    if rest:
        frag += "-" + "-".join(rest)
    return f"{kind}:{year}/{number}#{frag}"


def normalize_citation(ref: str) -> Optional[str]:
    """Map a human citation to a provision_id. None if unknown."""
    s = (ref or "").strip()
    if not s:
        return None
    if _PROVISION_ID_RE.match(s):
        return s
    m = _FSMA_RE.search(s)
    if m:
        return f"ukpga:2000/8#s_{m.group(1).lower()}"
    m = _EU_REG_RE.search(s)
    if m:
        year, num, art = m.group(1), m.group(2), m.group(3)
        return f"celex:3{year}R{num.zfill(4)}#art_{art.lower()}"
    return None


def lookup_citation(ref: str, chunks: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    pid = normalize_citation(ref)
    if not pid:
        return None
    for c in chunks:
        if c.get("provision_id") == pid:
            return c
    return None


def parse_clml(xml_bytes: bytes, *, source_url: str = "") -> tuple[str, list[dict[str, Any]]]:
    root = etree.fromstring(xml_bytes)
    uri = source_url or _document_uri(root)
    kind, year, number = _kind_from_uri(uri) if uri else ("ukpga", "0000", "0")
    document_id = f"{kind}:{year}/{number}"

    records: list[dict[str, Any]] = []

    def walk(el, ancestors: list[tuple[str, str]]) -> None:
        local = _local(el.tag)
        if local in PROVISION_TAGS:
            num = _direct_pnumber(el)
            chain = ancestors + [(local, num or local.lower())]
            text = _local_text(el)
            if text:
                records.append(
                    {
                        "chain": chain,
                        "text": text,
                        "level": local,
                    }
                )
            for child in el:
                walk(child, chain)
            return
        for child in el:
            walk(child, ancestors)

    walk(root, [])

    parts: list[str] = []
    chunks: list[dict[str, Any]] = []
    cursor = 0
    for rec in records:
        text = rec["text"]
        start = cursor
        end = start + len(text)
        pid = provision_id(kind, year, number, rec["chain"])
        section_number = rec["chain"][0][1] if rec["chain"] else ""
        parent_path = " > ".join(f"{lv} {n}" for lv, n in rec["chain"])
        chunks.append(
            {
                "chunk_id": pid,
                "provision_id": pid,
                "document_id": document_id,
                "section_number": section_number,
                "parent_path": parent_path,
                "level": rec["level"],
                "text": text,
                "start": start,
                "end": end,
                "source_url": uri,
                "licence": LICENCE,
                "source_tier": SOURCE_TIER,
                "jurisdiction": "UK",
                "document_type": "legislation",
            }
        )
        parts.append(text)
        cursor = end + 1  # newline between provisions

    canonical = "\n".join(parts)
    if canonical:
        canonical += "\n"

    digest = sha256_text(canonical)
    for c in chunks:
        c["doc_hash"] = digest
        c["snapshot_id"] = digest
        c["content_sha"] = sha256_text(c["text"])
        sliced = canonical[c["start"] : c["end"]]
        if sliced != c["text"]:
            raise RuntimeError(
                f"Offset round-trip failed for {c['provision_id']}: "
                f"start={c['start']} end={c['end']}"
            )

    return canonical, chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse UK CLML into provision chunks")
    parser.add_argument("--xml", required=True, help="Path to CLML XML")
    parser.add_argument("--out", default=None, help="Optional JSONL output path")
    args = parser.parse_args()

    path = Path(args.xml)
    xml_bytes = path.read_bytes()
    canonical, chunks = parse_clml(xml_bytes)
    print(f"{path.name}: {len(chunks)} provisions, {len(canonical)} chars")
    for c in chunks:
        print(f"  {c['provision_id']}  ({c['level']})  {c['text'][:72]!r}")

    if args.out:
        out = Path(args.out)
        out.write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks),
            encoding="utf-8",
        )
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
