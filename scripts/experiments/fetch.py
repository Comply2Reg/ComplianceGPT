"""Quarantined: download legislation.gov.uk XML only. Does NOT write data/chunks.jsonl.

Parse with: python scripts/clml.py --xml data/raw/fsma2000.xml
"""

from __future__ import annotations

import pathlib
import time

import requests

HEADERS = {"User-Agent": "Comply2Reg-research/0.1"}

DOCS = [
    ("fsma2000", "https://www.legislation.gov.uk/ukpga/2000/8/data.xml"),
    ("mlr2017", "https://www.legislation.gov.uk/uksi/2017/692/data.xml"),
    ("psr2017", "https://www.legislation.gov.uk/uksi/2017/752/data.xml"),
    ("emr2011", "https://www.legislation.gov.uk/uksi/2011/99/data.xml"),
]


def main() -> int:
    raw = pathlib.Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for doc_id, url in DOCS:
        print("fetching", doc_id, flush=True)
        r = requests.get(url, headers=HEADERS, timeout=90)
        r.raise_for_status()
        dest = raw / f"{doc_id}.xml"
        dest.write_bytes(r.content)
        print(f"  wrote {dest} ({len(r.content)} bytes)")
        time.sleep(1)
    print("DONE — XML only. Run scripts/clml.py to parse. Does not write chunks.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
