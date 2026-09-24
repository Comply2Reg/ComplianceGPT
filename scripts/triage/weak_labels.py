#!/usr/bin/env python3
"""Free labels from the registry and title heuristics — the sanity baseline.

    python -m scripts.triage.weak_labels --corpus-dir <corpus> \\
        --out data/uk/labels --version v1

Not training data. report.py compares these with the model labels so a
prompt regression shows up as a drop in agreement on the non-MIXED sources.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import Corpus, write_jsonl  # noqa: E402
from scripts.triage.taxonomy import CLASS_TO_STRATUM, functions_for  # noqa: E402

log = logging.getLogger("triage.weak_labels")

_TITLE_RULES = (
    (
        re.compile(
            r"\b(final notice|decision notice|warning notice|fine[sd]?|penalt|"
            r"prohibition order|censure)\b",
            re.I,
        ),
        "A6",
    ),
    (
        re.compile(
            r"\b(consultation|call for (input|evidence)|discussion paper|"
            r"\bCP\d{1,2}/\d{1,3}|\bDP\d{1,2}/\d{1,3})",
            re.I,
        ),
        "A3",
    ),
    (
        re.compile(
            r"\b(policy statement|\bPS\d{1,2}/\d{1,3}|handbook notice|"
            r"instrument \d{4}|made rules?)\b",
            re.I,
        ),
        "A2",
    ),
    (
        re.compile(
            r"\b(supervisory statement|statement of policy|\bSS\d{1,2}/\d{1,3}|"
            r"\bSoP\d{1,2}/\d{1,3}|finalised guidance|\bFG\d{1,2}/\d{1,3})",
            re.I,
        ),
        "A4",
    ),
    (
        re.compile(
            r"\b(dear (ceo|cro|cfo|chair)|portfolio letter|supervisory "
            r"(priorities|strategy))\b",
            re.I,
        ),
        "A5",
    ),
    (
        re.compile(
            r"\b(market study|thematic review|multi-firm review|stress test)\b", re.I
        ),
        "A10",
    ),
    (
        re.compile(
            r"\b(speech|remarks|blog|press release|working paper|research|"
            r"podcast|interview)\b",
            re.I,
        ),
        "A11",
    ),
    (re.compile(r"\b(market notice|operational notice)\b", re.I), "A9"),
    (
        re.compile(
            r"\b(regulations? \d{4}|order \d{4}|\bSI \d{4}/\d+|statutory "
            r"instrument|act \d{4})\b",
            re.I,
        ),
        "A1",
    ),
    (
        re.compile(
            r"\b(merger|acquisition|phase [12]|remedies|competition act|"
            r"cartel|price cap|market investigation)\b",
            re.I,
        ),
        "A10",
    ),
)

PRIORITY_BY_CLASS = {
    "A1": "P1",
    "A2": "P1",
    "A7": "P1",
    "A8": "P1",
    "A12": "P1",
    "A3": "P2",
    "A4": "P2",
    "A5": "P2",
    "A6": "P2",
    "A10": "P2",
    "A13": "P2",
    "A14": "P2",
    "A9": "P3",
    "A11": "P3",
}


def heuristic_class(title: str) -> Optional[str]:
    for pattern, cls in _TITLE_RULES:
        if pattern.search(title or ""):
            return cls
    return None


def weak_label(row: Dict) -> Dict:
    prior = row.get("alert_class")
    source = "registry"
    cls = prior if prior and prior != "MIXED" else None
    if cls is None:
        cls = heuristic_class(row.get("title") or "")
        source = "title" if cls else "none"
    return {
        "doc_id": int(row["id"]),
        "document_id": (f"{str(row['regulator']).lower()}:"
                        f"{row['document_type']}#{row['id']}"),
        "doc_hash": row["sha256"],
        "regulator": row["regulator"],
        "document_type": row["document_type"],
        "tier": row.get("tier"),
        "alert_class": cls,
        "alert_class_source": source,
        "stratum": CLASS_TO_STRATUM.get(cls) if cls else None,
        "priority": PRIORITY_BY_CLASS.get(cls) if cls else None,
        "primary_functions": functions_for(cls) if cls else [],
        "secondary_functions": functions_for(cls, ("S", "I")) if cls else [],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus-dir", type=Path, default=Path(cfg.TRIAGE["corpus_dir"]))
    ap.add_argument("--out", type=Path, default=Path(cfg.TRIAGE["labels_dir"]))
    ap.add_argument("--version", default="v1")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    corpus = Corpus(args.corpus_dir)
    corpus.check()
    labels = [weak_label(r) for r in corpus.manifest()]
    n = write_jsonl(args.out / f"weak_{args.version}.jsonl", labels)
    by_source = {}
    for lab in labels:
        by_source[lab["alert_class_source"]] = (
            by_source.get(lab["alert_class_source"], 0) + 1
        )
    log.info("wrote %d weak labels -> %s | by source %s", n, args.out, by_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
