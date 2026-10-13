"""GREEN documents from the crawler's exported corpora — the "repo snapshots".

RegNano builds tasks only from openly-licensed text. Unlike the triage
labeller there is no --allow-amber override at all: a task set is published
training data, and AMBER/RED text may not enter it under any sign-off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

KIT = Path(__file__).resolve().parents[4] / "09-inventory-kit" / "c2r-inventory-kit"
DEFAULT_CORPORA = [KIT / "data" / "us_corpus", KIT / "data" / "alert_corpus"]
ALLOWED_TIERS = frozenset({"GREEN"})


@dataclass(frozen=True)
class Doc:
    source_id: str
    regulator: str
    document_type: str
    title: str
    release_date: str
    tier: str
    licence: str
    url: str
    doc_hash: str
    text: str


def iter_docs(corpora: List[Path] = DEFAULT_CORPORA) -> Iterator[Doc]:
    """Every GREEN, non-duplicate document with canonical text on disk."""
    for corpus in corpora:
        manifest = corpus / "manifest.jsonl"
        if not manifest.exists():
            continue
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("tier") not in ALLOWED_TIERS or row.get("duplicate_of"):
                    continue
                path = corpus / "canonical" / f"{row['id']}.txt"
                if not path.exists():
                    continue
                yield Doc(
                    source_id=f"{corpus.name}:{row['id']}",
                    regulator=row["regulator"],
                    document_type=row["document_type"],
                    title=row["title"],
                    release_date=row.get("release_date") or "",
                    tier=row["tier"],
                    licence=row["licence"],
                    url=row.get("url") or "",
                    doc_hash=row["sha256"],
                    text=path.read_text(encoding="utf-8"),
                )
