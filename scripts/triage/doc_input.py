"""What the labeller shows the model for one document.

Not the whole text: the head, the section headings, and the few chunks that
usually carry scope and dates. ~3k tokens in, whatever the document's length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HEAD_CHARS = 6000
MAX_HEADINGS = 40
APPLICABILITY_CHUNKS = 3
APPLICABILITY_CHARS = 400
_PART_RE = re.compile(r"\s*\(part \d+\)\s*$", re.IGNORECASE)
APPLICABILITY_RE = re.compile(
    r"who (this|it) applies to|application|scope|next steps|implementation|"
    r"deadline|comes into force|commencement|timetable|when|transitional",
    re.IGNORECASE,
)


@dataclass
class DocInput:
    doc_id: int
    document_id: str
    doc_hash: str
    regulator: str
    document_type: str
    title: str
    release_date: Optional[str]
    prior_alert_class: Optional[str]
    tier: str
    head: str
    headings: List[str] = field(default_factory=list)
    applicability: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        parts = [
            f"Regulator: {self.regulator}",
            f"Publication type: {self.document_type}",
            f"Title: {self.title}",
            f"Published: {self.release_date or 'unknown'}",
        ]
        if self.prior_alert_class and self.prior_alert_class != "MIXED":
            parts.append(
                f"Registry prior for alert class: {self.prior_alert_class} "
                "(a hint from the source list, not ground truth)"
            )
        if self.headings:
            parts.append("Section headings:\n- " + "\n- ".join(self.headings))
        if self.applicability:
            parts.append(
                "Scope / dates passages:\n" + "\n---\n".join(self.applicability)
            )
        parts.append("Document text (beginning):\n" + self.head)
        return "\n\n".join(parts)


def _clean_heading(section: str) -> str:
    return _PART_RE.sub("", (section or "").strip())


def build_doc_input(row: Dict, canonical: str, chunks: List[Dict]) -> DocInput:
    headings: List[str] = []
    seen = set()
    for c in chunks:
        h = _clean_heading(c.get("section", ""))
        if h and h.lower() != "body" and h not in seen:
            seen.add(h)
            headings.append(h[:120])
        if len(headings) >= MAX_HEADINGS:
            break

    applicability: List[str] = []
    for c in chunks:
        if len(applicability) >= APPLICABILITY_CHUNKS:
            break
        if APPLICABILITY_RE.search(c.get("section", "") or ""):
            text = (c.get("text") or "").strip()
            if text:
                applicability.append(text[:APPLICABILITY_CHARS])

    return DocInput(
        doc_id=int(row["id"]),
        document_id=f"{str(row['regulator']).lower()}:{row['document_type']}#{row['id']}",
        doc_hash=row["sha256"],
        regulator=row["regulator"],
        document_type=row["document_type"],
        title=row.get("title") or "",
        release_date=row.get("release_date"),
        prior_alert_class=row.get("alert_class"),
        tier=str(row.get("tier", "")).upper(),
        head=canonical[:HEAD_CHARS],
        headings=headings,
        applicability=applicability,
    )
