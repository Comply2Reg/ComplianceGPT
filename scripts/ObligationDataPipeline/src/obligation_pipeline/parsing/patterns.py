"""
Regex patterns for legal document hierarchy (Indian Act style).

Configurable for multiple jurisdictions (RBI, SEBI, IRDA, MAS, etc.).
"""

import re
from typing import Any

# Hierarchy levels aligned with ontology: act, chapter, section, subsection, clause, explanation, proviso, definition
LEGAL_PATTERNS: dict[str, list[dict[str, Any]]] = {
    "part": [
        {"re": re.compile(r"PART\s+([IVXLC]+|\d+)\s*[.—\-]\s*(.*?)(?=\n|$)", re.IGNORECASE | re.DOTALL), "label_grp": 1, "title_grp": 2},
        {"re": re.compile(r"Part\s+([IVXLC]+|\d+)\s*[.—\-]\s*(.*?)(?=\n|$)", re.DOTALL), "label_grp": 1, "title_grp": 2},
    ],
    "chapter": [
        {"re": re.compile(r"CHAPTER\s+([IVXLC0-9]+)\s*[.—\-]?\s*(.*?)(?=\n\n|$)", re.IGNORECASE | re.DOTALL), "label_grp": 1, "title_grp": 2},
        {"re": re.compile(r"Chapter\s+([IVXLC0-9]+)\s*[.—\-]?\s*(.*?)(?=\n\n|$)", re.DOTALL), "label_grp": 1, "title_grp": 2},
    ],
    "section": [
        {"re": re.compile(r"Section\s+(\d+[A-Za-z]?(?:\(\d+\))?)\s*[.—\-]?\s*(.*?)(?=Section\s+\d|CHAPTER|PART|SCHEDULE|Explanation|Proviso|$)", re.IGNORECASE | re.DOTALL), "label_grp": 1, "title_grp": 2},
        {"re": re.compile(r"^\s*(\d+)\s*[.—]\s+([^\n]+)(?=\n)", re.MULTILINE), "label_grp": 1, "title_grp": 2},
    ],
    "subsection": [
        {"re": re.compile(r"^\s*\(([a-z0-9]+)\)\s+(.*?)(?=^\s*\([a-z0-9]+\)|Section\s+\d|$)", re.MULTILINE | re.DOTALL), "label_grp": 1, "title_grp": 2},
        {"re": re.compile(r"\((\d+)\)\s+(.*?)(?=\(\d+\)|Section\s+\d|$)", re.DOTALL), "label_grp": 1, "title_grp": 2},
    ],
    "clause": [
        {"re": re.compile(r"^\s*\(([a-z])\)\s+(.*?)(?=^\s*\([a-z]\)|^\s*\([a-z0-9]+\)|Explanation|Proviso|Section\s+\d|$)", re.MULTILINE | re.DOTALL), "label_grp": 1, "title_grp": 2},
        {"re": re.compile(r"^\s*\(([ivx]+)\)\s+(.*?)(?=^\s*\([ivx]+\)|Explanation|Proviso|$)", re.MULTILINE | re.IGNORECASE | re.DOTALL), "label_grp": 1, "title_grp": 2},
    ],
    "explanation": [
        {"re": re.compile(r"Explanation\s*[.—]*\s*([IVXLC0-9]*)\s*[.—\-]*\s*(.*?)(?=Explanation|Proviso|Section\s+\d|CHAPTER|$)", re.IGNORECASE | re.DOTALL), "label_grp": 1, "title_grp": 2},
    ],
    "proviso": [
        {"re": re.compile(r"Proviso\s*[.—]*\s*(.*?)(?=Explanation|Section\s+\d|CHAPTER|$)", re.IGNORECASE | re.DOTALL), "label_grp": 1, "title_grp": 1},
    ],
    "definition": [
        {"re": re.compile(r"^\s*\(([a-z]+)\)\s+[\"\u201c\u201d]?(.*?)[\"\u201c\u201d]\s+means\s+(.*?)(?=^\s*\([a-z]+\)|Section\s+\d|$)", re.MULTILINE | re.IGNORECASE | re.DOTALL), "label_grp": 1, "title_grp": 2},
    ],
    "schedule": [
        {"re": re.compile(r"SCHEDULE\s+([IVXLC0-9]*)\s*[.—\-]?\s*(.*?)(?=\n\n|$)", re.IGNORECASE | re.DOTALL), "label_grp": 1, "title_grp": 2},
    ],
}
