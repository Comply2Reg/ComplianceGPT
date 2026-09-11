"""Parse S3 URIs from database file_path values.

Adapted from the GraphRAG project's ingestion.s3_uri helper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

S3_URI_PATTERN = re.compile(r"^s3://([^/]+)/(.+)$")


@dataclass(frozen=True)
class ParsedS3Uri:
    bucket: str
    key: str

    @property
    def full_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(file_path: str) -> Optional[ParsedS3Uri]:
    """
    Parse file_path as s3://bucket/key.

    Returns None if not a full S3 URI (caller may treat value as a key only).
    """
    if not file_path or not str(file_path).strip():
        return None
    m = S3_URI_PATTERN.match(str(file_path).strip())
    if not m:
        return None
    return ParsedS3Uri(bucket=m.group(1), key=m.group(2))


def key_extension(key: str) -> str:
    """Return lowercase extension including dot, or empty string."""
    name = key.rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def redact_s3_location(bucket: Optional[str], key: Optional[str]) -> str:
    """Safe diagnostic: show key basename only, never full sensitive paths."""
    base = (key or "").rsplit("/", 1)[-1] if key else "?"
    bucket_set = "yes" if bucket else "no"
    return f"bucket_configured={bucket_set} object_basename={base!r}"
