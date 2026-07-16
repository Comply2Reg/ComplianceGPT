"""Parse S3 URIs (e.g. s3://bucket/key/path/to/file.pdf)."""

import re
from dataclasses import dataclass
from typing import Optional

S3_URI_PATTERN = re.compile(r"^s3://([^/]+)/(.+)$")


@dataclass
class ParsedS3Uri:
    bucket: str
    key: str

    @property
    def full_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(file_path: str) -> Optional[ParsedS3Uri]:
    """
    Parse file_path as either a full s3:// URI or a plain key.
    Returns ParsedS3Uri(bucket, key) for s3:// URIs, or None if not an S3 URI
    (caller should treat file_path as key and use config bucket).
    """
    if not file_path or not file_path.strip():
        return None
    file_path = file_path.strip()
    m = S3_URI_PATTERN.match(file_path)
    if m:
        return ParsedS3Uri(bucket=m.group(1), key=m.group(2))
    return None
