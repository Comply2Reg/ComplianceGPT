"""Read-only S3 helpers using boto3.

Adapted from GraphRAG GraphRag/src/ingestion/s3.py.
Only head_object / download_file / get_object-style reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from config import get_aws_region, resolve_default_bucket
from s3_uri import redact_s3_location


def get_s3_client():
    """
    Create an S3 client via standard boto3 credential resolution.

    If AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are in the environment
    (e.g. via .env), boto3 picks them up. Otherwise profile / instance role.
    """
    kwargs: dict[str, Any] = {}
    region = get_aws_region()
    if region:
        kwargs["region_name"] = region

    profile = None
    import os

    if os.environ.get("AWS_PROFILE", "").strip():
        profile = os.environ["AWS_PROFILE"].strip()

    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
        return session.client("s3")
    return boto3.client("s3", **kwargs)


def resolve_bucket_and_key(
    s3_bucket: Optional[str],
    s3_key: Optional[str],
) -> tuple[str, str]:
    if not s3_key or not str(s3_key).strip():
        raise ValueError("Missing S3 object key / file_path")
    bucket = s3_bucket or resolve_default_bucket()
    if not bucket:
        raise ValueError(
            "No S3 bucket available: file_path is not an s3:// URI and no "
            "AWS_S3_BUCKET_REGU_LENS / AWS_S3_BUCKET / S3_BUCKET env var is set"
        )
    return bucket, s3_key.strip()


def head_object_safe(bucket: str, key: str, s3_client=None) -> dict[str, Any]:
    client = s3_client or get_s3_client()
    return client.head_object(Bucket=bucket, Key=key)


def download_object(
    bucket: str,
    key: str,
    local_path: Path,
    s3_client=None,
) -> int:
    """
    Download object to local_path (read-only). Returns ContentLength.
    Raises RuntimeError with safe diagnostics on failure.
    """
    client = s3_client or get_s3_client()
    loc = redact_s3_location(bucket, key)
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        raise RuntimeError(f"S3 head_object failed ({code}); {loc}") from None
    except BotoCoreError as e:
        raise RuntimeError(f"S3 head_object failed ({type(e).__name__}); {loc}") from None

    size = int(head.get("ContentLength") or 0)
    if size <= 0:
        raise RuntimeError(f"S3 object is empty; {loc}")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(local_path))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        raise RuntimeError(f"S3 download_file failed ({code}); {loc}") from None
    except BotoCoreError as e:
        raise RuntimeError(f"S3 download_file failed ({type(e).__name__}); {loc}") from None

    written = local_path.stat().st_size
    if written <= 0:
        raise RuntimeError(f"Downloaded file is empty; {loc}")
    return written
