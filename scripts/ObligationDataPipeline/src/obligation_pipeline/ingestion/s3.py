"""Download PDF files from S3."""

import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.ingestion.models import DocumentRecord

logger = logging.getLogger(__name__)


def get_s3_client(settings: Settings):
    """Create S3 client from settings (credentials or profile)."""
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    elif settings.aws_profile:
        kwargs["profile_name"] = settings.aws_profile
    return boto3.client("s3", **kwargs)


def download_pdf(
    settings: Settings,
    record: DocumentRecord,
    local_path: Path,
    s3_client=None,
) -> tuple[bool, Optional[str]]:
    """
    Download a single PDF from S3 to local_path.
    Returns (success, error_message).
    """
    if s3_client is None:
        s3_client = get_s3_client(settings)
    bucket = record.s3_bucket if record.s3_bucket else settings.s3_bucket
    key = record.s3_key

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        logger.debug("Downloaded s3://%s/%s -> %s", bucket, key, local_path)
        return True, None
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = str(e)
        logger.warning("S3 download failed for %s: %s", key, msg)
        return False, f"{code}: {msg}"
    except OSError as e:
        logger.warning("File write failed for %s: %s", local_path, e)
        return False, str(e)
