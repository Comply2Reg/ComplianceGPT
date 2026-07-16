"""Fetch document metadata (S3 keys / file_path) from SQL database."""

import logging
from typing import Iterator

from sqlalchemy import text
from sqlalchemy.engine import create_engine
from sqlalchemy.engine import Engine

from src.obligation_pipeline.config import Settings
from src.obligation_pipeline.ingestion.models import DocumentRecord
from src.obligation_pipeline.ingestion.s3_uri import parse_s3_uri

logger = logging.getLogger(__name__)


def get_engine(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine from settings."""
    url = settings.get_database_url()
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def _quote_ident(settings: Settings, name: str) -> str:
    """Quote identifier for the configured DB driver (MySQL backticks, PostgreSQL double-quotes)."""
    if settings.db_driver == "mysql":
        return f"`{name}`"
    return f'"{name}"'


def fetch_document_metadata(settings: Settings) -> Iterator[DocumentRecord]:
    """
    Yield document metadata from the configured table.
    Uses file_path (S3 URI or key); when ACT_ONLY is True, filters to
    UPPER(document_type) LIKE 'ACT%' OR UPPER(document_subtype) LIKE 'ACT%'.
    """
    engine = get_engine(settings)
    def q(name: str) -> str:
        return _quote_ident(settings, name)
    reg_col = settings.regulator_column
    key_col = settings.s3_key_column
    table = settings.documents_table
    type_col = settings.document_type_column
    subtype_col = settings.document_subtype_column

    columns = [
        "id",
        reg_col,
        key_col,
        "title",
        type_col,
        subtype_col,
        "release_date",
        "url",
        "pdf_url",
    ]
    col_list = ", ".join(q(c) for c in columns)
    query = f"SELECT {col_list} FROM {q(table)}"
    params: dict = {}

    if settings.act_only:
        query += f" WHERE (UPPER({q(type_col)}) LIKE :act_prefix) OR (UPPER({q(subtype_col)}) LIKE :act_prefix2)"
        params["act_prefix"] = "ACT%"
        params["act_prefix2"] = "ACT%"
    elif settings.document_type_filter:
        query += f" WHERE {q(type_col)} = :doc_type_filter"
        params["doc_type_filter"] = settings.document_type_filter

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        for row in result.mappings():
            reg_raw = row.get(reg_col)
            regulator = settings.normalize_regulator(reg_raw if reg_raw is not None else None)
            if not regulator:
                logger.warning("Skipping row with unsupported regulator: %s", reg_raw)
                continue
            file_path_val = row.get(key_col)
            if not file_path_val or not str(file_path_val).strip():
                logger.warning("Skipping row with empty file_path")
                continue

            file_path = str(file_path_val).strip()
            parsed = parse_s3_uri(file_path)
            if parsed:
                s3_key = parsed.key
                s3_bucket = parsed.bucket
            else:
                s3_key = file_path
                s3_bucket = None

            release_val = row.get("release_date")
            release_date = None
            if release_val is not None:
                release_date = release_val.isoformat() if hasattr(release_val, "isoformat") else str(release_val)

            record = DocumentRecord(
                s3_key=s3_key,
                regulator=regulator,
                s3_bucket=s3_bucket,
                row_id=row.get("id"),
                title=row.get("title"),
                document_type=row.get(type_col),
                document_subtype=row.get(subtype_col),
                release_date=release_date,
                source_url=row.get("url"),
                pdf_url=row.get("pdf_url"),
                extra_metadata={
                    k: v for k, v in row.items() if v is not None and k not in (reg_col, key_col, "id", "title", type_col, subtype_col, "release_date", "url", "pdf_url")
                },
            )
            yield record

    logger.info("Finished fetching document metadata from database")
