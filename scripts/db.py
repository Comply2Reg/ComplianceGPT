"""Read-only access to regulation_documents metadata.

Patterns adapted from GraphRAG GraphRag/src/ingestion/db.py.
Never writes to the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import DOCUMENTS_TABLE, get_database_url, get_db_driver
from s3_uri import parse_s3_uri


@dataclass
class DocumentMeta:
    id: int
    regulator: Optional[str]
    document_type: Optional[str]
    document_subtype: Optional[str]
    title: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    source_site: Optional[str]
    file_path: Optional[str]
    metadata_json: Any
    s3_bucket: Optional[str]
    s3_key: Optional[str]


def get_engine() -> Engine:
    return create_engine(get_database_url(), pool_pre_ping=True)


def _quote_ident(name: str) -> str:
    if get_db_driver() == "mysql":
        return f"`{name}`"
    return f'"{name}"'


def fetch_documents_by_ids(
    document_ids: list[int] | tuple[int, ...],
    expected_document_type: Optional[str] = None,
    table: str = DOCUMENTS_TABLE,
) -> list[DocumentMeta]:
    """
    SELECT read-only metadata for the given IDs.

    If expected_document_type is set, also filter in SQL.
    """
    if not document_ids:
        return []

    q = _quote_ident
    columns = [
        "id",
        "regulator",
        "document_type",
        "document_subtype",
        "title",
        "url",
        "pdf_url",
        "source_site",
        "file_path",
        "metadata_json",
    ]
    col_list = ", ".join(q(c) for c in columns)

    # Named bind params for each id (portable across drivers).
    id_params = {f"id{i}": int(doc_id) for i, doc_id in enumerate(document_ids)}
    id_placeholders = ", ".join(f":id{i}" for i in range(len(document_ids)))

    sql = (
        f"SELECT {col_list} FROM {q(table)} "
        f"WHERE {q('id')} IN ({id_placeholders})"
    )
    params: dict[str, Any] = dict(id_params)
    if expected_document_type is not None:
        sql += f" AND {q('document_type')} = :doc_type"
        params["doc_type"] = expected_document_type

    engine = get_engine()
    rows: list[DocumentMeta] = []
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        for row in result.mappings():
            file_path = row.get("file_path")
            file_path_str = str(file_path).strip() if file_path is not None else None
            parsed = parse_s3_uri(file_path_str) if file_path_str else None
            if parsed:
                s3_bucket, s3_key = parsed.bucket, parsed.key
            elif file_path_str:
                s3_bucket, s3_key = None, file_path_str
            else:
                s3_bucket, s3_key = None, None

            rows.append(
                DocumentMeta(
                    id=int(row["id"]),
                    regulator=row.get("regulator"),
                    document_type=row.get("document_type"),
                    document_subtype=row.get("document_subtype"),
                    title=row.get("title"),
                    url=row.get("url"),
                    pdf_url=row.get("pdf_url"),
                    source_site=row.get("source_site"),
                    file_path=file_path_str,
                    metadata_json=row.get("metadata_json"),
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                )
            )
    return rows
