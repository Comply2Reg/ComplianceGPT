"""Safe configuration for the Stage-1 data pipeline.

Loads environment variable NAMES via python-dotenv / os.environ.
Never logs or prints secret values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CANONICAL_DIR = DATA_DIR / "canonical"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# Central source selection — extend here for FCA / other sources later.
SOURCES: dict[str, dict] = {
    "esma_newsletter": {
        "label": "ESMA newsletters",
        "document_type": "ESMA newsletter",
        "ids": [4424, 7673, 8162],
    },
    # Future example (not active):
    # "fca_newsletter": {
    #     "label": "FCA newsletters",
    #     "document_type": "FCA newsletter",
    #     "ids": [],
    # },
}

ACTIVE_SOURCE = "esma_newsletter"
TARGET_DOCUMENT_IDS = list(SOURCES[ACTIVE_SOURCE]["ids"])
TARGET_DOCUMENT_TYPE = SOURCES[ACTIVE_SOURCE]["document_type"]

DOCUMENTS_TABLE = "regulation_documents"

# Env var names expected for DB (GraphRAG DB_* and this project's MYSQL_*).
DB_ENV_NAMES = (
    "DATABASE_URL",
    "DB_DRIVER",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_DATABASE",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
)

# Env var names for AWS / S3 (this project + GraphRAG aliases).
AWS_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_PROFILE",
    "AWS_S3_BUCKET",
    "AWS_S3_BUCKET_REGU_LENS",
    "S3_BUCKET",
)


def load_env(dotenv_path: Optional[Path] = None) -> None:
    """Load .env from project root if present. Does not print values."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    path = dotenv_path or (ROOT / ".env")
    if path.is_file():
        load_dotenv(path, override=False)
    else:
        load_dotenv(override=False)


def env_present(name: str) -> bool:
    val = os.environ.get(name)
    return bool(val is not None and str(val).strip() != "")


def _mysql_env_complete() -> bool:
    return all(
        env_present(k)
        for k in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD")
    )


def _db_star_env_complete() -> bool:
    return all(
        env_present(k)
        for k in ("DB_DRIVER", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    )


def missing_required_db_config() -> list[str]:
    """Return missing DB config keys (names only)."""
    if env_present("DATABASE_URL") or _mysql_env_complete() or _db_star_env_complete():
        return []
    if any(env_present(k) for k in ("MYSQL_HOST", "MYSQL_USER", "MYSQL_DATABASE")):
        required = [
            "MYSQL_HOST",
            "MYSQL_PORT",
            "MYSQL_DATABASE",
            "MYSQL_USER",
            "MYSQL_PASSWORD",
        ]
        return [k for k in required if not env_present(k)]
    required = ["DB_DRIVER", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    return [k for k in required if not env_present(k)]


def get_database_url() -> str:
    """Build DB URL from env. Raises if required config is missing."""
    missing = missing_required_db_config()
    if missing:
        raise RuntimeError(
            "Database configuration incomplete. Missing env vars: "
            + ", ".join(missing)
            + ". Set DATABASE_URL, or MYSQL_* , or DB_* . "
            "Do not paste secrets into chat."
        )

    if env_present("DATABASE_URL"):
        return os.environ["DATABASE_URL"].strip()

    # Prefer this project's MYSQL_* naming when present.
    if _mysql_env_complete():
        user = os.environ["MYSQL_USER"].strip()
        password = quote_plus(os.environ["MYSQL_PASSWORD"])
        host = os.environ["MYSQL_HOST"].strip()
        port = os.environ["MYSQL_PORT"].strip()
        name = os.environ["MYSQL_DATABASE"].strip()
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"

    driver = os.environ["DB_DRIVER"].strip().lower()
    user = os.environ["DB_USER"].strip()
    password = quote_plus(os.environ["DB_PASSWORD"])
    host = os.environ["DB_HOST"].strip()
    port = os.environ["DB_PORT"].strip()
    name = os.environ["DB_NAME"].strip()

    if driver in ("postgresql", "postgres"):
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    if driver == "mysql":
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"
    raise RuntimeError(f"Unsupported DB_DRIVER: {driver!r} (use postgresql or mysql)")


def get_db_driver() -> str:
    if env_present("DATABASE_URL"):
        url = os.environ["DATABASE_URL"].strip().lower()
        if url.startswith("mysql"):
            return "mysql"
        return "postgresql"
    if _mysql_env_complete():
        return "mysql"
    return os.environ.get("DB_DRIVER", "postgresql").strip().lower()


def get_aws_region() -> Optional[str]:
    if env_present("AWS_REGION"):
        return os.environ["AWS_REGION"].strip()
    return None


def resolve_default_bucket() -> Optional[str]:
    """
    Resolve default S3 bucket from env when file_path is a key (not s3:// URI).

    Preference order matches this project's .env naming, then GraphRAG's S3_BUCKET.
    Returns the value for internal use only — callers must not print it.
    """
    for name in ("AWS_S3_BUCKET_REGU_LENS", "AWS_S3_BUCKET", "S3_BUCKET"):
        if env_present(name):
            return os.environ[name].strip()
    return None


def bucket_env_status() -> str:
    """Safe status string: which bucket env names are set (not values)."""
    set_names = [
        n
        for n in ("AWS_S3_BUCKET_REGU_LENS", "AWS_S3_BUCKET", "S3_BUCKET")
        if env_present(n)
    ]
    if not set_names:
        return "none of AWS_S3_BUCKET_REGU_LENS / AWS_S3_BUCKET / S3_BUCKET are set"
    return "set: " + ", ".join(set_names)


@dataclass(frozen=True)
class SourceSelection:
    source_key: str
    document_type: str
    ids: tuple[int, ...]

    @property
    def label(self) -> str:
        return SOURCES[self.source_key].get("label", self.source_key)


def get_source_selection(
    source_key: Optional[str] = None,
    ids: Optional[list[int]] = None,
) -> SourceSelection:
    key = source_key or ACTIVE_SOURCE
    if key not in SOURCES:
        raise ValueError(
            f"Unknown source {key!r}. Known sources: {', '.join(SOURCES)}"
        )
    cfg = SOURCES[key]
    selected_ids = tuple(ids) if ids is not None else tuple(cfg["ids"])
    return SourceSelection(
        source_key=key,
        document_type=cfg["document_type"],
        ids=selected_ids,
    )


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
