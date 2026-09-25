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
    # UK regulatory alerts captured by c2r-inventory-kit and exported with its
    # scripts/export_alert_corpus.py. Unlike the ESMA source there is no fixed
    # id list: documents arrive continuously, so pass --ids explicitly or leave
    # it empty to validate everything present in chunks.jsonl.
    "uk_alerts": {
        "label": "UK regulatory alerts",
        "document_type": None,      # spans FCA/PRA/BoE/HMT/CMA/DRCF types
        "ids": [],
        "profile": "uk_alerts",     # scripts/quality.py SourceProfile
        "corpus_dir": "data/alert_corpus",
    },
    # legislation.gov.uk provisions parsed by scripts/clml.py.
    "uk_legislation": {
        "label": "UK legislation (CLML)",
        "document_type": "legislation",
        "ids": [],
        "profile": "uk_legislation",
    },
}

ACTIVE_SOURCE = "esma_newsletter"
TARGET_DOCUMENT_IDS = list(SOURCES[ACTIVE_SOURCE]["ids"])
TARGET_DOCUMENT_TYPE = SOURCES[ACTIVE_SOURCE]["document_type"]


def get_profile_name(source_key: Optional[str] = None) -> str:
    """Quality-gate profile for a source (see scripts/quality.py)."""
    cfg = SOURCES.get(source_key or ACTIVE_SOURCE, {})
    return cfg.get("profile", "esma_newsletter")

DOCUMENTS_TABLE = "regulation_documents"

# ── Alert-triage labelling and datasets (scripts/triage/) ────────────────────
# Env var NAMES only; the same names the obligation pipeline on `main` uses.
OPENAI_ENV_NAMES = ("OPENAI_API_KEY", "OPENAI_MODEL")

TRIAGE = {
    # The Stage-1 corpus c2r-inventory-kit exports (sibling repo in the club).
    "corpus_dir": str(ROOT.parent.parent / "09-inventory-kit" / "c2r-inventory-kit"
                      / "data" / "alert_corpus"),
    "labels_dir": str(DATA_DIR / "uk" / "labels"),
    "datasets_dir": str(DATA_DIR / "uk" / "datasets"),
    "gold_dir": str(DATA_DIR / "uk" / "gold"),
    "obligation_dir": str(DATA_DIR / "uk" / "obligation"),
    "default_model": "gpt-4o-2024-08-06",
}

# ── Target models for fine-tuning (scripts/triage/render.py, train_triage.py) ──
# Chosen 2026-09-25 (docs/uk-datasets.md "Model choice"). Everything model-
# specific — chat template markers for response-only loss, budget, gating —
# lives here so the fallback is a flag, not a rewrite. Templates themselves are
# never hand-built: render() calls the tokenizer's apply_chat_template.
MODELS = {
    "qwen3-4b-instruct": {
        "hf_id": "Qwen/Qwen3-4B-Instruct-2507",
        "licence": "Apache-2.0",
        "max_context": 262144,
        "max_seq_length": 2048,          # head-of-document triage records
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "supports_system": True,
        "thinking_switch": False,        # non-thinking model; nothing to disable
        "gated": False,
        # Apple Silicon path (scripts/triage/train_mlx.py): the community 4-bit
        # MLX conversion of the same weights.
        "mlx_id": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    },
    "gemma-4-e4b-it": {
        "hf_id": "google/gemma-4-E4B-it",
        "licence": "Apache-2.0",
        "max_context": 131072,
        "max_seq_length": 2048,
        "instruction_part": "<|turn>user\n",
        "response_part": "<|turn>model\n",
        "supports_system": True,
        "thinking_switch": True,         # template takes enable_thinking
        "gated": True,                   # needs HF_TOKEN
        "mlx_id": None,                  # no MLX conversion registered yet
    },
}
FOCUS_MODEL = "qwen3-4b-instruct"

TRAIN = {
    "epochs": 3, "learning_rate": 1e-4, "seed": 3407, "per_device_batch": 1,
    "grad_accum": 8, "lora_r": 16, "lora_alpha": 16,
}
# mlx-lm lora defaults for a 24 GB Apple Silicon machine (train_mlx.py).
TRAIN_MLX = {
    "iters": 600, "batch_size": 1, "num_layers": 16, "learning_rate": 1e-4,
    "steps_per_eval": 100, "save_every": 100,
}

# USD per 1K tokens (prompt, completion); used only for the cost estimate.
PRICE_PER_1K = {
    "gpt-4o-2024-08-06": (0.0025, 0.010),
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
}

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
