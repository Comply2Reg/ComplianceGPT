"""Environment-backed configuration for obligation data pipeline.

Merged: GraphRag infrastructure settings (DB, S3, chunking, ontology, LLM parsing)
      + Obligation extraction settings (OpenAI, balancing, confidence weights).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --------------------------------------------------
    # Database (document metadata)
    # --------------------------------------------------
    db_driver: str = Field(default="mysql", alias="DB_DRIVER")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=3306, alias="DB_PORT")
    db_name: str = Field(default="c2rdb", alias="DB_NAME")
    db_user: str = Field(default="root", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # --------------------------------------------------
    # S3 (PDF downloads)
    # --------------------------------------------------
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="ap-south-1", alias="AWS_REGION")
    s3_bucket: str = Field(default="regulens-uploaded-docs", alias="S3_BUCKET")
    aws_profile: Optional[str] = Field(default=None, alias="AWS_PROFILE")

    # --------------------------------------------------
    # Ingestion pipeline
    # --------------------------------------------------
    output_base: Path = Field(default=Path("./data/ingested"), alias="OUTPUT_BASE")
    regulator_column: str = Field(default="regulator", alias="REGULATOR_COLUMN")
    s3_key_column: str = Field(default="file_path", alias="S3_KEY_COLUMN")
    documents_table: str = Field(default="regulation_documents", alias="DOCUMENTS_TABLE")
    act_only: bool = Field(default=False, alias="ACT_ONLY")
    document_type_column: str = Field(default="document_type", alias="DOCUMENT_TYPE_COLUMN")
    document_subtype_column: str = Field(default="document_subtype", alias="DOCUMENT_SUBTYPE_COLUMN")
    document_type_filter: Optional[str] = Field(default=None, alias="DOCUMENT_TYPE_FILTER")
    all_regulators: bool = Field(default=True, alias="ALL_REGULATORS")
    regulators: str = Field(default="SEBI,RBI,IRDA", alias="REGULATORS")
    regulator_aliases: Optional[str] = Field(default=None, alias="REGULATOR_ALIASES")

    @property
    def REGULATORS(self) -> tuple[str, ...]:
        """Parsed list of allowed regulator codes (uppercase)."""
        return tuple(s.strip().upper() for s in self.regulators.split(",") if s.strip()) or ("SEBI", "RBI", "IRDA")

    def _get_regulator_aliases(self) -> dict[str, str]:
        """Parse REGULATOR_ALIASES (key:value,key:value) into a dict."""
        if not self.regulator_aliases:
            return {}
        out: dict[str, str] = {}
        for part in self.regulator_aliases.split(","):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                out[k.strip().upper()] = v.strip().upper()
        return out

    def normalize_regulator(self, value: Optional[str]) -> Optional[str]:
        """Map DB regulator value to standard code."""
        if not value:
            return None
        upper = value.strip().upper()
        if not upper:
            return None
        if self.all_regulators:
            return upper
        aliases = self._get_regulator_aliases()
        mapped = aliases.get(upper, upper)
        return mapped if mapped in self.REGULATORS else None

    def get_database_url(self) -> str:
        """Build database URL from components or use DATABASE_URL if set."""
        if self.database_url:
            return self.database_url
        if self.db_driver == "postgresql":
            return (
                f"postgresql://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        if self.db_driver == "mysql":
            return (
                f"mysql+pymysql://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        raise ValueError(f"Unsupported DB_DRIVER: {self.db_driver}")

    # --------------------------------------------------
    # Structural parsing & chunking
    # --------------------------------------------------
    chunks_output_dir: Path = Field(default=Path("./data/chunks"), alias="CHUNKS_OUTPUT_DIR")
    ingest_manifest_path: Optional[Path] = Field(default=None, alias="INGEST_MANIFEST_PATH")
    chunk_strategy: str = Field(default="hybrid", alias="CHUNK_STRATEGY")
    max_chunk_chars: int = Field(default=6000, alias="MAX_CHUNK_CHARS")
    min_chunk_chars: int = Field(default=200, alias="MIN_CHUNK_CHARS")
    overlap_chars: int = Field(default=150, alias="OVERLAP_CHARS")
    skip_pages: int = Field(default=0, alias="PDF_SKIP_PAGES")
    chunking_write_manifest: bool = Field(default=True, alias="CHUNKING_WRITE_MANIFEST")
    chunking_skip_existing: bool = Field(default=False, alias="CHUNKING_SKIP_EXISTING")

    # Ontology-driven pipeline
    ontology_chunking: bool = Field(default=True, alias="ONTOLOGY_CHUNKING")
    graph_output_dir: Path = Field(default=Path("./data/graph"), alias="GRAPH_OUTPUT_DIR")
    traceability_index_path: Optional[Path] = Field(default=None, alias="TRACEABILITY_INDEX_PATH")

    # --------------------------------------------------
    # LLM-assisted structural parsing (optional)
    # --------------------------------------------------
    use_llm_structure_parser: bool = Field(default=False, alias="USE_LLM_STRUCTURE_PARSER")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-2024-08-06", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=4000, alias="LLM_MAX_TOKENS")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")

    def get_llm_api_key(self) -> Optional[str]:
        """Return API key for structural parsing LLM, falling back to OPENAI_API_KEY."""
        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_provider in {"openai", "azure"}:
            return self.openai_api_key
        return None

    # --------------------------------------------------
    # OpenAI: Obligation Extraction
    # --------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    use_openai: bool = Field(default=True, alias="USE_OPENAI")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-2024-08-06", alias="OPENAI_MODEL")
    max_concurrency: int = Field(default=2, alias="MAX_CONCURRENCY")
    max_retries: int = Field(default=4, alias="MAX_RETRIES")
    max_input_tokens: int = Field(default=4096, alias="MAX_INPUT_TOKENS")
    max_output_tokens: int = Field(default=1024, alias="MAX_OUTPUT_TOKENS")
    pipeline_version: str = Field(default="obl-pipeline-v2", alias="PIPELINE_VERSION")
    prompt_version: str = Field(default="v2.0", alias="PROMPT_VERSION")
    processing_node: str = Field(default="local-dev", alias="PROCESSING_NODE")

    # --------------------------------------------------
    # Dataset balancing
    # --------------------------------------------------
    target_obligation_ratio: float = Field(default=0.50, alias="TARGET_OBLIGATION_RATIO")
    target_non_obligation_ratio: float = Field(default=0.20, alias="TARGET_NON_OBLIGATION_RATIO")
    target_neutral_ratio: float = Field(default=0.30, alias="TARGET_NEUTRAL_RATIO")

    # --------------------------------------------------
    # Confidence score weights (composite scoring)
    # --------------------------------------------------
    confidence_w_classification: float = Field(default=0.25, alias="CONFIDENCE_W_CLASSIFICATION")
    confidence_w_modality: float = Field(default=0.15, alias="CONFIDENCE_W_MODALITY")
    confidence_w_completeness: float = Field(default=0.20, alias="CONFIDENCE_W_COMPLETENESS")
    confidence_w_reasoning: float = Field(default=0.15, alias="CONFIDENCE_W_REASONING")
    confidence_w_grounding: float = Field(default=0.25, alias="CONFIDENCE_W_GROUNDING")

    # --------------------------------------------------
    # Vector DB
    # --------------------------------------------------
    enable_vector_ingest: bool = Field(default=True, alias="ENABLE_VECTOR_INGEST")
    embedding_provider: str = Field(default="sentence_transformers", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    vector_db_type: str = Field(default="chroma", alias="VECTOR_DB_TYPE")
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="obligation_chunks", alias="CHROMA_COLLECTION")

    # --------------------------------------------------
    # Graph DB (Neo4j)
    # --------------------------------------------------
    enable_graph_ingest: bool = Field(default=True, alias="ENABLE_GRAPH_INGEST")
    graph_db_type: str = Field(default="neo4j", alias="GRAPH_DB_TYPE")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")

    # Ingest input path override
    chunks_ontology_path: Optional[Path] = Field(default=None, alias="CHUNKS_ONTOLOGY_PATH")
