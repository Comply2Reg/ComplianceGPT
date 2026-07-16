"""Vector database ingestion helpers for obligation records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .schema import ObligationRecord


def _embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    if settings.embedding_provider == "openai" and settings.openai_api_key:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.embeddings.create(model=settings.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    vectors = model.encode(texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def ingest_records_to_chroma(records: list[ObligationRecord], settings: Settings) -> int:
    if not records:
        return 0
    if settings.vector_db_type.lower() != "chroma":
        return 0
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    persist_dir = Path(settings.chroma_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [r.metadata.source_id for r in records]
    docs = [r.input_text for r in records]
    embeddings = _embed_texts(docs, settings)
    metas: list[dict[str, Any]] = []
    for r in records:
        metas.append(
            {
                "classification": r.classification,
                "document_id": r.metadata.source_id.rsplit("_", 1)[0] if "_" in r.metadata.source_id else r.metadata.source_id,
                "document_title": r.metadata.regulation_name,
                "chunk_id": r.metadata.source_id,
                "page_number": -1,
                "vector_id": r.metadata.source_id,
                "confidence_score": 1.0,
                "confidence_tier": "unknown",
            }
        )
    batch = 100
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=docs[i : i + batch],
            metadatas=metas[i : i + batch],
        )
    return len(ids)
