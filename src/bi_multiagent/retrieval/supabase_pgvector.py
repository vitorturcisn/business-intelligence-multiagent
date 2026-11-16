"""Supabase Postgres/pgvector backend for the Knowledge Analyst.

The local FAISS index remains the default backend. This module is only loaded
when KNOWLEDGE_VECTOR_BACKEND=supabase. It uses the same local Hugging Face
embedding model as the FAISS implementation to keep vector dimensions aligned.

On Windows, psycopg's async driver is incompatible with the default
ProactorEventLoop. The synchronous PGVectorStore facade internally uses async
SQLAlchemy/psycopg operations, so configure SelectorEventLoop before creating
its engine. Linux/macOS behavior is unchanged.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any


def configure_psycopg_event_loop() -> bool:
    """Use SelectorEventLoop for psycopg async internals on Windows.

    Returns True only when this call changes the process-wide event-loop
    policy. It must run before PGEngine/PGVectorStore creates an async engine.
    """
    if sys.platform != "win32":
        return False

    selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy is None:  # pragma: no cover - Python without Windows policy
        return False

    current_policy = asyncio.get_event_loop_policy()
    if isinstance(current_policy, selector_policy):
        return False

    asyncio.set_event_loop_policy(selector_policy())
    return True


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Add it to the local .env file."
        )
    return value


def get_supabase_connection_uri(*, sqlalchemy: bool = False) -> str:
    """Return the Supabase database URI, optionally adapted for SQLAlchemy."""
    raw = _required_env("SUPABASE_DB_URI")
    parts = urlsplit(raw)
    if parts.scheme not in {"postgres", "postgresql", "postgresql+psycopg"}:
        raise ValueError("SUPABASE_DB_URI must use a PostgreSQL connection URI.")

    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("sslmode", os.getenv("SUPABASE_SSLMODE", "require"))
    scheme = "postgresql+psycopg" if sqlalchemy else "postgresql"
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def vector_config() -> dict[str, Any]:
    schema = os.getenv("SUPABASE_VECTOR_SCHEMA", "public").strip()
    table = os.getenv("SUPABASE_VECTOR_TABLE", "business_knowledge_vectors").strip()
    dimension = int(os.getenv("SUPABASE_VECTOR_SIZE", "384"))
    for label, identifier in (("schema", schema), ("table", table)):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
            raise ValueError(f"Invalid SQL {label} identifier: {identifier!r}")
    if dimension != 384:
        raise ValueError(
            "SUPABASE_VECTOR_SIZE must be 384: the configured multilingual MiniLM "
            "embedding model and SQL migration both use 384 dimensions."
        )
    return {"schema": schema, "table": table, "dimension": dimension}


def _build_embeddings():
    # Same model and normalization as the current FAISS backend, without
    # importing FAISS into an Agent Server deployment that uses Supabase.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """Create and cache the LangChain PGVectorStore handle for this process."""
    # Must be set before PGEngine/SQLAlchemy creates an async psycopg engine.
    configure_psycopg_event_loop()

    try:
        from langchain_postgres import PGEngine, PGVectorStore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Supabase backend requires langchain-postgres. Install the retrieval extra."
        ) from exc

    config = vector_config()
    engine = PGEngine.from_connection_string(
        url=get_supabase_connection_uri(sqlalchemy=True)
    )
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=config["table"],
        schema_name=config["schema"],
        embedding_service=_build_embeddings(),
        id_column="id",
        content_column="content",
        embedding_column="embedding",
        metadata_json_column="metadata",
    )


def search_supabase_knowledge_base(query: str, k: int = 4) -> str:
    """Search business knowledge in Supabase and format source-grounded hits."""
    if not query or not query.strip():
        raise ValueError("The retrieval query cannot be empty.")
    if k < 1:
        raise ValueError("k must be positive.")

    matches = get_vector_store().similarity_search_with_score(query, k=k)
    if not matches:
        return "Nenhum documento relevante foi encontrado no Supabase PGVector."

    blocks = []
    for position, (document, score) in enumerate(matches, start=1):
        source = document.metadata.get("source", "Fonte desconhecida")
        chunk = document.metadata.get("chunk_index", "n/a")
        blocks.append(
            f"Resultado {position}\nFonte: {source}\nTrecho: {chunk}\n"
            f"Score/distância: {float(score):.4f}\nConteúdo:\n{document.page_content}"
        )
    return "\n\n---\n\n".join(blocks)
