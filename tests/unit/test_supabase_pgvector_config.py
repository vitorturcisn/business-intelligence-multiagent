from __future__ import annotations

import pytest

from bi_multiagent.retrieval.supabase_pgvector import (
    get_supabase_connection_uri,
    vector_config,
)


def test_pgvector_config_defaults(monkeypatch):
    for key in ("SUPABASE_VECTOR_SCHEMA", "SUPABASE_VECTOR_TABLE", "SUPABASE_VECTOR_SIZE"):
        monkeypatch.delenv(key, raising=False)
    assert vector_config() == {
        "schema": "public",
        "table": "business_knowledge_vectors",
        "dimension": 384,
    }


def test_pgvector_rejects_invalid_table_identifier(monkeypatch):
    monkeypatch.setenv("SUPABASE_VECTOR_TABLE", "docs;drop table users")
    with pytest.raises(ValueError):
        vector_config()


def test_pgvector_dimension_must_match_configured_embedding_model(monkeypatch):
    monkeypatch.setenv("SUPABASE_VECTOR_SIZE", "384")
    assert vector_config()["dimension"] == 384
    monkeypatch.setenv("SUPABASE_VECTOR_SIZE", "768")
    with pytest.raises(ValueError, match="must be 384"):
        vector_config()


def test_connection_uri_adds_ssl_and_sqlalchemy_driver(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URI", "postgresql://user:pass@db.example.test:5432/postgres")
    assert get_supabase_connection_uri(sqlalchemy=True) == (
        "postgresql+psycopg://user:pass@db.example.test:5432/postgres?sslmode=require"
    )


def test_connection_uri_preserves_explicit_ssl(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URI", "postgresql://user:pass@db.example.test:5432/postgres?sslmode=verify-full")
    assert "sslmode=verify-full" in get_supabase_connection_uri()
