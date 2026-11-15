from contextlib import contextmanager

import pytest
from langgraph.checkpoint.postgres import PostgresSaver

from bi_multiagent.persistence import postgres


class FakeSaver:
    def __init__(self):
        self.setup_called = False

    def setup(self):
        self.setup_called = True


def make_fake_context(captured):
    @contextmanager
    def fake_from_conn_string(uri):
        captured["uri"] = uri
        captured["saver"] = FakeSaver()
        captured["closed"] = False

        try:
            yield captured["saver"]
        finally:
            captured["closed"] = True

    return staticmethod(fake_from_conn_string)


def test_postgres_checkpointer_manages_connection_lifecycle(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        postgres,
        "POSTGRES_CHECKPOINTER_URI",
        "postgresql://user:pass@localhost:5432/database",
    )

    monkeypatch.setattr(
        PostgresSaver,
        "from_conn_string",
        make_fake_context(captured),
    )

    with postgres.postgres_checkpointer(setup=True) as saver:
        assert captured["uri"].startswith("postgresql://")
        assert captured["closed"] is False
        assert saver is captured["saver"]
        assert saver.setup_called is True

    assert captured["closed"] is True


def test_postgres_checkpointer_skips_setup_by_default(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        postgres,
        "POSTGRES_CHECKPOINTER_URI",
        "postgresql://user:pass@localhost:5432/database",
    )

    monkeypatch.setattr(
        PostgresSaver,
        "from_conn_string",
        make_fake_context(captured),
    )

    with postgres.postgres_checkpointer() as saver:
        assert saver.setup_called is False

    assert captured["closed"] is True


def test_postgres_checkpointer_closes_connection_after_exception(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        postgres,
        "POSTGRES_CHECKPOINTER_URI",
        "postgresql://user:pass@localhost:5432/database",
    )

    monkeypatch.setattr(
        PostgresSaver,
        "from_conn_string",
        make_fake_context(captured),
    )

    with (
        pytest.raises(ValueError, match="simulated failure"),
        postgres.postgres_checkpointer(),
    ):
        raise ValueError("simulated failure")

    assert captured["closed"] is True


def test_postgres_checkpointer_requires_connection_uri(monkeypatch):
    monkeypatch.setattr(
        postgres,
        "POSTGRES_CHECKPOINTER_URI",
        None,
    )

    with (
        pytest.raises(RuntimeError, match="POSTGRES_CHECKPOINTER_URI"),
        postgres.postgres_checkpointer(),
    ):
        pass
