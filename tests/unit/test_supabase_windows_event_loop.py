from __future__ import annotations

import asyncio

from bi_multiagent.retrieval import supabase_pgvector


def test_event_loop_policy_is_unchanged_off_windows(monkeypatch):
    monkeypatch.setattr(supabase_pgvector.sys, "platform", "linux")
    original_policy = asyncio.get_event_loop_policy()
    assert supabase_pgvector.configure_psycopg_event_loop() is False
    assert asyncio.get_event_loop_policy() is original_policy


def test_windows_uses_selector_policy_before_pgvector_engine(monkeypatch):
    class FakeSelectorPolicy:
        pass

    marker_policy = object()
    installed = []

    monkeypatch.setattr(supabase_pgvector.sys, "platform", "win32")
    monkeypatch.setattr(
        supabase_pgvector.asyncio,
        "WindowsSelectorEventLoopPolicy",
        FakeSelectorPolicy,
        raising=False,
    )
    monkeypatch.setattr(
        supabase_pgvector.asyncio,
        "get_event_loop_policy",
        lambda: marker_policy,
    )
    monkeypatch.setattr(
        supabase_pgvector.asyncio,
        "set_event_loop_policy",
        lambda policy: installed.append(policy),
    )

    assert supabase_pgvector.configure_psycopg_event_loop() is True
    assert len(installed) == 1
    assert isinstance(installed[0], FakeSelectorPolicy)
