"""Tests for the response cache: unit + integration with the server.

Coverage:
* Fingerprinting is stable and order-independent for the message list.
* LRU eviction kicks in at max_entries.
* Persistence round-trip via a temp file.
* Cache-aware routing: a fake Claude backend is hit only once for the
  same request fingerprint; the second call returns from cache.
* Per-session requests intentionally bypass the cache (output depends
  on W̃ state, not the prompt).
* /v1/cache/stats and DELETE /v1/cache work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from axiom_engine import server as srv
from axiom_engine.config import AxiomConfig
from axiom_engine.response_cache import ResponseCache, cache_from_env, fingerprint


def _tiny_cfg() -> AxiomConfig:
    return AxiomConfig(
        d_model=16,
        n_layers=2,
        num_heads=2,
        vocab_size=64,
        lr_inner=1e-3,
        max_context_tokens=8,
    )


# ----------------------------------------------------------------------
# Unit tests — fingerprinting and LRU semantics
# ----------------------------------------------------------------------


def test_fingerprint_is_stable():
    a = fingerprint(model="m", max_tokens=10, prompt="hi")
    b = fingerprint(model="m", max_tokens=10, prompt="hi")
    assert a == b


def test_fingerprint_differs_on_meaningful_changes():
    base = fingerprint(model="m", max_tokens=10, prompt="hi")
    assert base != fingerprint(model="m", max_tokens=11, prompt="hi")
    assert base != fingerprint(model="m", max_tokens=10, prompt="bye")
    assert base != fingerprint(model="other", max_tokens=10, prompt="hi")


def test_fingerprint_is_message_order_sensitive():
    a = fingerprint(
        model="m", max_tokens=10,
        messages=[{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
    )
    b = fingerprint(
        model="m", max_tokens=10,
        messages=[{"role": "user", "content": "second"}, {"role": "user", "content": "first"}],
    )
    assert a != b


def test_lru_eviction():
    cache = ResponseCache(max_entries=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k3", "v3")  # evicts k1
    assert cache.get("k1") is None
    assert cache.get("k2") == "v2"
    assert cache.get("k3") == "v3"


def test_lru_promotes_on_get():
    cache = ResponseCache(max_entries=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    assert cache.get("k1") == "v1"  # promotes k1
    cache.put("k3", "v3")            # evicts k2 (least-recent)
    assert cache.get("k1") == "v1"
    assert cache.get("k2") is None
    assert cache.get("k3") == "v3"


def test_stats_track_hits_and_misses():
    cache = ResponseCache(max_entries=4)
    cache.put("k", "v")
    cache.get("k")
    cache.get("k")
    cache.get("missing")
    s = cache.stats()
    assert s.entries == 1
    assert s.hits == 2
    assert s.misses == 1


def test_persistence_round_trip(tmp_path: Path):
    path = tmp_path / "cache.json"
    cache = ResponseCache(max_entries=4, persist_path=path)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload == {"k1": "v1", "k2": "v2"}

    restored = ResponseCache(max_entries=4, persist_path=path)
    assert restored.get("k1") == "v1"
    assert restored.get("k2") == "v2"


def test_cache_from_env_disabled(monkeypatch):
    monkeypatch.delenv("AXIOM_CACHE", raising=False)
    monkeypatch.delenv("AXIOM_CACHE_PATH", raising=False)
    assert cache_from_env() is None


def test_cache_from_env_in_memory(monkeypatch):
    monkeypatch.setenv("AXIOM_CACHE", "1")
    monkeypatch.delenv("AXIOM_CACHE_PATH", raising=False)
    cache = cache_from_env()
    assert cache is not None
    assert cache.persist_path is None


def test_cache_from_env_persistent(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("AXIOM_CACHE", raising=False)
    monkeypatch.setenv("AXIOM_CACHE_PATH", str(tmp_path / "c.json"))
    cache = cache_from_env()
    assert cache is not None
    assert cache.persist_path == tmp_path / "c.json"


# ----------------------------------------------------------------------
# Integration — cache routing via the FastAPI server
# ----------------------------------------------------------------------


class _CountingFakeClaude:
    """Stand-in backend that counts calls so we can prove cache hits skipped it."""

    def __init__(self) -> None:
        self.model = "fake-claude"
        self.generate_calls = 0
        self.chat_calls = 0

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.generate_calls += 1
        return f"reply#{self.generate_calls}:{prompt}"

    def generate_chat(self, messages, max_tokens, system=None):
        self.chat_calls += 1
        joined = "|".join(m.content for m in messages)
        return f"chat#{self.chat_calls}:{joined}"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(srv, "AxiomConfig", _tiny_cfg)
    srv._sessions.clear()
    srv.set_claude_backend(None)
    srv.set_response_cache(None)

    with TestClient(srv.app) as test_client:
        yield test_client

    srv._pipeline = None
    srv._sessions.clear()
    srv.set_claude_backend(None)
    srv.set_response_cache(None)


def test_cache_hit_skips_claude_backend(client):
    fake = _CountingFakeClaude()
    srv.set_claude_backend(fake)
    srv.set_response_cache(ResponseCache(max_entries=8))

    body = {
        "messages": [{"role": "user", "content": "repeated query"}],
        "max_tokens": 16,
    }
    r1 = client.post("/v1/chat/completions", json=body)
    r2 = client.post("/v1/chat/completions", json=body)
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["choices"][0]["message"]["content"] == r2.json()["choices"][0]["message"]["content"]
    assert fake.chat_calls == 1, "second identical request should hit cache, not Claude"

    stats = client.get("/v1/cache/stats").json()
    assert stats["enabled"] is True
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1


def test_cache_hit_skips_for_messages_endpoint(client):
    fake = _CountingFakeClaude()
    srv.set_claude_backend(fake)
    srv.set_response_cache(ResponseCache(max_entries=8))

    body = {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping anthropic"}],
    }
    r1 = client.post("/v1/messages", json=body)
    r2 = client.post("/v1/messages", json=body)
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["content"][0]["text"] == r2.json()["content"][0]["text"]
    assert fake.chat_calls == 1


def test_cache_distinguishes_different_prompts(client):
    fake = _CountingFakeClaude()
    srv.set_claude_backend(fake)
    srv.set_response_cache(ResponseCache(max_entries=8))

    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "a"}], "max_tokens": 4,
    })
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "b"}], "max_tokens": 4,
    })
    assert fake.chat_calls == 2

    stats = client.get("/v1/cache/stats").json()
    assert stats["entries"] == 2
    assert stats["misses"] == 2


def test_session_requests_bypass_cache(client):
    """Per-session generation depends on W̃ state, so caching is unsafe."""
    fake = _CountingFakeClaude()
    srv.set_claude_backend(fake)
    srv.set_response_cache(ResponseCache(max_entries=8))

    session_id = client.post("/v1/sessions", json={}).json()["session_id"]
    body = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": "stateful"}],
        "max_tokens": 4,
    }
    client.post("/v1/chat/completions", json=body)
    client.post("/v1/chat/completions", json=body)
    assert fake.chat_calls == 2, "session-aware requests must not be cached"

    stats = client.get("/v1/cache/stats").json()
    assert stats["entries"] == 0


def test_cache_clear_endpoint(client):
    fake = _CountingFakeClaude()
    srv.set_claude_backend(fake)
    srv.set_response_cache(ResponseCache(max_entries=8))

    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4,
    })
    assert client.get("/v1/cache/stats").json()["entries"] == 1

    cleared = client.delete("/v1/cache").json()
    assert cleared == {"cleared": True}

    stats = client.get("/v1/cache/stats").json()
    assert stats["entries"] == 0
    assert stats["hits"] == 0


def test_cache_stats_when_disabled(client):
    srv.set_response_cache(None)
    body = client.get("/v1/cache/stats").json()
    assert body == {"enabled": False}

    cleared = client.delete("/v1/cache").json()
    assert cleared == {"cleared": False}
