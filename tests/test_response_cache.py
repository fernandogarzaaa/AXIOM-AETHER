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


def test_stats_hit_rate_value_is_pinned():
    """Mutation: hits/total -> hits*total must not survive."""
    cache = ResponseCache(max_entries=4)
    cache.put("k", "v")
    cache.get("k")
    cache.get("k")
    cache.get("missing")
    d = cache.stats().to_dict()
    assert d["hit_rate"] == pytest.approx(2 / 3)
    assert d["hit_rate"] != 2 * 3  # guards against hits*total mutant


def test_stats_hit_rate_zero_when_no_requests():
    d = ResponseCache(max_entries=4).stats().to_dict()
    assert d["hit_rate"] == 0.0
    assert d["hits"] == 0
    assert d["misses"] == 0


def test_fingerprint_nested_key_order_independent():
    """Mutation: sort_keys=True -> False must not survive.

    Nested message dicts with keys in different insertion orders must
    fingerprint identically; only sort_keys=True guarantees that.
    """
    a = fingerprint(
        model="m", max_tokens=10,
        messages=[{"role": "user", "content": "hello"}],
    )
    b = fingerprint(
        model="m", max_tokens=10,
        messages=[{"content": "hello", "role": "user"}],
    )
    assert a == b


def test_lru_put_existing_key_promotes():
    """Mutation: removing move_to_end from put() must not survive.

    Re-putting an existing key must promote it to most-recently-used.
    """
    cache = ResponseCache(max_entries=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k1", "v1-updated")  # re-put promotes k1
    cache.put("k3", "v3")          # must evict k2, not k1
    assert cache.get("k1") == "v1-updated"
    assert cache.get("k2") is None
    assert cache.get("k3") == "v3"


def test_clear_persists_empty_to_disk(tmp_path: Path):
    """Mutation: removing _persist_locked() from clear() must not survive."""
    path = tmp_path / "cache.json"
    cache = ResponseCache(max_entries=4, persist_path=path)
    cache.put("k1", "v1")
    assert json.loads(path.read_text()) == {"k1": "v1"}
    cache.clear()
    assert path.exists()
    assert json.loads(path.read_text()) == {}


def test_load_corrupt_file_starts_empty(tmp_path: Path):
    """Mutation: except (OSError, JSONDecodeError) must not survive."""
    path = tmp_path / "cache.json"
    path.write_text("this is not json{{{", encoding="utf-8")
    cache = ResponseCache(max_entries=4, persist_path=path)
    assert cache.stats().entries == 0
    assert cache.get("anything") is None


def test_load_unreadable_file_starts_empty(tmp_path: Path, monkeypatch):
    """Mutation: except OSError in _load_from_disk must not survive."""
    path = tmp_path / "cache.json"
    path.write_text('{"k": "v"}', encoding="utf-8")

    def _boom(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", _boom)
    cache = ResponseCache(max_entries=4, persist_path=path)
    assert cache.stats().entries == 0


def test_persist_failure_is_swallowed(tmp_path: Path, monkeypatch):
    """Mutation: except OSError in _persist_locked must not survive."""
    path = tmp_path / "cache.json"
    cache = ResponseCache(max_entries=4, persist_path=path)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)
    cache.put("k", "v")  # must not raise
    assert cache.get("k") == "v"  # in-memory copy still works


def test_cache_from_env_invalid_max_entries_falls_back(monkeypatch):
    """Mutation: except ValueError in cache_from_env must not survive."""
    from axiom_engine.response_cache import DEFAULT_MAX_ENTRIES

    monkeypatch.setenv("AXIOM_CACHE", "1")
    monkeypatch.setenv("AXIOM_CACHE_MAX_ENTRIES", "not-a-number")
    monkeypatch.delenv("AXIOM_CACHE_PATH", raising=False)
    cache = cache_from_env()
    assert cache is not None
    assert cache.max_entries == DEFAULT_MAX_ENTRIES


def test_max_entries_zero_clamped_to_one():
    """Mutation: max(1, max_entries) -> max(2, max_entries) must not survive."""
    cache = ResponseCache(max_entries=0)
    assert cache.max_entries == 1
    cache.put("k", "v")
    assert cache.get("k") == "v"


def test_clear_resets_hit_miss_counters():
    """Mutation: hits=0 -> 1 / misses=0 -> 1 in clear() must not survive."""
    cache = ResponseCache(max_entries=4)
    cache.put("k", "v")
    cache.get("k")
    cache.get("missing")
    assert cache.stats().hits == 1
    assert cache.stats().misses == 1
    cache.clear()
    d = cache.stats().to_dict()
    assert d["hits"] == 0
    assert d["misses"] == 0
    assert d["hit_rate"] == 0.0


def test_load_skips_non_string_entries(tmp_path: Path):
    """Mutation: isinstance and->or in _load_from_disk must not survive.

    JSON object keys always deserialize as str, so the value check is the
    live one: {"k3": 456} must be skipped while {"k1": "v1"} loads.
    """
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"k1": "v1", "k3": 456}), encoding="utf-8")
    cache = ResponseCache(max_entries=4, persist_path=path)
    assert cache.get("k1") == "v1"
    assert cache.get("k3") is None  # non-str value skipped
    assert cache.stats().entries == 1


def test_fingerprint_non_ascii_stable():
    """Mutation: ensure_ascii=False -> True must not survive."""
    a = fingerprint(model="m", max_tokens=10, prompt="héllo wörld 🎉")
    b = fingerprint(model="m", max_tokens=10, prompt="héllo wörld 🎉")
    assert a == b
    # ensure_ascii=True would produce a different (escaped) encoding
    import hashlib

    canonical = {
        "model": "m", "max_tokens": 10, "prompt": "héllo wörld 🎉",
        "messages": None, "system": None,
    }
    escaped = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()
    assert a != escaped


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
