"""End-to-end tests for the OpenAI and Anthropic adapter surfaces.

The tests use a *tiny* AxiomConfig so the untrained model fits on CPU in
under a second; we only verify wire-format correctness, not output
quality. The Claude backend is exercised against an in-process fake so
no real API key or network call is required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

import pytest
import torch
from fastapi.testclient import TestClient

from axiom_engine import server as srv
from axiom_engine.config import AxiomConfig
from axiom_engine.inference import InferencePipeline


# ----------------------------------------------------------------------
# Pipeline fixture — tiny model so tests stay fast on CPU
# ----------------------------------------------------------------------


def _tiny_cfg() -> AxiomConfig:
    return AxiomConfig(
        d_model=16,
        n_layers=2,
        num_heads=2,
        vocab_size=64,
        lr_inner=1e-3,
        max_context_tokens=8,
    )


@pytest.fixture
def client(monkeypatch):
    """Boot the FastAPI app with a tiny in-memory pipeline.

    The server's lifespan instantiates a full 4096-dim / 32-layer model by
    default — far too large for CPU CI. We patch ``AxiomConfig`` in the
    server module so the lifespan allocates the tiny test config instead.
    """
    monkeypatch.setattr(srv, "AxiomConfig", _tiny_cfg)
    srv._sessions.clear()
    srv.set_claude_backend(None)

    with TestClient(srv.app) as test_client:
        yield test_client

    srv._pipeline = None
    srv._sessions.clear()
    srv.set_claude_backend(None)


# ----------------------------------------------------------------------
# Fake Claude backend — avoids any network call
# ----------------------------------------------------------------------


@dataclass
class _FakeClaude:
    """Stand-in for ClaudeBackend with the same generate / generate_chat API."""

    model: str = "fake-claude"
    last_prompt: str = ""
    last_messages: List[object] = None  # type: ignore[assignment]
    last_system: str = ""

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.last_prompt = prompt
        return f"[claude:{self.model}] {prompt[:32]} (max={max_tokens})"

    def generate_chat(self, messages, max_tokens, system=None):
        self.last_messages = list(messages)
        self.last_system = system or ""
        joined = " | ".join(m.content for m in messages)
        sys_tag = f"[sys={system}] " if system else ""
        return f"{sys_tag}[claude:{self.model}] {joined[:48]}"


# ----------------------------------------------------------------------
# OpenAI surface
# ----------------------------------------------------------------------


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "axiom-ttt-v1"


def test_chat_completion_local(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "axiom-ttt-v1",
            "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)


def test_completion_local(client):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "axiom test", "max_tokens": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert isinstance(body["choices"][0]["text"], str)


# ----------------------------------------------------------------------
# Sessions + adapt round-trip (local pipeline)
# ----------------------------------------------------------------------


def test_session_create_and_adapt_local(client):
    create = client.post("/v1/sessions", json={})
    assert create.status_code == 200
    session_id = create.json()["session_id"]

    adapt = client.post(
        "/v1/adapt",
        json={
            "session_id": session_id,
            "corpus": ["one two three", "four five six"],
        },
    )
    assert adapt.status_code == 200
    assert adapt.json()["corpus_documents"] == 2

    chat = client.post(
        "/v1/chat/completions",
        json={
            "session_id": session_id,
            "messages": [{"role": "user", "content": "go"}],
            "max_tokens": 2,
        },
    )
    assert chat.status_code == 200


def test_checkpoint_round_trip_local(client):
    session_id = client.post("/v1/sessions", json={}).json()["session_id"]
    client.post(
        "/v1/adapt",
        json={"session_id": session_id, "corpus": ["alpha beta gamma"]},
    )

    ckpt = client.get(f"/v1/sessions/{session_id}/checkpoint")
    assert ckpt.status_code == 200
    body = ckpt.json()
    assert body["version"] == 1
    assert len(body["layers"]) > 0

    restored = client.put(f"/v1/sessions/{session_id}/checkpoint", json=body)
    assert restored.status_code == 200


# ----------------------------------------------------------------------
# Anthropic /v1/messages surface
# ----------------------------------------------------------------------


def test_messages_local_string_content(client):
    resp = client.post(
        "/v1/messages",
        json={
            "model": "axiom-ttt-v1",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi there"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"
    assert isinstance(body["content"][0]["text"], str)
    assert body["id"].startswith("msg_")


def test_messages_local_block_content(client):
    """Anthropic clients send content as a list of blocks — must flatten correctly."""
    resp = client.post(
        "/v1/messages",
        json={
            "max_tokens": 4,
            "system": "you are a tester",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one "},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content"][0]["text"] is not None


# ----------------------------------------------------------------------
# Claude backend routing
# ----------------------------------------------------------------------


def test_chat_completion_routes_to_claude_backend(client):
    fake = _FakeClaude(model="fake-claude-haiku")
    srv.set_claude_backend(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "say hi"},
            ],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["message"]["content"]
    assert "[claude:fake-claude-haiku]" in text
    assert fake.last_messages is not None
    assert any(m.content == "say hi" for m in fake.last_messages)


def test_messages_routes_to_claude_backend(client):
    fake = _FakeClaude(model="fake-claude-sonnet")
    srv.set_claude_backend(fake)

    resp = client.post(
        "/v1/messages",
        json={
            "max_tokens": 32,
            "system": "stay on topic",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"][0]["text"].startswith("[sys=stay on topic]")
    assert fake.last_system == "stay on topic"


def test_messages_routes_to_claude_backend_with_block_content(client):
    fake = _FakeClaude()
    srv.set_claude_backend(fake)

    resp = client.post(
        "/v1/messages",
        json={
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "alpha "},
                        {"type": "text", "text": "beta"},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert fake.last_messages[0].content == "alpha beta"


# ----------------------------------------------------------------------
# Claude backend unit: message normalization
# ----------------------------------------------------------------------


def test_claude_backend_normalizes_system_messages():
    from axiom_engine.claude_backend import ClaudeMessage, _normalize_messages

    chat, system = _normalize_messages(
        [
            ClaudeMessage(role="system", content="be helpful"),
            ClaudeMessage(role="user", content="hello"),
            ClaudeMessage(role="assistant", content="hi"),
            ClaudeMessage(role="user", content="how are you?"),
        ]
    )
    assert system == "be helpful"
    assert [m["role"] for m in chat] == ["user", "assistant", "user"]


def test_claude_backend_forces_user_first_turn():
    from axiom_engine.claude_backend import ClaudeMessage, _normalize_messages

    chat, _ = _normalize_messages([ClaudeMessage(role="assistant", content="hi")])
    assert chat[0]["role"] == "user"
    assert chat[1]["role"] == "assistant"
