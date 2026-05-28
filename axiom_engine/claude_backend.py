"""Claude (Anthropic Messages API) generation backend.

Optional adapter that replaces the local Axiom-TTT generation path with
Anthropic-hosted Claude. Useful when:

* The local TTT engine has no trained checkpoint and emits gibberish
  via the SHA-256-hashing ``SimpleTokenizer`` — Claude provides real
  text while the rest of the Axiom HTTP surface (``/v1/models``,
  ``/v1/chat/completions``, ``/v1/messages``) keeps the same protocol.
* You want to keep client integrations on Axiom's wire format while
  swapping the underlying generator.

Important caveat
----------------
When this backend is active, ``/v1/adapt`` and the W_tilde lifecycle
become **no-ops with respect to actual generation** — Claude is a
remote frozen model and we cannot push gradient updates into it.
Sessions still exist (so checkpoint round-trips and session IDs
remain compatible), but they no longer influence output.

Activation
----------
The backend is opt-in via ``AXIOM_BACKEND=claude`` plus the standard
``ANTHROPIC_API_KEY`` (or by constructing ``ClaudeBackend`` directly
and passing it to the server). When inactive, generation falls back
to the local InferencePipeline path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class ClaudeMessage:
    """Minimal chat message shape consumed by the backend."""

    role: str
    content: str


@dataclass
class ClaudeBackend:
    """Anthropic Messages API backend for the Axiom server.

    Parameters
    ----------
    model:
        Anthropic model identifier. Defaults to a current Haiku snapshot.
    api_key:
        Optional override; falls back to the ``ANTHROPIC_API_KEY`` env var.
    default_system:
        System prompt injected when the request does not supply one.
    """

    model: str = DEFAULT_CLAUDE_MODEL
    api_key: Optional[str] = None
    default_system: Optional[str] = None
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "ClaudeBackend requires the 'anthropic' package: "
                "pip install anthropic"
            ) from exc

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ClaudeBackend requires an API key — set ANTHROPIC_API_KEY "
                "or pass api_key=... explicitly."
            )
        self._client = Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Generation entry points used by the server
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Single-turn completion from a bare prompt string."""
        return self.generate_chat(
            [ClaudeMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
            system=self.default_system,
        )

    def generate_chat(
        self,
        messages: List[ClaudeMessage],
        max_tokens: int,
        system: Optional[str] = None,
    ) -> str:
        """Multi-turn completion. Non-user/assistant roles are folded into system."""
        anth_messages, folded_system = _normalize_messages(messages)
        effective_system = system or folded_system or self.default_system

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anth_messages,
        }
        if effective_system:
            kwargs["system"] = effective_system

        response = self._client.messages.create(**kwargs)
        return _extract_text(response)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _normalize_messages(
    messages: List[ClaudeMessage],
) -> tuple[list[dict], Optional[str]]:
    """Split out system messages and coerce alternating user/assistant turns.

    Anthropic requires the conversation in ``messages`` to begin with a
    ``user`` turn and only contain ``user``/``assistant`` roles. System
    prompts are passed via the top-level ``system`` field. We fold any
    ``system`` messages (or unknown roles) into a single system string.
    """
    system_parts: list[str] = []
    chat: list[dict] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role in ("user", "assistant"):
            chat.append({"role": msg.role, "content": msg.content})
        else:
            system_parts.append(f"[{msg.role}] {msg.content}")

    if not chat:
        chat = [{"role": "user", "content": ""}]
    elif chat[0]["role"] != "user":
        chat.insert(0, {"role": "user", "content": ""})

    folded = "\n\n".join(p for p in system_parts if p) or None
    return chat, folded


def _extract_text(response: object) -> str:
    """Pull plain text out of an Anthropic Messages response."""
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


# ----------------------------------------------------------------------
# Environment-driven factory
# ----------------------------------------------------------------------


def backend_from_env() -> Optional[ClaudeBackend]:
    """Return a configured backend when ``AXIOM_BACKEND=claude``, else None."""
    if os.environ.get("AXIOM_BACKEND", "").lower() != "claude":
        return None
    model = os.environ.get("AXIOM_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
    system = os.environ.get("AXIOM_CLAUDE_SYSTEM") or None
    return ClaudeBackend(model=model, default_system=system)
