"""Response-fingerprint cache for the Axiom HTTP server.

Wraps any generation path (local TTT pipeline, Claude backend, future
ones) with a deterministic memoizing layer. Identical request
fingerprints return the cached response without invoking the underlying
generator — when the generator is the hosted Claude API, that
translates directly into Anthropic token savings on repeated prompts.

Activation
----------
* ``AXIOM_CACHE=1`` — in-memory LRU cache, default 1024 entries.
* ``AXIOM_CACHE_PATH=/path/to/cache.json`` — persistent across restarts
  (implies cache enabled).
* ``AXIOM_CACHE_MAX_ENTRIES=N`` — override default eviction threshold.

Fingerprinting
--------------
Cache keys are SHA-256 over a canonical JSON of:

    {"model": ..., "system": ..., "messages": [...],
     "prompt": ..., "max_tokens": ...}

``temperature`` is intentionally *not* part of the key — the cache only
makes sense for deterministic (temperature=0 or near-zero) usage. Skip
the cache or send a unique nonce for sampling-heavy workloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger("axiom.cache")

DEFAULT_MAX_ENTRIES = 1024


@dataclass
class CacheStats:
    entries: int
    hits: int
    misses: int

    def to_dict(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {
            "entries": self.entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class ResponseCache:
    """Thread-safe LRU cache mapping request fingerprints to generated text.

    Parameters
    ----------
    max_entries:
        Soft upper bound; LRU eviction kicks in when exceeded.
    persist_path:
        Optional file path. When set, the cache is loaded on construction
        and rewritten on every ``put`` (best-effort — IO failures are
        logged but never raised).
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._lock = Lock()
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self.max_entries = max(1, max_entries)
        self.persist_path = persist_path
        self.hits = 0
        self.misses = 0
        if persist_path is not None:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Lookup / mutation
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            self._persist_locked()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self._persist_locked()

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(entries=len(self._cache), hits=self.hits, misses=self.misses)

    # ------------------------------------------------------------------
    # Persistence (best-effort)
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        assert self.persist_path is not None
        if not self.persist_path.exists():
            return
        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cache load failed (%s); starting empty", exc)
            return
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(key, str) and isinstance(value, str):
                    self._cache[key] = value

    def _persist_locked(self) -> None:
        """Caller must hold ``self._lock``."""
        if self.persist_path is None:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(
                json.dumps(dict(self._cache), ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("cache persist failed: %s", exc)


# ----------------------------------------------------------------------
# Fingerprinting
# ----------------------------------------------------------------------


def fingerprint(
    *,
    model: str,
    max_tokens: int,
    prompt: Optional[str] = None,
    messages: Optional[list[Any]] = None,
    system: Optional[str] = None,
) -> str:
    """Stable SHA-256 fingerprint over the request inputs that matter."""
    canonical = {
        "model": model,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "messages": messages,
        "system": system,
    }
    serialized = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Environment-driven factory
# ----------------------------------------------------------------------


def cache_from_env() -> Optional[ResponseCache]:
    """Return a configured cache when env vars enable one, else ``None``."""
    path_env = os.environ.get("AXIOM_CACHE_PATH")
    enabled = os.environ.get("AXIOM_CACHE", "").lower() in ("1", "true", "yes", "on")
    if not (enabled or path_env):
        return None

    try:
        max_entries = int(os.environ.get("AXIOM_CACHE_MAX_ENTRIES", DEFAULT_MAX_ENTRIES))
    except ValueError:
        max_entries = DEFAULT_MAX_ENTRIES

    persist_path = Path(path_env) if path_env else None
    return ResponseCache(max_entries=max_entries, persist_path=persist_path)
