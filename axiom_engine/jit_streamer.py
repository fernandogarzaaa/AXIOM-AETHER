from __future__ import annotations

import asyncio
import hashlib
import html
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import Tensor


_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_NOISE_RE = re.compile(r"^\s{0,3}(?:#|\*|>|-{3,}|={3,}|\[.*\]\(.*\))\s*$")
_AVG_DOC_LENGTH = 64.0
_TOKEN_SEED_MODULUS = 2**31
_BM25_WEIGHT = 0.7
_COSINE_WEIGHT = 0.3


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _clean_text(text: str) -> str:
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _MARKDOWN_NOISE_RE.match(stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _cosine_from_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _bm25_like_score(query_terms: Sequence[str], doc_terms: Sequence[str]) -> float:
    if not doc_terms:
        return 0.0
    doc_freq = Counter(doc_terms)
    q = Counter(query_terms)
    score = 0.0
    k1 = 1.5
    b = 0.75
    dl = len(doc_terms)
    avgdl = _AVG_DOC_LENGTH
    for term, qtf in q.items():
        tf = doc_freq.get(term, 0)
        if tf == 0:
            continue
        term_weight = 1.0 / (1.0 + tf)
        denom = tf + k1 * (1 - b + b * (dl / avgdl))
        score += qtf * (term_weight * (tf * (k1 + 1)) / max(denom, 1e-6))
    return score


def _deterministic_token_vector(token: str, d_model: int) -> Tensor:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % _TOKEN_SEED_MODULUS
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    vec = torch.randn(d_model, generator=g, dtype=torch.float32, device=torch.device("cpu"))
    vec = vec / (vec.norm(p=2) + 1e-8)
    return vec


def _encode_tokens(tokens: Iterable[str], d_model: int, device: torch.device) -> Tensor:
    vectors = [_deterministic_token_vector(tok, d_model) for tok in tokens]
    if not vectors:
        return torch.zeros(1, d_model, device=device)
    return torch.stack(vectors, dim=0).to(device)


def process_and_pack_context(
    query: str,
    raw_documents: Sequence[str],
    d_model: int,
    max_context_tokens: int,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> Tensor:
    device = device or torch.device("cpu")

    query_terms = _tokenize(query)
    query_counts = Counter(query_terms)

    scored_lines = []
    seen = set()
    for doc in raw_documents:
        cleaned = _clean_text(doc)
        for line in cleaned.splitlines():
            line_norm = line.strip().lower()
            if not line_norm or line_norm in seen:
                continue
            seen.add(line_norm)
            line_terms = _tokenize(line_norm)
            bm25_score = _bm25_like_score(query_terms, line_terms)
            cos_score = _cosine_from_counts(query_counts, Counter(line_terms))
            score = _BM25_WEIGHT * bm25_score + _COSINE_WEIGHT * cos_score
            if score > 0:
                scored_lines.append((score, line_norm))

    scored_lines.sort(key=lambda x: x[0], reverse=True)

    selected_tokens: List[str] = []
    for _, line in scored_lines:
        line_tokens = _tokenize(line)
        if not line_tokens:
            continue
        for tok in line_tokens:
            selected_tokens.append(tok)
            if len(selected_tokens) >= max_context_tokens:
                break
        if len(selected_tokens) >= max_context_tokens:
            break

    context = _encode_tokens(selected_tokens, d_model=d_model, device=device)
    context = context.unsqueeze(0).repeat(batch_size, 1, 1)
    return context


@dataclass
class NeuralSearchClient:
    """Async neural retrieval client boundary (integrate concrete provider externally)."""

    latency_ms: int = 30

    async def fetch_markdown(self, query: str) -> List[str]:
        await asyncio.sleep(self.latency_ms / 1000.0)
        return [
            f"# Result\nTechnical context for: {query}\n- API behavior\n- update notes\n- examples",
            f"## Documentation\n{query} implementation details and known failure modes.",
            f"### Changelog\nRecent changes affecting {query} and migration guidance.",
        ]


class JITStreamer:
    def __init__(self, search_client: NeuralSearchClient) -> None:
        self.search_client = search_client

    @staticmethod
    def decompose_query(query: str, max_subqueries: int = 4) -> List[str]:
        terms = [t for t in _tokenize(query) if len(t) > 3]
        if not terms:
            return [query]
        subqueries = [query]
        for term in terms[: max_subqueries - 1]:
            subqueries.append(f"{query} {term}")
        return subqueries

    async def collect_context(self, query: str) -> List[str]:
        subqueries = self.decompose_query(query)
        tasks = [self.search_client.fetch_markdown(sq) for sq in subqueries]
        results = await asyncio.gather(*tasks)
        docs: List[str] = []
        for docs_batch in results:
            docs.extend(docs_batch)
        return docs

    async def context_tensor(
        self,
        query: str,
        d_model: int,
        max_context_tokens: int,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> Tensor:
        docs = await self.collect_context(query)
        return process_and_pack_context(
            query=query,
            raw_documents=docs,
            d_model=d_model,
            max_context_tokens=max_context_tokens,
            batch_size=batch_size,
            device=device,
        )
