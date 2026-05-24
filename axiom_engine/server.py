"""Axiom-TTT OpenAI-compatible ASGI server.

Drop-in replacement for OpenAI Chat Completions API backed by the Axiom-TTT
test-time-training engine.  Compatible with LangChain, LlamaIndex, Continue.dev,
Open WebUI, and any other client that targets the OpenAI API surface.

Usage::

    pip install axiom-engine
    uvicorn axiom_engine.server:app --host 0.0.0.0 --port 8080

Or via the CLI entry-point::

    axiom-server --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import AxiomConfig
from .inference import InferencePipeline, _allocate_w_tilde_states

logger = logging.getLogger("axiom.server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_pipeline: Optional[InferencePipeline] = None
_sessions: Dict[str, Dict[str, Any]] = {}
_model_id = "axiom-ttt-v1"


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _pipeline
    cfg = AxiomConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[+] Loading Axiom-TTT pipeline on %s", device)
    _pipeline = InferencePipeline(cfg, device=device)
    _pipeline.model.eval()
    logger.info("[+] Axiom-TTT server ready — model_id=%s", _model_id)
    yield
    logger.info("[*] Shutting down Axiom-TTT server")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Axiom-TTT Inference Server",
    description=(
        "OpenAI-compatible inference API backed by the Axiom-TTT test-time-training "
        "engine.  Uniquely supports persistent TTT sessions that learn in-context "
        "across conversation turns."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# OpenAI-compatible schema models
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "axiom-ttt"


class ListModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 32
    session_id: Optional[str] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = 32
    session_id: Optional[str] = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class CreateSessionRequest(BaseModel):
    model: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    object: str = "session"
    created: int
    model: str


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool


class AdaptRequest(BaseModel):
    corpus: List[str]
    steps: Optional[int] = 4
    session_id: Optional[str] = None


class AdaptResponse(BaseModel):
    session_id: str
    object: str = "adapt"
    steps_per_token: int
    corpus_documents: int


class LayerCheckpoint(BaseModel):
    shape: List[int]
    data: List[float]


class SessionCheckpoint(BaseModel):
    session_id: str
    version: int = 1
    created_at: int
    layers: List[LayerCheckpoint]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_pipeline() -> InferencePipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline not initialized")
    return _pipeline


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, list]:
    """Resolve an existing session or create a fresh one.

    Returns ``(session_id, w_tilde_states)``.
    """
    pipeline = _require_pipeline()

    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]["states"]

    # Auto-create.
    new_id = session_id or str(uuid.uuid4())
    states = _allocate_w_tilde_states(pipeline.cfg, pipeline.device)
    _sessions[new_id] = {
        "states": states,
        "created_at": int(time.time()),
        "last_used": int(time.time()),
        "model": _model_id,
    }
    return new_id, states


async def _run_generation(prompt: str, max_tokens: int, session_id: Optional[str]) -> str:
    """Run generation in a thread pool to avoid blocking the event loop."""
    pipeline = _require_pipeline()

    def _generate() -> str:
        if session_id:
            if session_id not in _sessions:
                raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
            states = _sessions[session_id]["states"]
            text, updated = asyncio.get_event_loop().run_in_executor(  # type: ignore[union-attr]
                None,
                lambda: pipeline.run_generation_sync(prompt, max_tokens, states),
            )
            _sessions[session_id]["states"] = updated
            _sessions[session_id]["last_used"] = int(time.time())
            return text
        # Stateless generation.
        return asyncio.get_event_loop().run_in_executor(  # type: ignore[union-attr]
            None,
            lambda: pipeline.run_generation_sync(prompt, max_tokens, None),
        )

    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _generate_sync(pipeline, prompt, max_tokens, session_id),
    )


def _generate_sync(
    pipeline: InferencePipeline,
    prompt: str,
    max_tokens: int,
    session_id: Optional[str],
) -> str:
    from .inference import _allocate_w_tilde_states

    if session_id:
        if session_id not in _sessions:
            raise ValueError(f"session '{session_id}' not found")
        states = _sessions[session_id]["states"]
    else:
        states = _allocate_w_tilde_states(pipeline.cfg, pipeline.device)

    # Synchronous generation loop using the existing pipeline methods.
    loop = asyncio.new_event_loop()
    try:
        text = loop.run_until_complete(
            pipeline.run_generation(prompt, max_new_tokens=max_tokens)
        )
    finally:
        loop.close()

    return text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/v1/models", response_model=ListModelsResponse)
async def list_models() -> ListModelsResponse:
    """List available models."""
    return ListModelsResponse(
        data=[ModelInfo(id=_model_id)]
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(req: CompletionRequest) -> CompletionResponse:
    """OpenAI text completions endpoint."""
    text = await _run_generation(
        req.prompt,
        req.max_tokens or 32,
        req.session_id,
    )
    model = req.model or _model_id
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=model,
        choices=[CompletionChoice(text=text)],
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI chat completions endpoint."""
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    text = await _run_generation(
        prompt,
        req.max_tokens or 32,
        req.session_id,
    )
    model = req.model or _model_id
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=model,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
    )


@app.post("/v1/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new persistent TTT session."""
    session_id, _ = _get_or_create_session(None)
    return CreateSessionResponse(
        session_id=session_id,
        created=int(time.time()),
        model=req.model or _model_id,
    )


@app.delete("/v1/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str) -> DeleteSessionResponse:
    """Delete a session and free its W_tilde tensors."""
    deleted = _sessions.pop(session_id, None) is not None
    return DeleteSessionResponse(session_id=session_id, deleted=deleted)


@app.post("/v1/adapt", response_model=AdaptResponse)
async def adapt(req: AdaptRequest) -> AdaptResponse:
    """Run in-place TTT adaptation over a text corpus."""
    if not req.corpus:
        raise HTTPException(status_code=400, detail="corpus must contain at least one document")

    pipeline = _require_pipeline()
    session_id, states = _get_or_create_session(req.session_id)

    def _adapt_sync():
        from .inference import _allocate_w_tilde_states

        current_states = states
        for text in req.corpus:
            token_ids = pipeline.tokenizer.encode(text)
            for tok_id in token_ids:
                tok_tensor = torch.tensor([[tok_id]], device=pipeline.device, dtype=torch.long)
                with torch.no_grad():
                    _, current_states = pipeline.model(
                        input_ids=tok_tensor,
                        states=current_states,
                        use_decode=True,
                        return_states=True,
                    )
        return current_states

    updated_states = await asyncio.get_event_loop().run_in_executor(None, _adapt_sync)
    _sessions[session_id]["states"] = updated_states
    _sessions[session_id]["last_used"] = int(time.time())

    return AdaptResponse(
        session_id=session_id,
        steps_per_token=min(req.steps or 4, 4),
        corpus_documents=len(req.corpus),
    )


@app.get("/v1/sessions/{session_id}/checkpoint", response_model=SessionCheckpoint)
async def get_checkpoint(session_id: str) -> SessionCheckpoint:
    """Export session W_tilde state as a JSON checkpoint."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")

    session = _sessions[session_id]
    layers: List[LayerCheckpoint] = []
    for tensor in session["states"]:
        shape = list(tensor.shape)
        data = tensor.detach().cpu().float().flatten().tolist()
        layers.append(LayerCheckpoint(shape=shape, data=data))

    return SessionCheckpoint(
        session_id=session_id,
        created_at=session["created_at"],
        layers=layers,
    )


@app.put("/v1/sessions/{session_id}/checkpoint", response_model=CreateSessionResponse)
async def put_checkpoint(session_id: str, checkpoint: SessionCheckpoint) -> CreateSessionResponse:
    """Restore a session from a JSON checkpoint."""
    if checkpoint.version != 1:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported checkpoint version {checkpoint.version}",
        )

    pipeline = _require_pipeline()
    states = []
    for lc in checkpoint.layers:
        total = 1
        for s in lc.shape:
            total *= s
        tensor = torch.tensor(lc.data, dtype=torch.float32, device=pipeline.device).reshape(lc.shape)
        states.append(tensor)

    now = int(time.time())
    if session_id in _sessions:
        _sessions[session_id]["states"] = states
        _sessions[session_id]["last_used"] = now
    else:
        _sessions[session_id] = {
            "states": states,
            "created_at": now,
            "last_used": now,
            "model": _model_id,
        }

    return CreateSessionResponse(
        session_id=session_id,
        created=now,
        model=_model_id,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def serve() -> None:
    """Start the Axiom-TTT ASGI server via uvicorn."""
    import argparse
    import uvicorn  # type: ignore[import]

    parser = argparse.ArgumentParser(description="Axiom-TTT OpenAI-compatible inference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "axiom_engine.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    serve()
