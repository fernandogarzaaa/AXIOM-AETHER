from __future__ import annotations

import argparse
import asyncio
import hashlib
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from .config import AxiomConfig
from .jit_streamer import JITContextStreamer, JITStreamer, NeuralSearchClient
from .kernel import AxiomTTTEngine


@dataclass
class SimpleTokenizer:
    vocab_size: int

    def encode(self, text: str) -> List[int]:
        toks = text.strip().split()
        if not toks:
            return [0]
        ids: List[int] = []
        for tok in toks:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            ids.append(int.from_bytes(digest[:8], byteorder="little", signed=False) % self.vocab_size)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        return " ".join(f"tok_{t}" for t in token_ids)


def _allocate_w_tilde_states(cfg: AxiomConfig, device: torch.device) -> List[Tensor]:
    """Allocate zeroed W_tilde tensors for all layers."""
    head_dim = cfg.d_model // cfg.num_heads
    return [
        torch.zeros(1, cfg.num_heads, head_dim, head_dim, device=device, dtype=torch.float32)
        for _ in range(cfg.n_layers)
    ]


class InferencePipeline:
    """End-to-end dual-cycle inference pipeline for the Axiom-TTT engine.

    Manages the full prefill → decode cycle with explicit W_tilde lifecycle:

    1. JIT context retrieval (async, sub-second).
    2. Parallel prefill over the retrieved context chunk.
    3. Zero-initialised W_tilde allocation for each of the N decoder layers.
    4. Autoregressive decoding loop with per-token W_tilde updates.
    5. State flush and GPU cache clear on completion.
    """

    def __init__(self, cfg: AxiomConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AxiomTTTEngine(cfg).to(self.device)
        self.model.eval()

        self.streamer = JITContextStreamer(
            vocab_size=cfg.vocab_size,
            max_context_tokens=cfg.max_context_tokens,
        )
        self.tokenizer = SimpleTokenizer(cfg.vocab_size)

    @torch.no_grad()
    async def run_generation(self, user_prompt: str, max_new_tokens: int = 32) -> str:
        """Full generation cycle: JIT fetch → prefill → decode → state flush.

        Args:
            user_prompt:    Input prompt string.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Decoded output string.
        """
        # Step 1: Acquire real-time external context token IDs.
        context_ids = await self.streamer.fetch_and_pack_context(user_prompt)
        context_tensor = torch.tensor([context_ids], device=self.device, dtype=torch.long)

        # Step 2: Prefill on the retrieved context (single parallel matrix step).
        self.model(input_ids=context_tensor, use_decode=False, return_states=False)

        # Also prefill on the user prompt tokens.
        prompt_ids = self.tokenizer.encode(user_prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        self.model(input_ids=prompt_tensor, use_decode=False, return_states=False)

        # Step 3: Allocate zeroed W_tilde for all layers: [1, Num_Heads, Head_Dim, Head_Dim].
        states: List[Tensor] = _allocate_w_tilde_states(self.cfg, self.device)

        # Prime: run decode on the last prompt token to get the first generated token.
        last_token_tensor = torch.tensor([[prompt_ids[-1]]], device=self.device, dtype=torch.long)
        logits, states = self.model(
            input_ids=last_token_tensor,
            states=states,
            use_decode=True,
            return_states=True,
        )
        next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated: List[int] = [next_token]

        # Step 4: Autoregressive decoding loop with W_tilde state accumulation.
        for _ in range(max_new_tokens - 1):
            step_tensor = torch.tensor([[generated[-1]]], device=self.device, dtype=torch.long)
            logits, states = self.model(
                input_ids=step_tensor,
                states=states,
                use_decode=True,
                return_states=True,
            )
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated.append(next_token)

        # Step 5: State flush — delete runtime weight matrices and free GPU memory.
        del states
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return self.tokenizer.decode(generated)

    # ------------------------------------------------------------------
    # Synchronous variants (used by the FastAPI server thread-pool path)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_core_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        states: Optional[List[Tensor]] = None,
    ) -> tuple[str, List[Tensor]]:
        """Synchronous prefill + autoregressive decode.

        Skips the async JIT context fetch — server callers run inside a
        thread-pool executor that cannot drive ``asyncio.run`` safely.
        """
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [0]

        prompt_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        self.model(input_ids=prompt_tensor, use_decode=False, return_states=False)

        if states is None:
            states = _allocate_w_tilde_states(self.cfg, self.device)

        last_token = torch.tensor([[prompt_ids[-1]]], device=self.device, dtype=torch.long)
        logits, states = self.model(
            input_ids=last_token,
            states=states,
            use_decode=True,
            return_states=True,
        )
        generated: List[int] = [int(torch.argmax(logits[:, -1, :], dim=-1).item())]

        for _ in range(max(0, max_new_tokens - 1)):
            step = torch.tensor([[generated[-1]]], device=self.device, dtype=torch.long)
            logits, states = self.model(
                input_ids=step,
                states=states,
                use_decode=True,
                return_states=True,
            )
            generated.append(int(torch.argmax(logits[:, -1, :], dim=-1).item()))

        return self.tokenizer.decode(generated), states

    def generate_sync(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Stateless synchronous generation."""
        text, states = self._generate_core_sync(prompt, max_new_tokens, states=None)
        del states
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return text

    def generate_with_session_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        states: List[Tensor],
    ) -> tuple[str, List[Tensor]]:
        """Stateful synchronous generation — returns (text, updated W_tilde states)."""
        return self._generate_core_sync(prompt, max_new_tokens, states=states)


class AxiomInferenceRunner:
    """CLI-facing inference runner; delegates to InferencePipeline internally."""

    def __init__(self, cfg: AxiomConfig, device: Optional[torch.device] = None) -> None:
        self._pipeline = InferencePipeline(cfg, device=device)

    @torch.no_grad()
    async def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        return await self._pipeline.run_generation(prompt, max_new_tokens=max_new_tokens)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Axiom-TTT streaming inference")
    parser.add_argument("prompt", type=str, help="User input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--lr-inner", type=float, default=1e-3)
    return parser


def _config_from_args(args: argparse.Namespace) -> AxiomConfig:
    return AxiomConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        lr_inner=args.lr_inner,
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = _config_from_args(args)

    runner = AxiomInferenceRunner(cfg)
    out = asyncio.run(runner.generate(args.prompt, max_new_tokens=args.max_new_tokens))
    print(out)


if __name__ == "__main__":
    main()

