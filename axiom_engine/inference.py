from __future__ import annotations

import argparse
import asyncio
import hashlib
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from .config import AxiomConfig
from .jit_streamer import JITStreamer, NeuralSearchClient
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


class AxiomInferenceRunner:
    def __init__(self, cfg: AxiomConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AxiomTTTEngine(cfg).to(self.device)
        self.streamer = JITStreamer(NeuralSearchClient())
        self.tokenizer = SimpleTokenizer(cfg.vocab_size)

    @torch.no_grad()
    async def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        batch_size = 1

        context = await self.streamer.context_tensor(
            query=prompt,
            d_model=self.cfg.d_model,
            max_context_tokens=self.cfg.max_context_tokens,
            batch_size=batch_size,
            device=self.device,
        )

        _, states = self.model(inputs_embeds=context, return_states=True)

        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)

        logits, states = self.model(input_ids=input_ids, states=states, return_states=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            step_ids = torch.tensor([[generated[-1]]], device=self.device, dtype=torch.long)
            logits, states = self.model(input_ids=step_ids, states=states, return_states=True)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token)

        self.model.reset_dynamic_state()
        return self.tokenizer.decode(generated)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Axiom-TTT streaming inference")
    parser.add_argument("prompt", type=str, help="User input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--lr-inner", type=float, default=1e-3)
    return parser


def _config_from_args(args: argparse.Namespace) -> AxiomConfig:
    return AxiomConfig(d_model=args.d_model, n_layers=args.n_layers, lr_inner=args.lr_inner)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = _config_from_args(args)

    runner = AxiomInferenceRunner(cfg)
    out = asyncio.run(runner.generate(args.prompt, max_new_tokens=args.max_new_tokens))
    print(out)


if __name__ == "__main__":
    main()
