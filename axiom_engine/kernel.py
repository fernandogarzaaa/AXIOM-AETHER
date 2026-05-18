from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import AxiomConfig
from .ttt_layer import TTTLinearLayer


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, multiplier: int = 4) -> None:
        super().__init__()
        hidden = d_model * multiplier
        self.up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = self.up(x)
        x_gate, x_val = x_proj.chunk(2, dim=-1)
        return self.down(F.silu(x_gate) * x_val)


class AxiomTTTBlock(nn.Module):
    def __init__(self, cfg: AxiomConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.ttt = TTTLinearLayer(cfg.d_model, lr_inner=cfg.lr_inner)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.ffn = SwiGLUFFN(cfg.d_model, multiplier=cfg.ffn_multiplier)

    def reset_state(self) -> None:
        self.ttt.reset_state()

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        return_state: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        ttt_out = self.ttt(self.norm1(x), state=state, return_state=True)
        x = x + ttt_out[0]
        x = x + self.ffn(self.norm2(x))

        if return_state:
            return x, ttt_out[1]
        return x


class AxiomTTTEngine(nn.Module):
    def __init__(self, cfg: AxiomConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([AxiomTTTBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def reset_dynamic_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()

    def forward_hidden(
        self,
        x: Tensor,
        states: Optional[List[Tensor]] = None,
        return_states: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        next_states: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            st = None if states is None else states[i]
            x, st_new = layer(x, state=st, return_state=True)
            next_states.append(st_new)
        x = self.final_norm(x)

        if return_states:
            return x, next_states
        return x

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        states: Optional[List[Tensor]] = None,
        return_states: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")

        x = inputs_embeds if inputs_embeds is not None else self.token_embedding(input_ids)
        hidden, next_states = self.forward_hidden(x, states=states, return_states=True)
        logits = self.lm_head(hidden)

        if return_states:
            return logits, next_states
        return logits
