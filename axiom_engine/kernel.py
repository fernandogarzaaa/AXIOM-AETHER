from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import AxiomConfig
from .ttt_layer import TTTLinearLayer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU-activated Feed-Forward Network.

    Intermediate dimension: ``int(2 * (d_model * 4 / 3) / 2)``
    Gate:  ``F.silu(w1(x)) * w3(x)``
    Down:  ``w2(gate_output)``
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden: int = int(2 * (d_model * 4 / 3) / 2)
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden, d_model, bias=False)  # down projection
        self.w3 = nn.Linear(d_model, hidden, bias=False)  # up projection

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AxiomBlock(nn.Module):
    """Single Pre-LN residual block: RMSNorm → TTTLinearLayer → SwiGLUFFN.

    Routes to parallel prefill or step-wise decode based on the ``use_decode``
    flag, enabling explicit branching for hardware-efficient scheduling.
    """

    def __init__(self, cfg: AxiomConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ttt = TTTLinearLayer(cfg.d_model, num_heads=cfg.num_heads, lr_inner=cfg.lr_inner)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = SwiGLUFFN(cfg.d_model)

    def forward(
        self,
        x: Tensor,
        W_tilde: Optional[Tensor] = None,
        use_decode: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x:          [B, T, d_model] (prefill) or [B, 1, d_model] (decode).
            W_tilde:    Dynamic weight state [B, H, D, D]; used only in decode mode.
            use_decode: Selects step-wise decode (True) or parallel prefill (False).

        Returns:
            Prefill: output [B, T, d_model].
            Decode:  (output [B, 1, d_model], updated W_tilde [B, H, D, D]).
        """
        normed: Tensor = self.norm1(x)

        if use_decode:
            # Step-wise decode: TTT layer returns (output, updated W_tilde).
            ttt_out, W_tilde_next = self.ttt(normed, W_tilde=W_tilde, use_decode=True)
            x = x + ttt_out
            x = x + self.ffn(self.norm2(x))
            return x, W_tilde_next
        else:
            # Parallel prefill: TTT layer returns only the output tensor.
            ttt_out = self.ttt(normed, use_decode=False)
            x = x + ttt_out
            x = x + self.ffn(self.norm2(x))
            return x


class AxiomTTTEngine(nn.Module):
    """Full model stack: Embedding → N × AxiomBlock → RMSNorm → LM Head."""

    def __init__(self, cfg: AxiomConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([AxiomBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def reset_dynamic_state(self) -> None:
        """No-op: dynamic state (W_tilde) is managed externally by the inference runner."""
        pass

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        states: Optional[List[Tensor]] = None,
        use_decode: bool = False,
        return_states: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Args:
            input_ids:     [B, T] token indices (mutually exclusive with inputs_embeds).
            inputs_embeds: [B, T, d_model] pre-computed embeddings.
            states:        Per-layer W_tilde tensors [B, H, D, D] for decode mode.
            use_decode:    Route all blocks to step-wise decode when True.
            return_states: Also return the updated per-layer state list when True.

        Returns:
            logits [B, T, vocab_size] and, if return_states=True, a list of updated
            W_tilde tensors (one per layer; empty list when use_decode=False).
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Exactly one of input_ids or inputs_embeds must be provided, "
                "not both or neither."
            )

        x: Tensor = (
            inputs_embeds if inputs_embeds is not None else self.token_embedding(input_ids)
        )

        next_states: List[Tensor] = []

        for i, block in enumerate(self.layers):
            W_tilde_i: Optional[Tensor] = None if states is None else states[i]
            if use_decode:
                x, W_tilde_next = block(x, W_tilde=W_tilde_i, use_decode=True)
                next_states.append(W_tilde_next)
            else:
                x = block(x, use_decode=False)

        x = self.final_norm(x)
        logits: Tensor = self.lm_head(x)

        if return_states:
            return logits, next_states
        return logits

