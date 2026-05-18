from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TTTLinearLayer(nn.Module):
    """Linear-time test-time training layer with per-token dynamic updates."""

    def __init__(self, d_model: int, lr_inner: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.lr_inner = lr_inner

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_G = nn.Linear(d_model, d_model, bias=False)

        self._dynamic_state: Optional[Tensor] = None

    def reset_state(self) -> None:
        self._dynamic_state = None

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        return_state: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            state: Optional starting dynamic matrix [batch, d_model, d_model]
            return_state: If True, return final dynamic matrix.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, d_model], got {tuple(x.shape)}")

        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        if state is None:
            W_tilde = torch.zeros(batch, d_model, d_model, device=x.device, dtype=x.dtype)
        else:
            if state.shape != (batch, d_model, d_model):
                raise ValueError(
                    f"state shape must be {(batch, d_model, d_model)}, got {tuple(state.shape)}"
                )
            W_tilde = state.clone()

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            k_t = self.W_K(x_t)
            v_t = self.W_V(x_t)
            g_t = self.W_G(x_t)

            k_col = k_t.unsqueeze(-1)
            pred_pre = torch.bmm(W_tilde, k_col).squeeze(-1)
            pred = F.silu(pred_pre) * g_t

            loss = 0.5 * (pred - v_t).pow(2).mean()
            grad = torch.autograd.grad(loss, W_tilde, create_graph=False, retain_graph=False)[0]

            with torch.no_grad():
                W_tilde.add_(grad, alpha=-self.lr_inner)

            q_t = self.W_Q(x_t)
            y_t = torch.bmm(W_tilde, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        self._dynamic_state = W_tilde.detach()

        if return_state:
            return y, W_tilde.detach()
        return y
