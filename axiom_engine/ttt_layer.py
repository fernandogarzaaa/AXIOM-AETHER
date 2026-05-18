from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TTTLinearLayer(nn.Module):
    """Multi-head linear-time test-time training layer.

    Supports two operational forms:

    - **Parallel prefill** via Gram-matrix dual-form (O(T·D²) total, no sequential
      dependency): key-key interactions are captured in a causal Gram matrix and used
      to correct the standard linear-attention output for cumulative W_tilde drift.

    - **Step-wise decode** with explicit per-token W_tilde updates: for each new
      token the reconstruction loss gradient is computed and used to take one
      in-place gradient descent step on the dynamic weight matrix.
    """

    def __init__(self, d_model: int, num_heads: int, lr_inner: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.lr_inner = lr_inner

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: Tensor) -> Tensor:
        """[B, T, d_model] → [B, H, T, D]."""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """[B, H, T, D] → [B, T, d_model]."""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    # ------------------------------------------------------------------
    # Form 1 – Parallel dual-form prefill
    # ------------------------------------------------------------------

    def _forward_prefill(self, x: Tensor) -> Tensor:
        """Vectorized dual-form context ingestion over a full sequence.

        Mathematically unrolls sequential gradient updates into a single matrix
        operation via the causal Gram matrix:

            gram[i,j] = η · kᵢ · kⱼ   (causal: j ≤ i)

        The Gram matrix captures how much W_tilde has drifted by time t as a
        result of all prior key-based gradient steps.  The product
        ``gram @ attn_logits`` encodes the historical correction that adjusts
        the standard linear-attention scores to account for this W_tilde evolution.

        Args:
            x: [Batch, Seq_Len, d_model]

        Returns:
            y: [Batch, Seq_Len, d_model]
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        assert x.shape == (B, T, self.d_model), (
            f"Prefill input shape mismatch: expected ({B}, {T}, {self.d_model}), "
            f"got {tuple(x.shape)}"
        )

        # Linear projections → multi-head format
        q: Tensor = self._split_heads(self.W_Q(x))  # [B, H, T, D]
        k: Tensor = self._split_heads(self.W_K(x))  # [B, H, T, D]
        v: Tensor = self._split_heads(self.W_V(x))  # [B, H, T, D]

        assert q.shape == k.shape == v.shape == (B, H, T, D), (
            f"Head projection shape mismatch: expected ({B}, {H}, {T}, {D})"
        )

        # Gram matrix of key interactions scaled by the inner learning rate.
        # gram[b,h,i,j] = η · kᵢ · kⱼ   →  [B, H, T, T]
        gram: Tensor = torch.matmul(k, k.transpose(-2, -1)) * self.lr_inner

        # Lower-triangular causal mask: key at position j must not influence
        # the W_tilde correction used at position i < j.
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        gram = gram.masked_fill(~causal_mask, 0.0)

        # Standard causal linear-attention logits Q Kᵀ.
        attn_logits: Tensor = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]
        attn_logits = attn_logits.masked_fill(~causal_mask, 0.0)

        # Historical update correction:
        # At each time step t the effective attention weight must account for
        # the cumulative in-place gradient updates that have modified W_tilde up
        # to that point.  The outer product gram @ attn_logits encodes this drift,
        # so subtracting it yields attention weights consistent with the evolving
        # hidden weight trajectory.
        correction: Tensor = torch.matmul(gram, attn_logits)   # [B, H, T, T]
        effective_weights: Tensor = attn_logits - correction    # [B, H, T, T]

        # Aggregate value vectors weighted by the corrected attention.
        y: Tensor = torch.matmul(effective_weights, v)          # [B, H, T, D]

        assert y.shape == (B, H, T, D), (
            f"Prefill output shape mismatch: expected ({B}, {H}, {T}, {D}), "
            f"got {tuple(y.shape)}"
        )

        # Merge heads and project to d_model.
        return self.out_proj(self._merge_heads(y))  # [B, T, d_model]

    # ------------------------------------------------------------------
    # Form 2 – Step-wise decoding with in-place W_tilde update
    # ------------------------------------------------------------------

    def _forward_decode(
        self,
        x: Tensor,
        W_tilde: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Single-token online gradient step on the dynamic weight matrix.

        For each new token:
          1. Compute the reconstruction loss on the current W_tilde:
                L = 0.5 · ‖W_tilde @ k_norm − v‖²
          2. Derive the exact gradient and apply one gradient descent step:
                W_tilde_next = W_tilde − η · ∂L/∂W_tilde
          3. Query the *updated* weight matrix:
                y_t = W_tilde_next @ q

        Args:
            x:       [Batch, 1, d_model]
            W_tilde: [Batch, Num_Heads, Head_Dim, Head_Dim]

        Returns:
            y:            [Batch, 1, d_model]
            W_tilde_next: [Batch, Num_Heads, Head_Dim, Head_Dim]
        """
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim

        assert S == 1, f"Decode expects seq_len=1, got {S}"
        assert x.shape == (B, 1, self.d_model), (
            f"Decode input shape mismatch: expected ({B}, 1, {self.d_model}), "
            f"got {tuple(x.shape)}"
        )
        assert W_tilde.shape == (B, H, D, D), (
            f"W_tilde shape mismatch: expected ({B}, {H}, {D}, {D}), "
            f"got {tuple(W_tilde.shape)}"
        )

        # Project single token and reshape to multi-head column vectors.
        q: Tensor = self._split_heads(self.W_Q(x))  # [B, H, 1, D]
        k: Tensor = self._split_heads(self.W_K(x))  # [B, H, 1, D]
        v: Tensor = self._split_heads(self.W_V(x))  # [B, H, 1, D]

        # Reshape to column vectors [B, H, D, 1] for batched matmul.
        k_vec: Tensor = k.squeeze(2).unsqueeze(-1)  # [B, H, D, 1]
        v_vec: Tensor = v.squeeze(2).unsqueeze(-1)  # [B, H, D, 1]
        q_vec: Tensor = q.squeeze(2).unsqueeze(-1)  # [B, H, D, 1]

        # L2-normalise the key vector for a numerically stable reconstruction target.
        k_norm: Tensor = F.normalize(k_vec, p=2, dim=-2)  # [B, H, D, 1]

        # Reconstruction: pred = W_tilde @ k_norm   →  [B, H, D, 1]
        pred: Tensor = torch.matmul(W_tilde, k_norm)

        # Error and gradient: ∂L/∂W_tilde = error · k_normᵀ   →  [B, H, D, D]
        error: Tensor = pred - v_vec
        grad: Tensor = torch.matmul(error, k_norm.transpose(-2, -1))

        # In-place gradient descent step on the dynamic weight matrix.
        W_tilde_next: Tensor = W_tilde - self.lr_inner * grad  # [B, H, D, D]

        # Query the updated matrix.
        y_t: Tensor = torch.matmul(W_tilde_next, q_vec)       # [B, H, D, 1]
        y_t = y_t.squeeze(-1).unsqueeze(2)                     # [B, H, 1, D]

        assert y_t.shape == (B, H, 1, D), (
            f"Decode output shape mismatch: expected ({B}, {H}, 1, {D}), "
            f"got {tuple(y_t.shape)}"
        )

        # Merge heads and project to d_model.
        y_out: Tensor = self.out_proj(self._merge_heads(y_t))  # [B, 1, d_model]
        return y_out, W_tilde_next

    # ------------------------------------------------------------------
    # Unified forward dispatcher
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        W_tilde: Optional[Tensor] = None,
        use_decode: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Route to the appropriate operational form.

        Args:
            x:          [B, T, d_model] (prefill) or [B, 1, d_model] (decode).
            W_tilde:    Dynamic weight matrix [B, H, D, D]. Required in decode mode;
                        if None, a zero matrix is allocated automatically.
            use_decode: When True, run step-wise decode; otherwise parallel prefill.

        Returns:
            Prefill: output tensor [B, T, d_model].
            Decode:  tuple (output [B, 1, d_model], updated W_tilde [B, H, D, D]).
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3-D input [batch, seq_len, d_model], "
                f"got {x.ndim}-D tensor with shape {tuple(x.shape)}"
            )

        if use_decode:
            if W_tilde is None:
                W_tilde = torch.zeros(
                    x.shape[0], self.num_heads, self.head_dim, self.head_dim,
                    device=x.device, dtype=x.dtype,
                )
            return self._forward_decode(x, W_tilde)

        return self._forward_prefill(x)

