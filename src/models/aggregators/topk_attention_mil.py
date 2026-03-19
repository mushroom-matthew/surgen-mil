from __future__ import annotations

import torch
import torch.nn as nn


class TopKAttentionMIL(nn.Module):
    """
    Sparse attention MIL: compute attention logits for all patches, keep top-k
    by logit, renormalise over those k, then aggregate only those k.

    This directly tests the sparse-signal hypothesis: if the positive class is
    determined by a small subset of patches, concentrating attention mass on
    the highest-scoring k patches should improve precision without hurting recall.

    k=-1 falls back to standard (full) attention MIL.
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        k: int = 16,
    ):
        super().__init__()
        self.k = k

        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, coords=None) -> dict[str, torch.Tensor]:
        """
        x: [N, D]
        """
        attn_logits = self.attention(x)  # [N, 1]

        n = x.shape[0]
        k = self.k if self.k > 0 else n
        k = min(k, n)

        if k < n:
            topk_idx = torch.topk(attn_logits.squeeze(-1), k, dim=0).indices  # [k]
            x_k = x[topk_idx]                        # [k, D]
            logits_k = attn_logits[topk_idx]          # [k, 1]
        else:
            topk_idx = torch.arange(n, device=x.device)
            x_k = x
            logits_k = attn_logits

        attn_weights = torch.softmax(logits_k, dim=0)              # [k, 1]
        slide_embedding = torch.sum(attn_weights * x_k, dim=0)     # [D]
        logit = self.classifier(slide_embedding).squeeze(-1)

        # Return full-length attention tensor (zeros for non-selected patches)
        full_weights = torch.zeros(n, device=x.device)
        full_weights[topk_idx] = attn_weights.squeeze(-1)

        return {
            "logit": logit,
            "slide_embedding": slide_embedding,
            "attention_weights": full_weights,
            "selected_indices": topk_idx,
        }
