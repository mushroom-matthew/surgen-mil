from __future__ import annotations

import torch
import torch.nn as nn


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL (Ilse et al., 2018).

    Replaces the single tanh attention branch with a gated formulation:

        U = tanh(W_U h)          # [N, attention_dim]
        V = sigmoid(W_V h)       # [N, attention_dim]  ← gate
        e = w^T (U ⊙ V)         # [N] raw scores
        a = softmax(e)           # [N] attention weights

    The element-wise product lets the gate suppress irrelevant patch
    dimensions before scoring, giving more expressive patch selection
    without extra depth in the classifier head.
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.U = nn.Linear(input_dim, attention_dim)   # tanh branch
        self.V = nn.Linear(input_dim, attention_dim)   # sigmoid gate
        self.w = nn.Linear(attention_dim, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, coords=None) -> dict[str, torch.Tensor]:
        """
        x: [N, D] patch embeddings for one slide
        """
        gate = torch.tanh(self.U(x)) * torch.sigmoid(self.V(x))  # [N, attention_dim]
        scores = self.w(gate)                                       # [N, 1]
        attn_weights = torch.softmax(scores, dim=0)                 # [N, 1]

        slide_embedding = torch.sum(attn_weights * x, dim=0)       # [D]
        logit = self.classifier(slide_embedding).squeeze(-1)

        return {
            "logit": logit,
            "slide_embedding": slide_embedding,
            "attention_weights": attn_weights.squeeze(-1),
        }
