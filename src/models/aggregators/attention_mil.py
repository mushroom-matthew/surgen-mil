from __future__ import annotations

import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

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
        attn_scores = self.attention(x)                    # [N, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)   # [N, 1]

        slide_embedding = torch.sum(attn_weights * x, dim=0)  # [D]
        logit = self.classifier(slide_embedding).squeeze(-1)

        return {
            "logit": logit,
            "slide_embedding": slide_embedding,
            "attention_weights": attn_weights.squeeze(-1),
        }
