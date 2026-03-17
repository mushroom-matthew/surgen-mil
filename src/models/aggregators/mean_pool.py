from __future__ import annotations

import torch
import torch.nn as nn


class MeanPoolMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: [N, D] patch embeddings for one slide
        returns:
            {
                "logit": [1],
                "slide_embedding": [D],
            }
        """
        slide_embedding = x.mean(dim=0)
        logit = self.classifier(slide_embedding).squeeze(-1)
        return {
            "logit": logit,
            "slide_embedding": slide_embedding,
        }
