from __future__ import annotations

import torch
import torch.nn as nn


class MeanVarPoolMIL(nn.Module):
    """
    Mean + std + max pooling MIL.

    Concatenates three statistics per feature dimension:
      slide_embedding = [mean(x), std(x), max(x)]   # [3*D]

    This directly encodes the probe result: the 3072-dim hand-crafted
    descriptor already separates the signal linearly, so we give the MLP
    the same statistics as input rather than learning to re-discover them
    via attention.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, coords=None) -> dict[str, torch.Tensor]:
        """
        x: [N, D] patch embeddings for one slide
        """
        # unbiased=False avoids nan for single-patch bags
        slide_embedding = torch.cat([
            x.mean(dim=0),
            x.std(dim=0, unbiased=False),
            x.max(dim=0).values,
        ], dim=-1)  # [3*D]

        logit = self.classifier(slide_embedding).squeeze(-1)
        return {
            "logit": logit,
            "slide_embedding": slide_embedding,
        }
