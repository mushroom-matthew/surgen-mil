from __future__ import annotations

import torch
import torch.nn as nn


class InstanceMeanMIL(nn.Module):
    """
    Instance MLP + mean pooling.

    Each patch is scored by a small MLP, then the slide logit is the
    simple mean of those scores.  This gives the model richer per-patch
    expressiveness than a linear projection without the instability of
    learned attention weights.

      z_i   = instance_mlp(h_i)    # [N, 1]
      logit = mean(z_i)            # scalar
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.instance_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, coords=None) -> dict[str, torch.Tensor]:
        """
        x: [N, D] patch embeddings for one slide
        """
        z = self.instance_mlp(x).squeeze(-1)   # [N]
        logit = z.mean()
        return {
            "logit": logit,
            "instance_scores": z,
        }
