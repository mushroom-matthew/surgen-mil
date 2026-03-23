from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateEncoder(nn.Module):
    """
    Encode per-patch (x, y) coordinates after per-slide normalisation.
    """

    def __init__(self, hidden_dim: int = 32, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords is None:
            raise ValueError("CoordinateEncoder requires coords")

        xy = coords.float()
        lo = xy.min(dim=0).values
        hi = xy.max(dim=0).values
        span = (hi - lo).clamp(min=1.0)
        xy = (xy - lo) / span
        return self.net(xy)
