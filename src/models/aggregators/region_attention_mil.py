from __future__ import annotations

import torch
import torch.nn as nn


class RegionAttentionMIL(nn.Module):
    """
    Two-level hierarchical MIL: region mean pooling → gated attention over regions.

    Level 1 — spatial binning:
        Divide the slide into an n_bins × n_bins grid using patch coordinates.
        Mean-pool patch embeddings within each non-empty grid cell → region embeddings.

    Level 2 — region attention:
        Apply gated attention (Ilse formulation) over region embeddings.
        Classify from the attention-weighted region embedding.

    This tests whether mesoscale spatial structure (neighbourhoods, tissue
    architecture) provides signal that patch-level attention misses.

    Args:
        n_bins: grid resolution per axis; total regions ≤ n_bins².
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        n_bins: int = 8,
    ):
        super().__init__()
        self.n_bins = n_bins

        # Gated attention over region embeddings
        self.U = nn.Linear(input_dim, attention_dim)
        self.V = nn.Linear(input_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _bin_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Map [N, 2] float coords → [N] integer region IDs in [0, n_bins²).
        Coordinates are normalised per-slide so the grid covers the tissue.
        """
        n = self.n_bins
        xy = coords.float()
        lo = xy.min(dim=0).values
        hi = xy.max(dim=0).values
        span = (hi - lo).clamp(min=1.0)

        bins = ((xy - lo) / span * n).long().clamp(0, n - 1)  # [N, 2]
        return bins[:, 1] * n + bins[:, 0]                     # [N]  row-major

    def _region_embeddings(
        self, x: torch.Tensor, region_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean-pool patch embeddings within each region → [R, D].
        """
        unique_ids = region_ids.unique()
        embeddings = []
        for rid in unique_ids:
            mask = region_ids == rid
            embeddings.append(x[mask].mean(dim=0))
        return torch.stack(embeddings, dim=0)  # [R, D]

    def forward(self, x: torch.Tensor, coords=None) -> dict[str, torch.Tensor]:
        """
        x:      [N, D] patch embeddings
        coords: [N, 2] patch coordinates (required)
        """
        if coords is None:
            raise ValueError("RegionAttentionMIL requires coords")

        region_ids = self._bin_coords(coords)          # [N]
        r = self._region_embeddings(x, region_ids)     # [R, D]

        # Gated attention over regions
        gate = torch.tanh(self.U(r)) * torch.sigmoid(self.V(r))  # [R, attention_dim]
        scores = self.w(gate)                                      # [R, 1]
        attn_weights = torch.softmax(scores, dim=0)                # [R, 1]

        slide_embedding = torch.sum(attn_weights * r, dim=0)      # [D]
        logit = self.classifier(slide_embedding).squeeze(-1)

        return {
            "logit": logit,
            "slide_embedding": slide_embedding,
            "region_attention_weights": attn_weights.squeeze(-1),
            "n_regions": torch.tensor(len(r)),
        }
