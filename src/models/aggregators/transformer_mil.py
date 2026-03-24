from __future__ import annotations

import torch
import torch.nn as nn

from src.models.aggregators.coord_encoder import CoordinateEncoder


class TransformerMIL(nn.Module):
    """
    Transformer-based MIL aggregator matching the SurGen paper (giaf086).

    Architecture:
      1. (Optional) Coordinate encoding: encode (x, y) → coord_embed_dim,
         concatenate with patch embeddings before projection
      2. Linear projection: (input_dim [+ coord_embed_dim]) → proj_dim, followed by ReLU
      3. Transformer encoder: n_layers layers, n_heads attention heads,
         ffn_dim feedforward dimension, dropout, layer_norm_eps
      4. Mean pooling across patches
      5. Linear classifier → 1 logit
    """

    def __init__(
        self,
        input_dim: int = 1024,
        proj_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.15,
        ln_eps: float = 1e-5,
        use_coords: bool = False,
        coord_hidden_dim: int = 32,
        coord_embed_dim: int = 32,
    ):
        super().__init__()
        self.use_coords = use_coords
        if use_coords:
            self.coord_encoder = CoordinateEncoder(
                hidden_dim=coord_hidden_dim,
                output_dim=coord_embed_dim,
            )
            effective_input_dim = input_dim + coord_embed_dim
        else:
            self.coord_encoder = None
            effective_input_dim = input_dim

        self.projection = nn.Sequential(
            nn.Linear(effective_input_dim, proj_dim),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            layer_norm_eps=ln_eps,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(proj_dim, 1)

    def forward(self, x: torch.Tensor, coords=None) -> dict[str, torch.Tensor]:
        """
        x:      [N, D] patch embeddings for one slide
        coords: [N, 2] patch coordinates (required when use_coords=True)
        """
        if self.use_coords:
            coord_emb = self.coord_encoder(coords)   # [N, coord_embed_dim]
            x = torch.cat([x, coord_emb], dim=-1)   # [N, D + coord_embed_dim]
        x = self.projection(x)           # [N, proj_dim]
        x = x.unsqueeze(0)               # [1, N, proj_dim]
        x = self.transformer(x)          # [1, N, proj_dim]
        slide_embedding = x.squeeze(0).mean(dim=0)  # [proj_dim]
        logit = self.classifier(slide_embedding).squeeze(-1)
        return {
            "logit": logit,
            "slide_embedding": slide_embedding,
        }
