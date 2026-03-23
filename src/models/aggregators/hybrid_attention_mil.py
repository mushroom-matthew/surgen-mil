from __future__ import annotations

import torch
import torch.nn as nn

from src.models.aggregators.coord_encoder import CoordinateEncoder


class HybridAttentionMIL(nn.Module):
    """
    Hybrid MIL aggregator that fuses mean pooling with multiple attention heads.

    This targets the current repo's central tension:
      - mean pooling is stable and already strong
      - learned attention can surface sparse signal but is seed-sensitive

    The model therefore keeps the stable slide-level mean embedding and adds
    up to a small number of attention-pooled embeddings. Each attention head
    produces its own bag summary, and the summaries are fused before the final
    classifier.

    Optional diversity regularisation penalises overlap between attention
    distributions so the heads do not collapse to the same patch subset.
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        n_attention_heads: int = 2,
        include_mean: bool = True,
        fusion: str = "concat",
        diversity_weight: float = 0.0,
        use_coords: bool = False,
        coord_hidden_dim: int = 32,
        coord_embed_dim: int = 32,
    ):
        super().__init__()
        if n_attention_heads < 1 or n_attention_heads > 5:
            raise ValueError("n_attention_heads must be in [1, 5]")
        if not include_mean and n_attention_heads < 1:
            raise ValueError("HybridAttentionMIL requires at least one aggregation branch")
        if fusion not in {"concat", "mean"}:
            raise ValueError("fusion must be 'concat' or 'mean'")

        self.n_attention_heads = n_attention_heads
        self.include_mean = include_mean
        self.fusion = fusion
        self.diversity_weight = diversity_weight
        self.use_coords = use_coords

        attn_input_dim = input_dim
        self.coord_encoder = None
        if self.use_coords:
            self.coord_encoder = CoordinateEncoder(
                hidden_dim=coord_hidden_dim,
                output_dim=coord_embed_dim,
            )
            attn_input_dim += coord_embed_dim

        self.attention = nn.Sequential(
            nn.Linear(attn_input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, n_attention_heads),
        )

        n_branches = n_attention_heads + int(include_mean)
        classifier_in = input_dim * n_branches if fusion == "concat" else input_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _diversity_penalty(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        attn_weights: [H, N]

        Penalise overlap between heads using the off-diagonal mass of the
        attention Gram matrix. Zero means perfectly orthogonal distributions.
        """
        if attn_weights.shape[0] <= 1:
            return attn_weights.new_zeros(())
        gram = attn_weights @ attn_weights.transpose(0, 1)  # [H, H]
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        off_diag = gram * (1.0 - eye)
        return off_diag.pow(2).mean()

    def forward(self, x: torch.Tensor, coords=None) -> dict[str, torch.Tensor]:
        """
        x: [N, D]
        """
        attn_input = x
        if self.use_coords:
            coord_embed = self.coord_encoder(coords)
            attn_input = torch.cat([x, coord_embed], dim=-1)

        attn_scores = self.attention(attn_input)                # [N, H]
        attn_weights = torch.softmax(attn_scores, dim=0)     # [N, H]

        branch_embeddings = []
        if self.include_mean:
            branch_embeddings.append(x.mean(dim=0))

        head_embeddings = []
        for h in range(self.n_attention_heads):
            w = attn_weights[:, h : h + 1]                   # [N, 1]
            head_embeddings.append(torch.sum(w * x, dim=0))  # [D]
        branch_embeddings.extend(head_embeddings)

        if self.fusion == "concat":
            slide_embedding = torch.cat(branch_embeddings, dim=0)
        else:
            slide_embedding = torch.stack(branch_embeddings, dim=0).mean(dim=0)

        logit = self.classifier(slide_embedding).squeeze(-1)

        out = {
            "logit": logit,
            "slide_embedding": slide_embedding,
            "attention_weights": attn_weights.mean(dim=1),
            "attention_weights_multi": attn_weights.transpose(0, 1),
        }

        if self.diversity_weight > 0:
            penalty = self._diversity_penalty(attn_weights.transpose(0, 1))
            out["aux_loss"] = self.diversity_weight * penalty
            out["attention_diversity_penalty"] = penalty.detach()

        return out
