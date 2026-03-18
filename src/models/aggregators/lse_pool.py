from __future__ import annotations

import torch
import torch.nn as nn


class LSEPoolMIL(nn.Module):
    """
    Log-Sum-Exp pooling MIL — correct formulation.

    Step 1: project each patch embedding to a scalar instance score.
    Step 2: pool those scalars with LSE (interpolates mean ↔ max via tau).
    Step 3: blend with plain mean for stability in low-data regimes.

      z_i  = instance_head(h_i)          # scalar per patch  [N]
      lse  = tau * logsumexp(z / tau)    # soft-max pooling  scalar
      logit = alpha * lse + (1-alpha) * z.mean()

    Large tau  → behaves like mean pooling (preferred for distributed signal)
    Small tau  → behaves like max pooling  (focused signal)
    alpha      → blend weight between lse and mean (fixed at 0.5)
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
        tau: float = 1.0,
        learn_tau: bool = True,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.log_tau = nn.Parameter(
            torch.tensor(float(tau)).log(),
            requires_grad=learn_tau,
        )
        self.alpha = alpha
        self.instance_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor, coords=None) -> dict[str, torch.Tensor]:
        """
        x: [N, D] patch embeddings for one slide
        """
        tau = self.log_tau.exp()
        z = self.instance_head(x).squeeze(-1)          # [N]
        lse = tau * torch.logsumexp(z / tau, dim=0)    # scalar
        logit = self.alpha * lse + (1 - self.alpha) * z.mean()
        return {
            "logit": logit,
            "instance_scores": z,
            "tau": tau,
        }
