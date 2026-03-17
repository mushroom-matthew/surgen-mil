from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class BCEWrapper(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logit, target)


class FocalWrapper(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(
            logit, target,
            alpha=self.alpha, gamma=self.gamma,
            reduction="mean",
        )


class BCEFocalWrapper(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lam_bce: float = 0.5,
        lam_focal: float = 0.5,
    ):
        super().__init__()
        self.bce = BCEWrapper(pos_weight=pos_weight)
        self.focal = FocalWrapper(alpha=alpha, gamma=gamma)
        self.lam_bce = lam_bce
        self.lam_focal = lam_focal

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lam_bce * self.bce(logit, target) + self.lam_focal * self.focal(logit, target)


def build_loss(
    cfg: dict,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
) -> nn.Module:
    name = cfg["name"]

    if name == "bce":
        return BCEWrapper(pos_weight=pos_weight)

    if name == "focal":
        return FocalWrapper(
            alpha=cfg.get("alpha", 0.25),
            gamma=cfg.get("gamma", 2.0),
        )

    if name == "bce_focal":
        return BCEFocalWrapper(
            pos_weight=pos_weight,
            alpha=cfg.get("alpha", 0.25),
            gamma=cfg.get("gamma", 2.0),
            lam_bce=cfg.get("lam_bce", 0.5),
            lam_focal=cfg.get("lam_focal", 0.5),
        )

    raise ValueError(f"Unknown loss: {name}")
