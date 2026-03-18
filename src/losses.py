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
    """Naive fixed-weight hybrid — raw scale, no normalisation."""
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


class NormalizedBCEFocalWrapper(nn.Module):
    """
    Batch-normalised hybrid: each loss is divided by its own detached magnitude
    before mixing, so lam_bce / lam_focal control a true proportion not a scale.

        L = lam * (L_bce / stop_grad(L_bce)) + (1-lam) * (L_focal / stop_grad(L_focal))
    """
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lam: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.bce = BCEWrapper(pos_weight=pos_weight)
        self.focal = FocalWrapper(alpha=alpha, gamma=gamma)
        self.lam = lam
        self.eps = eps

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_bce   = self.bce(logit, target)
        l_focal = self.focal(logit, target)
        l_bce_n   = l_bce   / (l_bce.detach()   + self.eps)
        l_focal_n = l_focal / (l_focal.detach() + self.eps)
        return self.lam * l_bce_n + (1.0 - self.lam) * l_focal_n


class CurriculumBCEFocalWrapper(nn.Module):
    """
    Curriculum hybrid: BCE-heavy early, focal-heavy late.

    schedule is a list of (up_to_epoch, lam_bce) pairs, evaluated in order.
    Default: epochs 1-3 → 0.8 BCE, 4-8 → 0.5, 9+ → 0.3.
    """
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        schedule: list[tuple[int, float]] | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.bce = BCEWrapper(pos_weight=pos_weight)
        self.focal = FocalWrapper(alpha=alpha, gamma=gamma)
        self.schedule = schedule or [(3, 0.8), (8, 0.5), (999, 0.3)]
        self.eps = eps
        self._epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _lam(self) -> float:
        for up_to, lam in self.schedule:
            if self._epoch <= up_to:
                return lam
        return self.schedule[-1][1]

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_bce   = self.bce(logit, target)
        l_focal = self.focal(logit, target)
        l_bce_n   = l_bce   / (l_bce.detach()   + self.eps)
        l_focal_n = l_focal / (l_focal.detach() + self.eps)
        lam = self._lam()
        return lam * l_bce_n + (1.0 - lam) * l_focal_n


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

    if name == "bce_focal_normalized":
        return NormalizedBCEFocalWrapper(
            pos_weight=pos_weight,
            alpha=cfg.get("alpha", 0.25),
            gamma=cfg.get("gamma", 2.0),
            lam=cfg.get("lam", 0.5),
        )

    if name == "bce_focal_curriculum":
        schedule = cfg.get("schedule", [(3, 0.8), (8, 0.5), (999, 0.3)])
        # yaml will give list-of-lists; convert to list-of-tuples
        schedule = [tuple(s) for s in schedule]
        return CurriculumBCEFocalWrapper(
            pos_weight=pos_weight,
            alpha=cfg.get("alpha", 0.25),
            gamma=cfg.get("gamma", 2.0),
            schedule=schedule,
        )

    raise ValueError(f"Unknown loss: {name}")
