from __future__ import annotations

from src.models.aggregators.attention_mil import AttentionMIL
from src.models.aggregators.gated_attention_mil import GatedAttentionMIL
from src.models.aggregators.instance_mean import InstanceMeanMIL
from src.models.aggregators.lse_pool import LSEPoolMIL
from src.models.aggregators.mean_pool import MeanPoolMIL
from src.models.aggregators.mean_var_pool import MeanVarPoolMIL
from src.models.aggregators.region_attention_mil import RegionAttentionMIL


def build_model(cfg):
    model_name = cfg["model"]["name"]

    if model_name == "mean_pool":
        return MeanPoolMIL(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
        )
    elif model_name == "attention_mil":
        return AttentionMIL(
            input_dim=cfg["model"]["input_dim"],
            attention_dim=cfg["model"]["attention_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
        )
    elif model_name == "gated_attention_mil":
        return GatedAttentionMIL(
            input_dim=cfg["model"]["input_dim"],
            attention_dim=cfg["model"]["attention_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
        )
    elif model_name == "region_attention_mil":
        return RegionAttentionMIL(
            input_dim=cfg["model"]["input_dim"],
            attention_dim=cfg["model"]["attention_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
            n_bins=cfg["model"].get("n_bins", 8),
        )
    elif model_name == "lse_pool":
        return LSEPoolMIL(
            input_dim=cfg["model"]["input_dim"],
            dropout=cfg["model"]["dropout"],
            tau=cfg["model"].get("tau", 1.0),
            learn_tau=cfg["model"].get("learn_tau", True),
            alpha=cfg["model"].get("alpha", 0.5),
        )
    elif model_name == "instance_mean":
        return InstanceMeanMIL(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
        )
    elif model_name == "mean_var_pool":
        return MeanVarPoolMIL(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
