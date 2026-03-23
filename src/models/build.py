from __future__ import annotations

from src.models.aggregators.attention_mil import AttentionMIL
from src.models.aggregators.gated_attention_mil import GatedAttentionMIL
from src.models.aggregators.hybrid_attention_mil import HybridAttentionMIL
from src.models.aggregators.instance_mean import InstanceMeanMIL
from src.models.aggregators.lse_pool import LSEPoolMIL
from src.models.aggregators.mean_pool import MeanPoolMIL
from src.models.aggregators.mean_var_pool import MeanVarPoolMIL
from src.models.aggregators.region_attention_mil import RegionAttentionMIL
from src.models.aggregators.topk_attention_mil import TopKAttentionMIL
from src.models.aggregators.transformer_mil import TransformerMIL


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
            use_coords=cfg["model"].get("use_coords", False),
            coord_hidden_dim=cfg["model"].get("coord_hidden_dim", 32),
            coord_embed_dim=cfg["model"].get("coord_embed_dim", 32),
        )
    elif model_name == "gated_attention_mil":
        return GatedAttentionMIL(
            input_dim=cfg["model"]["input_dim"],
            attention_dim=cfg["model"]["attention_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
        )
    elif model_name == "hybrid_attention_mil":
        return HybridAttentionMIL(
            input_dim=cfg["model"]["input_dim"],
            attention_dim=cfg["model"].get("attention_dim", 128),
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
            n_attention_heads=cfg["model"].get("n_attention_heads", 2),
            include_mean=cfg["model"].get("include_mean", True),
            fusion=cfg["model"].get("fusion", "concat"),
            diversity_weight=cfg["model"].get("diversity_weight", 0.0),
            use_coords=cfg["model"].get("use_coords", False),
            coord_hidden_dim=cfg["model"].get("coord_hidden_dim", 32),
            coord_embed_dim=cfg["model"].get("coord_embed_dim", 32),
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
    elif model_name == "topk_attention_mil":
        return TopKAttentionMIL(
            input_dim=cfg["model"]["input_dim"],
            attention_dim=cfg["model"]["attention_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            dropout=cfg["model"]["dropout"],
            k=cfg["model"].get("k", 16),
        )
    elif model_name == "transformer_mil":
        return TransformerMIL(
            input_dim=cfg["model"]["input_dim"],
            proj_dim=cfg["model"].get("proj_dim", 512),
            n_layers=cfg["model"].get("n_layers", 2),
            n_heads=cfg["model"].get("n_heads", 2),
            ffn_dim=cfg["model"].get("ffn_dim", 2048),
            dropout=cfg["model"].get("dropout", 0.15),
            ln_eps=cfg["model"].get("ln_eps", 1e-5),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
