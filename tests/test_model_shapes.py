"""Test that each model type produces correct output shapes."""
import pytest
import torch

from src.models.build import build_model

MODEL_CONFIGS = [
    {"name": "mean_pool", "input_dim": 1024, "hidden_dim": 64, "dropout": 0.0},
    {"name": "attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0},
    {"name": "attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "use_coords": True, "coord_hidden_dim": 16, "coord_embed_dim": 8},
    {"name": "gated_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0},
    {"name": "hybrid_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "n_attention_heads": 2, "include_mean": True, "fusion": "concat"},
    {"name": "hybrid_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "n_attention_heads": 2, "include_mean": True, "fusion": "concat", "use_coords": True, "coord_hidden_dim": 16, "coord_embed_dim": 8},
    {"name": "instance_mean", "input_dim": 1024, "hidden_dim": 64, "dropout": 0.0},
    {"name": "mean_var_pool", "input_dim": 1024, "hidden_dim": 64, "dropout": 0.0},
    {"name": "lse_pool", "input_dim": 1024, "dropout": 0.0},
    {"name": "topk_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "k": 4},
    {"name": "region_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "n_bins": 4},
    {"name": "transformer_mil", "input_dim": 1024, "proj_dim": 256, "dropout": 0.0},
]

# Models that return slide_embedding
MODELS_WITH_SLIDE_EMBEDDING = {
    "mean_pool", "attention_mil", "gated_attention_mil",
    "hybrid_attention_mil", "mean_var_pool", "topk_attention_mil", "region_attention_mil", "transformer_mil",
}


@pytest.mark.parametrize("model_cfg", MODEL_CONFIGS, ids=[c["name"] for c in MODEL_CONFIGS])
def test_model_output_logit(model_cfg):
    cfg = {
        "model": model_cfg,
        "optimizer": {"lr": 1e-4},
        "loss": {"name": "bce"},
        "training": {"seed": 0},
    }
    model = build_model(cfg)
    model.eval()

    N = 10
    x = torch.randn(N, 1024)
    coords = torch.arange(N * 2, dtype=torch.float32).view(N, 2)

    with torch.no_grad():
        out = model(x, coords=coords)

    assert "logit" in out, f"{model_cfg['name']} missing 'logit'"
    assert out["logit"].numel() == 1, f"{model_cfg['name']} logit is not scalar"


@pytest.mark.parametrize("model_cfg", [
    c for c in MODEL_CONFIGS if c["name"] in MODELS_WITH_SLIDE_EMBEDDING
], ids=[c["name"] for c in MODEL_CONFIGS if c["name"] in MODELS_WITH_SLIDE_EMBEDDING])
def test_model_slide_embedding(model_cfg):
    cfg = {
        "model": model_cfg,
        "optimizer": {"lr": 1e-4},
        "loss": {"name": "bce"},
        "training": {"seed": 0},
    }
    model = build_model(cfg)
    model.eval()

    N = 10
    x = torch.randn(N, 1024)
    coords = torch.arange(N * 2, dtype=torch.float32).view(N, 2)

    with torch.no_grad():
        out = model(x, coords=coords)

    assert "slide_embedding" in out, f"{model_cfg['name']} missing 'slide_embedding'"
    assert out["slide_embedding"].dim() == 1, f"{model_cfg['name']} slide_embedding is not 1D"


@pytest.mark.parametrize("model_cfg", [
    {"name": "attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0},
    {"name": "attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "use_coords": True, "coord_hidden_dim": 16, "coord_embed_dim": 8},
    {"name": "gated_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0},
    {"name": "hybrid_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "n_attention_heads": 2, "include_mean": True, "fusion": "concat"},
    {"name": "hybrid_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "n_attention_heads": 2, "include_mean": True, "fusion": "concat", "use_coords": True, "coord_hidden_dim": 16, "coord_embed_dim": 8},
    {"name": "topk_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "k": 4},
], ids=["attention_mil", "attention_mil_coords", "gated_attention_mil", "hybrid_attention_mil", "hybrid_attention_mil_coords", "topk_attention_mil"])
def test_attention_weights_shape(model_cfg):
    cfg = {"model": model_cfg, "optimizer": {}, "loss": {}, "training": {}}
    model = build_model(cfg)
    model.eval()

    N = 10
    x = torch.randn(N, 1024)
    coords = torch.arange(N * 2, dtype=torch.float32).view(N, 2)

    with torch.no_grad():
        out = model(x, coords=coords)

    assert "attention_weights" in out, f"{model_cfg['name']} missing 'attention_weights'"
    attn = out["attention_weights"].squeeze()
    assert attn.shape == (N,) or attn.shape[0] <= N, \
        f"{model_cfg['name']} attention_weights shape {attn.shape} unexpected"


@pytest.mark.parametrize("model_cfg", [
    {"name": "attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "use_coords": True},
    {"name": "hybrid_attention_mil", "input_dim": 1024, "attention_dim": 64, "hidden_dim": 64, "dropout": 0.0, "n_attention_heads": 2, "include_mean": True, "fusion": "concat", "use_coords": True},
], ids=["attention_coords_requires_coords", "hybrid_coords_requires_coords"])
def test_coord_aware_models_require_coords(model_cfg):
    cfg = {"model": model_cfg, "optimizer": {}, "loss": {}, "training": {}}
    model = build_model(cfg)
    x = torch.randn(10, 1024)
    with pytest.raises(ValueError):
        model(x, coords=None)
