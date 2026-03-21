"""Minimal FastAPI inference wrapper for SurGen MIL.

Endpoints:
    GET  /health   — liveness check
    GET  /models   — list loaded models and their metadata
    POST /predict  — run slide-level MSI/MMR inference from patch embeddings

Usage (single model via env vars):
    MODEL_CONFIG=configs/uni_mean_fair.yaml \\
    MODEL_CHECKPOINT=outputs/uni_mean_fair/runs/latest/checkpoint.pt \\
    uvicorn api:app --host 0.0.0.0 --port 8000

Usage (auto-discover all runs in outputs/):
    uvicorn api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.build import build_model

app = FastAPI(
    title="SurGen MIL",
    description="Slide-level MSI/MMR prediction from precomputed UNI patch embeddings.",
    version="0.1.0",
)

# name -> {model, cfg, temperature, thresholds, device, checkpoint}
_registry: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_entry(cfg_path: Path, ckpt_path: Path, device: torch.device) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    run_dir = ckpt_path.parent
    metrics_path = run_dir / "metrics.json"
    temperature = 1.0
    if metrics_path.exists():
        with open(metrics_path) as f:
            temperature = json.load(f).get("temperature", 1.0)

    thresholds_path = run_dir / "thresholds.json"
    thresholds = (
        json.loads(thresholds_path.read_text())
        if thresholds_path.exists()
        else {"default": 0.5}
    )

    return {
        "model": model,
        "cfg": cfg,
        "temperature": temperature,
        "thresholds": thresholds,
        "device": device,
        "checkpoint": str(ckpt_path),
    }


def _discover_runs(outputs_dir: Path) -> list[tuple[str, Path, Path]]:
    """Yield (name, config_path, checkpoint_path) for every valid run in outputs/."""
    results = []
    for metrics_path in sorted(outputs_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        ckpt = run_dir / "checkpoint.pt"
        if not ckpt.exists():
            ckpt = run_dir / "model.pt"
        cfg_path = run_dir / "config.yaml"
        if ckpt.exists() and cfg_path.exists():
            # e.g. "uni_mean_fair/runs/001"
            name = "/".join(run_dir.parts[-3:])
            results.append((name, cfg_path, ckpt))
    return results


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_config = os.environ.get("MODEL_CONFIG")
    env_ckpt = os.environ.get("MODEL_CHECKPOINT")

    if env_config and env_ckpt:
        cfg_path, ckpt_path = Path(env_config), Path(env_ckpt)
        try:
            entry = _load_entry(cfg_path, ckpt_path, device)
            _registry[cfg_path.stem] = entry
            print(f"Loaded model '{cfg_path.stem}' from {ckpt_path}")
        except Exception as exc:
            print(f"WARNING: failed to load model from env vars: {exc}")
        return

    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("No outputs/ directory found and no env vars set — no models loaded.")
        return

    for name, cfg_path, ckpt_path in _discover_runs(outputs_dir):
        try:
            _registry[name] = _load_entry(cfg_path, ckpt_path, device)
            print(f"Loaded model '{name}'")
        except Exception as exc:
            print(f"WARNING: skipping '{name}': {exc}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(_registry)}


@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "name": name,
                "aggregator": entry["cfg"]["model"]["name"],
                "input_dim": entry["cfg"]["model"].get("input_dim", 1024),
                "temperature": entry["temperature"],
                "thresholds": entry["thresholds"],
                "checkpoint": entry["checkpoint"],
            }
            for name, entry in _registry.items()
        ]
    }


class PredictRequest(BaseModel):
    features: list[list[float]]
    """Patch embeddings as a 2-D array of shape [N, D] (e.g. N patches, D=1024 for UNI)."""
    coords: Optional[list[list[float]]] = None
    """Optional spatial coordinates [N, 2]. Zeros used when omitted."""
    model_name: Optional[str] = None
    """Name of the model to use (from GET /models). Defaults to the first loaded model."""


class PredictResponse(BaseModel):
    prob: float
    """Calibrated probability of MSI-high / MMR-loss."""
    label: int
    """Binary prediction (1 = positive) using optimal_youden threshold."""
    threshold_used: float
    model_name: str
    temperature: float


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest):
    if not _registry:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Set MODEL_CONFIG and MODEL_CHECKPOINT env vars.",
        )

    name = req.model_name or next(iter(_registry))
    if name not in _registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Available: {list(_registry)}",
        )

    entry = _registry[name]
    device = entry["device"]

    x = torch.tensor(req.features, dtype=torch.float32, device=device)  # [N, D]
    if req.coords is not None:
        coords = torch.tensor(req.coords, dtype=torch.float32, device=device)
    else:
        coords = torch.zeros(x.shape[0], 2, dtype=torch.float32, device=device)

    out = entry["model"](x, coords=coords)
    logit = out["logit"].view(())
    prob = float(torch.sigmoid(logit / entry["temperature"]).item())

    threshold = entry["thresholds"].get(
        "optimal_youden", entry["thresholds"].get("default", 0.5)
    )
    return PredictResponse(
        prob=prob,
        label=int(prob >= threshold),
        threshold_used=threshold,
        model_name=name,
        temperature=entry["temperature"],
    )
