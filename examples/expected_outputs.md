# Expected Training Outputs

A completed training run writes the following files:

```
outputs/<model_name>/runs/001/
  config.yaml        # copy of the config used for this run
  model.pt           # model weights at the best validation checkpoint
  metrics.json       # test AUROC, AUPRC, temperature, case-level metrics
  predictions.csv    # slide_id, label, prob, split for all slides
  history.json       # per-epoch: train_loss, val_auroc, val_auprc, val_auprc_ema
  training_curve.png # loss and AUROC curves
```

## metrics.json structure

```json
{
  "test": {
    "auroc": 0.82,
    "auprc": 0.71
  },
  "test_case_level": {
    "max": {"auroc": 0.85, "auprc": 0.74},
    "mean": {"auroc": 0.83, "auprc": 0.72},
    "noisy_or": {"auroc": 0.84, "auprc": 0.73}
  },
  "temperature": 1.23
}
```

## predictions.csv structure

| Column | Description |
|--------|-------------|
| `slide_id` | Slide identifier (e.g. `SR1482_40X_HE_T1_0`) |
| `label` | Ground truth (0 or 1) |
| `prob` | Temperature-scaled probability of MSI/MMR loss |
| `split` | `train`, `val`, or `test` |

## history.json structure

One entry per epoch:
```json
[
  {
    "epoch": 1,
    "train_loss": 0.65,
    "val_auroc": 0.71,
    "val_auprc": 0.58,
    "val_auprc_ema": 0.58
  },
  ...
]
```

## Run numbering

Each call to `python train.py --config ...` creates a new numbered run directory (`001`, `002`, ...).
A `latest` symlink always points to the most recent run.
