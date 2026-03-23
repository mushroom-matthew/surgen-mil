# src/train/

Placeholder for training loop utilities.

Currently, the full training loop lives in `train.py` at the project root.
This package would be the right home if the training logic is split into:

- `engine.py` — per-epoch train/eval loops, decoupled from I/O and config
- `callbacks.py` — early stopping, checkpoint saving, metric logging as composable callbacks
- `mixup.py` or `augment.py` — slide-level or bag-level augmentation strategies
- `distributed.py` — multi-GPU or multi-node training setup

Refactoring into this structure makes sense once the training loop needs to support
multiple experiment types (e.g. multi-task, contrastive, fine-tuning UNI).
