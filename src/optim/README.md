# src/optim/

Placeholder for optimizer and learning rate scheduler utilities.

Currently, optimizer and scheduler construction lives inline in `train.py`.
This package would be the right home if any of the following are added:

- Custom optimizers (e.g. SAM, Lion, lookahead wrappers)
- Shared scheduler factories (cosine, linear warmup, cyclic) extracted from `train.py`
- Parameter group builders (e.g. layer-wise learning rate decay for fine-tuning)
- Gradient clipping or gradient norm logging utilities

At the current scale (one training script, two optimizer types), inline construction
in `train.py` is sufficient.
