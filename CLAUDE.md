# Project: SurGen MIL

This repository is for a 1-week computational pathology take-home project.

## Objective
Build a modular experiment framework for slide-level MSI/MMR prediction from precomputed UNI patch embeddings provided with the SurGen dataset.

Dataset root on remote machine:
- /mnt/data-surgen/

## Design priorities
Keep scope tight, but architecture reusable.

The code should make it easy to swap:
- feature provider
- bag sampler
- aggregation model
- loss function
- optimizer

## Initial executed scope
Focus on a small experiment matrix:
1. UNI + mean pooling + BCE + AdamW
2. UNI + attention MIL + BCE + AdamW
3. UNI + attention MIL + weighted BCE + AdamW

## Core abstractions
- FeatureProvider: returns slide embeddings, coords, label, metadata
- BagSampler: random subset or full bag
- Aggregator: mean pooling or attention MIL
- Loss factory
- Optimizer factory

## Expected pipeline
feature provider -> bag sampler -> aggregator -> classifier head -> loss

## Metrics
- AUROC
- AUPRC
- confusion matrix

## Notes
- Do not overbuild.
- Prefer clarity and extensibility over framework complexity.
- Prioritize getting to first training run quickly.
