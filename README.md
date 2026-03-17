# surgen-mil

Modular MIL experimentation framework for slide-level MSI/MMR prediction on the SurGen dataset using precomputed UNI embeddings.

## Initial goals
- Reusable architecture for:
  - feature providers
  - bag samplers
  - aggregators
  - losses
  - optimizers
- Tight initial experiment scope:
  - UNI + mean pooling
  - UNI + attention MIL
  - optional weighted BCE ablation
