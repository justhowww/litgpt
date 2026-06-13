# Byte-domain extension

This package contains the H.264-specific implementation layered on top of
LitGPT:

- `data.py`: Annex-B indexing and AR/FIM dataset construction.
- `reconstruction.py`: decoder-level AR/FIM evaluation.
- `mrt.py`: online minimum-risk candidate generation and scoring.
- `training.py`: byte loss, validation, and training lifecycle hooks.

User-facing commands live under `scripts/byte/`. The former modules
`litgpt.data.byte_data`, `litgpt.eval.byte_reconstruction`, and
`litgpt.byte_mrt`, plus the former root-level script names, are compatibility
wrappers. New code should import from `litgpt.byte` and invoke
`scripts/byte/*`.

The generic LitGPT model and pretraining loop retain only the integration
points required for region/offset embeddings and `ByteTrainingRuntime`.
