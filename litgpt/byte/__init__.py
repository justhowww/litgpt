"""Byte-domain video modeling extensions for LitGPT.

The package owns H.264 dataset construction, decoder-level evaluation, and
minimum-risk training. Legacy import paths remain available as compatibility
modules while callers migrate to ``litgpt.byte``.
"""

from litgpt.byte.data import ByteDataConfig, ByteDataModule, ByteSliceDataset
from litgpt.byte.mrt import MRTConfig
from litgpt.byte.reconstruction import ReconstructionEvalConfig
from litgpt.byte.training import byte_training_loss

__all__ = [
    "ByteDataConfig",
    "ByteDataModule",
    "ByteSliceDataset",
    "MRTConfig",
    "ReconstructionEvalConfig",
    "byte_training_loss",
]
