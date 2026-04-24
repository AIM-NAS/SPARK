# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# PyTorch-preferred re-export layer mirroring original CLRS models.py.
# Public API is *identical* to the original file; we only switch the import
# source to *_torch modules when available, falling back to the originals.
# ============================================================================

"""The CLRS Algorithmic Reasoning Benchmark (Torch façade)."""

# Baselines
try:  # Prefer Torch port
  from ._src.baselines import BaselineModel, BaselineModelChunked  # type: ignore
except Exception:  # Fallback to original
  from ._src.baselines import BaselineModel, BaselineModelChunked  # type: ignore

# Nets
try:
  from ._src.nets import Net, NetChunked  # type: ignore
except Exception:
  from ._src.nets import Net, NetChunked  # type: ignore

# Processors (exposed for convenience in original API)
try:
  from ._src.processors import GAT, MPNN  # type: ignore
except Exception:
  from ._src.processors import GAT, MPNN  # type: ignore

__all__ = (
    "BaselineModel",
    "BaselineModelChunked",
    "GAT",
    "MPNN",
    "Net",
    "NetChunked",
)
