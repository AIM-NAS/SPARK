# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# PyTorch-aligned reimplementation of CLRS model.py with identical API.
# - Keeps the same public class name `Model`, method signatures and behavior.
# - Additionally inherits from `torch.nn.Module` to integrate with Torch while
#   preserving abstract interface used across CLRS code.
#
from __future__ import annotations

import abc
from typing import Dict, List, Optional, Union

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # Fallback to plain object if torch is unavailable

from . import probing
from . import samplers
from . import specs


Result = Dict[str, probing.DataPoint]


class Model(abc.ABC, nn.Module):
  """Abstract base class for CLRS3-B models (Torch-compatible)."""

  def __init__(self, spec: Union[specs.Spec, List[specs.Spec]]):
    """Set up the problem, prepare to predict on first task."""
    # Initialize nn.Module if available
    try:
      super().__init__()  # nn.Module.__init__
    except TypeError:
      # If nn.Module is not available (fallback path), just proceed.
      pass
    if not isinstance(spec, list):
      spec = [spec]
    self._spec = spec

  @abc.abstractmethod
  def predict(self, features: samplers.Features) -> Result:
    """Make predictions about the current task."""
    raise NotImplementedError

  @abc.abstractmethod
  def feedback(self, feedback: Optional[samplers.Feedback]):
    """Advance to the next task, incorporating any available feedback."""
    raise NotImplementedError
