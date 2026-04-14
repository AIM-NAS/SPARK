# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Torch-aligned __init__ that mirrors the public API of CLRS while preferring
# our PyTorch ports ( *_torch.py ) when present. If a Torch module is missing,
# we fall back to the original counterpart to keep import compatibility.
#
"""The CLRS Algorithmic Reasoning Benchmark (Torch port façade)."""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Top-level submodule exposure (unchanged public surface)
# -----------------------------------------------------------------------------
# `models` is a top-level module in CLRS. In our port it should expose
# BaselineModel/BaselineModelChunked with identical APIs.
from clrs_pytorch import models  # Expect this to be a Torch port in your environment

# Algorithms remain framework-agnostic
from . import algorithms

# Optional text utils (may be absent in downstream forks)
try:
  from ._src import clrs_text  # type: ignore
except Exception:  # pragma: no cover
  clrs_text = None  # sentinel

# -----------------------------------------------------------------------------
# Prefer Torch versions; gracefully fallback to originals
# -----------------------------------------------------------------------------
# decoders / processors / specs
try:
  from ._src import decoders as decoders  # type: ignore
except Exception:  # pragma: no cover
  from ._src import decoders  # type: ignore

try:
  from ._src import processors as processors  # type: ignore
except Exception:  # pragma: no cover
  from ._src import processors  # type: ignore

try:
  from ._src import specs as specs  # type: ignore
except Exception:  # pragma: no cover
  from ._src import specs  # type: ignore

# dataset
try:
  from ._src.dataset import (
      chunkify,
      CLRSDataset,
      create_chunked_dataset,
      create_dataset,
      get_clrs_folder,
      get_dataset_gcp_url,
  )
except Exception:  # pragma: no cover
  from ._src.dataset import (
      chunkify,
      CLRSDataset,
      create_chunked_dataset,
      create_dataset,
      get_clrs_folder,
      get_dataset_gcp_url,
  )

# evaluation
try:
  from ._src.evaluation import evaluate, evaluate_hints  # type: ignore
except Exception:  # pragma: no cover
  from ._src.evaluation import evaluate, evaluate_hints  # type: ignore

# model (low-level) — not the same as `models`
try:
  from ._src.model import Model  # type: ignore
except Exception:  # pragma: no cover
  from ._src.model import Model  # type: ignore

# probing utils
try:
  from ._src.probing import (
      DataPoint,
      predecessor_to_cyclic_predecessor_and_first,
  )
except Exception:  # pragma: no cover
  from ._src.probing import (
      DataPoint,
      predecessor_to_cyclic_predecessor_and_first,
  )

# processors factory (provided by processors module)
from ._src.processors import get_processor_factory  # torch/original alias above

# samplers
try:
  from ._src.samplers import (
      build_sampler,
      CLRS30,
      Features,
      Feedback,
      process_permutations,
      process_pred_as_input,
      process_random_pos,
      Sampler,
      Trajectory,
  )
except Exception:  # pragma: no cover
  from ._src.samplers import (
      build_sampler,
      CLRS30,
      Features,
      Feedback,
      process_permutations,
      process_pred_as_input,
      process_random_pos,
      Sampler,
      Trajectory,
  )

# public constants/types from specs
ALGO_IDX_INPUT_NAME = specs.ALGO_IDX_INPUT_NAME
CLRS_30_ALGS_SETTINGS = specs.CLRS_30_ALGS_SETTINGS
Location = specs.Location
OutputClass = specs.OutputClass
Spec = specs.Spec
SPECS = specs.SPECS
Stage = specs.Stage
Type = specs.Type

__version__ = "2.0.3-torch"

__all__ = (
    "ALGO_IDX_INPUT_NAME",
    "build_sampler",
    "chunkify",
    "CLRS30",
    "CLRS_30_ALGS_SETTINGS",
    "create_chunked_dataset",
    "create_dataset",
    "clrs_text",
    "get_clrs_folder",
    "get_dataset_gcp_url",
    "get_processor_factory",
    "DataPoint",
    "predecessor_to_cyclic_predecessor_and_first",
    "process_permutations",
    "process_pred_as_input",
    "process_random_pos",
    "specs",
    "evaluate",
    "evaluate_hints",
    "Features",
    "Feedback",
    "Location",
    "Model",
    "Sampler",
    "Spec",
    "SPECS",
    "Stage",
    "Trajectory",
    "Type",
    "algorithms",
    "decoders",
    "processors",
    "models",
)
