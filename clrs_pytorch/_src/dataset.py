# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# PyTorch/Numpy reimplementation of CLRS dataset.py with identical public API
# and data flow semantics. This file removes TF/TFDS/JAX runtime deps while
# preserving behavior for: `_preprocess`, `chunkify`, `create_dataset`,
# `create_chunked_dataset`, and the light-weight builder `CLRSDataset`.
#
# Notes:
# - Uses numpy for array ops; relies on clrs._src.probing/samplers/specs types.
# - `CLRSDataset` is a minimal in-memory generator that mirrors TFDS builder
#   logic (name/split handling, sample counts, and per-algorithm settings).
# - Shapes follow the original: inputs/outputs are [B, ...] in full samples;
#   hints are [T, B, ...]. In chunked mode all tensors become [T, B, ...].

from __future__ import annotations

import dataclasses
import functools
from typing import Dict, Iterator, List, Tuple

import numpy as np

# --- Torch -> NumPy helper (safe no-op if torch not available)
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

def _to_np(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

from . import probing
from . import samplers
from . import specs

# ------------------------------ Helpers -------------------------------------

def _correct_axis_filtering(tensor: np.ndarray, index: int, name: str):
    """Match TFDS slicing behavior: hints keep time first dimension."""
    if 'hint_' in name:
        return tensor[:, index]
    else:
        return tensor[index]


# ------------------------------ TFDS config shim ----------------------------

@dataclasses.dataclass
class CLRSConfig:
    """Simple stand-in for tfds.core.BuilderConfig used in original code."""
    name: str
    split: str = ''


DEFAULT_BUILDER_CONFIGS: List[CLRSConfig] = []


def _build_default_builder_configs():
    for split in ['train', 'val', 'test']:
        for alg in specs.CLRS_30_ALGS:
            DEFAULT_BUILDER_CONFIGS.append(CLRSConfig(name=f'{alg}_{split}', split=split))


_build_default_builder_configs()


class CLRSDataset:
    """In-memory generator matching the original TFDS builder behavior."""

    VERSION = '1.0.0'
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}
    BUILDER_CONFIGS = DEFAULT_BUILDER_CONFIGS

    _instantiated_dataset: Dict[str, np.ndarray] | None = None
    _instantiated_dataset_name: str = ''
    _instantiated_dataset_split: str = ''

    def __init__(self, builder_config: CLRSConfig):
        self._builder_config = builder_config

    # ------- internal helpers (logic mirrored from the original) ------------
    def _num_samples(self, algorithm_name: str) -> int:
        num = samplers.CLRS30[self._builder_config.split]['num_samples']
        if self._builder_config.split != 'train':
            num *= specs.CLRS_30_ALGS_SETTINGS[algorithm_name]['num_samples_multiplier']
        return int(num)

    def _create_data(self, single_sample: bool):
        algorithm_name = '_'.join(self._builder_config.name.split('_')[:-1])
        num_samples = self._num_samples(algorithm_name)
        sampler, _ = samplers.build_sampler(
            algorithm_name,
            seed=samplers.CLRS30[self._builder_config.split]['seed'],
            num_samples=num_samples,
            length=samplers.CLRS30[self._builder_config.split]['length'],
        )
        sampled = sampler.next(batch_size=1 if single_sample else None)
        data = {'input_' + t.name: t.data for t in sampled.features.inputs}
        data['lengths'] = sampled.features.lengths
        data.update({'output_' + t.name: t.data for t in sampled.outputs})
        data.update({'hint_' + t.name: t.data for t in sampled.features.hints})
        self._instantiated_dataset = data

    # ---------------------------- Public API --------------------------------
    def split_generators(self) -> Dict[str, Iterator[Tuple[str, Dict[str, np.ndarray]]]]:
        if (self._instantiated_dataset_name != self._builder_config.name or
            self._instantiated_dataset_split != self._builder_config.split):
            self._create_data(single_sample=False)
            self._instantiated_dataset_name = self._builder_config.name
            self._instantiated_dataset_split = self._builder_config.split
        return {self._builder_config.split: self._generate_examples()}

    def _generate_examples(self) -> Iterator[Tuple[str, Dict[str, np.ndarray]]]:
        assert self._instantiated_dataset is not None
        algorithm_name = '_'.join(self._builder_config.name.split('_')[:-1])
        for i in range(self._num_samples(algorithm_name)):
            data = {k: _correct_axis_filtering(v, i, k)
                    for k, v in self._instantiated_dataset.items()}
            yield str(i), data


# ------------------------------ URL helpers ---------------------------------

def _get_clrs_file_name():
    return f'CLRS30_v{CLRSDataset.VERSION}.tar.gz'


def get_dataset_gcp_url():
    return f'https://storage.googleapis.com/dm-clrs/{_get_clrs_file_name()}'


def get_clrs_folder():
    return f'CLRS30_v{CLRSDataset.VERSION}'


# ----------------------------- Preprocess -----------------------------------

def _preprocess(data_point: Dict[str, np.ndarray], algorithm: str | None = None):
    """Convert sampled inputs (a flat dict) into Feedback with DataPoints.
    Mirrors the original semantics:
      - names are split on first prefix (input_/output_/hint_)
      - hints are [T,B,...] and require time/batch axis swap for DataPoint
    """
    inputs: List[probing.DataPoint] = []
    outputs: List[probing.DataPoint] = []
    hints: List[probing.DataPoint] = []
    lengths = None

    for name, data in data_point.items():
        if name == 'lengths':
            lengths = _to_np(data)                # <- 确保是 numpy
            continue

        parts = name.split('_')
        clean_name = '_'.join(parts[1:])
        stage, location, dp_type = specs.SPECS[algorithm][clean_name]

        # 安全获取 stage 名称（兼容 Enum 或 str）
        stage_name = stage.name.lower() if hasattr(stage, 'name') else str(stage).lower()
        assert stage_name == parts[0]             # 和前缀对齐

        data = _to_np(data)                       # <- 确保是 numpy
        if stage_name == 'hint':
            data = np.swapaxes(data, 0, 1)        # [T,B,...] -> keep T first for DP

        dp = probing.DataPoint(clean_name, location, dp_type, data)
        if stage_name == 'input':
            inputs.append(dp)
        elif stage_name == 'output':
            outputs.append(dp)
        else:  # 'hint'
            hints.append(dp)

    return samplers.Feedback(
        samplers.Features(tuple(inputs), tuple(hints), lengths),
        tuple(outputs)
    )

# ----------------------------- Chunkify -------------------------------------

def _zeros_like(arr: np.ndarray) -> np.ndarray:
    return np.zeros_like(arr)


def chunkify(dataset: Iterator[samplers.Feedback], chunk_length: int):
    """Generator of fixed-length chunks from full-trajectory samples.

    Args:
      dataset: full-sample dataset as numpy iterator.
      chunk_length: time length of chunks.
    Yields:
      Fixed-timelength chunks of data. Each tensor of inputs, hints and outputs
      has dimensions [chunk_length, batch_size, ...]. `is_first`/`is_last` mark
      sample boundaries.
    """
    def _get_batch():
        d = next(dataset)
        return (list(d.features.inputs), list(d.features.hints), list(d.outputs), d.features.lengths.astype(int))

    inputs, hints, outputs, lengths = _get_batch()

    # Infer batch_size from first NODE/EDGE input (same as original)
    batch_size = None
    for inp in inputs:
        if inp.location in [specs.Location.NODE, specs.Location.EDGE]:
            batch_size = inp.data.shape[0]
            break
    assert batch_size is not None

    # Initialize empty chunk buffers (match tree_map behavior)
    def _mk_io_chunk(dp: probing.DataPoint):
        x = dp.data
        return np.zeros((chunk_length,) + x.shape, dtype=x.dtype)

    def _mk_hint_chunk(dp: probing.DataPoint):
        x = dp.data  # [T,B,...]
        return np.zeros((chunk_length,) + x.shape[1:], dtype=x.dtype)

    chunk_inputs = [_mk_io_chunk(dp) for dp in inputs]
    chunk_outputs = [_mk_io_chunk(dp) for dp in outputs]
    chunk_hints = [_mk_hint_chunk(dp) for dp in hints]

    # queues of pending batches to fill chunks
    inputs_q: List[List[probing.DataPoint]] = [inputs]
    hints_q: List[List[probing.DataPoint]] = [hints]
    outputs_q: List[List[probing.DataPoint]] = [outputs]
    left: List[np.ndarray] = [lengths.copy()]
    lengths_q: List[np.ndarray] = [lengths.copy()]

    while True:
        # Reset chunk buffers
        chunk_inputs = [np.zeros_like(x) for x in chunk_inputs]
        chunk_outputs = [np.zeros_like(x) for x in chunk_outputs]
        chunk_hints = [np.zeros_like(x) for x in chunk_hints]
        start_mark = np.zeros((chunk_length, batch_size), dtype=int)
        end_mark = np.zeros((chunk_length, batch_size), dtype=int)

        # Gather enough full batches to fill a chunk
        while np.any(np.sum(left, axis=0) < chunk_length):
            inp, hh, out, ll = _get_batch()
            inputs_q.append(inp)
            hints_q.append(hh)
            outputs_q.append(out)
            left.append(ll.copy())
            lengths_q.append(ll.copy())

        # Fill the chunk per batch element
        for i in range(batch_size):
            total, idx = 0, 0
            while total < chunk_length:
                to_add = min(int(left[idx][i]), chunk_length - total)
                if to_add:
                    start = int(lengths_q[idx][i] - left[idx][i])
                    assert start >= 0
                    # copy inputs/outputs (no time dim)
                    for j, dp in enumerate(inputs_q[idx]):
                        src = dp.data  # [B, ...]
                        dest = chunk_inputs[j]
                        assert np.all(dest[total:, i:] == 0)
                        dest[total:total + to_add, i] = src[i]
                    for j, dp in enumerate(outputs_q[idx]):
                        src = dp.data  # [B, ...]
                        dest = chunk_outputs[j]
                        assert np.all(dest[total:, i:] == 0)
                        dest[total:total + to_add, i] = src[i]
                    # copy hints (have time dim)
                    for j, dp in enumerate(hints_q[idx]):
                        src = dp.data  # [T, B, ...]
                        dest = chunk_hints[j]
                        assert np.all(dest[total:, i:] == 0)
                        dest[total:total + to_add, i] = src[start:start + to_add, i]
                    if start == 0:
                        start_mark[total, i] = 1
                    total += to_add
                    left[idx][i] -= to_add
                    assert left[idx][i] >= 0
                    if left[idx][i] == 0:
                        end_mark[total - 1, i] = 1
                idx += 1
            assert total == chunk_length

        while left and np.all(left[0] == 0):
            inputs_q.pop(0)
            hints_q.pop(0)
            outputs_q.pop(0)
            left.pop(0)
            lengths_q.pop(0)

        yield samplers.Feedback(
            samplers.FeaturesChunked(tuple(chunk_inputs), tuple(chunk_hints), start_mark, end_mark),
            tuple(chunk_outputs),
        )


# ------------------------ Dataset factory functions -------------------------

def create_dataset(folder: str, algorithm: str, split: str, batch_size: int):
    """Return an *endless* numpy iterator of full-trajectory batches.

    This mirrors the TFDS pipeline in spirit by sampling from CLRS samplers
    with the same split-specific seeds/lengths, repeating forever, and
    batching by `batch_size`.
    Returns: (iterator, num_samples, spec)
    """
    # Build a per-sample generator using CLRS sampler
    num_samples = samplers.CLRS30[split]['num_samples']
    if split != 'train':
        num_samples *= specs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
    sampler, _ = samplers.build_sampler(
        algorithm,
        seed=samplers.CLRS30[split]['seed'],
        num_samples=num_samples,
        length=samplers.CLRS30[split]['length'],
    )

    def _iter():
        while True:
            fb = sampler.next(batch_size=batch_size)
            # Convert to a flat dict as in TFDS sample, then preprocess back
            data = {'input_' + t.name: t.data for t in fb.features.inputs}
            data['lengths'] = fb.features.lengths
            data.update({'output_' + t.name: t.data for t in fb.outputs})
            data.update({'hint_' + t.name: t.data for t in fb.features.hints})
            yield _preprocess(data, algorithm=algorithm)

    return _iter(), num_samples, specs.SPECS[algorithm]


def create_chunked_dataset(folder: str, algorithm: str, split: str, batch_size: int, chunk_length: int):
    """Return an endless iterator of chunked batches + spec.

    Equivalent to original TFDS->as_numpy_iterator()->chunkify pipeline.
    """
    it, _num, _spec = create_dataset(folder, algorithm, split, batch_size)
    return chunkify(it, chunk_length), specs.SPECS[algorithm]
