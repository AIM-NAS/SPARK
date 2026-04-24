
# Copyright 2025.
# Converted to PyTorch-aligned tests.

"""Unit tests for `losses.py` (PyTorch port)."""

from typing import Generator

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import torch
import torch.nn.functional as F

from clrs_pytorch._src import dataset
from clrs_pytorch._src import losses
from clrs_pytorch._src import probing
from clrs_pytorch._src import samplers
from clrs_pytorch._src import specs

_Array = np.ndarray
_Location = specs.Location


def _make_sampler(algo: str, nb_nodes: int) -> samplers.Sampler:
  sampler, _ = samplers.build_sampler(
      algo,
      seed=samplers.CLRS30['val']['seed'],
      num_samples=samplers.CLRS30['val']['num_samples'],
      length=nb_nodes,
  )
  return sampler


def _make_iterable_sampler(
    algo: str, batch_size: int,
    nb_nodes: int) -> Generator[samplers.Feedback, None, None]:
  sampler = _make_sampler(algo, nb_nodes)
  while True:
    yield sampler.next(batch_size)


def _permute_along_axis(x: torch.Tensor, axis: int, seed: int) -> torch.Tensor:
  torch.manual_seed(seed)
  idx = torch.randperm(x.size(axis))
  return x.index_select(axis, idx)


def _as_pred_data(x, nb_nodes, seed, batch_axis):
  """Fake a prediction from a data point (torch)."""
  data = x.data
  if isinstance(data, np.ndarray):
    data = torch.from_numpy(data)
  # Permute along batch axis to make the prediction different.
  data = _permute_along_axis(data, batch_axis, seed)
  # Extend to one-hot for pointer types.
  if x.type_ == specs.Type.POINTER:
    data = F.one_hot(data.long(), nb_nodes).float()
  return data


def _mask_datapoint(x, seed, t_axis=None):
  """Add some masking to data (torch)."""
  torch.manual_seed(seed)
  data = x.data
  if isinstance(data, np.ndarray):
    data = torch.from_numpy(data)

  if x.type_ == specs.Type.MASK:
    # mask some data at random
    mask_shape = list(data.shape)
    if t_axis is not None:
      mask_shape[t_axis] = 1
    mask = (torch.rand(mask_shape) < 0.2).to(data.dtype)
    data = torch.where(mask.bool(), torch.full_like(data, specs.OutputClass.MASKED), data)
  elif x.type_ in [specs.Type.CATEGORICAL, specs.Type.MASK_ONE]:
    # mask some data at random (all categories together)
    mask_shape = list(data.shape)[:-1]
    if t_axis is not None:
      mask_shape[t_axis] = 1
    mask = (torch.rand(mask_shape) < 0.2).to(data.dtype)
    data = torch.where(mask[..., None].bool(),
                       torch.full_like(data, specs.OutputClass.MASKED), data)
  return probing.DataPoint(name=x.name, location=x.location, type_=x.type_,
                           data=data)


def _create_data(algo, nb_nodes):
  batch_size = 8
  ds = _make_iterable_sampler(algo, batch_size, nb_nodes)
  full_sample = next(ds)

  chunk_length = int(full_sample.features.lengths[0])
  chunked_ds = dataset.chunkify(
      _make_iterable_sampler(algo, batch_size, nb_nodes),
      chunk_length)
  chunk_sample = next(chunked_ds)
  return full_sample, chunk_sample


class FullVsChunkLossesTest(parameterized.TestCase):
  """Test that the full and chunked versions of the losses match."""

  # Test two algorithms with fixed-length, covering all data types
  @parameterized.parameters('dfs', 'floyd_warshall')
  def test_output_loss(self, algo):
    nb_nodes = 16
    full_sample, chunk_sample = _create_data(algo, nb_nodes)

    # Calculate output loss.
    for truth_full, truth_chunked in zip(full_sample.outputs,
                                         chunk_sample.outputs):
      chunk_output_loss = losses.output_loss_chunked(
          truth=_mask_datapoint(truth_chunked, seed=0),
          pred=_as_pred_data(truth_chunked, nb_nodes, 0, 1),
          is_last=chunk_sample.features.is_last,
          nb_nodes=nb_nodes,
      )
      full_output_loss = losses.output_loss(
          truth=_mask_datapoint(truth_full, seed=0),
          pred=_as_pred_data(truth_full, nb_nodes, 0, 0),
          nb_nodes=nb_nodes,
      )
      np.testing.assert_allclose(float(chunk_output_loss), float(full_output_loss), rtol=1e-4)

  @parameterized.parameters('dfs', 'floyd_warshall')
  def test_hint_loss(self, algo):
    nb_nodes = 16
    full_sample, chunk_sample = _create_data(algo, nb_nodes)
    for truth_full, truth_chunked in zip(full_sample.features.hints,
                                         chunk_sample.features.hints):
      # Hints copied exactly
      if isinstance(truth_full.data, torch.Tensor):
        np.testing.assert_array_equal(truth_full.data.detach().cpu().numpy(),
                                      truth_chunked.data.detach().cpu().numpy())
      else:
        np.testing.assert_array_equal(truth_full.data, truth_chunked.data)
      pred = _as_pred_data(truth_chunked, nb_nodes, 0, 1)
      chunk_hint_loss = losses.hint_loss_chunked(
          truth=_mask_datapoint(truth_chunked, seed=1, t_axis=0),
          pred=pred,
          is_first=chunk_sample.features.is_first,
          nb_nodes=nb_nodes,
      )

      # For full, pass list of preds excluding first time step
      if isinstance(pred, torch.Tensor):
        full_preds = list(pred[1:])
      else:
        full_preds = list(pred[1:])
      full_hint_loss = losses.hint_loss(
          truth=_mask_datapoint(truth_full, 1, t_axis=0),
          preds=full_preds,
          lengths=full_sample.features.lengths,
          nb_nodes=nb_nodes,
      )
      np.testing.assert_allclose(float(chunk_hint_loss), float(full_hint_loss), rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
