# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Strict PyTorch-aligned port of `baselines_test.py`.
# The logic, assertions, and test structure are kept 1:1 where applicable.
# This version relies solely on the PyTorch port under `clrs_pytorch`.
# ==============================================================================

import copy
import contextlib
import functools
from typing import Any, Dict, Generator, Iterable, Tuple

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

# Use the local PyTorch port (no JAX/Haiku/Chex).
from . import baselines
from . import dataset
from . import probing
from . import processors
from . import samplers
from . import specs


_Array = np.ndarray


# ----------------------------- small helpers ---------------------------------

def _to_numpy(x: Any) -> _Array:
  try:  # torch.Tensor
    import torch  # local import so tests don't require torch when unused
    if isinstance(x, torch.Tensor):
      return x.detach().cpu().numpy()
  except Exception:
    pass
  # numpy array or scalar
  return np.array(x)


def _tree_map(fn, x, y):
  """Map `fn` over two nested dicts/lists/tuples with matching structure."""
  if isinstance(x, dict) and isinstance(y, dict):
    assert x.keys() == y.keys(), "Mismatched param tree keys"
    return {k: _tree_map(fn, x[k], y[k]) for k in x}
  if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
    assert len(x) == len(y), "Mismatched param tree lengths"
    mapped = [_tree_map(fn, a, b) for a, b in zip(x, y)]
    return type(x)(mapped)
  return fn(x, y)


def _tree_flatten(x):
  if isinstance(x, dict):
    out = []
    for k in sorted(x.keys()):
      out.extend(_tree_flatten(x[k]))
    return out
  if isinstance(x, (list, tuple)):
    out = []
    for v in x:
      out.extend(_tree_flatten(v))
    return out
  return [x]


def _error(x, y):
  x = _to_numpy(x)
  y = _to_numpy(y)
  return np.sum(np.abs(x - y))


@contextlib.contextmanager
def _fake_jit():
  """No-op context to mirror `chex.fake_jit()` semantics in tests."""
  yield


# ----------------------------- data utilities --------------------------------

def _make_sampler(algo: str, length: int) -> samplers.Sampler:
  sampler, _spec = samplers.build_sampler(
      algo,
      seed=samplers.CLRS30['val']['seed'],
      num_samples=samplers.CLRS30['val']['num_samples'],
      length=length,
  )
  return sampler


def _without_permutation(feedback: samplers.Feedback) -> samplers.Feedback:
  """Replace SHOULD_BE_PERMUTATION outputs with POINTER (as original test)."""
  outputs = []
  for x in feedback.outputs:
    if x.type_ != specs.Type.SHOULD_BE_PERMUTATION:
      outputs.append(x)
      continue
    assert x.location == specs.Location.NODE
    outputs.append(probing.DataPoint(name=x.name, location=x.location,
                                     type_=specs.Type.POINTER, data=x.data))
  return feedback._replace(outputs=outputs)


def _make_iterable_sampler(
    algo: str, batch_size: int, length: int
) -> Generator[samplers.Feedback, None, None]:
  sampler = _make_sampler(algo, length)
  while True:
    yield _without_permutation(sampler.next(batch_size))


def _remove_permutation_from_spec(spec):
  new_spec = {}
  for k in spec:
    if (spec[k][1] == specs.Location.NODE and
        spec[k][2] == specs.Type.SHOULD_BE_PERMUTATION):
      new_spec[k] = (spec[k][0], spec[k][1], specs.Type.POINTER)
    else:
      new_spec[k] = spec[k]
  return new_spec


# ---------------------------------- tests ------------------------------------
class BaselinesTest(parameterized.TestCase):

  def test_full_vs_chunked(self):
    """Test that chunking does not affect gradients."""

    batch_size = 4
    length = 8
    algo = 'insertion_sort'
    spec = _remove_permutation_from_spec(specs.SPECS[algo])

    rng_key = 42  # integer RNG key for torch port

    full_ds = _make_iterable_sampler(algo, batch_size, length)
    chunked_ds = dataset.chunkify(
        _make_iterable_sampler(algo, batch_size, length), length)
    double_chunked_ds = dataset.chunkify(
        _make_iterable_sampler(algo, batch_size, length), length * 2)

    full_batches = [next(full_ds) for _ in range(2)]
    chunked_batches = [next(chunked_ds) for _ in range(2)]
    double_chunk_batch = next(double_chunked_ds)

    with _fake_jit():  # parity with chex.fake_jit()
      processor_factory = processors.get_processor_factory(
          'mpnn', use_ln=False, nb_triplet_fts=0)
      common_args = dict(processor_factory=processor_factory, hidden_dim=8,
                         learning_rate=0.01,
                         decode_hints=True, encode_hints=True)

      b_full = baselines.BaselineModel(
          spec, dummy_trajectory=full_batches[0], **common_args)
      b_full.init(full_batches[0].features, seed=42)
      full_params = copy.deepcopy(b_full.params)
      full_loss_0 = b_full.feedback(rng_key, full_batches[0])
      b_full.params = copy.deepcopy(full_params)
      full_loss_1 = b_full.feedback(rng_key, full_batches[1])
      new_full_params = copy.deepcopy(b_full.params)

      b_chunked = baselines.BaselineModelChunked(
          spec, dummy_trajectory=chunked_batches[0], **common_args)
      b_chunked.init([[chunked_batches[0].features]], seed=42)
      chunked_params = copy.deepcopy(b_chunked.params)

      # param tree equality at init
      _tree_map(lambda a, b: np.testing.assert_array_equal(_to_numpy(a), _to_numpy(b)),
                full_params, chunked_params)

      chunked_loss_0 = b_chunked.feedback(rng_key, chunked_batches[0])
      b_chunked.params = copy.deepcopy(chunked_params)
      chunked_loss_1 = b_chunked.feedback(rng_key, chunked_batches[1])
      new_chunked_params = copy.deepcopy(b_chunked.params)

      b_chunked.params = copy.deepcopy(chunked_params)
      double_chunked_loss = b_chunked.feedback(rng_key, double_chunk_batch)

    # losses match
    np.testing.assert_allclose(full_loss_0, chunked_loss_0, rtol=1e-4)
    np.testing.assert_allclose(full_loss_1, chunked_loss_1, rtol=1e-4)
    np.testing.assert_allclose(full_loss_0 + full_loss_1,
                               2 * double_chunked_loss,
                               rtol=1e-4)

    # gradients non-zero on full pass
    param_change = np.mean([
        _error(_to_numpy(a), _to_numpy(b))
        for a, b in zip(_tree_flatten(full_params), _tree_flatten(new_full_params))
    ])
    self.assertGreater(param_change, 0.1)

    # full vs chunked params identical after equivalent updates
    _tree_map(lambda a, b: np.testing.assert_allclose(_to_numpy(a), _to_numpy(b), rtol=1e-4),
              new_full_params, new_chunked_params)

  def test_multi_vs_single(self):
    """Test that multi = single when training one algorithm only."""

    batch_size = 4
    length = 16
    algos = ['insertion_sort', 'activity_selector', 'bfs']
    spec = [_remove_permutation_from_spec(specs.SPECS[algo]) for algo in algos]
    rng_key = 42

    full_ds = [_make_iterable_sampler(algo, batch_size, length) for algo in algos]
    full_batches = [next(ds) for ds in full_ds]
    full_batches_2 = [next(ds) for ds in full_ds]

    with _fake_jit():
      processor_factory = processors.get_processor_factory(
          'mpnn', use_ln=False, nb_triplet_fts=0)
      common_args = dict(processor_factory=processor_factory, hidden_dim=8,
                         learning_rate=0.01,
                         decode_hints=True, encode_hints=True)

      b_single = baselines.BaselineModel(
          spec[0], dummy_trajectory=full_batches[0], **common_args)
      b_multi = baselines.BaselineModel(
          spec, dummy_trajectory=full_batches, **common_args)
      b_single.init(full_batches[0].features, seed=0)
      b_multi.init([f.features for f in full_batches], seed=0)

      single_params = []
      single_losses = []
      multi_params = []
      multi_losses = []

      single_params.append(copy.deepcopy(b_single.params))
      single_losses.append(b_single.feedback(rng_key, full_batches[0]))
      single_params.append(copy.deepcopy(b_single.params))
      single_losses.append(b_single.feedback(rng_key, full_batches_2[0]))
      single_params.append(copy.deepcopy(b_single.params))

      multi_params.append(copy.deepcopy(b_multi.params))
      multi_losses.append(b_multi.feedback(rng_key, full_batches[0], algorithm_index=0))
      multi_params.append(copy.deepcopy(b_multi.params))
      multi_losses.append(b_multi.feedback(rng_key, full_batches_2[0], algorithm_index=0))
      multi_params.append(copy.deepcopy(b_multi.params))

    # losses equal
    np.testing.assert_array_equal(single_losses, multi_losses)
    # loss decreased
    assert single_losses[1] < single_losses[0]

    # param trees inclusions & equality on shared modules
    def _is_subset(subset: Dict[str, Any], superset: Dict[str, Any]) -> bool:
      return set(subset.keys()).issubset(set(superset.keys()))

    for single, multi in zip(single_params, multi_params):
      assert _is_subset(single, multi)
      for module_name, params in single.items():
        _tree_map(lambda a, b: np.testing.assert_array_equal(_to_numpy(a), _to_numpy(b)),
                  params, multi[module_name])

    # params change only for trained algorithm in multi
    for module_name, params in multi_params[0].items():
      changes = _tree_map(_error, params, multi_params[1][module_name])
      # sum nested dict of scalars
      def _sum_tree(v):
        if isinstance(v, dict):
          return sum(_sum_tree(x) for x in v.values())
        if isinstance(v, (list, tuple)):
          return sum(_sum_tree(x) for x in v)
        return float(v)
      param_change = _sum_tree(changes)
      if module_name in single_params[0]:
        assert param_change > 1e-3
      else:
        assert param_change == 0.0

  @parameterized.parameters(True, False)
  def test_multi_algorithm_idx(self, is_chunked):
    """Test that algorithm selection works as intended."""

    batch_size = 4
    length = 8
    algos = ['insertion_sort', 'activity_selector', 'bfs']
    spec = [_remove_permutation_from_spec(specs.SPECS[algo]) for algo in algos]
    rng_key = 42

    if is_chunked:
      ds = [dataset.chunkify(_make_iterable_sampler(algo, batch_size, length), 2 * length)
            for algo in algos]
    else:
      ds = [_make_iterable_sampler(algo, batch_size, length) for algo in algos]
    batches = [next(d) for d in ds]

    processor_factory = processors.get_processor_factory(
        'mpnn', use_ln=False, nb_triplet_fts=0)
    common_args = dict(processor_factory=processor_factory, hidden_dim=8,
                       learning_rate=0.01,
                       decode_hints=True, encode_hints=True)
    if is_chunked:
      baseline = baselines.BaselineModelChunked(spec, dummy_trajectory=batches, **common_args)
      baseline.init([[f.features for f in batches]], seed=0)
    else:
      baseline = baselines.BaselineModel(spec, dummy_trajectory=batches, **common_args)
      baseline.init([f.features for f in batches], seed=0)

    # compute param changes when training each algorithm
    def _change(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, float]:
      changes: Dict[str, float] = {}
      for module_name, params in x.items():
        diffs = _tree_map(_error, params, y[module_name])
        # sum nested dict
        def _sum_tree(v):
          if isinstance(v, dict):
            return sum(_sum_tree(xx) for xx in v.values())
          if isinstance(v, (list, tuple)):
            return sum(_sum_tree(xx) for xx in v)
          return float(v)
        changes[module_name] = _sum_tree(diffs)
      return changes

    param_changes = []
    for algo_idx in range(len(algos)):
      init_params = copy.deepcopy(baseline.params)
      _ = baseline.feedback(
          rng_key,
          batches[algo_idx],
          algorithm_index=(0, algo_idx) if is_chunked else algo_idx)
      param_changes.append(_change(init_params, baseline.params))

    # Non-changing params correspond to enc/dec of non-trained algorithms
    unchanged = [[k for k in pc if pc[k] == 0] for pc in param_changes]

    def _get_other_algos(algo_idx: int, modules: Iterable[str]):
      return set([k for k in modules if '_construct_encoders_decoders' in k
                  and f'algo_{algo_idx}' not in k])

    for algo_idx in range(len(algos)):
      expected_unchanged = _get_other_algos(algo_idx, baseline.params.keys())
      self.assertNotEqual(len(expected_unchanged), 0)
      self.assertSetEqual(expected_unchanged, set(unchanged[algo_idx]))


if __name__ == '__main__':
  absltest.main()
