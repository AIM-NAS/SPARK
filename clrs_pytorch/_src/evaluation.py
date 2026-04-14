# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# PyTorch/Numpy reimplementation of CLRS evaluation.py with identical API & logic.
# Differences from the original:
# - Accepts both numpy arrays and torch.Tensors inside probing.DataPoint.data.
#   Internally converts to numpy for metric computation to preserve exact math.
# - Keeps function names, signatures, control flow and returned structures
#   strictly aligned with the original implementation.
#
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

from . import probing
from . import specs

_Array = np.ndarray  # for typing parity; we operate on numpy internally
Result = Dict[str, probing.DataPoint]


def _to_np(x):
    """Convert torch.Tensor or numpy array to numpy array without copying."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        # Detach to avoid grad, move to CPU if needed, then to numpy
        return x.detach().cpu().numpy()
    return x


def fuse_perm_and_mask(perm: probing.DataPoint,
                       mask: probing.DataPoint) -> probing.DataPoint:
  """Replace permutation pointers active in the mask with self-pointers.

  Args:
    perm: a node permutation_pointer; data shape is expected to be
      [..., N, N], ideally one-hot over the last two dimensions.
    mask: a mask_one over nodes; data shape is expected to be [..., N].
  Returns:
    A node pointer with shape [..., N].
  """
  assert perm.type_ == specs.Type.PERMUTATION_POINTER
  assert perm.location == specs.Location.NODE
  assert mask.name == perm.name + '_mask'
  assert mask.type_ == specs.Type.MASK_ONE
  assert mask.location == specs.Location.NODE
  perm_data = _to_np(perm.data)
  mask_data = _to_np(mask.data)
  assert perm_data.shape[-1] == perm_data.shape[-2]
  assert perm_data.shape[:-1] == mask_data.shape
  data = np.where(mask_data > 0.5,
                  np.arange(perm_data.shape[-1]),  # self-pointers
                  np.argmax(perm_data, axis=-1))   # original pointers
  return probing.DataPoint(name=perm.name,
                           type_=specs.Type.POINTER,
                           location=perm.location,
                           data=data)


def _reduce_permutations_tuple(
    targets: Tuple[probing.DataPoint, ...]) -> Tuple[probing.DataPoint, ...]:
  """Reduce node pointer + mask_one permutation to just node pointer."""
  out_targets = []
  n_perms = 0
  i = 0
  while i < len(targets):
    truth = targets[i]
    if truth.type_ != specs.Type.PERMUTATION_POINTER:
      out_targets.append(truth)
      i += 1
      continue
    truth_mask = targets[i + 1]
    out_targets.append(fuse_perm_and_mask(truth, truth_mask))
    i += 2
    n_perms += 1

  assert len(out_targets) == len(targets) - n_perms
  return tuple(out_targets)


def _reduce_permutations_dict(predictions: Result) -> Result:
  """Reduce node pointer + mask_one permutation to just node pointer."""
  out_preds = {}
  n_perms = 0
  for k, pred in predictions.items():
    if (k.endswith('_mask') and k[:-5] in predictions and
        predictions[k[:-5]].type_ == specs.Type.PERMUTATION_POINTER):
      # This mask will be processed with its associated permutation datapoint
      continue
    if pred.type_ != specs.Type.PERMUTATION_POINTER:
      out_preds[k] = pred
      continue
    pred_mask = predictions[k + '_mask']
    out_preds[k] = fuse_perm_and_mask(pred, pred_mask)
    n_perms += 1

  assert len(out_preds) == len(predictions) - n_perms
  return out_preds


def evaluate_hints(
    hints: Tuple[probing.DataPoint, ...],
    lengths: _Array,
    hint_preds: List[Result],
) -> Dict[str, _Array]:
  """Evaluate hint predictions."""
  evals = {}
  hints = _reduce_permutations_tuple(hints)
  hint_preds = [_reduce_permutations_dict(h) for h in hint_preds]
  for truth in hints:
    assert truth.name in hint_preds[0]
    eval_along_time = [_evaluate(truth, p[truth.name],
                                 idx=i+1, lengths=lengths)
                       for (i, p) in enumerate(hint_preds)]
    evals[truth.name] = np.sum(
        [x * np.sum(i+1 < lengths)
         for i, x in enumerate(eval_along_time)]) / np.sum(lengths - 1)
    evals[truth.name + '_along_time'] = np.array(eval_along_time)

  # Unlike outputs, hints sometimes include scalars (no meaningful global score).
  return evals


def evaluate(
    outputs: Tuple[probing.DataPoint, ...],
    predictions: Result,
) -> Dict[str, float]:
  """Evaluate output predictions."""
  evals: Dict[str, float] = {}
  outputs = _reduce_permutations_tuple(outputs)
  predictions = _reduce_permutations_dict(predictions)
  for truth in outputs:
    assert truth.name in predictions
    pred = predictions[truth.name]
    evals[truth.name] = _evaluate(truth, pred)
  # Return a single scalar score that is the mean of all output scores.
  evals['score'] = sum([float(v.item() if hasattr(v, 'item') else v)
                        for v in evals.values()]) / len(evals)
  return evals


def _evaluate(truth, pred, idx=None, lengths=None):
  """Evaluate single prediction of hint or output."""
  assert pred.name == truth.name
  assert pred.location == truth.location
  assert pred.type_ == truth.type_

  if truth.type_ not in _EVAL_FN:
    raise ValueError('Invalid type')
  truth_data = _to_np(truth.data)
  pred_data = _to_np(pred.data)
  if idx is not None:
    if np.all(idx >= lengths):
      return 0.
    truth_data = truth_data[idx][idx < lengths]
    pred_data = pred_data[idx < lengths]
  return _EVAL_FN[truth.type_](pred_data, truth_data)


def _eval_one(pred, truth):
  mask = np.all(truth != specs.OutputClass.MASKED, axis=-1)
  return np.sum(
      (np.argmax(pred, -1) == np.argmax(truth, -1)) * mask) / np.sum(mask)


def _mask_fn(pred, truth):
  """Evaluate outputs of type MASK, and account for any class imbalance."""
  mask = (truth != specs.OutputClass.MASKED).astype(np.float32)

  # Use F1 score for the masked outputs to address any imbalance
  tp = np.sum((((pred > 0.5) * (truth > 0.5)) * 1.0) * mask)
  fp = np.sum((((pred > 0.5) * (truth < 0.5)) * 1.0) * mask)
  fn = np.sum((((pred < 0.5) * (truth > 0.5)) * 1.0) * mask)

  # Protect against division by zero
  if tp + fp > 0:
    precision = tp / (tp + fp)
  else:
    precision = np.float32(1.0)
  if tp + fn > 0:
    recall = tp / (tp + fn)
  else:
    recall = np.float32(1.0)

  if precision + recall > 0.0:
    f_1 = 2.0 * precision * recall / (precision + recall)
  else:
    f_1 = np.float32(0.0)

  return f_1

_EVAL_FN = {
    specs.Type.SCALAR:
        lambda pred, truth: np.mean((pred - truth)**2),
    specs.Type.MASK: _mask_fn,
    specs.Type.MASK_ONE:
        _eval_one,
    specs.Type.CATEGORICAL:
        _eval_one,
    specs.Type.POINTER:
        lambda pred, truth: np.mean((pred == truth) * 1.0),
}
