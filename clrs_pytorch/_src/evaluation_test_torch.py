
# Copyright 2025.
# Converted to PyTorch-aligned tests.

"""Unit tests for `evaluation.py` (PyTorch port)."""

from absl.testing import absltest
import numpy as np
import torch
import torch.nn.functional as F

from clrs_pytorch._src import evaluation
from clrs_pytorch._src import probing
from clrs_pytorch._src import specs


class EvaluationTest(absltest.TestCase):

  def test_reduce_permutations(self):
    torch.manual_seed(0)
    b = 8
    n = 16
    pred = torch.stack([torch.randperm(n) for _ in range(b)], dim=0)
    heads = torch.randint(low=0, high=n, size=(b,), dtype=torch.long)

    perm = probing.DataPoint(name='test',
                             type_=specs.Type.PERMUTATION_POINTER,
                             location=specs.Location.NODE,
                             data=F.one_hot(pred, n).float())
    mask = probing.DataPoint(name='test_mask',
                             type_=specs.Type.MASK_ONE,
                             location=specs.Location.NODE,
                             data=F.one_hot(heads, n).float())
    output = evaluation.fuse_perm_and_mask(perm=perm, mask=mask)
    expected_output = pred.clone()
    expected_output[torch.arange(b), heads] = heads
    assert output.name == 'test'
    assert output.type_ == specs.Type.POINTER
    assert output.location == specs.Location.NODE
    np.testing.assert_allclose(output.data.detach().cpu().numpy(),
                               expected_output.detach().cpu().numpy())


if __name__ == '__main__':
  absltest.main()
