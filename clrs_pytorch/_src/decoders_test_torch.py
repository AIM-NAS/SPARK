
# Copyright 2025.
# Converted to PyTorch-aligned tests.

"""Unit tests for `decoders.py` (PyTorch port)."""

from absl.testing import absltest
import numpy as np
import torch

from clrs_pytorch._src import decoders


class DecodersTest(absltest.TestCase):

  def test_log_sinkhorn(self):
    torch.manual_seed(42)
    x = torch.randn(10, 10)
    y = torch.exp(decoders.log_sinkhorn(x, steps=10, temperature=1.0,
                                        zero_diagonal=False,
                                        noise_rng_key=None))
    np.testing.assert_allclose(y.sum(dim=-1).detach().cpu().numpy(), 1., atol=1e-4)
    np.testing.assert_allclose(y.sum(dim=-2).detach().cpu().numpy(), 1., atol=1e-4)

  def test_log_sinkhorn_zero_diagonal(self):
    torch.manual_seed(42)
    x = torch.randn(10, 10)
    y = torch.exp(decoders.log_sinkhorn(x, steps=10, temperature=1.0,
                                        zero_diagonal=True,
                                        noise_rng_key=None))
    np.testing.assert_allclose(y.sum(dim=-1).detach().cpu().numpy(), 1., atol=1e-4)
    np.testing.assert_allclose(y.sum(dim=-2).detach().cpu().numpy(), 1., atol=1e-4)
    np.testing.assert_allclose(torch.diagonal(y, 0).sum().item(), 0., atol=1e-4)


if __name__ == '__main__':
  absltest.main()
