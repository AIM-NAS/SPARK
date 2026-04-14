# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Strict PyTorch-aligned port of clrs_test.py. The test remains framework-
# agnostic and simply verifies the public import surface. Keep 1:1 API.

from absl.testing import absltest
import clrs_pytorch


class ClrsTest(absltest.TestCase):
  """Test CLRS can be imported correctly."""

  def test_import(self):
    self.assertTrue(hasattr(clrs_pytorch, 'Model'))


if __name__ == '__main__':
  absltest.main()
