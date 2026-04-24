# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Strict PyTorch-aligned port of `processors_test.py`.
# Logic and assertions are kept 1:1 where applicable, without JAX/Haiku/Chex.
# ==============================================================================

from absl.testing import absltest
import torch

from . import processors


class MemnetTest(absltest.TestCase):

  def test_simple_run_and_check_shapes(self):
    batch_size = 64
    vocab_size = 177
    embedding_size = 64
    sentence_size = 11
    memory_size = 320
    linear_output_size = 128
    num_hops = 2
    use_ln = True

    torch.manual_seed(42)

    model = processors.MemNetFull(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        sentence_size=sentence_size,
        memory_size=memory_size,
        linear_output_size=linear_output_size,
        num_hops=num_hops,
        use_ln=use_ln,
    )
    model.eval()

    # Integer token ids (as in original test)
    queries = torch.ones((batch_size, sentence_size), dtype=torch.long)
    stories = torch.ones((batch_size, memory_size, sentence_size), dtype=torch.long)

    with torch.no_grad():
      # The PyTorch port exposes a standard forward; for strict compatibility we
      # call `_apply` if it exists (matching the original test), otherwise fall back
      # to `forward`.
      if hasattr(model, "_apply") and callable(getattr(model, "_apply")):
        out = model._apply(queries, stories)  # type: ignore[attr-defined]
      else:
        out = model(queries, stories)

    # Shape and dtype checks (mirror chex.assert_shape / assert_type)
    self.assertEqual(list(out.shape), [batch_size, vocab_size])
    self.assertEqual(out.dtype, torch.float32)


if __name__ == '__main__':
  absltest.main()
