import torch
import torch.nn as nn


# ===== BEGIN EVOLVE PGN REGION =====
class ConvBase(nn.Module):
  def __init__(self, input_size=40, output_size=10):
    super(ConvBase, self).__init__()
    self.linear = nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.linear(x)
# ===== END EVOLVE PGN REGION =====

def build_model(input_size=40, output_size=10):
    return ConvBase(output_size=output_size)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(8, 40)
    y = model(x)
    print("Output shape:", y.shape)