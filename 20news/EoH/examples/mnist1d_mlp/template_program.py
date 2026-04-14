import torch
import torch.nn as nn


# ===== BEGIN EVOLVE PGN REGION =====
class ConvBase(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super(ConvBase, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.linear1(x).relu()
        h = h + self.linear2(h).relu()
        return self.linear3(h)


# ===== END EVOLVE PGN REGION =====

def build_model(input_size=40, output_size=10):
    return ConvBase(output_size=output_size)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(8, 40)
    y = model(x)
    print("Output shape:", y.shape)