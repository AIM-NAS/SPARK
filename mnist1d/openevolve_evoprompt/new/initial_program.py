import torch
import torch.nn as nn


# ===== BEGIN EVOLVE PGN REGION =====
class ConvBase(nn.Module):
    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(linear_in, output_size)
    def forward(self, x, verbose=False):
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1)
        return self.linear(h3)
# ===== END EVOLVE PGN REGION =====

def build_model(input_size=40, output_size=10):
    return ConvBase(output_size=output_size)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(8, 40)
    y = model(x)
    print("Output shape:", y.shape)