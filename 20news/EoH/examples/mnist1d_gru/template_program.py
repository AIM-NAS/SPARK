import torch
import torch.nn as nn


# ===== BEGIN EVOLVE PGN REGION =====
class ConvBase(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=6, bidirectional=True):
        super(ConvBase, self).__init__()

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1


        self.linear = nn.Linear(40 * hidden_size * num_dirs, output_size)

    def forward(self, x, h0=None):

        x = x.unsqueeze(-1)

        num_dirs = 2 if self.bidirectional else 1
        if h0 is None:
            h0 = torch.zeros(num_dirs, x.shape[0], self.hidden_size, device=x.device)

        output, _ = self.gru(x, h0)
        output = output.reshape(output.shape[0], -1)
        return self.linear(output)
# ===== END EVOLVE PGN REGION =====

def build_model(input_size=40, output_size=10):
    return ConvBase(input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(8, 40)
    y = model(x)
    print("Output shape:", y.shape)