import torch
import torch.nn as nn



# ===== BEGIN EVOLVE PGN REGION =====
class SimplePolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


# ===== END EVOLVE PGN REGION =====

def build_model(input_dim=4, output_dim=2):
    """
    Build and return a policy network for CartPole-v1.

    Constraints:
    1. Must return an instance of torch.nn.Module.
    2. Input tensor shape: [batch_size, 4]
    3. Output tensor shape: [batch_size, 2]
    4. Output should be raw logits for 2 discrete actions.
    """
    return SimplePolicyNet()