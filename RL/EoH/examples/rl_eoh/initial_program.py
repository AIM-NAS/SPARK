import torch
import torch.nn as nn


# ===== BEGIN EVOLVE PGN REGION =====
def build_policy_net(input_dim, output_dim):
    hidden_dim = 32
    policy_net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )
    return policy_net
# ===== END EVOLVE PGN REGION =====


class SimplePolicyNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.net = build_policy_net(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


def build_model(input_dim=4, output_dim=2):
    return SimplePolicyNet(input_dim=input_dim, output_dim=output_dim)
