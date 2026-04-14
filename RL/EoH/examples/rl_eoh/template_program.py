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
    """
    Build and return a policy network for CartPole-v1.

    Constraints:
    1. Must return an instance of torch.nn.Module.
    2. Input tensor shape: [batch_size, 4]
    3. Output tensor shape: [batch_size, 2]
    4. Output should be raw logits for 2 discrete actions.
    """
    return SimplePolicyNet(input_dim=input_dim, output_dim=output_dim)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(8, 4)
    y = model(x)
    print("Output shape:", y.shape)
