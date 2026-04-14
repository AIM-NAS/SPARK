import torch.nn as nn
def build_policy_net(input_dim, output_dim):
    hidden_dim = 32
    policy_net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )
    return policy_net

