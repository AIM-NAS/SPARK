from implementation import funsearch
import torch
import torch.nn as nn

@funsearch.evolve
def get_model_config(input_dim=4, output_dim=2):
    hidden_dim = 32
    activation = "tanh"
    num_layers = 2
    return {
        "hidden_dim": hidden_dim,
        "activation": activation,
        "num_layers": num_layers,
    }

def build_model(input_dim=4, output_dim=2):
    cfg = get_model_config(input_dim=input_dim, output_dim=output_dim)

    class SimplePolicyNet(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=32, output_dim=2, activation="tanh", num_layers=2):
            super().__init__()

            act_map = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "gelu": nn.GELU,
                "elu": nn.ELU,
            }
            act = act_map.get(activation, nn.Tanh)

            layers = []
            in_dim = input_dim
            for _ in range(max(1, num_layers - 1)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(act())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return SimplePolicyNet(
        input_dim=input_dim,
        hidden_dim=cfg["hidden_dim"],
        output_dim=output_dim,
        activation=cfg["activation"],
        num_layers=cfg["num_layers"],
    )