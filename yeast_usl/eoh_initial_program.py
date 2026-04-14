# Initial model template for Yeast task. No sklearn dependency.
import torch
import torch.nn as nn


def build_model(input_dim: int = 8, num_classes: int = 10) -> nn.Module:
    """Initial MLP for the Yeast protein localization task.

    EoH version exposes a function-based interface.
    """
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )
