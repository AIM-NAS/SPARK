import torch
import torch.nn as nn

# ===== BEGIN EVOLVE PGN REGION =====
def build_candidate_net(input_dim, num_classes, hidden_dim=32, dropout=0.1):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )
    return model
# ===== END EVOLVE PGN REGION =====


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = build_candidate_net(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x):
        return self.net(x)
