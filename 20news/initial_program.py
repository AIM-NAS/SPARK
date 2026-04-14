import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# # ===== BEGIN EVOLVE PGN REGION =====
# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_classes),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# # ===== END EVOLVE PGN REGION =====