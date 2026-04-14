# Initial model template for Yeast task. No sklearn dependency.
import torch
import torch.nn as nn


# ===== BEGIN EVOLVE PGN REGION =====
class SimpleMLP(nn.Module):
    """Initial MLP for the Yeast protein localization task.

    Input:  [batch, 8]
    Output: [batch, 10]
    """

    def __init__(self, input_dim: int = 8, num_classes: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.norm1 = nn.LayerNorm(32)
        self.act1 = nn.GELU()

        self.linear2 = nn.Linear(32, 32)
        self.norm2 = nn.LayerNorm(32)
        # Residual projection: input -> second hidden layer (post-norm skip)
        self.res_proj = nn.Linear(input_dim, 32, bias=False)

        # Direct adaptive projection: remove bottleneck compression
        # 32 -> num_classes (saves ~600 params, preserves rank, improves gradient flow)
        self.output_proj = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act1(self.norm1(self.linear1(x)))
        # Post-norm residual: norm(linear2(h1)) + res_proj(x), then SiLU
        h2 = self.linear2(h1)
        h2 = self.norm2(h2) + self.res_proj(x)
        h2 = nn.functional.silu(h2)  # SiLU instead of GELU for better dynamic range on sparse features
        out = self.output_proj(h2)
        return out
# ===== END EVOLVE PGN REGION =====