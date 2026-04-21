from __future__ import annotations
from typing import Tuple, Dict
import math
import torch
import torch.nn as nn

# ===========================================================================
# Evolution contract for the LLM/agent (READ ME):
# Goal: evolve BOTH the network topology (CandidateNet) and the linear operator
#       algorithm (LinearLoopLayer) to achieve HIGHER ACCURACY and LOWER LATENCY.
#
# Hard constraints:
# - I/O contract must stay identical:
#     input:  B x 3 x 32 x 32  (CIFAR-10 images)
#     output: B x 10 logits
# - Keep the public API identical (build_model() → (nn.Module, meta: Dict)).
# - You may change internal math/paths as long as the program runs and trains.
#
#
# Reporting for the controller:
# - If you introduce new knobs (e.g., more layers or operators), also reflect them
#   into meta["arch_signature"] / meta["arch_feature_vec"] so the controller can
#   score/guide future edits.
# ===========================================================================

IN_DIM = 3 * 32 * 32
NUM_CLASSES = 10

# If you choose to do A or B, please modify the linearloop layer
class LinearLoopLayer(nn.Module):
    """
    A linear transform with multiple algorithmic compute paths (low-rank / grouped /
    dense), intended as a playground for algorithm-level evolution.

    WHAT MUST STAY:
    - Input/Output tensor shapes per call.
    - Bias semantics if bias=True.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, lowrank_rank: int = 0,
                 groups: int = 1, sparsity: float = 1.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(lowrank_rank)       # EVO knob: >0 activates low-rank path
        self.groups = max(1, int(groups))   # EVO knob: >1 activates grouped path
        self.sparsity = float(sparsity)     # EVO knob: 0<rho<=1 controls sampling

        # Parameters for different paths. You may reparameterize as needed,
        # but keep the forward contract (linear map with optional bias).
        if self.rank > 0:
            # Low-rank: W ≈ V @ U  (out x in) with rank = r
            self.U = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.V = nn.Parameter(torch.empty(self.out_features, self.rank))
            nn.init.kaiming_uniform_(self.U, a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.V, a=5 ** 0.5)
            with torch.no_grad():
                self.U.mul_(1.0 / math.sqrt(self.in_features))
                self.V.mul_(1.0 / math.sqrt(max(1, self.rank)))
            self.weight = None
        else:
            # Dense path: W (out x in)
            self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            self.U = None
            self.V = None

        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        if self.bias is not None:
            bound = (self.in_features) ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B = x.size(0)
            x = x.reshape(B, -1)
        else:
            B = x.size(0)

        rho = max(0.0, min(1.0, self.sparsity))
        keep_stride = 1 if rho >= 0.999 else max(1, int(round(1.0 / max(1e-6, rho))))

        rows = []

        if self.rank > 0:
            # ---- Low-rank path ----
            for b in range(B):
                xb = x[b]

                if keep_stride == 1:
                    t = (self.U * xb.unsqueeze(0)).sum(dim=1)  # [rank]
                else:
                    idx = torch.arange(0, self.in_features, keep_stride, device=xb.device)
                    t = (self.U[:, idx] * xb[idx].unsqueeze(0)).sum(dim=1)


                row = (self.V * t.unsqueeze(0)).sum(dim=1)  # [out_features]
                if self.bias is not None:
                    row = row + self.bias
                rows.append(row)
            return torch.stack(rows, dim=0)

        if self.groups > 1:
            # ---- Grouped dense path ----
            step = (self.in_features + self.groups - 1) // self.groups
            for b in range(B):
                xb = x[b]
                row_elems = []
                for j in range(self.out_features):
                    g = j % self.groups
                    start = g * step
                    end = min(self.in_features, start + step)
                    if keep_stride == 1:
                        s = (xb[start:end] * self.weight[j, start:end]).sum()
                    else:
                        idx = torch.arange(start, end, keep_stride, device=xb.device)
                        s = (xb[idx] * self.weight[j, idx]).sum()
                    if self.bias is not None:
                        s = s + self.bias[j]
                    row_elems.append(s)
                rows.append(torch.stack(row_elems))
            return torch.stack(rows, dim=0)

        # ---- Plain dense path ----
        for b in range(B):
            xb = x[b]
            row_elems = []
            if keep_stride == 1:

                for j in range(self.out_features):
                    s = (xb * self.weight[j]).sum()
                    if self.bias is not None:
                        s = s + self.bias[j]
                    row_elems.append(s)
            else:
                idx_all = torch.arange(0, self.in_features, keep_stride, device=xb.device)
                for j in range(self.out_features):
                    s = (xb[idx_all] * self.weight[j, idx_all]).sum()
                    if self.bias is not None:
                        s = s + self.bias[j]
                    row_elems.append(s)
            rows.append(torch.stack(row_elems))
        return torch.stack(rows, dim=0)

# If you choose to do C or D, please modify the CandidateNet
class CandidateNet(nn.Module):
    """
    Minimal classifier *skeleton* intended for topology evolution.

    WHAT MUST STAY:
    - Final output: (B, NUM_CLASSES) logits.
    - build_model() must still return (model, meta) with valid signatures.
    """
    def __init__(self, hidden_dim: int = 0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        if self.hidden_dim > 0:
            # Two-layer reference head (you may replace/extend this topology).
            self.fc1 = LinearLoopLayer(IN_DIM, self.hidden_dim, bias=True,
                                       lowrank_rank=0, groups=1, sparsity=1.0)
            self.act = nn.ReLU(inplace=True)
            self.fc2 = LinearLoopLayer(self.hidden_dim, NUM_CLASSES, bias=True,
                                       lowrank_rank=0, groups=1, sparsity=1.0)
        else:
            # Single-layer reference head (you may replace/extend this topology).
            self.fc = LinearLoopLayer(IN_DIM, NUM_CLASSES, bias=True,
                                      lowrank_rank=0, groups=1, sparsity=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # You may internal-reshape and use richer modules, but return logits Nx10.
        if x.dim() == 4:
            b = x.size(0)
            x = x.view(b, -1)
        if self.hidden_dim > 0:
            return self.fc2(self.act(self.fc1(x)))
        else:
            return self.fc(x)


from typing import Tuple, Dict

def build_model() -> Tuple[nn.Module, Dict]:
    """
    Factory for the evolutionary loop.
    MUST:
    - Return (model, meta) where meta includes arch_signature/arch_feature_vec.
    - If you add new architectural knobs, report them here so the controller
      can guide search (coverage, anchors, directional edits).
    """
    model = CandidateNet(hidden_dim=0)  # Reference start; evolution may replace.

    # -------------------- Report hyper-parameters/knobs -----------------------
    hp = {}
    hp["in_dim"] = IN_DIM
    hp["num_classes"] = NUM_CLASSES
    hp["hidden_dim"] = int(getattr(model, "hidden_dim", 0) or 0)

    # Expose first linear-like stage knobs (update this if you change topology).
    if hp["hidden_dim"] > 0 and hasattr(model, "fc1"):
        fc_like = model.fc1
    else:
        fc_like = getattr(model, "fc", None)

    if fc_like is not None:
        hp["lowrank_rank"] = int(getattr(fc_like, "rank", 0) or 0)
        hp["groups"] = int(getattr(fc_like, "groups", 1) or 1)
        hp["sparsity"] = float(getattr(fc_like, "sparsity", 1.0))
    else:
        hp["lowrank_rank"] = 0
        hp["groups"] = 1
        hp["sparsity"] = 1.0

    meta = {"hyperparams": hp}

    def _count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    # ---------------- Signature / feature vector for controller ---------------
    arch_signature = {
        "hidden_dim": hp["hidden_dim"],
        "lowrank_rank": int(hp.get("lowrank_rank", 0)),
        "groups": int(hp.get("groups", 1)),
        "sparsity_nonzero_ratio": float(hp.get("sparsity", 1.0)),
        "in_dim": hp["in_dim"],
        "num_classes": hp["num_classes"],
        "param_count": _count_params(model),
        "layering": "1xLoopLinear" if hp["hidden_dim"] == 0 else "2xLoopLinear",
        # If you add new controls (e.g., norm/attention depth), also include them.
    }
    arch_feature_vec = [
        float(hp["hidden_dim"]),
        float(hp.get("lowrank_rank", 0)),
        float(hp.get("groups", 1)),
        float(hp.get("sparsity", 1.0)),
        float(hp["in_dim"]),
        float(hp["num_classes"]),
        # You may append new scaled features for better directional guidance.
    ]
    meta["arch_signature"] = arch_signature
    meta["arch_feature_vec"] = arch_feature_vec

    return model, meta
