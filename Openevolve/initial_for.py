from __future__ import annotations
from typing import Tuple, Dict
import torch
import torch.nn as nn

# ===========================================================================
# Evolution contract for the LLM/agent (READ ME):
# Goal: invent a linear operator implementation that uses FEWER scalar
# multiplications than the naive triple-loop, while preserving the I/O contract.
#
# Hard constraints:
# - I/O must stay identical:
#     input:  B x 3 x 32 x 32  (CIFAR-10)
#     output: B x 10 logits
# - Keep the public API identical (build_model() -> (nn.Module, meta: Dict)).
# - Keep the layer linear; bias semantics unchanged if bias=True.
#
# Design notes for evolution:
# - We expose two paths in LinearLoopLayer:
#     * algo="optimized" (default): vectorized dense path (fast baseline)
#     * algo="naive":     triple-for loop (algorithmic baseline)
# - You may REPLACE the "optimized" path with your own lower-multiplication
#   formulation (e.g., factorizations, block/Kronecker, Toeplitz/FFT, etc.).
# - Do NOT change data/evaluator interfaces.
# ===========================================================================

IN_DIM = 3 * 32 * 32
NUM_CLASSES = 10


class LinearLoopLayer(nn.Module):
    """
    Linear transform with two compute paths:

      - algo="optimized" (default): vectorized dense path for speed
      - algo="naive": triple-for loop (B x out x in), as an algorithmic baseline

    You are encouraged to replace the "optimized" path with a formulation that
    reduces the number of scalar multiplications while keeping the same I/O and
    linearity. Do NOT introduce non-linear ops inside this layer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 algo: str = "optimized"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.algo = str(algo)

        # Parameters: standard weight/bias; you may re-parameterize in evolution.
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        if self.bias is not None:
            bound = (self.in_features) ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.reshape(x.size(0), -1)  # [B, IN_DIM]
        B = x.size(0)

        if self.algo == "naive":
            # -------- Naive triple-loop: B x out x in --------
            # This is intentionally slow and serves as an algorithmic baseline.
            # You must replace this with a lower-multiplication formulation
            out = x.new_zeros(B, self.out_features)
            W = self.weight
            b = self.bias
            for bidx in range(B):
                xb = x[bidx]  # [in]
                for j in range(self.out_features):
                    acc = 0.0
                    # multiply-accumulate along in_features
                    for i in range(self.in_features):
                        acc += float(xb[i]) * float(W[j, i])
                    if b is not None:
                        acc += float(b[j])
                    out[bidx, j] = acc
            return out
        else:
            # -------- Optimized/vectorized dense path (baseline) --------
            # You may replace this with a lower-multiplication formulation,
            # but keep it vectorized to pass stage1 latency screening.
            y = x @ self.weight.t()
            if self.bias is not None:
                y = y + self.bias
            return y


class CandidateNet(nn.Module):
    """
    Minimal classifier skeleton intended for topology evolution.

    WHAT MUST STAY:
    - Final output: (B, NUM_CLASSES) logits.
    - build_model() must return (model, meta) with valid signatures.
    """
    def __init__(self, hidden_dim: int = 0, head_algo: str = "optimized"):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.head_algo = str(head_algo)

        if self.hidden_dim > 0:
            # Two-layer reference head (you may replace/extend this topology).
            self.fc1 = LinearLoopLayer(IN_DIM, self.hidden_dim, bias=True, algo=self.head_algo)
            self.act = nn.ReLU(inplace=True)
            self.fc2 = LinearLoopLayer(self.hidden_dim, NUM_CLASSES, bias=True, algo=self.head_algo)
        else:
            # Single-layer reference head (you may replace/extend this topology).
            self.fc = LinearLoopLayer(IN_DIM, NUM_CLASSES, bias=True, algo=self.head_algo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        if self.hidden_dim > 0:
            return self.fc2(self.act(self.fc1(x)))
        else:
            return self.fc(x)


def build_model() -> Tuple[nn.Module, Dict]:
    """
    Factory for the evolutionary loop.

    MUST:
    - Return (model, meta) where meta includes arch_signature/arch_feature_vec.
    - If you add new architectural knobs, report them here so the controller
      can guide search (coverage, anchors, directional edits).
    """
    # Start with a single-layer head and the fast baseline algo.
    model = CandidateNet(hidden_dim=0, head_algo="optimized")

    # -------------------- Report hyper-parameters/knobs -----------------------
    hp: Dict[str, float] = {}
    hp["in_dim"] = IN_DIM
    hp["num_classes"] = NUM_CLASSES
    hp["hidden_dim"] = int(getattr(model, "hidden_dim", 0) or 0)

    # For compatibility with the evaluator’s MAC estimator, provide defaults:
    hp["lowrank_rank"] = 0     # we do not expose this knob by default
    hp["groups"] = 1           # we do not expose this knob by default
    hp["sparsity"] = 1.0       # full density by default

    # Also report which algo is used in the current classification head.
    if hp["hidden_dim"] > 0 and hasattr(model, "fc1"):
        head = model.fc1
    else:
        head = getattr(model, "fc", None)
    algo = getattr(head, "algo", "optimized") if head is not None else "optimized"
    algo_id = 0 if algo == "optimized" else 1  # 0=optimized, 1=naive (numeric for feature vec)

    meta: Dict = {"hyperparams": hp}

    def _count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    # ---------------- Signature / feature vector for controller ---------------
    arch_signature = {
        "in_dim": hp["in_dim"],
        "num_classes": hp["num_classes"],
        "hidden_dim": hp["hidden_dim"],
        "algo": algo,
        "param_count": _count_params(model),
        "layering": "1xLoopLinear" if hp["hidden_dim"] == 0 else "2xLoopLinear",
    }
    arch_feature_vec = [
        float(hp["hidden_dim"]),
        float(hp.get("lowrank_rank", 0)),
        float(hp.get("groups", 1)),
        float(hp.get("sparsity", 1.0)),
        float(hp["in_dim"]),
        float(hp["num_classes"]),
        float(algo_id),
    ]
    meta["arch_signature"] = arch_signature
    meta["arch_feature_vec"] = arch_feature_vec

    return model, meta
