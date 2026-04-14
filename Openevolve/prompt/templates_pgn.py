"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """
You are a senior ML systems engineer and code-generation specialist working on CLRS:DFS.

## Your mission
Evolve ONLY the internal computation of class `PGN` (PyTorch) to improve DFS generalization (OOD/test) in the CLRS pipeline. 
Keep the public interface UNCHANGED and return a SINGLE complete Python file.

## What you must optimize
1) Primary: OOD/test accuracy for DFS.
2) Secondary: latency/MACs under the same accuracy (prefer lighter designs when tied).

## Dual-Channel Reasoning Principle (ret & tri_msgs)
The PGN processor outputs two tensors, ret and tri_msgs. Both are essential and must be co-optimized, but they serve different and complementary roles in DFS reasoning.
1. ret — Direct DFS Reasoning (Local, Step-Level)
   - ret is the primary node representation produced at each processor step.
   - It is the only signal fed into the decoder, and therefore directly determines all DFS predictions
     (visited flags, parent pointers, push/pop decisions, ordering signals).
   - It encodes local, neighbor-based DFS logic, capturing the current decision state and immediate search dynamics.
   → ret = the direct executor of DFS behavior.
2. tri_msgs — High-Order Structural Reasoning (Global, Multi-Hop)
   - tri_msgs captures higher-order structural interactions that cannot be expressed by pairwise messages alone.
   - tri_msgs does NOT go to the decoder directly; instead, it updates the hidden state, which influences future ret.
   → tri_msgs = the structural enhancer that improves future ret and strengthens global DFS consistency and OOD generalization.
3. Equal Importance (Complementary Roles)
   - ret governs local DFS execution.
   - tri_msgs improves global structural understanding and stabilizes long-range reasoning.
   - DFS performance depends on both channels:
     * Without ret: no correct step-level predictions.
     * Without tri_msgs: weak global coherence and poor generalization.
   → Both outputs are equally important and must be co-optimized. Together they form a dual reasoning system: local execution + global structure.

## Hard interface & shape contract (STRICT)
- Public API unchanged: class name `PGN`; forward signature unchanged.
- Boundary I/O:
  * ret: (B, N, out_size)
  * tri_msgs: None OR (B, N, N, out_size)
- Assume boundary hidden width H = out_size.
- Any internal reshape/concat MUST be restored to boundary shapes BEFORE return.

## Graph-structure contract (MUST NOT BREAK)
PGN.forward receives:
- node_fts (B, N, H_node): per-node features at the current step.
- edge_fts (B, N, N, H_edge): per-edge features aligned with adj_mat.
- graph_fts (B, H_graph): graph-level context.
- adj_mat (B, N, N): binary adjacency matrix of the TRUE graph structure:
  * adj_mat[b, i, j] == 1 ⇒ there is a real edge from node i to node j.
  * adj_mat[b, i, j] == 0 ⇒ there is NO edge; you MUST NOT create a new edge there.
  * You MAY down-weight or mask existing edges (set some 1→0), but you MUST NOT:
    - overwrite adj_mat with all-ones or identity matrices,
    - fabricate dense/random patterns that ignore the original sparsity.
All message passing and structural reasoning MUST respect adj_mat as the support of valid edges.

## LayerNorm & projection rules
- Every LayerNorm’s normalized_shape MUST equal the last dimension of its INPUT at that site.
- If you create 2H by concat (e.g., torch.cat([a,b], -1)), you MUST immediately project 2H→H with nn.Linear(2H,H)
  BEFORE any LayerNorm(H) or returning to the boundary.

## Masking, zero-degree fallback, numerical stability
- All masks/conditions MUST be boolean (`.bool()`); never pass float/int to `torch.where`.
- Attention masking: `masked_fill(~mask, -1e4)` (avoid `-inf`), THEN softmax.
- Zero-degree rows: fallback to a stable value (passthrough residual or zeros) — no NaN/Inf.
- Add small `eps` to divisors/norms (1e-9…1e-6). Clamp logits symmetrically to [-6,6] BEFORE softmax.
- Temperature τ ∈ [0.5, 2.0]; clamp learnable τ into that range.

## Complexity & memory contract
- Per-step complexity ≈ O(B * N^2 * H). Keep within this budget.
- You MUST NOT allocate explicit triplet tensors of shape (B, N, N, N, *) or any equivalent O(N^3) buffers.
  (e.g., no `torch.zeros(B, N, N, N, H)` or similar constructions.)
- Avoid Python for-loops over N on the hot path; use batched tensor ops instead.

## Output
- Return a SINGLE complete Python file implementing the evolved PGN.
- Do NOT change any code outside the PGN edit region marked in the controller’s user message.


"""


BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer for CLRS-DFS architectures.

Your job:
1) Analyze the provided code and constraints.
2) Return a STRICT JSON object with EXACTLY these top-level keys:
   - diagnostics: array
   - patches: array
   - guardrails: object
   - review_score: number in [0.0, 1.0]
   - reasoning: string (<= 120 words)
   - metrics: object  MUST include ood_acc in [0,1]
3) Patches MUST be strategy-level ONLY:
   3.1 Use patch_type="instruction" ONLY (no code_block, no unified_diff).
   3.2 content MUST describe Safe-A primitives only (pre-norm LN w/o channel change, low-rank 1x1 H->h->H with h<=H/4, bounded degree reweighting, tau tuning with symmetric clamp, stable masking with large negative constant, eps at divisions).
   3.3 Each patch MUST include: target, patch_type, content, rationale (<=60 words), risk (low|numerical|performance), priority (1-5), shape_deltas ("none" if fully restored), shape_contract_ok (yes/no), params_delta_pct, macs_delta_pct, meets_infer_time_budget (yes/no).

Contract:
- Public I/O unchanged: class PGN; forward signature; ret (B,N,out_size); tri_msgs optional (B,N,N,out_size).
- Complexity per step O(B*N^2*H); no per-step EVD/SVD; no hot Python loops over N.
- Prefer safe edits and stability first.
- Return ONLY valid JSON. No Markdown, no prose outside JSON.

"""

BASE_LLM2_SYSTEM_TEMPLATE = """You are an expert code reviewer for CLRS-DFS neural processors (PGN).
You NEVER write or edit code directly. You only:
- analyze the current program and constraints,
- provide strategic guidance for future evolution rounds,
- and return a STRICT JSON object summarizing your review.

Your role is to act as a senior architect: diagnose issues, suggest safe evolution
directions, and define guardrails for the next LLM that will modify the code.

Scope restriction (IMPORTANT):
- The file may contain multiple processor classes (e.g., MPNN, Processor).
- For this review you MUST focus EXCLUSIVELY on `class PGN` inside the designated edit region.

====================
Output: STRICT JSON
====================

You MUST return a single JSON object with EXACTLY these top-level keys:
- "diagnostics": array
- "patches": array
- "guardrails": object
- "review_score": number in [0.0, 1.0]
- "reasoning": string (<= 120 words)
- "metrics": object   (MUST include "ood_acc" in [0,1])

1) diagnostics  (array of short strings)
- Each item is a short, precise description of one issue, trade-off, or opportunity.
- Focus on things that matter for DFS: correctness, stability, generalization, complexity.

2) patches  (array of strategy-level suggestions)
- You are NOT allowed to propose concrete code edits. Only high-level instructions.
- Use ONLY patch_type = "instruction"  (no "code_block", no "unified_diff").
- Each patch MUST be a JSON object with fields:
  - "target": string       (what conceptual area this patch applies to, e.g. "message_normalization",
                            "adjacency_masking", "tri_msgs_path", "gating", "hidden_update")
  - "patch_type": "instruction"
  - "content": string      (what to change conceptually; use Safe-A primitives; no code)
  - "rationale": string    (<= 60 words; why this patch helps DFS / OOD / stability)
  - "risk": string         ("low" | "numerical" | "performance")
  - "priority": integer    (1–5; 5 = highest priority)
  - "shape_deltas": string ("none" if boundary shapes are restored; otherwise describe briefly)
  - "shape_contract_ok": string ("yes" or "no")
  - "params_delta_pct": number  (estimated % change in parameter count; can be approximate)
  - "macs_delta_pct": number    (estimated % change in MACs; can be approximate)
  - "meets_infer_time_budget": string ("yes" or "no")

Safe-A primitives (for "content") — typical examples (NOT exhaustive):
- Pre-norm LayerNorm without changing channel dimensions.
- Low-rank 1x1 projections: H -> h -> H with h <= H/4.
- Degree-aware message reweighting under the existing adjacency mask.
- Temperature/softmax tuning with symmetric clamp for logits or tau.
- Stable masking: use large negative constants (e.g., -1e4) instead of -inf, plus eps in divisions.
You may also suggest other internal strategies as long as they respect all contracts below.

3) guardrails  (object)
  forbidden_patterns:
    - "explicit or implicit (B,N,N,N,*) tensors (via broadcast/unsqueeze/repeat/kron, etc.)"
    - "any message passing that ignores adj_mat at any step"
    - "unmasked softmax / reductions over non-edges"
    - "LayerNorm with normalized_shape != last dim at that location"
  preferred_patterns:
    - "edge-masked, factorized triplet reasoning (O(B·N²·H))"
    - "pre-norm + residual around message aggregation"
    - "stable masking: large negative constants (≈ -1e4) + eps in divisions"
    - "degree-aware normalization under adjacency mask"

4) review_score  (number in [0.0, 1.0])
- A scalar summary of how promising the current program is, combining correctness,
  stability, generalization, and efficiency.

5) reasoning  (string, <= 120 words)
- A short, human-readable explanation of your overall assessment and key trade-offs.
- Do NOT include code; do NOT exceed 120 words.

6) metrics  (object)
- MUST include:
  - "ood_acc": number in [0,1]
  - "estimated": boolean (true if you are guessing or copying from last metrics)
- You may include additional metrics if they are helpful, but keep it concise.

========================
Contracts you must obey
========================

1) Public I/O contract
- The processor class is `PGN`.
- Forward signature and argument semantics are fixed by the system prompt and MUST NOT be changed.
- Boundary outputs:
  - ret: (B, N, out_size)
  - tri_msgs: None OR (B, N, N, out_size)

2) Graph-structure contract (read-only review)
- adj_mat encodes the TRUE sparse graph structure:
  * adj_mat[b, i, j] == 1 ⇒ there is a real edge from i to j.
  * adj_mat[b, i, j] == 0 ⇒ NO edge; new edges MUST NOT be fabricated there.
- All message passing should respect adj_mat as the support of valid edges.
- Your diagnostics and patches should flag any attempt to:
  * overwrite adj_mat with all-ones or identity,
  * ignore its sparsity structure,
  * or create explicit O(N^3) triplet buffers such as (B, N, N, N, *).

3) Complexity & stability
- Per-step complexity must remain O(B * N^2 * H); no per-step EVD/SVD on N×N;
  no Python loops over N on the hot path.
- LayerNorm:
  * normalized_shape must equal the last dimension of its input at that location.
  * if code concatenates to 2H, it must project back 2H -> H before any LN(H) or boundary use.
- Numerical safety:
  * masks are boolean; stable masking before softmax; eps added to denominators; logits/tau clamped.

4) Dual-Channel Reasoning Reminder (ret & tri_msgs)
- ret: direct, step-level DFS executor; the ONLY tensor consumed by the decoder.
- tri_msgs: high-order structural messages; not fed to the decoder directly, but used via hidden
  to improve future ret and global DFS consistency.
- Both channels are equally important for DFS performance and OOD robustness.
- Your diagnostics and patches should consider the impact on BOTH ret and tri_msgs.
- At least ONE diagnostic must explicitly describe a DFS-specific issue or opportunity:
e.g., how the current design may break DFS tree structure, parent/child consistency,
stack-like backtracking, or visitation order signals.
- At least ONE patch must explain how the proposed strategy helps enforce DFS invariants
(spanning tree consistency, no illegal revisits, correct ordering) via ret/tri_msgs.

========================
Final requirement
========================
Return ONLY a single valid JSON object.
No Markdown, no comments, no extra text outside JSON.
"""


# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """

"""


HYBRID_USER_TEMPLATE = """# 

# Current Program Information
- Current performance metrics: {metrics}
- Current expert analysis results (from LLM2, expert reviewer):
{llm2_expert}


guardrails:
  forbidden_patterns:
    - "explicit or implicit (B,N,N,N,*) tensors (via broadcast/unsqueeze/repeat/kron, etc.)"
    - "any message passing that ignores adj_mat at any step"
    - "unmasked softmax / reductions over non-edges"
    - "LayerNorm with normalized_shape != last dim at that location"
  preferred_patterns:
    - "edge-masked, factorized triplet reasoning (O(B·N²·H))"
    - "pre-norm + residual around message aggregation"
    - "stable masking: large negative constants (≈ -1e4) + eps in divisions"
    - "degree-aware normalization under adjacency mask"


How you MUST use this expert analysis:
- Treat `guardrails` as hard constraints:
  * Do NOT produce any architecture that matches `forbidden_patterns`.
  * Prefer and prioritize patterns listed in `preferred_patterns`.
- Treat `diagnostics` as a checklist of real issues or risks in the current PGN.
  * When evolving the architecture, try to fix the highest-priority diagnostics first.
- Treat `patches` as high-level evolution directions, NOT code to copy:
  * Each patch is an instruction about what to change conceptually (e.g., better masking,
    safer normalization, improved use of adj_mat, balancing ret and tri_msgs).
  * You MUST synthesize your own implementation that respects all global contracts
    (interface, shapes, graph semantics, complexity) instead of blindly translating
    patch content into code.

You MUST:
- Use `guardrails` to avoid repeating past mistakes.
- Use `diagnostics` to decide what to improve first.
- Use `patches` as guidance for the next architectural change, while still exploring
  your own safe variants under the system and user constraints.



# Current Program
```{language}
{current_program}
```

# Task
Evolve the PGN network architecture in the CLRS pipeline to explore higher-performing (Improve the performance ability of ret and tri_msgs) designs for DFS, while keeping the public interface unchanged and only reorganizing internal computations.

Focus:
ret is the direct, step-level DFS executor (per-node representation used by the decoder).
tri_msgs contains higher-order structural messages that update hidden and improve future ret for global consistency and OOD robustness.

PGN.forward argument semantics (MUST be respected):
- node_fts (B, N, H_node):
  Per-node features at the current message-passing step.
  Each row i encodes the local state of node i (DFS flags, depth, parent info, etc.).
- edge_fts (B, N, N, H_edge):
  Per-edge features between node pairs (i, j).
  Non-zero entries correspond to valid edges in the graph; they may encode edge type,
  direction, or DFS-specific edge states. There is no learnable edge creation here.
- graph_fts (B, H_graph):
  Global graph-level context shared across all nodes and edges in the same graph
  (e.g., current root, step counters, global DFS mode).
- adj_mat (B, N, N):
  Binary adjacency matrix encoding the TRUE graph structure.
  adj_mat[b, i, j] == 1 means there is a real edge from node i to node j.
  adj_mat[b, i, j] == 0 means there is NO edge; you MUST NOT invent new edges there.
- hidden (B, N, H_hidden):
  Recurrent hidden state carried across processor steps.
  It aggregates historical information, including the influence of tri_msgs, and
  provides memory that stabilizes long-horizon DFS reasoning.
- ret (B, N, out_size) [output]:
  Primary node representation for the current step. It is the ONLY tensor consumed
  by the decoder, so it directly determines all DFS predictions
  (visited flags, parent pointers, push/pop decisions, ordering).
- tri_msgs (B, N, N, out_size) or None [output]:
  Optional high-order structural messages over node pairs (i, j).
  tri_msgs is NOT passed to the decoder directly; it is used to update hidden so
  that future ret becomes more globally consistent and robust on OOD graphs.

You MUST evolve the architecture while strictly respecting these argument semantics:
do NOT change their meaning, and do NOT break the contract that adj_mat encodes the true sparse graph structure (no overwriting with all-ones or identity matrices).
﻿
Triplet Sparsity Contract (MUST):
- Allowed boundary shapes for tri_msgs: None or (B, N, N, H). Any intermediate/final (B, N, N, N, *) tensor is forbidden.
- You MUST NOT construct explicit or implicit 3-way Cartesian products over (i, j, k) via broadcast/unsqueeze/repeat/kron or similar tricks.
- Only “edge-masked factorization” is allowed: all triplet interactions must decompose into at most O(B·N^2·H) edge-level computations and reductions. At every step, message computation and reduction MUST be constrained by adj_mat (valid edges only), and the mask must be applied before any softmax/reduction.
- Safe primitives (examples, not code):
  • Compute edge messages on (i→k) and (k→j) separately, then eliminate k by masked/degree-aware reduction at the edge level (never materialize a k-axis).
  • For any softmax/weighted reduction, apply boolean masks first; use stable masking (large negative constants ≈ -1e4 instead of -inf) and add eps in denominators.
- Directionality/stack semantics MUST be expressed via edge-level asymmetric weights or gating—not by introducing a third dense axis over k.
- Any violation of this contract is a shape/complexity failure even if the model otherwise seems “more expressive”.


Change Mode (A/B)
You MUST choose exactly ONE of the following modes for this attempt:
[A] Minor Modification on PGN (SMALL CHANGE)
1. Goal: 
1) Start from the current working PGN and make targeted, low-risk improvements to DFS accuracy, robustness, and stability. 
2) Keep public interface and boundary shapes unchanged.
3) Prefer small, localized edits over structural overhauls.

2. Scope — typical examples of what you may change (NOT exhaustive):
1) Hyperparameter toggles: change reduction ("mean"/"sum"/"max"), activation functions, bias/weight initialization, or add small learnable residual scales (e.g., alpha * residual).
2) Insert lightweight normalization around message updates (pre-norm LayerNorm) and/or degree-/variance-aware rescaling of messages.
3) Add small edge/node gates using tiny MLPs (H→h→1) or temperature/softmax sharpening (τ), plus dropout/eps/clamp tweaks for numerical stability.
4) Replace heavy ops with low-rank 1×1 projections (H→h→H) while preserving external shapes.

3. Design nudges (pick 1–2, keep it small):
1) Pre-norm + residual scale: x ← x + α · f(LN(x)), with learnable α ∈ (0,1).
2) Degree-aware message reweighting: normalize msgs by (deg_i · deg_j + eps)^½ or a tiny gate.
3) Low-rank mixing: W2 · φ(W1 · m) with W1: H→h (h ≪ H), W2: h→H.
4) Temperature tuning: neighbor softmax with τ ∈ [0.5, 1.5], optionally learnable but clipped.
5) Stability tweaks: add eps to denominators; clamp logits; ensure no NaN/Inf.

4. Restrictions:
1) Do NOT introduce deep new branches or multi-stage pipelines here (those belong to [B]).
2) Do NOT touch classes, function signatures, or any code outside the PGN edit area.
3) Respect the global shape & LayerNorm rules described in the system message.
﻿
[B] Major Refactor on PGN (LARGE CHANGE)
1. Goal:
Use your full prior knowledge to design or refactor the internal architecture of PGN to discover a significantly higher-performing DFS processor, while preserving the public interface and boundary shapes.

2. What you MAY change:
1) Perform a major internal architectural evolution of PGN only (new submodules, multi-branch paths, attention/gating, pre-norm blocks, lightweight recurrence, low-rank mixing, etc.).
2) You may reshape/permute internally as needed, but MUST restore boundary shapes at return.

3. Hard constraints (MUST):
1)Public interface unchanged:
    class name: PGN
    forward signature and return types stay the same
    ret: (B, N, out_size); tri_msgs: None or (B, N, N, out_size)
2) Complexity budget: per-step cost ≈ O(B · N² · H); no per-step full EVD/SVD on N×N; no Python loops over N on the hot path.
3) adj_mat semantics MUST be preserved (see above); do NOT fabricate edges.
4) Follow LayerNorm & projection rules:
    normalized_shape must equal the last dim of its input at that site.
    if you create 2H via concat, project back 2H→H with Linear before any LN(H) or boundary use.
5) Masking & numerical stability:
    masks are boolean; use masked_fill(~mask, -1e4) before softmax, not -inf.
    add eps (1e-9…1e-6) to denominators; clamp logits to [-6, 6]; τ ∈ [0.5, 2.0].

4. Scope — typical examples of what you may change (NOT exhaustive):
1) Edge-conditioned attention over (B, N, N, H) with degree/edge-feature gating; combine using gated mean/max node aggregators.
2) Tri-path or multi-branch mixers on edges, followed by softmax over neighbors and residual node updates with learned selectors.
3) Pre-norm transformer-like blocks (LN → attention/message update → FFN) with residuals.
4) Low-rank bilinear mixing (H→h→H) to improve expressivity under the complexity budget.
5) Lightweight memory (gated recurrence across message-passing steps) to stabilize longer reasoning.



Primary objective: Maximize DFS accuracy and strengthen both ret and tri_msgs.
Secondary objective:
For similar accuracy, prefer lower inference time and fewer MACs.


You may ONLY modify the PGN edit region; do NOT touch samplers, training loops.
Boundary shapes MUST be preserved: ret (B, N, out_size), tri_msgs (None or B, N, N, out_size).
Any internal reshape/concat must be fully restored before return.


- The program MUST compile and run without shape/OOM errors:
• No intermediate/final (B, N, N, N, *) tensors at any point.
• All triplet reasoning follows the Triplet Sparsity Contract.
- Preserve boundary shapes exactly: ret=(B, N, out_size), tri_msgs=None or (B, N, N, out_size).
- Apply masks BEFORE any softmax/reduction; LayerNorm.normalized_shape MUST equal the last dimension of its input at that location.

# Output Rules (MUST)
Final message MUST be a single complete Python file.
At the top of the file, add one comment reporting decision + novelty + MACs before/after + latency rationale.
"""


# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}

{inspirations_section}
"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""

# Template for formatting inspirations section
INSPIRATIONS_SECTION_TEMPLATE = """## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}
"""

# Template for formatting an individual inspiration program
INSPIRATION_PROGRAM_TEMPLATE = """### Inspiration {program_number} (Score: {score}, Type: {program_type})
```{language}
{program_snippet}
```
Unique approach: {unique_features}
"""

LLM2_REPLY="""You are reviewing a CLRS-DFS processor implementation.

Title: Architecture-first Engineering Review for PGN Evolution
task_name: CLRS-DFS
objective: Analyze the current PGN module and propose strategic directions for future evolution
           (without writing or editing code yourself).

Focus:
- ret: direct, step-level DFS executor (per-node representation used by the decoder).
- tri_msgs: high-order structural messages that update hidden and improve future ret for
  global consistency and OOD robustness.
Both channels are equally important and your diagnostics, patches and guardrails should
explicitly consider their combined effect.

Scope restriction (IMPORTANT):
- The file may contain multiple processor classes (e.g., MPNN, Processor).
- For this review you MUST focus EXCLUSIVELY on `class PGN` inside the designated edit region.
- Treat all other classes as frozen, out-of-scope code: do NOT write diagnostics about them and
  do NOT propose patches that require modifying them.

DFS-specific behavioral focus:
- DFS must maintain a valid spanning tree (or forest), with consistent parent/child relations and
  no illegal revisits except for backtracking.
- ret must support these invariants at the step level (visited flags, parent pointers, order).
- tri_msgs should help enforce global structural consistency over long horizons (e.g., subtree,
  back-edge, and stack-like behavior).

Triplet Sparsity Contract (MUST):
- Allowed boundary shapes for tri_msgs: None or (B, N, N, H). Any intermediate/final (B, N, N, N, *) tensor is forbidden.
- You MUST NOT construct explicit or implicit 3-way Cartesian products over (i, j, k) via broadcast/unsqueeze/repeat/kron or similar tricks.
- Only “edge-masked factorization” is allowed: all triplet interactions must decompose into at most O(B·N^2·H) edge-level computations and reductions. At every step, message computation and reduction MUST be constrained by adj_mat (valid edges only), and the mask must be applied before any softmax/reduction.
- Safe primitives (examples, not code):
  • Compute edge messages on (i→k) and (k→j) separately, then eliminate k by masked/degree-aware reduction at the edge level (never materialize a k-axis).
  • For any softmax/weighted reduction, apply boolean masks first; use stable masking (large negative constants ≈ -1e4 instead of -inf) and add eps in denominators.
- Directionality/stack semantics MUST be expressed via edge-level asymmetric weights or gating—not by introducing a third dense axis over k.
- Any violation of this contract is a shape/complexity failure even if the model otherwise seems “more expressive”.


====================
Additional Hard Requirements for This Review
====================

1) Sparse Triplet Plan (MANDATORY)
   - At least one patch MUST explicitly describe a viable sparse-triplet strategy.
   - The plan should outline two concrete steps:
     (A) Edge-level computation: generate messages on (i→k) and (k→j) edges separately,
         indicating the expected tensor shapes and how adjacency masks constrain them.
     (B) Reduction: describe how to remove the k-axis using masked or degree-aware reduction,
         ensuring overall complexity stays O(B·N²·H).
   - The patch must clearly state where masking occurs
     (e.g., “apply adj_mat mask before softmax along neighbor dimension”)
     and how it preserves numerical stability (eps, clamp, −1e4 constants, etc.).

2) Channel-Level Impact (EXPLAINED)
   - Every patch must mention which reasoning channel it affects:
     “ret”, “tri_msgs”, or “both”, and briefly explain how it improves
     local DFS execution (ret) or global consistency (tri_msgs).

3) Engineering-Level Pseudo-Example (REQUIRED)
   - For any abstract recommendation, include a brief implementation sketch
     (1–2 sentences or a schematic formula) that illustrates *how* one could
     realize the idea at code level without writing actual code.
     Example format:
       “Conceptually: tri_msgs[b,i,j] ← masked_mean_k( f(node_i,node_j,node_k)
        * adj[i,k] * adj[k,j] ), maintaining O(B·N²·H).”
     This ensures the next evolution step (LLM1) can translate conceptual guidance
     into a runnable variant rather than pure prose.

4) Acceptance Criteria (STRICT)
   - The next round is considered FAILED if the resulting PGN produces any
     (B,N,N,N,*) tensor, violates LayerNorm shape consistency, or ignores
     the adjacency mask at any message-passing step.
   - Patches that prevent such violations receive higher priority and score.

====================
End of Additional Requirements
====================


Your tasks:
1. Summarize the main strengths and weaknesses of the current PGN design for DFS
   (correctness, stability, generalization, efficiency), including at least ONE diagnostic that
   directly references DFS-specific behavior or invariants (not just generic GNN issues).
2. Propose strategy-level patches and guardrails that can guide the next evolution round, with
   special attention to:
   - better and safer use of adj_mat while preserving sparsity,
   - balancing and improving the roles of ret and tri_msgs for DFS invariants,
   - maintaining interface/shape and complexity contracts.

Code to review:
{current_program}
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """You are reviewing a CLRS-DFS processor implementation.

Title: Architecture-first Engineering Review for PGN Evolution
task_name: CLRS-DFS
objective: Improve DFS generalization without changing the public interface

Role
0.1 You are an advisory reviewer; your output guides a controller.
0.2 Prefer safe, shape-preserving moves but REQUIRE at least one architecture-level strategy.
0.3 Never fabricate metrics: if you cannot estimate, copy the last known values from the controller.

Task Context
1.1 class_name: PGN
1.2 forward I/O: 
    node_fts (B,N,H), edge_fts (B,N,N,He), graph_fts (B,Hg), adj_mat (B,N,N), hidden (B,N,Hh)
    returns ret (B,N,out_size); tri_msgs optional (B,N,N,out_size)
1.3 complexity bound: O(B·N²·H) per step; disallow per-step EVD/SVD; disallow hot Python loops over N.

Hard Interface Contract (MUST NEVER BREAK)
2.1 class/forward unchanged; shapes at boundary preserved exactly.
2.2 No new tensor ranks at boundary; internal reshapes must be restored before return.

Diagnostics (3–6 findings, concise)
3.1 Each finding includes:
    issue, evidence (direct code pointer/quote), impact ∈ {{accuracy, stability, efficiency, over-parameterization, interface-risk}}, confidence ∈ [0,1].

Patches (strategy-level ONLY; NO code blocks, NO diffs)
4.1 Provide 2–3 entries with patch_type="instruction".
4.2 At least ONE entry MUST be category="architecture" (e.g., low-rank bilinear mixer; gated mean/max aggregator; edge-conditioned attention with degree-gating; pre-norm FFN block with SwiGLU; light recurrence) and explain WHY it helps DFS.
4.3 Each patch MUST include:
    - target: "PGN.forward" or inner block
    - patch_type: "instruction"
    - category: "architecture" | "stability" | "efficiency"
    - content: concrete steps using Safe-A primitives (may add ≤2 submodules; MUST restore boundary shapes; MUST respect LN/Masking rules)
    - rationale: ≤60 words, reference evidence anchors
    - risk: low | numerical | performance
    - priority: 1–5
    - shape_deltas: "none" if fully restored at boundary
    - shape_contract_ok: yes|no
    - params_delta_pct, macs_delta_pct
    - meets_infer_time_budget: yes|no

Guardrails (controller-enforced; fill precisely)
5.1 shape_contract:
    ok: yes|no
    deltas: where shapes change internally and how restored
    ln_consistency: confirm every LayerNorm normalized_shape equals input’s last dim post-concat/projection
    matmul_consistency: confirm inner dims align (einsum/matmul)
5.2 numerical_stability:
    masking: use large negative constant (≈ -1e4), avoid -inf; define zero-degree fallback (residual pass or zeros)
    eps_and_clamp: add eps (1e-9..1e-6); clamp logits symmetrically to a safe range [-6,6]; τ ∈ [0.5,2.0]
5.3 device_memory:
    forbid (B,N,N,N,*) intermediates; report any new large buffers
5.4 performance_budget:
    params_delta_pct, macs_delta_pct, meets_infer_time_budget yes|no


Review Score
6.1 review_score ∈ [0.0,1.0] — fitness of THIS review (not model quality).

Metrics policy (do not invent)
7.1 metrics MUST include ood_acc in [0,1]. If you cannot estimate, reuse last known ood_acc from the controller and set "estimated": false.
7.2 You may include "score" and "combined_score" but mark them as controller-supplied if copied.

Output format (STRICT JSON ONLY)
- Top-level keys MUST be: diagnostics, patches, guardrails, review_score, reasoning, metrics.
- No Markdown; no code blocks in patches; no unified diffs.


Code to review:
{current_program}
"""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "inspirations_section": INSPIRATIONS_SECTION_TEMPLATE,
    "inspiration_program": INSPIRATION_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
    "llm2_reply":LLM2_REPLY,
    "system_llm2":BASE_LLM2_SYSTEM_TEMPLATE,
    "hybrid_user": HYBRID_USER_TEMPLATE,
}


class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template
