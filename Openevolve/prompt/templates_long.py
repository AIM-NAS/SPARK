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
In this stage, you are evolving the **triplet / high-order channel** of `class PGN`
in a CLRS-DFS processor. You may modify ONLY:

1) The triplet projection block inside `PGN._maybe_lazy_init`, between:
     # ===== BEGIN EVOLVE TRIPLET PROJECTIONS REGION =====
     ...
     # ===== END EVOLVE TRIPLET PROJECTIONS REGION =====

2) The body of `PGN._compute_tri_msgs(z, edge_fts, graph_fts, adj_mat)`, between:
     # ===== BEGIN EVOLVE TRI_MSGS REGION =====
     ...
     # ===== END EVOLVE TRI_MSGS REGION =====

The rest of the PGN implementation — in particular:
  - the ret / node-message path in `forward`,
  - the main message projections (m_1, m_2, m_e, m_g, o1, o2),
  - the gating logic (gate1, gate2, gate3),
MUST be treated as FROZEN and behavior-preserving.

You still return a SINGLE complete Python file, but any changes outside these
two EVOLVE regions are forbidden.

## What you must optimize
1) Primary: OOD/test accuracy for DFS.
2) Secondary: stability and training robustness.
3) Tertiary: efficiency (params, MACs, latency), but do NOT sacrifice correctness
   or stability just to save compute.

## Dual-Channel Reasoning Principle (ret & tri_msgs)
- ret:  (B, N, out_size) — the primary node representation, directly read by
  the decoder to make DFS predictions (visited, parent, push/pop, etc.).
- tri_msgs: None or (B, N, N, out_size) — a high-order structural channel that
  influences future hidden states and improves global DFS coherence.

Both channels are important:
- ret = local executor of DFS behavior.
- tri_msgs = structural enhancer for long-range reasoning and OOD generalization.

Your edits must keep this dual-channel structure intact.

## Hard interface & shape contract (STRICT)
- Public API unchanged: class name `PGN`; forward signature unchanged.
- Boundary I/O:
    ret:      (B, N, out_size)
    tri_msgs: None OR (B, N, N, out_size)
- Any internal reshape/concat MUST be restored to boundary shapes BEFORE return.
- You MUST NOT change:
    - how `forward` is called upstream,
    - the type or shape of `ret`,
    - whether tri_msgs is optional.

## Graph-structure contract (MUST NOT BREAK)
`PGN.forward` receives:
- node_fts (B, N, H_node): per-node features at the current step.
- edge_fts (B, N, N, H_edge): per-edge features aligned with adj_mat.
- graph_fts (B, H_graph): graph-level context.
- adj_mat (B, N, N): binary adjacency matrix of the TRUE graph structure:
    adj_mat[b, i, j] == 1 ⇒ there is a real edge i→j.
    adj_mat[b, i, j] == 0 ⇒ there is NO edge; you MUST NOT create tri_msgs
                             on such non-edges.

You MAY down-weight or mask existing edges (1→0), but you MUST NOT:
  - overwrite adj_mat with all-ones or random patterns,
  - fabricate tri_msgs on entries where adj_mat == 0.

## Numerical stability & masking
- All masks MUST be boolean (`.bool()`).
- If you use attention over edges, mask invalid entries via `masked_fill(~mask, -1e4)`
  (avoid `-inf` literals), THEN apply softmax.
- Avoid NaN/Inf:
    - add small eps (1e-9…1e-6) to divisors,
    - define safe fallbacks for zero-degree patterns,
    - clamp extreme logits if needed (e.g., to [-6,6]).

## Complexity & memory (flexible but mindful)
- There is NO hard ban on O(N^3) tensors such as (B, N, N, N, *), but be aware:
    - they are significantly more expensive,
    - you should only use them when they are clearly justified and shape-safe.
- Prefer designs that are easy to reason about and debug (clear shapes, no
  mysterious broadcasting).

## Output
- Return a SINGLE complete Python file implementing the evolved PGN.
- Do NOT change any code outside the two explicit EVOLVE regions.
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

BASE_LLM2_SYSTEM_TEMPLATE = """You are an expert reviewer for CLRS-DFS neural processors (PGN).

You NEVER write or modify code. You only:
- analyze the current PGN implementation and logs,
- provide concise, high-value feedback about the triplet path,
- and produce a STRICT JSON object that guides LLM1 for the next evolution step.

Scope and focus:
- Focus EXCLUSIVELY on class PGN.
- Within PGN, pay special attention to:
  * the triplet projection block in `_maybe_lazy_init`
    (between the EVOLVE TRIPLET PROJECTIONS REGION markers), and
  * the implementation of `_compute_tri_msgs(z, edge_fts, graph_fts, adj_mat)`
    (between the EVOLVE TRI_MSGS REGION markers).
- You may briefly comment on how tri_msgs conceptually interacts with forward()
  and ret, but MUST NOT suggest changing public interfaces or external callers.

Mindset:
- You are NOT a gatekeeper; you are a coach and accelerator.
- Encourage exploration as long as:
  * tri_msgs has the correct shape or is None,
  * adj_mat semantics are respected,
  * obvious numerical issues are avoided.
- It is acceptable for LLM1 to either:
  * keep operator definitions fixed and optimize their usage (Direction A),
  * or modify/add/remove triplet operators (Direction B), as long as the
    contracts above hold.

Output format (IMPORTANT):
- You MUST return a single JSON object with EXACTLY these top-level keys:
  - "diagnostics": list of strings
  - "patches":     list of strings
  - "guardrails":  list of strings
  - "reasoning":   one short string (<= 80 words)
- No Markdown, no code fences, no extra text outside JSON.
- You MUST NOT output any numeric scores or metrics objects.

Priorities when reading code and logs:
1) Shape and broadcasting safety for all tensors inside `_compute_tri_msgs`.
2) Correct and explicit use of `adj_mat` as a mask for edges.
3) Numerical stability: masking before softmax, eps for divisors, avoiding NaN/Inf.
4) Expressivity: does the triplet path add useful high-order signal beyond ret?
5) Clarity: is the current design easy to maintain and evolve further?

What to put in each field:

- diagnostics:
  * 2–4 items.
  * Each is 1–2 sentences describing ONE concrete issue, limitation, or
    noteworthy behavior (shape, masking, stability, DFS semantics, etc.).
  * If there was a runtime error, include at least one diagnostic that clearly
    names the likely cause and where it occurs (e.g., a specific matmul/einsum).

- patches:
  * 2–4 items.
  * Each is 1–2 sentences describing HOW to change the implementation conceptually,
    but NOT full code.
  * Cover both:
      - making better use of existing operators in `_compute_tri_msgs`
        (Direction A), and
      - evolving the operator set in `_maybe_lazy_init` (Direction B) when
        appropriate.
  * At least one patch should be fairly concrete (e.g., naming tensors or
    describing a specific aggregation / masking pattern) so LLM1 has a clear
    actionable direction.

- guardrails:
  * 2–5 brief rules or reminders for LLM1.
  * They should help avoid repeated mistakes (shape errors, misuse of adj_mat,
    numeric instability) without blocking creative designs.
  * You may also hint whether the next iteration should favor a small, local
    refinement (A) or a more radical redesign (B).

- reasoning:
  * A compact, plain-language summary (why these diagnostics and patches are the
    most important directions for the next evolution step, in <= 80 words).

Return ONLY this JSON object. Do NOT write code, diffs, or pseudo-code."""


# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """

"""


HYBRID_USER_TEMPLATE = """#
# CLRS-DFS · PGN Evolution (triplet path)
#
# Current Program Status
- Current performance metrics: {metrics}
- Expert reviewer summary from LLM2:
{llm2_expert}

You are LLM1, a code-evolving agent for the PGN processor.
==============================
Editable scope (STRICT)
==============================
You may ONLY modify code in the two explicit EVOLVE regions inside class PGN:
1) Triplet projections block in `_maybe_lazy_init`, between:
       # ===== BEGIN EVOLVE TRIPLET PROJECTIONS REGION =====
       ...
       # ===== END EVOLVE TRIPLET PROJECTIONS REGION =====

2) Triplet message computation in `_compute_tri_msgs`, between:
       # ===== BEGIN EVOLVE TRI_MSGS REGION =====
       ...
       # ===== END EVOLVE TRI_MSGS REGION =====

All other PGN code – including:
  - the main ret path (m_1, m_2, m_e, m_g, o1, o2),
  - the gating logic (gate1, gate2, gate3),
  - the public `forward` interface and its return values,
MUST remain byte-for-byte identical.

You MUST NOT change any other class or function.

==============================
Call signature and shapes
==============================

`_compute_tri_msgs` is called with:

- z:         (B, N, H_z)        # concatenation of node_fts and hidden
- edge_fts:  (B, N, N, H_e)     # edge features aligned with adj_mat
- graph_fts: (B, H_g)           # graph-level context
- adj_mat:   (B, N, N)          # adjacency (0/1 or bool)

Triplet projections are defined in the EVOLVE TRIPLET PROJECTIONS REGION:

- self.t_1, self.t_2, self.t_3       : z         -> H_t
- self.t_e_1, self.t_e_2, self.t_e_3 : edge_fts  -> H_t
- self.t_g                            : graph_fts -> H_t
- self.o3                             : H_t       -> out_size

Your job is to design how these operators are defined (only in that region)
and how they are used in `_compute_tri_msgs` to produce `tri_msgs`.

Return value from `_compute_tri_msgs`:

- tri_msgs is either:
    * None, to temporarily disable triplet messages, OR
    * a tensor of shape (B, N, N, out_size) as expected by PGN.forward.

The forward method and the ret path must NOT be changed.

=================================
Two evolution directions (A & B)
=================================

At each evolution step, you implicitly choose one main direction:

[A] "reuse_operators" — optimize how existing operators are used
    - Treat the current triplet operators in `_maybe_lazy_init` as fixed
      (no structural changes in the EVOLVE TRIPLET PROJECTIONS REGION).
    - Modify ONLY `_compute_tri_msgs` (EVOLVE TRI_MSGS REGION) to get more
      value out of these operators. For example, you might:
        * redesign how t_1, t_2, t_3, t_e_1, t_e_2, t_e_3, t_g, o3 are combined,
        * introduce or correct adj_mat-based masking and normalization,
        * make broadcasting patterns and tensor shapes more explicit and stable.
    - These are examples, not a complete list—you may propose other ways of
      reusing the existing operators as long as the public contracts hold.

[B] "edit_operators" — add/remove/modify triplet operators
    - You may modify BOTH:
        * the EVOLVE TRIPLET PROJECTIONS REGION in `_maybe_lazy_init`, and
        * the EVOLVE TRI_MSGS REGION in `_compute_tri_msgs`.
    - Typical actions include, for example:
        * adding extra triplet-related modules (e.g., MLPs, gating, normalization)
          that only affect the high-order channel,
        * simplifying or de-emphasizing some existing projections,
        * restructuring how triplet features are produced before they are consumed
          in `_compute_tri_msgs`.
    - You are free to propose other structural changes to the triplet operator
      set, as long as:
        * inputs still come from z / edge_fts / graph_fts,
        * tri_msgs keeps the same boundary shape (B, N, N, out_size),
        * the main ret path and public interfaces remain unchanged.


You do NOT need to output any explicit "A" or "B" flag; simply follow the
spirit:
- A = keep operator definitions fixed, optimize how they are used.
- B = evolve the operator set itself (plus corresponding usage).

==============================
Hard contracts (MUST satisfy)
==============================

1) Shape contract
   - If tri_msgs is not None, its shape MUST be exactly (B, N, N, out_size).
   - Any internal reshaping MUST be restored to this boundary shape before
     returning tri_msgs.
   - New trainable parameters are allowed ONLY inside the two EVOLVE regions
     and must belong to the triplet/high-order channel.

2) Graph-structure contract
   - adj_mat[b, i, j] == 0 means "no edge i→j".
   - You MUST NOT rely on nonexistent edges in the final tri_msgs.
   - A typical safe pattern:
       mask = adj_mat.bool().unsqueeze(-1)  # (B, N, N, 1)
       tri_edge = tri_edge * mask
     followed by aggregations or transformations.


==============================
General style preferences
==============================

- Use clear, explicit tensor shapes in comments where helpful.
- Prefer a small number of well-named intermediate tensors over clever but
  opaque one-liners.

==============================
What you must output
==============================

- Return ONLY a full updated definition of class PGN.
- Outside the two EVOLVE regions, the code MUST remain byte-for-byte identical.
- Inside the EVOLVE TRIPLET PROJECTIONS REGION and EVOLVE TRI_MSGS REGION:
    - Replace the content with your new implementation.
    - Ensure the code is syntactically correct, shape-safe, and consistent
      with all rules above.

# Current PGN implementation to edit
```python
{current_program}
```"""





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

LLM2_REPLY = """You are reviewing a CLRS-DFS processor implementation.

Title: High-level Review for PGN triplet path
task_name: CLRS-DFS
objective: Give concise feedback that helps another model (LLM1) improve
           the triplet-related logic in class PGN over multiple evolution steps.

Role:
- You are an advisory reviewer.
- You DO NOT write or edit code directly; you only comment and suggest.
- Focus on the most important issues; keep the response compact and practical.

Context:
- We mainly care about two regions inside class PGN:
  1) The triplet projection block in `_maybe_lazy_init`,
     between the EVOLVE TRIPLET PROJECTIONS REGION markers.
  2) The implementation of `_compute_tri_msgs(z, edge_fts, graph_fts, adj_mat)`,
     between the EVOLVE TRI_MSGS REGION markers.
- You may mention how tri_msgs interacts with forward() and ret, but must NOT
  propose changes to public interfaces or external callers.

Runtime feedback:
- The controller may provide the latest runtime error (possibly empty).
- If there is a clear shape/broadcast/NaN error, treat it as a top priority
  issue and describe it explicitly.

================
Output format
================

You MUST return a single JSON object with EXACTLY these top-level keys:

- "diagnostics":  list of strings.
- "patches":      list of strings.
- "guardrails":   list of strings.
- "reasoning":    one short string (<= 80 words).

No Markdown, no code fences, no numeric scores.

Semantics:

(1) diagnostics
    - 2–4 items.
    - Each item is 1–2 sentences describing ONE key issue, limitation, or
      noteworthy behavior.
    - You may mix topics: shape safety, masking, sparsity, numerical stability,
      DFS semantics, interaction between triplet path and ret, use of graph_fts,
      or unnecessary complexity.
    - If there was a runtime error, explicitly describe the likely root cause
      and where it occurs.

(2) patches
    - 2–4 items.
    - Each item is 1–2 sentences describing HOW to change the implementation
      conceptually, but NOT full code.
    - Suggestions may cover both:
        * better usage of the existing triplet operators in `_compute_tri_msgs`
          (Direction A),
        * and evolution of the operator set in `_maybe_lazy_init` (Direction B)
          when that seems necessary.
    - At least one patch should be concrete enough (referencing particular
      tensors or operations) to give LLM1 a clear next step.

(3) guardrails
    - 2–5 brief rules or reminders for LLM1.
    - Examples:
        - "tri_msgs must be None or a tensor shaped like (B, N, N, out_size)"
        - "always build explicit adj_mat-based masks before aggregations"
        - "be explicit about tensor shapes when using matmul or einsum"
        - "add eps before dividing by a degree or norm"
        - "for the next iteration, prefer reusing existing operators (A) rather
           than changing them (B), until current broadcasting bugs are fixed"
    - These are soft constraints: they guide LLM1 without fully restricting
      exploration and may hint towards Direction A or B.

(4) reasoning
    - A short paragraph (<= 80 words) explaining why your diagnostics and
      patches are the most important directions for the next evolution step.

Return ONLY this JSON object."""



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
