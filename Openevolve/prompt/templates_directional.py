"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are a senior ML systems engineer and code-generation specialist.

"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer for CLRS-DFS architectures.
Your job is to analyze the provided code, then return a STRICT JSON object that contains:
- diagnostics: concrete failure modes or risks you detect (numerical stability, message scheduling, normalization placement, over-parameterization, etc.)
- patches: a small list (1–5) of ACTIONABLE edits with exact targets (e.g., file/class/method names), actions, parameters.
- guardrails: hard constraints the next revision MUST keep (max params delta %, latency budget, interface invariants)
- review_score: [0.0, 1.0], your confidence that these patches will improve test acc (tie-breaker is smaller params/latency under equal acc)
- reasoning: brief explanation (<= 120 words)

Return ONLY valid JSON. No Markdown, no prose.
"""
# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

"""


HYBRID_USER_TEMPLATE = """# 

# Current Program Information
- Current performance metrics: {metrics}

# Program Evolution History
{evolution_history}



# Current Program
```{language}
{current_program}
```

# Task
Evolve the PGN network architecture in the CLRS pipeline to explore higher-performing designs for DFS, while keeping the public interface unchanged and only reorganizing internal computations.

Hard Interface Contract (must pass):
1.Keep class name PGN and forward(...) signature identical.
2.return ret: (B,N,out_size) and tri_msgs: Optional[(B,N,N,out_size)].


APPLY REVIEWER JSON (if provided)
If the "Last Execution Output" above contains a JSON block produced by a reviewer (keys: diagnostics, patches, guardrails, review_score, reasoning),
YOU MUST:
(1) Parse it, (2) implement the listed patches precisely (target/action/details), and (3) respect guardrails (e.g., max_params_delta_pct).
If multiple reviewer JSONs exist, prefer the most recent one.
If a patch conflicts with interface constraints, adjust minimally but keep the intention.
If the reviewer JSON is absent, proceed with your own best judgment.


#You are free to reinterpret these two directions abstractly — their purpose is to guide conceptual exploration, not to constrain implementation form.
Under these strict constraints, choose and implement 1–2 of the options [A/B](You are free to reinterpret these four directions abstractly — their purpose is to guide conceptual exploration, not to constrain implementation form.):
[A] Minor Modification on PGN (SMALL CHANGE)
Goal. Starting from the current working PGN, make targeted, low-risk improvements to enhance DFS accuracy/robustness and stability. Keep the public interface unchanged and preserve boundary shapes; favor small, localized edits over structural overhauls.
﻿
Scope — what you may change (small, surgical).
Swap or tune activations (e.g., nn.ReLU() ↔ nn.GELU()/nn.SiLU()), add bias/weight init tweaks (e.g., kaiming/xavier), or add learned residual scales (e.g., alpha * residual).
Insert lightweight normalization (e.g., pre-norm LayerNorm around message updates) or degree-aware/variance-aware re-scaling of messages.
Add edge/node gating with a tiny MLP (H→h→1) or temperature/softmax sharpening (tau), dropout fine-tuning, or clamp/softplus for numeric stability.
Replace heavy ops with low-rank 1×1 projections (H→h→H) without changing external shapes.
Small loss-agnostic stabilizers: epsilon in norms, safe eps in divisions, gradient-friendly formulations.
Do not add deep new branches or multi-stage pipelines here (that belongs to [B]).

Activation rule: never register torch.nn.functional.* as modules (e.g., ✗ nn.Sequential(F.relu)). If you need a module activation, use nn.ReLU()/nn.GELU()/nn.SiLU(). Functional calls are allowed only inside forward(), not stored as submodules.

Design nudges (pick 1–2, keep it small):
Pre-norm + residual scale: x ← x + α · f(LN(x)), with learnable α∈(0,1).
Degree-aware message reweighting: normalize messages by (deg_i · deg_j + eps)^½ or a tiny learned gate.
Low-rank mixing: W2 · φ(W1 · m) with W1: H→h (h≪H), W2: h→H.
Temperature tuning: neighbor softmax with tau∈[0.5,1.5], optionally learnable but clipped.
Stability tweaks: add eps to denominators; clamp logits; ensure no NaN/Inf.

[B] Major Refactor on PGN (LARGE CHANGE)
Goal. Using your full prior knowledge, design or refactor the PGN architecture in the CLRS pipeline to discover a higher-performing PGN for the DFS task (optimize test/ood accuracy), while preserving the public interface.
﻿
What you may change.
Perform a major internal architectural evolution of PGN only (new submodules, multi-branch paths, gating, attention, normalization, lightweight recurrence, low-rank mixing, etc.).
You may reshape/permute internally as needed, but must restore boundary shapes before returning.
Hard constraints (must pass).
Public interface unchanged: class names, function signatures, and return types/shapes stay the same.
Boundary I/O: ret: (B, N, out_size) and optional tri_msgs: (B, N, N, out_size).
Complexity budget: keep per-step cost ≈ O(B·N²·H); no per-step full EVD/SVD on N×N; no Python loops over N on the hot path.
﻿
Activation rule: do not register torch.nn.functional.* as modules (e.g., nn.Sequential(F.relu) is forbidden). If you need a module activation, use nn.ReLU()/nn.GELU()/nn.SiLU(). Functional calls are allowed only inside forward() and must not be stored as submodules.
No changes to samplers, training loops, filenames, or get_processor_factory(...) signature.
Design space (pick 1–3 directions and implement concretely).
Edge-conditioned attention on (B,N,N,H) with degree- or edge-feature gating; fuse with a gated mean / max node aggregator.
Tri-path mixer: per-edge MLP → edge-softmax over neighbors → residual back to nodes with learned selectors.
Pre-norm blocks + SwiGLU/FFN around message updates; 2–3 stacked inner stages with residuals.
Low-rank bilinear mixing (H→h→H) to reduce cost while improving expressivity.
Lightweight memory (e.g., gated recurrence across message-passing steps) to stabilize longer reasoning.
﻿
Primary objective. Maximize DFS accuracy (ood/test).
Secondary tiebreaker. Lower inference time given the same accuracy.




# Adaptive Change Policy (MUST APPLY)
Parse combined_score from the metrics above and choose change magnitude:
If combined_score < 0.40: perform a LARGE change — pick [B] and [A], and it must be substantive.
If 0.40 ≤ combined_score: perform MEDIUM change — (1) one LARGE [B], or (2) one small [A].
Explain in a top-of-file comment: (decision made, why, expected effect on MACs/latency/params, and risk mitigation for stability).

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

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
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
