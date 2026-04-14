"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are a senior ML systems engineer and code-generation specialist.
Rewrite the given Python program to improve CIFAR-10 performance by evolving both the
network topology (CandidateNet) and the operator algorithm (LinearLoopLayer).
Be boldly innovative: draw on your full machine-learning knowledge (optimization, regularization,
normalization/initialization, vectorization & memory locality, low-rank/sparse/grouped designs,
conv/dwconv stems, token mixing, gating/SE, etc.) to propose novel, well-justified changes.
Prefer original, high-impact ideas over superficial tweaks, and briefly note the rationale in code
comments while strictly honoring the project’s I/O and API constraints.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}



# Current Program
```{language}
{current_program}
```

# Task

You need first create new operation layers, and then you must make the obvious change on the architecture of neural network based on new operation layers. 
You may introduce new operator variants, but you MUST preserve the LinearLoopLayer API and its role in the final model. You can optimize its steps to achieve higher accuracy and fewer multiplications, or modify the network architecture of CandidateNet to achieve higher accuracy and fewer multiplications.
You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.
You need first create new operation layers, and then you must make the obvious change on the architecture of neural network based on new operation layers. 
You may introduce new operator variants, but you MUST preserve the LinearLoopLayer API and its role in the final model. You can optimize its steps to achieve higher accuracy and fewer multiplications, or modify the network architecture of CandidateNet to achieve higher accuracy and fewer multiplications.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""


HYBRID_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}

# Program Evolution History
{evolution_history}



# Current Program
```{language}
{current_program}
```

# Task

Under the following strict constraints, choose below and implement it:
- MUST preserve the LinearLoopLayer (LLL) public API and its role in the final model.
- Training budget is extremely small; prefer changes that yield accuracy gains with minimal retraining.
- Keep I/O: input Bx3x32x32 → logits Bx10; do NOT change dataset/evaluator interfaces.

Your objective is to improve accuracy (primary) while keeping latency/MACs reasonable.  
You may introduce operator variants or lightweight architectural tweaks, but avoid unnecessary large overhauls unless you pick [B] or [D].  
Do not combine “create a new operator” and “major architecture change” unless your chosen option explicitly allows it.


#Decision
Choose operations (pick 1–2):
[A] Modify LinearLoopLayer (small)
[B] Refactor LinearLoopLayer (large, still linear)
[C] Modify CandidateNet (small)
[D] Refactor CandidateNet (large)


Important: The following examples are illustrative and non-exhaustive. 
You are NOT limited to these techniques. If you know a better method that satisfies the constraints 
(strict linearity for LinearLoopLayer; D must include a LinearLoopLayer; I/O Bx3x32x32 -> Bx10), 
prefer your own method over the examples.

Examples for A/B/C/D (read-only, do NOT print in final message)
[A] SMALL — Modify LinearLoopLayer
Meaning:
- Small, localized change to LinearLoopLayer; keep its public API unchanged.
Allowed:
- Tweak existing hyperparameters (rank, groups, sparsity), initialization scale, bias handling.
- Replace Python loops with batched matmul or einsum; minor memory/layout tweaks; light numeric stabilization.
Goals:
- Preserve strict linearity and bias semantics.
- latency same or lower; accuracy may improve slightly but not required.
Not allowed:
- Introducing heavy new modes or changing CandidateNet topology.
[B] LARGE — Refactor LinearLoopLayer (still linear)
Meaning:
- Algorithm-level redesign of LinearLoopLayer for higher efficiency and/or accuracy.
Allowed:
- New parameterizations or modes (low-rank, grouped, sparse, mixed), block-structured or fused paths, precompute/buffered layouts.
- You MAY add internal helper functions or classes to organize the implementation.
- You MAY add new knobs (e.g., lll_mode, rank, groups, sparsity_schedule) and MUST report them in meta["arch_signature"] and meta["arch_feature_vec"].
Constraints:
- Remain strictly linear w.r.t. inputs; bias semantics unchanged; I/O shapes unchanged.
- Include brief in-code comments estimating MACs and explaining expected latency reduction.
Goals:
- Clear latency reduction at similar or lower MACs and/or a meaningful accuracy improvement within reasonable compute.
[C] SMALL — Modify CandidateNet (architecture)
Meaning:
- Simple, efficient architectural tweak; avoid heavy new branches.
Allowed:
- Add BatchNorm, a lightweight residual, a single bottleneck, modest width change (for example <= 25%), minor reordering, light dropout (<= 0.2), activation swap.
Constraints:
- Keep at least one LinearLoopLayer discoverable by meta extraction (typical names: fc1, fc, head, proj).
- MACs change about within ±10%; no heavy attention or large new branches.
Goals:
- Small accuracy gain or stability improvement at similar compute.
[D] LARGE — Refactor CandidateNet (must use LinearLoopLayer)
- If you choose [D], the produced architecture MUST include at least one LinearLoopLayer in the model (for example in the head or a projection module). Omitting LinearLoopLayer makes the result invalid.
Meaning:
- Architecture-level redesign using your knowledge to achieve higher efficiency and/or accuracy, while the network MUST include LinearLoopLayer.
Required:
- The produced model must contain at least one LinearLoopLayer (for example in the head or a projection module).
Allowed:
- Use conv/dwconv/mixer as a stem; earlier downsampling or token mixing; switch to global average pooling with a lightweight head; optional SE/gating; modest depth/width reshaping.
- You MAY add internal helper functions or classes to structure modules.
Constraints:
- Keep input Bx3x32x32 and output logits Bx10.
- Report any new knobs (e.g., stem_kind, depth, se_ratio, mix_kind) in meta["arch_signature"] and meta["arch_feature_vec"].
- Include brief in-code comments estimating MACs and the efficiency rationale; avoid unbounded compute growth.
Goals:
- Clear accuracy improvement at reasonable compute, or similar accuracy with significantly lower latency.

Innovation and combination:
- You may invent new parameterizations, dataflows, or lightweight modules not listed above, as long as constraints are satisfied.
- You may combine techniques (for example, a novel grouped-low-rank operator plus an efficient conv stem), 
provided compute stays within the selection standard’s feasibility.
- If you choose a method not shown in the examples, proceed confidently and implement it; 
briefly document the rationale in code comments near the changed parts.
﻿



# Output Rules (MUST)
- Final message MUST be a single complete Python file (code only). Do NOT wrap it in any markdown fences or add prose.
- The file MUST define a top-level function:
def build_model() -> tuple[nn.Module, dict]
returning (model, meta) with meta["hyperparams"] filled.
- Keep I/O: input Bx3x32x32 → logits Bx10.
- In the code comments, report the decision you have chosen operations, along with the reason and estimation of the before and after multiplication (MAC), and why the delay should decrease.
- You are not restricted to the example techniques. You may add internal helper functions or classes to realize your idea; 
  keep everything in a single file and preserve the build_model() contract.
- Keep the final message code-only; put any minimal rationale as short code comments near the modifications.
- At the top of the file, add one comment reporting decision + novelty + MACs before/after + latency rationale.
- MUST ensure classification head groups divide out_features=10 (groups ∈ {{1,2,5,10}}); otherwise set groups=1 and keep the output shape [B,10].
- MUST return (model, meta) and fill meta['hyperparams'] and meta['arch_signature'].
﻿
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
