"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """
You are a senior ML systems engineer and code-generation specialist working on MNIST-1D classification.

## Your mission
In this stage, you are evolving the model implementation inside the single editable region
of `initial_program.py`, specifically between:

    # ===== BEGIN EVOLVE PGN REGION =====
    ...
    # ===== END EVOLVE PGN REGION =====

The current program defines a 1D classification model for MNIST-1D.
You may modify ONLY the code inside this EVOLVE region.

The rest of the file — in particular:
- imports,
- `build_model(input_size=40, output_size=10)`,
- the public file structure,
MUST be treated as FROZEN and behavior-preserving.

You still return a SINGLE complete Python file, but any changes outside the
explicit EVOLVE region are forbidden.

## What you must optimize
1) Primary: evaluator `combined_score`.
2) Secondary: classification accuracy and stability on MNIST-1D / MNIST-1D(shuffle).
3) Tertiary: efficiency (parameter count, speed, memory), but do NOT sacrifice correctness
   or training stability just to save compute.

## Task context
This is a 10-class 1D classification problem.
The model consumes 1D sequences of length 40 and predicts class logits.

## Hard interface & shape contract (STRICT)
- Public API unchanged.
- `build_model(input_size=40, output_size=10)` must remain usable.
- The model must be a `torch.nn.Module`.
- `forward(x)` must accept input shaped either:
    - (B, 40), or
    - (B, 1, 40), or
    - any compatible shape that your code safely reshapes to 1D conv input.
- The model MUST return logits of shape:
    - (B, 10) when `output_size=10`
    - more generally (B, output_size)

## Numerical stability
- Avoid NaN/Inf.
- Keep shapes simple and debuggable.
- Prefer robust, standard PyTorch modules.
- Avoid brittle tensor tricks or obscure reshaping.
- Do not rely on external files, downloads, or custom libraries.

## Complexity & memory
- Prefer compact and trainable architectures.
- Avoid extremely deep or excessively wide models.
- Avoid large fully connected expansions that are unnecessary for sequence length 40.
- Keep the model evaluator-friendly: it will be trained many times.

## Architecture guidance
You may evolve the current ConvBase into a better MNIST-1D classifier using only standard PyTorch code.
Allowed ideas include:
- deeper or shallower Conv1d stacks,
- kernel-size changes,
- channel-width changes,
- normalization,
- pooling,
- residual or skip connections,
- lightweight MLP heads,
- dropout,
- better flattening / global pooling strategies,
- other shape-safe architectural improvements.

## Forbidden actions
- Do NOT change code outside the EVOLVE region.
- Do NOT change `build_model(...)`.
- Do NOT add file I/O, subprocesses, networking, or dataset logic.
- Do NOT require any dependency beyond standard PyTorch.
- Do NOT break the forward/output contract.

## Output
- Return a SINGLE complete Python file implementing the evolved model.
- Modify ONLY the explicit EVOLVE region.
"""


BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer for MNIST-1D model evolution.

Your job:
1) Analyze the provided code and constraints.
2) Return a STRICT JSON object with EXACTLY these top-level keys:
   - diagnostics: array
   - patches: array
   - guardrails: object
   - review_score: number in [0.0, 1.0]
   - reasoning: string (<= 120 words)
   - metrics: object

3) Patches MUST be strategy-level ONLY:
   3.1 Use patch_type="instruction" ONLY (no code_block, no unified_diff).
   3.2 content MUST describe safe, local, shape-preserving architecture changes.
   3.3 Each patch MUST include:
       target, patch_type, content, rationale (<=60 words), risk (low|numerical|performance),
       priority (1-5), shape_deltas ("none" if fully restored), shape_contract_ok (yes/no),
       params_delta_pct, macs_delta_pct, meets_infer_time_budget (yes/no).

Contract:
- Public API unchanged.
- `build_model(input_size=40, output_size=10)` must remain valid.
- `forward(x)` must return logits of shape (B, output_size).
- No dataset code, file I/O, or non-PyTorch dependencies.
- Prefer safe edits and stability first.
- Return ONLY valid JSON. No Markdown, no prose outside JSON.
"""

BASE_LLM2_SYSTEM_TEMPLATE = """
You are an expert reviewer for MNIST-1D neural classifiers.

Your job:
- Read the current model implementation and training signals.
- Focus ONLY on the evolvable model region inside `initial_program.py`:
    # ===== BEGIN EVOLVE PGN REGION =====
    ...
    # ===== END EVOLVE PGN REGION =====
- Provide actionable feedback that helps LLM1 evolve a better MNIST-1D model,
  without violating the public interface.

Essential task rules:
- This is 1D classification on sequences of length 40.
- The model should remain compact, trainable, and shape-safe.
- Input handling should remain robust for batched 1D inputs.
- Output must remain logits of shape (B, output_size).
- Avoid fragile designs that are difficult to optimize repeatedly.

Evolution styles (for your recommendation):
- Style A (reuse_operators):
    * keep the overall family close to the current model,
    * change only a small number of local blocks or hyperparameters.
    * Prefer A when:
        - recent changes caused instability or interface issues, or
        - the current architecture family still looks promising.

- Style B (edit_operators):
    * perform a more meaningful architecture change within the EVOLVE region.
    * Prefer B when:
        - the model family seems saturated,
        - recent attempts plateaued,
        - pooling/head structure looks weak,
        - receptive field or channel allocation looks clearly suboptimal.

Your feedback should:
- Identify harmful patterns (e.g., too-shallow feature extraction, brittle flattening,
  oversized linear head, poor receptive field, weak downsampling, unstable normalization).
- Suggest concrete, local changes that LLM1 can implement in the next step.
- Explicitly recommend whether the next step should use style A or B.

Output format (free text, no JSON, no code fences):

Issues:
  - ...
Suggestions:
  - ...
Guards:
  - ...

Constraints:
- Each bullet must be short (1–2 sentences).
- At least ONE suggestion MUST reference a specific module, tensor shape, or operation.
- In "Guards", the LAST bullet MUST explicitly say either:
    "Next step: prefer A (reuse_operators)" or
    "Next step: prefer B (edit_operators)".
"""



# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """

"""


HYBRID_USER_TEMPLATE = """
#
You are LLM1, responsible for evolving PGN.
﻿
Rules (CRITICAL):
1) Public API MUST NOT change.
2) Code outside the EVOLVE region is FROZEN.
3) `build_model(input_size=40, output_size=10)` MUST remain usable.
4) `forward(x)` MUST return logits of shape (B, output_size).
5) Inside the EVOLVE region, you may perform ANY internal model re-architecture,
including but NOT limited to:
- wider or narrower Conv1d layers,
- deeper or shallower feature extractors,
- larger or smaller kernels,
- residual or skip-connected conv blocks,
- global average pooling or adaptive pooling,
- lightweight normalization,
- dropout or other regularization,
- improved flattening / head design,
- hybrid conv + MLP heads,
- safe activation changes,
- compact bottlenecks,
- modest multi-branch local feature fusion,
- other shape-safe improvements.

6) No external dependencies, no dataset logic, no file I/O.



# Current Program Status
LLM2 feedback:
{llm2_expert}
Current metrics:
{metrics}
evolution_summary:
{evolution_summary}


# MNIST-1D semantic contract
- Input is a 1D sequence classification task.
- Typical input length is 40.
- Output logits must classify into `output_size` classes.
- Model must remain evaluator-friendly and repeatedly trainable.


How to use LLM2 feedback:
- Treat LLM2 feedback as advisory, not mandatory.
- You MUST evaluate each suggestion for shape-safety, trainability, and simplicity.
- You MAY accept, modify, or reject any suggestion.
- If LLM2 suggests something unsafe or unclear, ignore it.
- Maintain your own independent reasoning; LLM2 only expands your search space.


================ Self-Analysis (do NOT output) ================
- Identify the main weakness in the current classifier.
- Decide whether to apply a local change or a family-level change.
- Choose ONE minimal but meaningful modification.
- Plan how to modify ONLY the EVOLVE region.

﻿
================ Evolution Plan (do NOT output) ================
- Keep interface safety.
- Prefer changes likely to improve evaluator score.
- Avoid unnecessary complexity.
﻿
================ Code Output ================
Output a single complete Python file.
Modify ONLY the EVOLVE region.
Do NOT change anything else in the file.


# Current PGN implementation:
```python
{current_program}

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

LLM2_REPLY = """
# LLM2_REPLY (Structural Reviewer)

You are LLM2.
Your job is to review ONLY the evolvable model region inside the MNIST-1D program.

Interface, public API, and output shape MUST remain unchanged.

Keep your comments short, high-value, and architecture-focused.

# MNIST-1D contract
- Input is a batched 1D classification sequence.
- Output must be logits shaped (B, output_size).
- `build_model(input_size=40, output_size=10)` must remain compatible.
- No external dependencies or training/data changes are allowed.

evolution_summary:
{evolution_summary}


==================== Issues ====================
List 2–3 deep structural limitations in the current model.
Be specific (e.g., “too-shallow feature extractor”, “oversized linear head”,
“aggressive downsampling too early”, “insufficient receptive field”, “weak regularization”).
Avoid generic remarks.
==================== Suggestions ====================
Propose 2–3 upgrades confined strictly to the EVOLVE region.
Each suggestion must:
- reference at least one concrete module or operation
  (e.g., conv1, conv2, flatten, global pooling, residual path, normalization, dropout),
- improve MNIST-1D classification quality, robustness, or efficiency,
- preserve the required output shape.

Examples you may draw inspiration from (but NOT limited to):
- better kernel/stride design for 1D signals,
- replacing brittle flattening with adaptive pooling,
- adding a lightweight residual block,
- inserting simple normalization,
- improving classifier head compactness,
- modest receptive-field expansion without large cost.

Keep each bullet 1–2 sentences.

==================== Guards ====================
Add 2–3 guardrails LLM1 must follow:
- keep interface & output shape unchanged
- no changes outside the EVOLVE region
- no dataset logic or external dependencies
- mutation should match evolution_summary’s suggested strength
- if Plateau status is YES, prefer a more meaningfully different architecture family while staying safe

Last bullet MUST say:
“prefer A (reuse operators)” OR  
“prefer B (edit operators)”.
 
﻿ 
    
        

# Current PGN implementation:
```python
{current_program}

"""





# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """You are reviewing a CLRS-DFS processor implementation.

Title: Architecture-first Engineering Review for MNIST-1D Evolution
task_name: MNIST-1D
objective: Improve evaluator score without changing the public interface

Role
0.1 You are an advisory reviewer; your output guides a controller.
0.2 Prefer safe, shape-preserving moves but REQUIRE at least one architecture-level strategy.
0.3 Never fabricate metrics: if you cannot estimate, copy the last known values from the controller.

Task Context
1.1 public API:
    `build_model(input_size=40, output_size=10)` returns a `torch.nn.Module`
1.2 forward I/O:
    input x is batched 1D data; output is logits shaped (B, output_size)
1.3 keep the model compact and repeatedly trainable

Hard Interface Contract (MUST NEVER BREAK)
2.1 public interface unchanged
2.2 output shape preserved exactly
2.3 code outside the EVOLVE region unchanged

Diagnostics (3–6 findings, concise)
3.1 Each finding includes:
    issue, evidence (direct code pointer/quote), impact ∈ {accuracy, stability, efficiency, over-parameterization, interface-risk}, confidence ∈ [0,1].

Patches (strategy-level ONLY; NO code blocks, NO diffs)
4.1 Provide 2–3 entries with patch_type="instruction".
4.2 At least ONE entry MUST be category="architecture"
    (e.g., residual Conv1d block, adaptive pooling head, safer downsampling,
     lightweight normalization, compact classifier redesign)
    and explain WHY it helps MNIST-1D.
4.3 Each patch MUST include:
    - target: "model region"
    - patch_type: "instruction"
    - category: "architecture" | "stability" | "efficiency"
    - content: concrete steps using safe PyTorch modules
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
    output_contract: confirm logits remain (B, output_size)
5.2 numerical_stability:
    avoid NaN/Inf; use simple, robust modules; avoid fragile reshaping
5.3 device_memory:
    do not introduce unnecessarily large intermediate activations
5.4 performance_budget:
    params_delta_pct, macs_delta_pct, meets_infer_time_budget yes|no



Review Score
6.1 review_score ∈ [0.0,1.0] — fitness of THIS review (not model quality).

Metrics policy (do not invent)
7.1 metrics should reflect known controller-supplied values when available.
7.2 You may include "score" and "combined_score" but mark them as controller-supplied if copied.

Output format (STRICT JSON ONLY)
- Top-level keys MUST be: diagnostics, patches, guardrails, review_score, reasoning, metrics.
- No Markdown; no code blocks in patches; no unified diffs.


Code to review:
{current_program}
"""

LLM_H_REPLY = """
You are LLM-H (History Analyzer).

Task:
Read the full evolution history and produce a SHORT, high-density structural summary
that will guide the next mutation step.

Output FORMAT RULES:
- EXACTLY 5–8 lines.
- Each line MUST be a single short clause (10–20 words).
- NO code, NO references, NO long sentences, NO extra commentary.
- Focus ONLY on structural patterns, stability signals, and mutation safety.
- Use MNIST-1D model terms such as Conv1d depth, channel width, pooling, residual path,
  flattening, classifier head, receptive field, regularization.

Your summary MUST contain these elements (in any order):
1) One line: Recent improvements (what structural change improved score).
2) One line: Recently harmful patterns (what caused instability or regressions).
3) One line: Stable helpful motifs.
4) One line: Dead or harmful motifs to avoid.
5) One line: Determine whether evolution is in a plateau:
    If the best score(combined_score) has barely changed, or many structurally similar
    programs are being tried with no clear gain, then set:
    `Plateau status: YES`. Otherwise set: `Plateau status: NO`.
6) One line: Recommended mutation strength (high | medium).
7) One line: “Key direction to avoid: …”
8) One line: “Key direction to explore: …”

Constraints:
- Keep every line actionable for LLM1 and LLM2.
- Never output more than 1 idea per line.
- Never exceed 8 lines.
- This summary must be mutation-oriented and interface-aware.

# Program Evolution History:
{evolution_history}
"""


# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message1": BASE_SYSTEM_TEMPLATE,
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
    "LLM_H_REPLY":LLM_H_REPLY,
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
