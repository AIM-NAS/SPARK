from pathlib import Path


class GetPrompts:
    def __init__(self, template_path=None):
        baseline = ""
        if template_path is not None and Path(template_path).exists():
            baseline = Path(template_path).read_text(encoding="utf-8")

        self.prompt_task = (
            "I need help evolving ONLY the neural network region of a PyTorch MNIST1D classifier. "
            "The generated region will be inserted into a fixed template program and evaluated on MNIST1D. "
            "The optimization target is combined_score = max_test_acc - 0.0001 * num_params."
        )

        # 这里保留接口字段，兼容 EoH，但不再鼓励输出 build_model
        self.prompt_func_name = "ConvBase"
        self.prompt_func_inputs = ["output_size"]
        self.prompt_func_outputs = ["logits"]

        self.prompt_inout_inf = (
            "You are NOT writing the full program. "
            "You are ONLY writing the code that replaces the evolve region inside the template. "
            "The region must define class ConvBase(nn.Module). "
            "ConvBase must be constructible as ConvBase(output_size=output_size). "
            "Its forward(self, x, verbose=False) must accept x with shape [batch, 40] "
            "and return logits with shape [batch, 10]."
        )

        self.prompt_other_inf = f"""
You are evolving ONLY a restricted code region inside a fixed PyTorch template for MNIST1D classification.

Your output will be inserted into an existing template program between:

# ===== BEGIN EVOLVE PGN REGION =====
... your output goes here ...
# ===== END EVOLVE PGN REGION =====

The rest of the template program is FIXED and must NOT be rewritten.

Your task is to output ONLY a complete replacement of that evolve region.

ABSOLUTE RULES:
1. Output ONLY valid Python code.
2. Output NO markdown fences.
3. Output NO explanations.
4. Output NO comments outside normal Python code.
5. Output NO natural language before or after the code.
6. Output NO import statements.
7. Output NO 'def build_model(...)'.
8. Output NO code outside the evolve region.
9. Output NO main function.
10. Output NO testing code.
11. Output NO placeholders.
12. Output NO pseudocode.

YOU MUST OUTPUT EXACTLY ONE COMPLETE CLASS DEFINITION WITH THIS HEADER:

class ConvBase(nn.Module):

The output MUST begin with exactly:
class ConvBase(nn.Module):

Do NOT output only method bodies.
Do NOT output only __init__ or only forward.
Do NOT omit the class header.

REQUIRED STRUCTURE:
- Define exactly one class named ConvBase that inherits from nn.Module.
- ConvBase must contain:
  1. def __init__(self, output_size, ...)
  2. def forward(self, x, verbose=False)
- The class must be constructible as:
  ConvBase(output_size=output_size)

INPUT/OUTPUT REQUIREMENTS:
- forward(self, x, verbose=False) will receive x with shape [B, 40].
- You must reshape x inside forward as needed.
- The returned tensor MUST have shape [B, 10] when output_size=10.
- The returned value must be a defined tensor expression.
- Never return an undefined variable.
- If you return logits, logits must be explicitly defined earlier.
- The safest pattern is to return self.linear(h) or another explicitly defined tensor.

ALLOWED LIBRARIES:
- You may use only torch and torch.nn symbols already available in the template.
- Do NOT import anything.

FORBIDDEN CONTENT:
- def build_model(
- import torch
- from torch
- if __name__ ==
- print(
- markdown code fences such as ```
- any explanation text
- any text like "Here is the code"
- any algorithm description outside Python code

OPTIMIZATION GOAL:
- Maximize combined_score = max_test_acc - 0.0001 * num_params
- Therefore prefer compact, trainable, shape-consistent models.

SAFETY / VALIDITY REQUIREMENTS:
- The model must be syntactically valid Python.
- The model must be executable.
- The model must not reference undefined variables.
- The model must not rely on unavailable modules.
- The model must not change external interfaces.
- The model must not assume any input length other than 40 unless handled correctly by the architecture.

VERY IMPORTANT:
- Your output is NOT the full file.
- Your output is ONLY the replacement code for the evolve region.
- Do NOT write build_model.
- Do NOT write imports.
- Do NOT write any text before 'class ConvBase(nn.Module):'

A valid output has this exact overall form:

class ConvBase(nn.Module):
    def __init__(self, output_size, ...):
        super(ConvBase, self).__init__()
        ...
    def forward(self, x, verbose=False):
        ...
        return ...

INVALID OUTPUT EXAMPLES:
- outputting explanations before the class
- outputting markdown fences
- outputting def build_model(...)
- outputting only def __init__(...)
- outputting only def forward(...)
- returning logits without defining logits
- returning model
- outputting imports

Now output ONLY the replacement evolve-region code.

{baseline}
""".strip()

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf