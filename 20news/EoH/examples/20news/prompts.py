from pathlib import Path


class GetPrompts:
    def __init__(self, template_path=None):
        baseline = ""
        if template_path and Path(template_path).exists():
            baseline = Path(template_path).read_text(encoding="utf-8")

        self.prompt_task = (
            "Evolve a neural network builder function for 20 Newsgroups text classification. "
            "Only generate the evolve-region function."
        )

        self.prompt_func_name = "build_candidate_net"
        self.prompt_func_inputs = ["input_dim", "num_classes", "hidden_dim", "dropout"]
        self.prompt_func_outputs = ["nn.Module"]

        self.prompt_inout_inf = """
        
        CRITICAL:

The output MUST start with:

def build_candidate_net

The first character MUST be 'd'.

DO NOT output anything before the function.

The function MUST:
- be <= 8 lines
- contain <= 2 Linear layers
- return nn.Sequential ONLY

If you violate this, your answer is invalid.
        The output MUST start with:

def build_candidate_net

The first character MUST be 'd'.

DO NOT output any text before the function.
DO NOT output explanations.
DO NOT output comments.
DO NOT output curly braces {{ }}.
You are ONLY generating a Python function.
If you output anything other than the function, your answer is invalid.
The function signature must be exactly:

def build_candidate_net(input_dim, num_classes, hidden_dim=128, dropout=0.2):

Inputs:
- input_dim: int
- num_classes: int
- hidden_dim: int
- dropout: float



You MUST output a Python dictionary EXACTLY in this format:

{{
  "algorithm": "",
  "code": "def build_candidate_net(input_dim, num_classes, hidden_dim=128, dropout=0.2):\n    return nn.Sequential(\n        nn.Linear(input_dim, hidden_dim),\n        nn.GELU(),\n        nn.Dropout(dropout),\n        nn.Linear(hidden_dim, num_classes)\n    )"
}}

Rules:
- "algorithm" can be empty string
- "code" MUST contain the full function
- Do NOT output anything else
- Do NOT output text before or after the dictionary

Output:
- MUST return a torch.nn.Module
- The returned module must map [B, input_dim] -> [B, num_classes]
""".strip()

        self.prompt_other_inf = """
        You must output ONLY raw Python code.

        STRICT RULES:
        - DO NOT output JSON
        - DO NOT output a dictionary
        - DO NOT output markdown
        - DO NOT output explanations
        - DO NOT output braces with algorithm descriptions
        - DO NOT output anything before or after the function

        You MUST output exactly one function with this header:

        def build_candidate_net(input_dim, num_classes, hidden_dim=128, dropout=0.2):

        The function must return an nn.Module.
        The returned module must map [B, input_dim] to [B, num_classes].

        A valid example is:

        def build_candidate_net(input_dim, num_classes, hidden_dim=128, dropout=0.2):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
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