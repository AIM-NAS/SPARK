from pathlib import Path


class GetPrompts:
    def __init__(self, template_path=None):
        baseline = ""
        if template_path is not None and Path(template_path).exists():
            baseline = Path(template_path).read_text(encoding="utf-8")

        self.prompt_task = (
            "I need help evolving ONLY the function build_candidate_net inside a fixed PyTorch Yeast classification template. "
            "The candidate will be evaluated on the UCI Yeast protein subcellular localization task. "
            "The optimization target is combined_score = max_val_acc - 0.0001 * num_params."
        )

        self.prompt_func_name = "build_candidate_net"
        self.prompt_func_inputs = ["input_dim", "num_classes", "hidden_dim", "dropout"]
        self.prompt_func_outputs = ["model"]

        self.prompt_inout_inf = (
            "Write exactly one Python function named build_candidate_net. "
            "It must accept the four inputs input_dim, num_classes, hidden_dim, and dropout. "
            "It must create a torch.nn.Module object stored in a variable named model and finally return model. "
            "The returned module must map [B, 8] to raw logits [B, 10]."
        )

        self.prompt_other_inf = f"""
        You are evolving ONLY a restricted function inside a fixed PyTorch template for Yeast protein classification.

        Your output must have exactly two parts:
        1. First line: one-sentence algorithm description inside braces, like:
        {{your one-sentence description here}}
        2. Then output exactly one Python function.

        Your output will replace only the code between:

        # ===== BEGIN EVOLVE PGN REGION =====
        ... your output goes here ...
        # ===== END EVOLVE PGN REGION =====

        The rest of the program is fixed and must NOT be rewritten.

        ABSOLUTE RULES:
        1. The first line must be a one-sentence description inside braces.
        2. After the brace description, output ONLY valid Python code.
        3. Output NO markdown fences.
        4. Output NO explanations other than the first brace description.
        5. Output NO import statements.
        6. Output NO class definitions.
        7. Output NO build_model definition.
        8. Output NO main function.
        9. Output NO testing code.
        10. Output NO placeholders.
        11. Output NO pseudocode.

        YOU MUST OUTPUT EXACTLY ONE COMPLETE FUNCTION WITH THIS HEADER:

        def build_candidate_net(input_dim, num_classes, hidden_dim=32, dropout=0.1):

        REQUIRED STRUCTURE:
        - Define exactly one function named build_candidate_net.
        - Inside the function, construct a torch.nn.Module and store it in a variable named model.
        - The final line of the function must be exactly: return model
        - Use exactly one return statement.

        INPUT/OUTPUT REQUIREMENTS:
        - The returned module will receive tensors with shape [B, 8].
        - The returned module must output raw logits with shape [B, 10].
        - Never apply softmax in the network.
        - Never reference undefined variables.
        - Prefer compact, trainable, stable architectures.
        - You must build model directly with nn.Sequential.
        - Do not define any inner classes.
        - Do not define any helper functions.
        - Do not define any variables outside build_candidate_net.
        - Use only simple layers such as nn.Linear, nn.ReLU, nn.Tanh, nn.LayerNorm, nn.BatchNorm1d, nn.Dropout.
        - Keep the architecture compact, with at most 3 Linear layers.

        VERY IMPORTANT:
        - The output variable name must be exactly model.
        - The last line of the function must be exactly: return model

        Reference template:
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
