from pathlib import Path


class GetPrompts:
    def __init__(self, template_path=None):
        baseline = ""
        if template_path is not None and Path(template_path).exists():
            baseline = Path(template_path).read_text(encoding="utf-8")

        self.prompt_task = (
            "I need help evolving ONLY the function build_policy_net inside a fixed PyTorch CartPole-v1 policy template. "
            "The candidate will be evaluated by REINFORCE training on CartPole-v1. "
            "The optimization target is combined_score = mean_test_return - 0.0001 * num_params."
        )

        self.prompt_func_name = "build_policy_net"
        self.prompt_func_inputs = ["input_dim", "output_dim"]
        self.prompt_func_outputs = ["policy_net"]

        self.prompt_inout_inf = (
            "Write exactly one Python function named build_policy_net. "
            "It must accept the two inputs input_dim and output_dim. "
            "It must create a torch.nn.Module object stored in a variable named policy_net and finally return policy_net. "
            "The returned module will later be wrapped by a fixed SimplePolicyNet class and must map [B, 4] to raw logits [B, 2]."
        )

        self.prompt_other_inf = f"""
        You are evolving ONLY a restricted function inside a fixed PyTorch template for CartPole-v1 reinforcement learning.

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

        def build_policy_net(input_dim, output_dim):

        REQUIRED STRUCTURE:
        - Define exactly one function named build_policy_net.
        - Inside the function, construct a torch.nn.Module and store it in a variable named policy_net.
        - The final line of the function must be exactly: return policy_net
        - Use exactly one return statement.

        INPUT/OUTPUT REQUIREMENTS:
        - The returned module will receive tensors with shape [B, 4].
        - The returned module must output raw logits with shape [B, 2].
        - Never apply softmax inside the network.
        - Never reference undefined variables.
        - Prefer compact, trainable, stable architectures.
        - You must build policy_net directly with nn.Sequential.
        - Do not define any inner classes.
        - Do not define any helper functions.
        - Do not define any variables outside build_policy_net.
        - Use only simple layers such as nn.Linear, nn.Tanh, nn.ReLU, nn.LayerNorm, nn.Dropout.

        VERY IMPORTANT:
        - The output variable name must be exactly policy_net.
        - The last line of the function must be exactly: return policy_net

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
