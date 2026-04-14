import re
import time
from ...llm.interface_LLM import InterfaceLLM

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs):

        # set prompt interface
        #getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking


        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    def get_prompt_i1(self):
        
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it as the complete replacement code for the evolve region. \
Do NOT implement a standalone function unless explicitly required. \
Your output must satisfy the required template constraints, input/output interface, and region-format constraints. "\
+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

        
    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it as the complete replacement code for the evolve region. \
Do NOT implement a standalone function unless explicitly required. \
Your output must satisfy the required template constraints, input/output interface, and region-format constraints. "\
+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_e2(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"\
"Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m1(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it as the complete replacement code for the evolve region. \
Do NOT implement a standalone function unless explicitly required. \
Your output must satisfy the required template constraints, input/output interface, and region-format constraints. "\
+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m2(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it as the complete replacement code for the evolve region. \
Do NOT implement a standalone function unless explicitly required. \
Your output must satisfy the required template constraints, input/output interface, and region-format constraints. "\
+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m3(self,indiv1):
        prompt_content = "First, you need to identify the main components in the function below. \
Next, analyze whether any of these components can be overfit to the in-distribution instances. \
Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. \
Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n"+indiv1['code']+"\n"\
+self.prompt_inout_inf+"\n"+"Do not give additional explanations."
        return prompt_content

    def _get_alg(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)

        def parse_response(resp: str):
            import json
            import ast
            import re

            resp = resp.strip()

            # 默认算法描述
            algorithm = "evolved neural architecture"

            # 去掉 markdown code fence
            if "```" in resp:
                blocks = re.findall(r"```(?:python)?\s*(.*?)```", resp, re.DOTALL)
                if len(blocks) > 0:
                    resp = blocks[0].strip()

            # 先尝试解析 JSON / Python dict 风格输出
            dict_candidates = [resp]
            if resp.startswith("{{") and resp.endswith("}}"):
                dict_candidates.append(resp[1:-1].strip())

            for cand in dict_candidates:
                data = None

                try:
                    data = json.loads(cand)
                except Exception:
                    pass

                if data is None:
                    try:
                        data = ast.literal_eval(cand)
                    except Exception:
                        pass

                if isinstance(data, dict):
                    alg = data.get("algorithm", algorithm)
                    code = data.get("code", None)

                    if isinstance(alg, str) and alg.strip():
                        algorithm = alg.strip()

                    if isinstance(code, str) and code.strip():
                        # 把 \\n、\\t 等反转义成真正代码
                        code = code.encode("utf-8").decode("unicode_escape").strip()
                        return code, algorithm

            # 再尝试提取 build_candidate_net 函数
            func_match = re.search(
                r"(def\s+build_candidate_net\s*\(.*?\):.*?)(?=\n\s*\n|\Z)",
                resp,
                re.DOTALL,
            )
            if func_match:
                code = func_match.group(1).strip()
                return code, algorithm

            # 兼容旧逻辑：ConvBase class
            class_match = re.search(
                r"(class\s+ConvBase\s*\(.*?\)\s*:.*?)(?=\n\s*#\s*=====\s*END|\Z)",
                resp,
                re.DOTALL,
            )
            if class_match:
                code = class_match.group(1).strip()
                return code, algorithm

            # 兼容旧逻辑：__init__ + forward
            method_match = re.search(
                r"((def\s+__init__\s*\(.*?)(?:\n.*?)*?def\s+forward\s*\(.*?)(?:\n.*)*)",
                resp,
                re.DOTALL,
            )
            if method_match:
                code = method_match.group(1).strip()
                return code, algorithm

            # fallback
            fallback = re.search(r"((class|def)\s+.*)", resp, re.DOTALL)
            if fallback:
                code = fallback.group(1).strip()
                return code, algorithm

            return None, None

        code, algorithm = parse_response(response)

        n_retry = 1
        while (code is None or algorithm is None):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 second and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)
            code, algorithm = parse_response(response)

            if n_retry > 3:
                break
            n_retry += 1

        if code is None or algorithm is None:
            raise ValueError(f"Failed to parse LLM response:\n{response}")

        return [code, algorithm]


    def i1(self):

        prompt_content = self.get_prompt_i1()


      
        [code_all, algorithm] = self._get_alg(prompt_content)


        return [code_all, algorithm]
    
    def e1(self,parents):
      
        prompt_content = self.get_prompt_e1(parents)


      
        [code_all, algorithm] = self._get_alg(prompt_content)



        return [code_all, algorithm]
    
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)


      
        [code_all, algorithm] = self._get_alg(prompt_content)



        return [code_all, algorithm]
    
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)


      
        [code_all, algorithm] = self._get_alg(prompt_content)



        return [code_all, algorithm]
    
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)


      
        [code_all, algorithm] = self._get_alg(prompt_content)


        return [code_all, algorithm]
    
    def m3(self,parents):
      
        prompt_content = self.get_prompt_m3(parents)


      
        [code_all, algorithm] = self._get_alg(prompt_content)


        return [code_all, algorithm]