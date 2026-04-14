import eoh
import eoh.llm.api_general as ag

print("eoh imported from:", eoh.__file__)
print("api_general imported from:", ag.__file__)
from eoh import eoh

import os
from eoh.utils.getParas import Paras

from prob import MNIST1DModelSearch


paras = Paras()
problem_local = MNIST1DModelSearch()

paras.set_paras(
    method="eoh",                 # ['ael', 'eoh']
    problem=problem_local,        # 鏈湴闂瀵硅薄
    llm_api_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
    llm_api_key=" ",
    llm_model="qwen-plus",
    ec_pop_size=4,# number of samples in each population
    ec_n_pop=100,# number of populations
    exp_n_proc=1,# ~~multi~~-core parallel
    exp_debug_mode=True,
    # 浣犵殑鍊欓€変唬鐮佹槸 PyTorch锛屼笉鏄?numba 鍑芥暟锛岃繖閲屽叧鎺?
    eva_numba_decorator=False,
)
print("llm_api_endpoint =", paras.llm_api_endpoint)
print("llm_model =", paras.llm_model)
print("debug_mode =", paras.exp_debug_mode)
evolution = eoh.EVOL(paras)
evolution.run()
