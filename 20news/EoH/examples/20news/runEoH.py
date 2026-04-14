import os
from eoh import eoh
from eoh.utils.getParas import Paras
from prob import News20SimpleMLPSearch

paras = Paras()

paras.set_paras(
    method="eoh",
    problem=News20SimpleMLPSearch(),
    llm_api_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
    llm_api_key=" ",
    llm_model="qwen-plus",
    ec_pop_size=2,
    ec_n_pop=100,
    exp_n_proc=1,
    exp_debug_mode=True,
)

evolution = eoh.EVOL(paras)
evolution.run()
