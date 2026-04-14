#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # 已经设置过



# --- begin local override ---
import sys, os, importlib
ROOT = os.path.abspath(os.path.dirname(__file__))         # .../openevolve_20250918
PKG_NAME = os.path.basename(ROOT)                         # "openevolve_20250918"
PARENT = os.path.dirname(ROOT)                            # 上一级目录

# 让 Python 先搜索上一级目录，从而可按包名导入当前目录
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

try:
    # 以当前目录名作为包名导入（要求当前目录有 __init__.py）
    pkg = importlib.import_module(PKG_NAME)
    # 将其注册为 "openevolve"，把所有 from openevolve... 重定向到这里
    sys.modules["openevolve"] = pkg
except Exception as e:
    print("WARNING: failed to prefer local package:", e, "\nROOT=", ROOT, "\nPARENT=", PARENT, "\nPKG_NAME=", PKG_NAME)
# --- end local override ---


import openevolve
print("openevolve is loaded from:", openevolve.__file__)

import sys
from openevolve.cli import main

if __name__ == "__main__":
    sys.exit(main())
