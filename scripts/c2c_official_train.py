#!/usr/bin/env python
"""Run C2C's official trainer (script/train/SFT_train.py of github.com/thu-nics/C2C, clone in $PREFILL_C2C_DIR) unchanged,
except that torch.autograd.set_detect_anomaly(True), which the script switches on at import, is made a no-op: it is a
debugging aid (records a stack trace per op, ~2x slower) and does not change any number.
Usage: python scripts/c2c_official_train.py --config <recipe.json>"""
import os
import runpy
import sys

import torch

C2C_DIR = os.environ.get("PREFILL_C2C_DIR", "/scratch/drjieliu_root/drjieliu/shengkai/prefillLLM/ext/C2C")
sys.path.insert(0, C2C_DIR)
torch.autograd.set_detect_anomaly = lambda *a, **k: None
script = os.path.join(C2C_DIR, "script", "train", "SFT_train.py")
sys.argv = [script] + sys.argv[1:]
os.chdir(C2C_DIR)
runpy.run_path(script, run_name="__main__")
