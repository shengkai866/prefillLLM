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

# Source (sharer / "teacher_model") too big for one GPU, e.g. Qwen3-30B-A3B (61 GB) on 48 GB L40S cards:
# PREFILL_C2C_TEACHER_MAXMEM="22GiB,40GiB" loads it split layer-wise over GPUs 0..n-1 (device_map="auto"), and three
# device-only changes keep the official training math as is: (1) RosettaModel.to(device) moves the receiver and the
# projectors but leaves the split sharer where it is, (2) every C2CProjector receives the sharer's K/V slices on its own
# GPU, (3) Qwen3-MoE gets prefill/moe.py's grouped expert loop (same per-expert math, one host sync per layer).
SPLIT = os.environ.get("PREFILL_C2C_TEACHER_MAXMEM", "")
if SPLIT:
    import json

    import transformers
    from rosetta.model.projector import C2CProjector
    from rosetta.model.wrapper import RosettaModel

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from prefill.models import patch_moe

    cfg_path = sys.argv[sys.argv.index("--config") + 1]
    TEACHER = os.path.abspath(json.load(open(cfg_path))["model"]["teacher_model"])
    caps = {i: c.strip() for i, c in enumerate(SPLIT.split(","))}
    _orig_fp = transformers.AutoModelForCausalLM.from_pretrained.__func__

    def _from_pretrained(cls, path, *a, **k):
        if os.path.abspath(str(path)) == TEACHER:
            k.update(device_map="auto", max_memory=caps)
            m = _orig_fp(cls, path, *a, **k)
            assert all(isinstance(d, int) for d in m.hf_device_map.values()), f"sharer does not fit the GPUs with caps {SPLIT}"
            patch_moe(m)
            print(f"[c2c split] sharer {path} over GPUs {sorted(set(str(d) for d in m.hf_device_map.values()))}", flush=True)
            return m
        return _orig_fp(cls, path, *a, **k)

    transformers.AutoModelForCausalLM.from_pretrained = classmethod(_from_pretrained)

    def _to(self, device):
        for m in self.model_list:
            if len(set(getattr(m, "hf_device_map", {}).values())) <= 1:
                m.to(device)
        self.projector_list.to(device)
        return self

    RosettaModel.to = _to
    _orig_proj = C2CProjector.forward

    def _proj_forward(self, source_kv, target_kv, *a, **k):
        dev = target_kv[0].device
        return _orig_proj(self, tuple(x.to(dev) for x in source_kv), target_kv, *a, **k)

    C2CProjector.forward = _proj_forward

    import atexit

    @atexit.register
    def _peak():
        peak = [f"{torch.cuda.max_memory_allocated(g) / 2**30:.1f}" for g in range(torch.cuda.device_count())]
        print(f"[c2c split] peak GPU memory (GiB) {peak}", flush=True)

# Memory test (smoke only): PREFILL_C2C_LONGEST=N trains on the N longest training samples instead of the whole set, so
# the few steps of a smoke run see the largest batches the real run can draw.
LONGEST = int(os.environ.get("PREFILL_C2C_LONGEST", "0"))
if LONGEST:
    import torch.utils.data as tud

    class _LongestLoader(tud.DataLoader):
        def __init__(self, dataset, *a, **k):
            if k.get("shuffle"):
                col = k["collate_fn"]

                def n_tok(i):
                    x = col([dataset[i]])["input_ids"]
                    return (x[0] if isinstance(x, list) else x).shape[-1]

                lens = sorted(((n_tok(i), i) for i in range(len(dataset))), reverse=True)[:LONGEST]
                print(f"[c2c longest] training on the {len(lens)} longest of {len(dataset)} samples: "
                      f"{[n for n, _ in lens]} tokens", flush=True)
                dataset = tud.Subset(dataset, [i for _, i in lens])
            super().__init__(dataset, *a, **k)

    tud.DataLoader = _LongestLoader
script = os.path.join(C2C_DIR, "script", "train", "SFT_train.py")
sys.argv = [script] + sys.argv[1:]
os.chdir(C2C_DIR)
runpy.run_path(script, run_name="__main__")
