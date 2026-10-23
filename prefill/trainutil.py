"""Shared training utilities: logging, LR schedule, periodic hybrid KL validation."""
import json
import math
import os
import time

import torch

from .kv import kl_teacher_student, student_forward, teacher_forward


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")
        self.t0 = time.time()

    def log(self, **kw):
        kw["elapsed_min"] = round((time.time() - self.t0) / 60, 2)
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)


def cosine_lr(step, total, warmup, base_lr, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def validate_hybrid(large, small, projector, sampler, n_seqs, prompt_len, cont_len, n_sink, n_tail, device):
    """Mean KL(teacher||hybrid), hybrid NLL, teacher NLL and top-1 agreement on held-out continuations."""
    projector.eval()
    prompt_len = min(prompt_len, sampler.max_len() - cont_len)  # adapt to what the held-out data offers
    kls, nll_h, nll_t, agree = [], [], [], []
    for _ in range(n_seqs):
        ids = sampler.sample(prompt_len + cont_len, at_least=prompt_len + cont_len).unsqueeze(0).to(device)
        t = teacher_forward(large, ids, prompt_len)
        s = student_forward(large, small, projector, ids, prompt_len, n_sink, n_tail)
        tgt = ids[:, prompt_len:].to(s.device)
        kls.append(kl_teacher_student(t, s).mean().item())
        nll_h.append(torch.nn.functional.cross_entropy(s.float().flatten(0, 1), tgt.flatten()).item())
        nll_t.append(torch.nn.functional.cross_entropy(t.float().flatten(0, 1), tgt.flatten()).item())
        agree.append((t.argmax(-1) == s.argmax(-1)).float().mean().item())
    projector.train()
    f = lambda x: sum(x) / max(1, len(x))
    return {"val_prompt_len": prompt_len, "val_kl": f(kls), "val_nll_hybrid": f(nll_h), "val_nll_teacher": f(nll_t), "val_top1_agree": f(agree)}
