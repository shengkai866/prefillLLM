#!/usr/bin/env python
"""
GPU test of prefill/rc_train.py (recompute-aware stage 2) on real weights of the current pair (env PREFILL_*):
  1. frac = 0   : student_forward_rc == kv.student_forward (HF path)                       KL < 2e-3
  2. frac = 0.1 : student_forward_rc == inference (RecomputeHybrid.prefill_sel attn_small sentences + continuation),
                  same recomputed tokens                                                     KL < 2e-3
  3. backward   : the projector gets gradient, the predicted residuals of recomputed tokens get exactly none
  4. peak memory of one training step (frac 0.2, ~8k prompt, checkpointed) is printed
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import TokenStore  # noqa: E402
from prefill.fast import FastQwen3, build_fold, kv_to_dynamic_cache  # noqa: E402
from prefill.kv import kl_teacher_student, student_forward, teacher_forward  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402
from prefill.rc_train import select_middle, student_forward_rc  # noqa: E402
from prefill.recompute import RecomputeHybrid  # noqa: E402


def main():
    proj_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt")
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    p, _ = Projector.load(proj_path, dev)
    L, S = FastQwen3(large), FastQwen3(small)
    boundary = torch.tensor([any(ch in tok.decode([i]) for ch in ".!?\n") for i in range(len(tok))], device=dev)
    rh = RecomputeHybrid(S, L, build_fold(p, L, "eager"), n_sink=8, n_tail=64, boundary=boundary)
    st = TokenStore("sd_mixtest", "all")
    pl = np.load(os.path.join(C.DATA_DIR, "sd_mixtest", "prompt_len.npy"))
    meta = [json.loads(l) for l in open(os.path.join(C.DATA_DIR, "sd_mixtest", "meta.jsonl"))]
    pick = {}
    for i, m in enumerate(meta):
        pick.setdefault(m["bucket"], i)
    ok = True
    p.eval()
    for b in ("4k", "8k"):
        i = pick[b]
        ids = st.get(i).to(dev)[None]
        P = int(pl[i])
        with torch.no_grad():
            t = teacher_forward(large, ids, P)
            s_hf = student_forward(large, small, p, ids, P, 8, 64)
            s0, _ = student_forward_rc(L, S, p, ids, P, 8, 64, 0.0, boundary, ckpt=False)
            kl0 = kl_teacher_student(s_hf, s0).mean().item()
            s1, idx = student_forward_rc(L, S, p, ids, P, 8, 64, 0.1, boundary, ckpt=False)
            kb, vb, lg0, n = rh.prefill_sel(ids[0, :P], "attn_small", 0.1, seg="sent")
            out = large(input_ids=ids[:, P:-1], past_key_values=kv_to_dynamic_cache(kb, vb), use_cache=True,
                        logits_to_keep=ids.shape[1] - P - 1)
            s_inf = torch.cat([lg0, out.logits[0]], 0)[None]
            kl1 = kl_teacher_student(s_inf, s1).mean().item()
            kl_t = [kl_teacher_student(t, x.to(t.device)).mean().item() for x in (s_hf, s0, s1, s_inf)]
        print(f"[{b}] P={P} | frac0: KL(hf||rc)={kl0:.2e} | frac0.1: KL(inference||rc)={kl1:.2e}, recomputed rc={idx.numel()} "
              f"inference={n} | KL to teacher hf/rc0/rc0.1/inf0.1 = " + " / ".join(f"{x:.4f}" for x in kl_t), flush=True)
        ok &= kl0 < 2e-3 and kl1 < 2e-3 and idx.numel() == n
    # backward + memory, frac 0.2 on the 8k example
    p.train()
    i = pick["8k"]
    ids = st.get(i).to(dev)[None]
    P = int(pl[i])
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    sel = select_middle(S, ids[0], P, 8, 64, 0.2, boundary)
    t = teacher_forward(large, ids, P)
    s, idx, h_hat = student_forward_rc(L, S, p, ids, P, 8, 64, 0.2, boundary, ckpt=True, return_hidden=True, sel=sel)
    h_hat.retain_grad()
    kl_teacher_student(t, s.to(t.device)).mean().backward()
    g = h_hat.grad[:, 0]  # [Ll, m, d]
    gn_rc = g[:, idx].abs().max().item()
    keep = torch.ones(g.shape[1], dtype=torch.bool, device=dev)
    keep[idx] = False
    gn_keep = g[:, keep].abs().max().item()
    gp = sum(x.grad.abs().sum().item() for x in p.parameters() if x.grad is not None)
    print(f"[backward] recomputed {idx.numel()} of {g.shape[1]} middle tokens | max |grad| recomputed rows {gn_rc:.2e}, "
          f"other rows {gn_keep:.2e} | sum |grad projector| {gp:.3e} | peak extra memory "
          f"{(torch.cuda.max_memory_allocated() - base) / 2**30:.1f} GiB, total peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    ok &= gn_rc == 0.0 and gn_keep > 0 and gp > 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
