"""
GPU check on real weights: the folds (fused Triton and eager) of ProjectorWide.from_projector(p) must give the same
middle K/V as the fold of the trained Projector p (bf16 precision), for w1_groups = 1 and = Ll.

Run (after sourcing slurm/env.sh):  python tests/check_wide_gpu.py [--projector path]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402
from prefill.fast import FastQwen3, build_fold  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.projector import Projector, ProjectorWide  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
ap.add_argument("--n", type=int, default=3000)
a = ap.parse_args()
dev = torch.device("cuda")
tok, small, large = load_pair(dev)
S, L = FastQwen3(small), FastQwen3(large)
p, _ = Projector.load(a.projector, dev)
p.eval()
g = torch.Generator().manual_seed(0)
ids = torch.randint(1000, 30000, (a.n,), generator=g).to(dev)
with torch.no_grad():
    hs = S.prefill(ids, collect=True, logits=False)[1][:, 8:]  # [Ls, n-8, d_s]
n = hs.shape[1]


def kv(proj, mode):
    f = build_fold(proj, L, mode=mode)
    k = torch.zeros(L.n_layers, L.Hkv, n, L.D, dtype=L.embed.dtype, device=dev)
    v = torch.zeros_like(k)
    with torch.no_grad():
        f(hs, L.cos[8:8 + n], L.sin[8:8 + n], k, v)
    return k.float(), v.float()


rel = lambda x, y: ((x - y).norm() / y.norm()).item()
ok = True
for mode in ("fused", "eager"):
    k0, v0 = kv(p, mode)
    for grp in (1, L.n_layers):
        w = ProjectorWide.from_projector(p, w1_groups=grp).eval()
        k1, v1 = kv(w, mode)
        e = max(rel(k1, k0), rel(v1, v0))
        print(f"[{mode}] w1_groups={grp}: rel diff vs trained Projector {e:.2e}", flush=True)
        ok &= e < 2e-2  # bf16 activations; the grouped path rounds mix-then-W1 instead of W1-then-mix
    del k0, v0
kf, vf = kv(p, "fused")
ke, ve = kv(p, "eager")
print(f"[ref] fused vs eager on the trained Projector: {max(rel(kf, ke), rel(vf, ve)):.2e} (scale of bf16 noise)")
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
