"""
GPU check on real weights (branch moe-layers): the folds (fused Triton and eager) of ProjectorMoE.from_projector(p)
must give the same middle K/V as the fold of the trained Projector p (bf16 precision), and the time of the fold
(two experts = grouped-W1 path: eager layer mix + one W1 GEMM per large layer) is printed next to the single projector's.

Run (after sourcing slurm/env.sh):  python tests/check_moe_gpu.py [--projector path] [--spans 18-26]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402
from prefill.fast import FastQwen3, build_fold  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.projector import Projector, ProjectorMoE, parse_spans  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
ap.add_argument("--spans", default="18-26")
ap.add_argument("--n", type=int, default=4096)
a = ap.parse_args()
dev = torch.device("cuda")
tok, small, large = load_pair(dev)
S, L = FastQwen3(small), FastQwen3(large)
p, _ = Projector.load(a.projector, dev)
p.eval()
m = ProjectorMoE.from_projector(p, spans=parse_spans(a.spans)).eval()
print(f"Projector {p.num_params()/1e6:.1f}M -> ProjectorMoE spans={a.spans} {m.num_params()/1e6:.1f}M "
      f"(experts: {[ex.cfg['Ll'] for ex in m.experts]} layers)")
g = torch.Generator().manual_seed(0)
ids = torch.randint(1000, 30000, (a.n,), generator=g).to(dev)
with torch.no_grad():
    hs = S.prefill(ids, collect=True, logits=False)[1][:, 8:]  # [Ls, n-8, d_s]
n = hs.shape[1]


def kv(proj, mode):
    f = build_fold(proj, L, mode=mode)
    k = torch.zeros(L.n_layers, L.Hkv, n, L.D, dtype=L.embed.dtype, device=dev)
    v = torch.zeros_like(k)
    ts = []
    with torch.no_grad():
        for _ in range(6):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            f(hs, L.cos[8:8 + n], L.sin[8:8 + n], k, v)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return k.float(), v.float(), 1e3 * sorted(ts[1:])[len(ts[1:]) // 2]


rel = lambda x, y: ((x - y).norm() / y.norm()).item()
ok = True
for mode in ("fused", "eager"):
    k0, v0, t0 = kv(p, mode)
    k1, v1, t1 = kv(m, mode)
    e = max(rel(k1, k0), rel(v1, v0))
    print(f"[{mode}] MoE(from_projector) vs trained Projector: rel diff {e:.2e}; fold of {n} tokens {t0:.1f} ms -> {t1:.1f} ms", flush=True)
    ok &= e < 2e-2  # bf16 activations; the grouped path rounds mix-then-W1 instead of W1-then-mix
    if mode == "fused":
        kf, vf = k0, v0
    else:
        print(f"[ref] fused vs eager on the trained Projector: {max(rel(kf, k0), rel(vf, v0)):.2e} (scale of bf16 noise)")
    del k1, v1
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
