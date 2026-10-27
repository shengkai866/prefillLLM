#!/usr/bin/env python
"""CPU test of the RaReCache baseline (prefill/rarecache.py) with tiny random Qwen3 models."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch

from prefill.fast import FastBig, FastQwen3, rope
from prefill.rarecache import RaReCache, RaReFitter, cache_rows, calibrate, derotate
from smoke_recompute import V, close, tiny  # noqa: E402  (also re-runs the recompute tests)

torch.manual_seed(2)

# 1) de-rotation inverts RoPE
S = FastQwen3(tiny(4, 64, 4, 2, True))
k = torch.randn(2, 50, 16)
close(derotate(rope(k, S.cos[:50], S.sin[:50]), S.cos[:50], S.sin[:50]), k, "derotate(rope(k)) == k", 1e-5)

# 2) the fitter recovers a planted linear map and picks the right source layers by single-source R^2
n_src, d, n_tgt, N = 4, 32, 3, 6000
X = torch.randn(N, n_src * d) * torch.linspace(0.5, 2, n_src * d) + 3
planted = [[0, 2], [1, 3], [2, 3]]
A = [torch.randn(2 * d, d) / d ** 0.5 for _ in range(n_tgt)]
c = [torch.randn(d) for _ in range(n_tgt)]
cols = lambda src: torch.cat([torch.arange(s * d, (s + 1) * d) for s in src])
Y = torch.cat([X[:, cols(p)] @ A[t] + c[t] + 0.01 * torch.randn(N, d) for t, p in enumerate(planted)], 1)
fitter = RaReFitter(n_src, d, n_tgt, d, "cpu")
for i in range(0, N, 1000):
    fitter.add(X[i:i + 1000], Y[i:i + 1000])
sol = fitter.solve(K=2, lam=1e-6, rank=8)
for t, p in enumerate(planted):
    assert sol[t]["src"] == p, (t, sol[t]["src"], p)
    close(sol[t]["W"], A[t], f"planted map {t} recovered", 2e-3)
    pred = (X[:, cols(p)] - sol[t]["xbar"]) @ sol[t]["W"] + sol[t]["ybar"]
    close(pred, Y[:, t * d:(t + 1) * d], f"planted map {t} predictions (Y has 0.01 noise)", 1e-2)
    assert sol[t]["r2"] > 0.999, sol[t]["r2"]
print("[ok] single-source R^2 picks the planted source layers")

# 3) real pipeline on tiny models: calibrate, map, score, recompute
small, large = tiny(4, 64, 4, 2, True), tiny(6, 96, 6, 2, False)
S, L = FastQwen3(small), FastQwen3(large)
prompts = [torch.randint(0, V, (int(n),)) for n in torch.randint(150, 400, (12,))]
fit = {"k": calibrate(S, L, prompts, "k", stride=2).solve(K=2, lam=0.01, rank=8),
       "v": calibrate(S, L, prompts, "v", stride=2).solve(K=2, lam=0.01, rank=8)}
rc = RaReCache(S, L, fit, chunk=100)
T = 300
ids = torch.randint(0, V, (T,))
kb, vb, score = rc.map_cache(ids)

# mapped cache == explicit map of the source cache rows (keys re-rotated at their positions)
ks, vs = cache_rows(S, ids)
for l in (0, 5):
    for sig, rows, buf in (("k", ks, kb), ("v", vs, vb)):
        e = fit[sig][l]
        x = rows[e["src"]].transpose(0, 1).reshape(T, -1)
        y = ((x - e["xbar"]) @ e["W"] + e["ybar"]).view(T, L.Hkv, L.D).transpose(0, 1)
        got = derotate(buf[l], L.cos[:T], L.sin[:T]) if sig == "k" else buf[l]
        close(got, y, f"mapped {sig} layer {l}", 1e-4)

# rank-disagreement score == squared gap between the full map and its rank-r restriction, summed over layers and K/V
ref = torch.zeros(T)
for sig, rows in (("k", ks), ("v", vs)):
    for e in fit[sig]:
        x = rows[e["src"]].transpose(0, 1).reshape(T, -1)
        Wr = e["W"] @ e["Vr"] @ e["Vr"].T
        full = x @ e["W"] + (e["ybar"] - e["xbar"] @ e["W"])
        restricted = x @ Wr + (e["ybar"] - e["xbar"] @ Wr)  # bias refitted to the calibration mean of the predictions
        ref += (full - restricted).pow(2).sum(-1)
close(score, ref, "rank-disagreement score == ||full - restricted||^2", 1e-3)

# rho = 1 recomputes every position -> exactly the large model's own prefill; rho = 0 runs only the last token
kb1, vb1, lg1, n1 = rc.prefill(ids, 1.0)
kbb, vbb, lgb = FastBig(L).prefill(ids)
assert n1 == T
close(lg1, lgb, "rho=1 logits == large prefill")
close(kb1, kbb, "rho=1 keys == large prefill")
close(vb1, vbb, "rho=1 values == large prefill")
_, _, _, n0 = rc.prefill(ids, 0.0)
assert n0 == 1, n0
_, _, _, n3 = rc.prefill(ids, 0.3)
assert n3 in (90, 91), n3  # ceil(0.3 T) top-scoring positions plus the last token (unless already chosen)
print("[ok] rho=0 runs only the last token; rho=0.3 runs ceil(0.3 T) (+ last)")
print("ALL RARECACHE TESTS PASSED")
