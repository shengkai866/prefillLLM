"""
CPU test of prefill.projector.ProjectorWide (untied W2 / grouped W1):
  1. from_projector() is exact: same forward output as the source Projector (w1_groups = 1, 3, Ll)
  2. the fold (prefill.fast.FoldedProjectorKV) of a converted projector equals the fold of the source projector
  3. after perturbing every group's W1 / every layer's W2, the fold still equals projector -> RMSNorm -> k/v -> k_norm -> RoPE
     computed directly (checks the grouped W1 path and the layer -> group map)
  4. save / Projector.load round trip, and gradients reach W1 / W2 / layer_mix

Run:  python tests/test_projector_wide.py
"""
import os
import sys
import tempfile

import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill.fast import FoldedProjectorKV, rms, rope  # noqa: E402
from prefill.projector import Projector, ProjectorWide  # noqa: E402


def fake_large(Ll, d_l, H, Hkv, D, dev, dt, eps=1e-6, seed=0):  # as tests/test_fold_fused.py (which needs a GPU to import)
    g = torch.Generator(device="cpu").manual_seed(seed)
    layers = [dict(wqkv=(torch.randn((H + 2 * Hkv) * D, d_l, generator=g) * d_l ** -0.5).to(dev, dt),
                   ln1=(1 + 0.1 * torch.randn(d_l, generator=g)).to(dev, dt),
                   kn=(1 + 0.1 * torch.randn(D, generator=g)).to(dev, dt)) for _ in range(Ll)]
    return types.SimpleNamespace(embed=torch.zeros(1, device=dev, dtype=dt), eps=eps, H=H, Hkv=Hkv, D=D, d=d_l, layers=layers)


def rope_table(n, D, dev, dt, base=1e6):
    inv = 1.0 / base ** (torch.arange(0, D, 2, dtype=torch.float32, device=dev) / D)
    f = torch.arange(n, dtype=torch.float32, device=dev)[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    return emb.cos().to(dt), emb.sin().to(dt)


torch.manual_seed(0)
Ls, Ll, d_s, d_l, d_mid, rank = 5, 6, 48, 96, 32, 8
H, Hkv, D = 4, 2, 16
dt = torch.float64
dev = torch.device("cpu")


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


p = Projector(Ls, Ll, d_s, d_l, d_mid=d_mid, rank=rank, in_norm=True).to(dt)
with torch.no_grad():
    p.layer_mix.add_(torch.randn_like(p.layer_mix))
    p.up.normal_(0, 0.1)
    p.bias.normal_(0, 0.1)
n = 11
hs = torch.randn(Ls, 1, n, d_s, dtype=dt) * torch.logspace(0, 2, Ls, dtype=dt)[:, None, None, None]
ref = p(hs)

# 1. exact conversion
for g in (1, 3, Ll):
    w = ProjectorWide.from_projector(p, w1_groups=g).to(dt)
    e = rel(w(hs), ref)
    print(f"[1] from_projector w1_groups={g}: rel diff {e:.2e}")
    assert e < 1e-12, e

# 2. fold of converted == fold of source
lf = fake_large(Ll, d_l, H, Hkv, D, dev, dt)
cos, sin = rope_table(n + 3, D, dev, dt)


def fold_kv(proj):
    f = FoldedProjectorKV(proj, lf)
    k = torch.zeros(Ll, Hkv, n, D, dtype=dt)
    v = torch.zeros_like(k)
    f(hs[:, 0], cos[3:3 + n], sin[3:3 + n], k, v)
    return k, v, f


k0, v0, _ = fold_kv(p)
for g in (1, 3, Ll):
    k1, v1, f = fold_kv(ProjectorWide.from_projector(p, w1_groups=g).to(dt))
    e = max(rel(k1, k0), rel(v1, v0))
    print(f"[2] fold(converted, w1_groups={g}) vs fold(source): rel diff {e:.2e}  grouped={f.grouped}")
    assert e < 1e-10, e


# 3. perturbed wide projector: fold vs direct computation
def direct_kv(proj):
    x = proj(hs)[:, 0]  # [Ll, n, d_l]
    ks, vs = [], []
    for l, lw in enumerate(lf.layers):
        xn = rms(x[l], lw["ln1"], lf.eps)
        wk = lw["wqkv"][H * D: H * D + Hkv * D]
        wv = lw["wqkv"][H * D + Hkv * D:]
        k = rms((xn @ wk.T).view(n, Hkv, D), lw["kn"], lf.eps).transpose(0, 1)
        ks.append(rope(k, cos[3:3 + n], sin[3:3 + n]))
        vs.append((xn @ wv.T).view(n, Hkv, D).transpose(0, 1))
    return torch.stack(ks), torch.stack(vs)


kd, vd = direct_kv(p)
e = max(rel(k0, kd), rel(v0, vd))
print(f"[3] source projector: fold vs direct rel diff {e:.2e}")
assert e < 1e-6, e  # the fold's layer-mix softmax is computed in fp32 (pre-existing), hence ~1e-8 rather than 1e-15
for g in (1, 3, Ll):
    w = ProjectorWide.from_projector(p, w1_groups=g).to(dt)
    with torch.no_grad():
        w.W1.add_(0.2 * torch.randn_like(w.W1) * w.W1.std())
        w.b1.add_(0.1 * torch.randn_like(w.b1))
        w.W2.add_(0.2 * torch.randn_like(w.W2) * w.W2.std())
        w.b2.add_(0.1 * torch.randn_like(w.b2))
    k1, v1, _ = fold_kv(w)
    kd, vd = direct_kv(w)
    e = max(rel(k1, kd), rel(v1, vd))
    print(f"[3] perturbed wide w1_groups={g}: fold vs direct rel diff {e:.2e}")
    assert e < 1e-6, e

# 4. save / load, gradients
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "p.pt")
    w.save(path, step=7)
    w2, ck = Projector.load(path)
    assert isinstance(w2, ProjectorWide) and ck["step"] == 7
    e = rel(w2.to(dt)(hs), w(hs))
    print(f"[4] save/load: {type(w2).__name__}, rel diff {e:.2e}")
    assert e < 1e-6  # load() builds an fp32 module (the training dtype), this test runs in fp64
    p.save(path)
    assert type(Projector.load(path)[0]) is Projector
w.zero_grad()
w(hs).pow(2).mean().backward()
for name in ("W1", "b1", "W2", "b2", "layer_mix"):
    gsum = getattr(w, name).grad.abs().sum().item()
    assert gsum > 0, name
print("[4] gradients reach W1, b1, W2, b2, layer_mix")
print(f"params: Projector {p.num_params()}, ProjectorWide(w1_groups={Ll}) {w.num_params()}")
print("ALL OK")
