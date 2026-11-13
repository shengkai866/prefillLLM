"""
CPU test of prefill.projector.ProjectorMoE (layer-routed mixture of projectors, branch moe-layers):
  1. routing: output layer l is exactly expert owner[l]'s output row local[l]; default init = Projector's depth alignment
  2. from_projector() is exact (same forward output as the source Projector), for several span sets
  3. the fold (prefill.fast.FoldedProjectorKV) of a converted projector equals the fold of the source projector, and
     after perturbing every expert differently it still equals projector -> RMSNorm -> k/v -> k_norm -> RoPE directly
  4. the experts are independent: a loss on the span layers gives no gradient to expert 0 and vice versa
  5. save / Projector.load round trip; new_projector(moe_spans=...) builds it

Run:  python tests/test_projector_moe.py
"""
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill.fast import FoldedProjectorKV, rms, rope  # noqa: E402
from prefill.models import new_projector  # noqa: E402
from prefill.projector import Projector, ProjectorMoE, parse_spans, rms_unit  # noqa: E402


def fake_large(Ll, d_l, H, Hkv, D, dev, dt, eps=1e-6, seed=0):  # as tests/test_projector_wide.py
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
Ls, Ll, d_s, d_l, d_mid, rank = 5, 8, 48, 96, 32, 8
H, Hkv, D = 4, 2, 16
dt = torch.float64
dev = torch.device("cpu")
SPANS = ["3-5", "0-1", "6-7", "2-3,6-6"]


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


assert parse_spans("18-26") == [[18, 26]] and parse_spans("2-3,6-6") == [[2, 3], [6, 6]]
n = 11
hs = torch.randn(Ls, 1, n, d_s, dtype=dt) * torch.logspace(0, 2, Ls, dtype=dt)[:, None, None, None]

# 1. routing and init
for in_norm in (False, True):
    m = ProjectorMoE(Ls, Ll, d_s, d_l, d_mid=d_mid, rank=rank, in_norm=in_norm, spans=parse_spans("3-5")).to(dt)
    with torch.no_grad():
        for ex in m.experts:
            ex.up.normal_(0, 0.1)
            ex.bias.normal_(0, 0.1)
    assert m.owner.tolist() == [0, 0, 0, 1, 1, 1, 0, 0] and m.local.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    assert [ex.cfg["Ll"] for ex in m.experts] == [5, 3]
    out = m(hs)
    x = rms_unit(hs) if in_norm else hs
    parts = [ex(x) for ex in m.experts]
    for l in range(Ll):
        assert torch.equal(out[l], parts[int(m.owner[l])][int(m.local[l])]), l
    assert torch.equal(m.layer_mix, Projector(Ls, Ll, d_s, d_l, d_mid=d_mid, rank=rank).layer_mix.to(dt))
print("[1] routing by layer and depth-proportional init OK (in_norm off / on)")
for bad in ("0-7", "2-4,4-5", "5-9"):
    try:
        ProjectorMoE(Ls, Ll, d_s, d_l, spans=parse_spans(bad))
        raise SystemExit(f"spans {bad} should be rejected")
    except AssertionError:
        pass

# 2. exact conversion
p = Projector(Ls, Ll, d_s, d_l, d_mid=d_mid, rank=rank, in_norm=True).to(dt)
with torch.no_grad():
    p.layer_mix.add_(torch.randn_like(p.layer_mix))
    p.up.normal_(0, 0.1)
    p.bias.normal_(0, 0.1)
ref = p(hs)
for sp in SPANS:
    m = ProjectorMoE.from_projector(p, spans=parse_spans(sp))
    e = rel(m(hs), ref)
    print(f"[2] from_projector spans={sp}: rel diff {e:.2e}, experts {[ex.cfg['Ll'] for ex in m.experts]}")
    assert e < 1e-12, e

# 3. folds
lf = fake_large(Ll, d_l, H, Hkv, D, dev, dt)
cos, sin = rope_table(n + 3, D, dev, dt)


def fold_kv(proj):
    f = FoldedProjectorKV(proj, lf)
    k = torch.zeros(Ll, Hkv, n, D, dtype=dt)
    v = torch.zeros_like(k)
    f(hs[:, 0], cos[3:3 + n], sin[3:3 + n], k, v)
    return k, v, f


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


k0, v0, _ = fold_kv(p)
for sp in SPANS:
    m = ProjectorMoE.from_projector(p, spans=parse_spans(sp))
    k1, v1, f = fold_kv(m)
    e = max(rel(k1, k0), rel(v1, v0))
    assert f.grouped and e < 1e-10, e
    with torch.no_grad():  # make the experts really different
        for ex in m.experts:
            for q in ex.parameters():
                q.add_(0.2 * torch.randn_like(q) * q.std())
    k1, v1, _ = fold_kv(m)
    kd, vd = direct_kv(m)
    e2 = max(rel(k1, kd), rel(v1, vd))
    print(f"[3] spans={sp}: fold(converted) vs fold(source) {e:.2e}; perturbed experts, fold vs direct {e2:.2e}")
    assert e2 < 1e-6, e2  # the fold's layer-mix softmax is fp32 (pre-existing)
    assert rel(k1, k0) > 1e-2  # the perturbation did change the result

# 4. independent experts
m = ProjectorMoE.from_projector(p, spans=parse_spans("3-5"))
m(hs)[3:6].pow(2).mean().backward()
g0 = sum(q.grad.abs().sum().item() for q in m.experts[0].parameters() if q.grad is not None)
g1 = sum(q.grad.abs().sum().item() for q in m.experts[1].parameters())
assert g0 == 0 and g1 > 0, (g0, g1)
m.zero_grad()
m(hs)[[0, 1, 2, 6, 7]].pow(2).mean().backward()
g0 = sum(q.grad.abs().sum().item() for q in m.experts[0].parameters())
g1 = sum(q.grad.abs().sum().item() for q in m.experts[1].parameters() if q.grad is not None)
assert g0 > 0 and g1 == 0, (g0, g1)
m.zero_grad()
m(hs).pow(2).mean().backward()
for name, q in m.named_parameters():
    assert q.grad is not None and q.grad.abs().sum() > 0, name
print("[4] span loss -> only the span expert gets gradients; full loss reaches every parameter of both")

# 5. save / load / factory
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "p.pt")
    m.save(path, step=7)
    m2, ck = Projector.load(path)
    assert isinstance(m2, ProjectorMoE) and ck["step"] == 7 and m2.cfg["spans"] == [[3, 5]] and m2.cfg["in_norm"] is True
    e = rel(m2.to(dt)(hs), m(hs))
    print(f"[5] save/load: {type(m2).__name__}, rel diff {e:.2e}")
    assert e < 1e-6  # load() builds an fp32 module
    p.save(path)
    assert type(Projector.load(path)[0]) is Projector
cfg = lambda L, d: types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=L, hidden_size=d))
q = new_projector(cfg(28, 64), cfg(36, 128), d_mid=32, rank=4, moe_spans="18-26")
assert isinstance(q, ProjectorMoE) and [ex.cfg["Ll"] for ex in q.experts] == [27, 9] and q.cfg["Ls"] == 29
assert type(new_projector(cfg(28, 64), cfg(36, 128), d_mid=32, rank=4)) is Projector
full = ProjectorMoE(29, 36, 1024, 4096)
print(f"params at real size (0.6B -> 8B): Projector {Projector(29, 36, 1024, 4096).num_params()/1e6:.1f}M, "
      f"ProjectorMoE 18-26 {full.num_params()/1e6:.1f}M = " + " + ".join(f"{ex.num_params()/1e6:.1f}M" for ex in full.experts))
print("ALL OK")
