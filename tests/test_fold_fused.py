"""
GPU test of prefill/fold_fused.FusedFoldKV against prefill/fast.FoldedProjectorKV on random weights of Qwen3-8B shape
(Ls=28 -> Ll=36, d_s=1024, d_l=4096, Hkv=8, D=128) and a Llama-style layer set (no k_norm).

  1. correctness: both implementations vs an fp64 evaluation of the same folded weights (same bf16 activations a_l);
     the fused kernel must be at least as close to fp64 as the eager path, and the two must agree to bf16 precision.
  2. speed: ms per call at 4k / 8k / 16k tokens (K/V written into a [Ll, Hkv, T, D] buffer slice, like stage_fold).

Run:  python tests/test_fold_fused.py [--lengths 4096,8192,16384] [--reps 5]
"""
import argparse
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill.fast import FoldedProjectorKV, rms, rope  # noqa: E402
from prefill.fold_fused import FusedFoldKV  # noqa: E402
from prefill.projector import Projector  # noqa: E402


def fake_large(Ll, d_l, H, Hkv, D, dev, dt, k_norm=True, eps=1e-6, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    layers = []
    for _ in range(Ll):
        wqkv = (torch.randn((H + 2 * Hkv) * D, d_l, generator=g) * d_l ** -0.5).to(dev, dt)
        ln1 = (1 + 0.1 * torch.randn(d_l, generator=g)).to(dev, dt)
        kn = (1 + 0.1 * torch.randn(D, generator=g)).to(dev, dt) if k_norm else None
        layers.append(dict(wqkv=wqkv, ln1=ln1, kn=kn))
    return types.SimpleNamespace(embed=torch.zeros(1, device=dev, dtype=dt), eps=eps, H=H, Hkv=Hkv, D=D, d=d_l, layers=layers)


def fake_projector(Ls, Ll, d_s, d_l, seed=0):
    torch.manual_seed(seed)
    p = Projector(Ls, Ll, d_s, d_l, in_norm=True)
    with torch.no_grad():
        p.layer_mix.add_(0.5 * torch.randn_like(p.layer_mix))
        p.up.copy_(0.02 * torch.randn_like(p.up))  # non-zero low-rank correction
        p.bias.copy_(0.1 * torch.randn_like(p.bias))
    return p


def rope_table(n, D, dev, dt, base=1e6):
    inv = 1.0 / base ** (torch.arange(0, D, 2, dtype=torch.float32, device=dev) / D)
    f = torch.arange(n, dtype=torch.float32, device=dev)[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    return emb.cos().to(dt), emb.sin().to(dt)


def reference_fp64(fold, a, cos, sin):
    """Exact (fp64) epilogue of the folded weights on the given bf16 activations a [Ll, n, d_mid]."""
    Ll, n, _ = a.shape
    Y = torch.baddbmm(fold.bcat.double(), a.double(), fold.Wcat.double())
    KV = fold.KV
    inv = torch.rsqrt(Y[..., 2 * KV:].pow(2).sum(-1, keepdim=True) / fold.d_l + fold.eps)
    k = (Y[..., :KV] * inv).view(Ll, n, fold.Hkv, fold.D)
    if fold.kn is not None:
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + fold.eps) * fold.kn.double()
    k = k.transpose(1, 2)
    v = (Y[..., KV:2 * KV] * inv).view(Ll, n, fold.Hkv, fold.D).transpose(1, 2)
    return rope(k, cos.double(), sin.double()), v


def err(x, ref):
    d = (x.double() - ref).abs()
    return d.max().item(), d.mean().item() / ref.abs().mean().item()


def check(name, Ll, Hkv, k_norm, dev, dt, variant=2, n=512, Ls=28, d_s=1024, d_l=4096, H=32, D=128):
    print(f"\n=== {name}: Ll={Ll} Hkv={Hkv} D={D} k_norm={k_norm}, n={n}, variant={variant}")
    lf = fake_large(Ll, d_l, H, Hkv, D, dev, dt, k_norm=k_norm)
    p = fake_projector(Ls, Ll, d_s, d_l)
    eager = FoldedProjectorKV(p, lf)
    fused = FusedFoldKV(p, lf, warm=False, variant=variant)
    torch.manual_seed(1)
    hs = (torch.randn(Ls, n, d_s, device=dev) * torch.logspace(0, 2, Ls, device=dev)[:, None, None]).to(dt)
    cos, sin = rope_table(n + 7, D, dev, dt)
    cos, sin = cos[7:], sin[7:]  # non-zero start offset, as in stage_fold
    T = n + 20
    s = 9
    mk = lambda: torch.full((Ll, Hkv, T, D), float("nan"), device=dev, dtype=dt)
    k0, v0, k1, v1 = mk(), mk(), mk(), mk()
    eager(hs, cos, sin, k0[:, :, s:s + n], v0[:, :, s:s + n])
    fused(hs, cos, sin, k1[:, :, s:s + n], v1[:, :, s:s + n])
    torch.cuda.synchronize()
    sl = (slice(None), slice(None), slice(s, s + n))
    assert torch.isnan(k1[:, :, :s]).all() and torch.isnan(k1[:, :, s + n:]).all(), "fused wrote outside its slice"
    assert not torch.isnan(k1[sl]).any() and not torch.isnan(v1[sl]).any(), "fused left NaNs in its slice"
    a = fused.activations(hs)
    a0 = fused.activations_eager(hs)
    da = (a.float() - a0.float()).abs()
    print(f"  activations fused-vs-eager: max={da.max():.2e} rel={da.mean() / a0.float().abs().mean():.2e}")
    assert da.mean() / a0.float().abs().mean() < 5e-3
    kr, vr = reference_fp64(fused, a, cos, sin)
    for nm, e, f, r in (("K", k0[sl], k1[sl], kr), ("V", v0[sl], v1[sl], vr)):
        me, re_ = err(e, r)
        mf, rf = err(f, r)
        mx, rx = err(f, e.double())
        print(f"  {nm}: |ref|max={r.abs().max():.3f}  eager-vs-fp64 max={me:.2e} rel={re_:.2e}   fused-vs-fp64 max={mf:.2e} rel={rf:.2e}"
              f"   fused-vs-eager max={mx:.2e} rel={rx:.2e}")
        assert rf <= 1.5 * re_ + 1e-4, f"{nm}: fused further from fp64 than eager"
        assert rx < 2e-2, f"{nm}: fused and eager disagree beyond bf16 noise"
    # partial layer count (used by diagnostics)
    k2, v2 = mk(), mk()
    fused(hs, cos, sin, k2[:4, :, s:s + n], v2[:4, :, s:s + n], n_layers=4)
    torch.cuda.synchronize()
    assert torch.equal(torch.nan_to_num(k2[:4][sl]), torch.nan_to_num(k1[:4][sl])), "n_layers path differs"
    print("  ok")
    del eager, fused, lf, p
    torch.cuda.empty_cache()


def bench(lengths, reps, dev, dt, Ls=28, Ll=36, d_s=1024, d_l=4096, H=32, Hkv=8, D=128):
    print(f"\n=== speed on {torch.cuda.get_device_name()} (ms per call, writes [Ll,Hkv,T,D] slice; median of {reps})")
    lf = fake_large(Ll, d_l, H, Hkv, D, dev, dt)
    p = fake_projector(Ls, Ll, d_s, d_l)
    eager = FoldedProjectorKV(p, lf)
    from prefill.fold_fused import _fold_kv_kernel, _fold_kv_kernel2
    fused1 = FusedFoldKV(p, lf, variant=1)  # warm(): autotune at 4k
    fused = FusedFoldKV(p, lf, variant=2)
    from prefill.fold_fused import _mix_silu_kernel
    for kern in (_fold_kv_kernel, _fold_kv_kernel2, _mix_silu_kernel):
        print(f"  {kern.fn.__name__} best: {kern.best_config}")
        tm = {c: (t[0] if isinstance(t, (list, tuple)) else t) for c, t in kern.configs_timings.items()}  # do_bench: [mean,..]
        for c, t in sorted(tm.items(), key=lambda x: x[1]):
            print(f"    {t:8.3f} ms  {c}")
    for n in lengths:
        hs = torch.randn(Ls, n, d_s, device=dev, dtype=dt)
        cos, sin = rope_table(n, D, dev, dt)
        kb = torch.empty(Ll, Hkv, n + 72, D, device=dev, dtype=dt)
        vb = torch.empty_like(kb)
        row = []
        for nm, f in (("eager", eager), ("fused1", fused1), ("fused", fused)):
            ts = []
            for i in range(reps + 2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                f(hs, cos, sin, kb[:, :, 8:8 + n], vb[:, :, 8:8 + n])
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            row.append(sorted(ts[2:])[len(ts[2:]) // 2])
        act = []
        for fn in (fused.activations_eager, fused.activations):  # activations alone (the part the fold kernel does not cover)
            ts = []
            for i in range(reps + 2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn(hs)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            act.append(sorted(ts[2:])[len(ts[2:]) // 2])
        print(f"  n={n:6d}: eager {row[0]:7.1f} ms   fused-v1 {row[1]:7.1f} ms   fused-v2 {row[2]:7.1f} ms   speedup {row[0] / row[2]:.2f}x"
              f"   (v2 = activations {act[1]:.1f} [eager act {act[0]:.1f}] + kernel {row[2] - act[1]:.1f})", flush=True)
        # parts of the activation step
        from prefill.projector import rms_unit
        H1 = F.linear(hs, fused.W1).contiguous()
        r = fused.row_rms(hs)
        r0 = torch.rsqrt(hs.float().pow(2).mean(-1) + 1e-6)
        assert ((r - r0).abs() / r0).max() < 1e-3, "row_rms differs from torch"
        parts = {"rms_unit (eager)": lambda: rms_unit(hs), "row_rms (triton)": lambda: fused.row_rms(hs),
                 "W1 linear": lambda: F.linear(hs, fused.W1), "mix+silu (triton)": lambda: fused.mix_silu(H1, r)}
        out = []
        for nm, fn in parts.items():
            ts = []
            for i in range(reps + 2):
                torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            out.append(f"{nm} {sorted(ts[2:])[len(ts[2:]) // 2]:.1f}")
        print(f"           activation parts (ms): " + ", ".join(out), flush=True)
        del H1, r, r0
        del hs, cos, sin, kb, vb
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="4096,8192,16384")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip_check", action="store_true")
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.bfloat16
    if not a.skip_check:
        for v in (1, 2):
            check("qwen3-8b-like", 36, 8, True, dev, dt, variant=v)
            check("llama-like (no k_norm)", 6, 8, False, dev, dt, variant=v, n=300)
    bench([int(x) for x in a.lengths.split(",")], a.reps, dev, dt)
    print("\nall done")


if __name__ == "__main__":
    main()
