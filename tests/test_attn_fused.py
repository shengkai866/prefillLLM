"""
GPU test of prefill/attn_fused.pos_attention (queries at arbitrary positions, causal by position, against a KV buffer)
against an fp32 reference and the original masked SDPA (fast.causal_mask_for + F.scaled_dot_product_attention).

  1. correctness on random bf16 data: Qwen3-8B heads (H=32, Hkv=8, D=128), Llama-1B-like D=64, sorted / unsorted
     positions, a compact buffer (rows != positions), tiny query counts; the kernel must be at least as close to fp32
     as SDPA.
  2. speed: ms for 36 layers' attention of the recompute call (sink 8 + 10 % of the middle + tail 64 queries against a
     T-row buffer), SDPA (incl. building the mask once) vs the kernel, at the given prompt lengths.

Run:  python tests/test_attn_fused.py [--lengths 4096,8192,16384,32768] [--reps 5]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill.attn_fused import _pos_attn_kernel, pos_attention  # noqa: E402
from prefill.fast import causal_mask_for  # noqa: E402


def positions(T, frac, g, dev, n_sink=8, n_tail=64):
    m = T - n_sink - n_tail
    mid = torch.randperm(m, generator=g)[:max(1, int(round(frac * m)))].sort().values + n_sink
    return torch.cat([torch.arange(n_sink), mid, torch.arange(T - n_tail, T)]).to(dev)


def ref_fp32(q, k, v, pos, scale):
    H, Q, D = q.shape
    rep = H // k.shape[0]
    kk = k.float().repeat_interleave(rep, 0)
    vv = v.float().repeat_interleave(rep, 0)
    s = torch.matmul(q.float(), kk.transpose(1, 2)) * scale
    keys = torch.arange(k.shape[1], device=q.device)
    s.masked_fill_(keys[None, None, :] > pos[None, :, None], float("-inf"))
    return torch.matmul(s.softmax(-1), vv)


def sdpa_masked(q, k, v, pos, scale, mask=None):
    H, Q, D = q.shape
    Hkv, T, _ = k.shape
    rep = H // Hkv
    if mask is None:
        mask = causal_mask_for(pos, T, q.dtype, q.device, rep)[None, None]
    o = F.scaled_dot_product_attention(q.reshape(Hkv, rep * Q, D)[None], k[None], v[None], attn_mask=mask, scale=scale)[0]
    return o.reshape(H, Q, D)


def err(x, ref):
    d = (x.float() - ref).abs()
    return d.max().item(), (d.mean() / ref.abs().mean()).item()


def check(name, H, Hkv, D, T, pos, dev, dt, k_rows=None, transposed_q=True):
    g = torch.Generator(device="cpu").manual_seed(T + D)
    Q = pos.shape[0]
    Tb = T if k_rows is None else k_rows
    if transposed_q:  # same layout as FastQwen3._qkv: [Q, H, D] viewed as [H, Q, D]
        q = torch.randn(Q, H, D, generator=g).to(dev, dt).transpose(0, 1)
    else:
        q = torch.randn(H, Q, D, generator=g).to(dev, dt)
    q = q * 2.0  # peakier softmax
    k = torch.randn(Hkv, Tb, D, generator=g).to(dev, dt)
    v = torch.randn(Hkv, Tb, D, generator=g).to(dev, dt)
    scale = D ** -0.5
    r = ref_fp32(q, k, v, pos, scale)
    o = pos_attention(q, k, v, pos, scale)
    s = sdpa_masked(q.contiguous(), k, v, pos, scale)
    torch.cuda.synchronize()
    assert not torch.isnan(o).any(), f"{name}: NaN"
    mo, ro = err(o, r)
    ms, rs = err(s, r)
    mx, rx = err(o, s.float())
    print(f"  {name:<44} Q={Q:5d} T={Tb:6d}: kernel-vs-fp32 max={mo:.2e} rel={ro:.2e}   sdpa-vs-fp32 max={ms:.2e} rel={rs:.2e}"
          f"   kernel-vs-sdpa rel={rx:.2e}", flush=True)
    assert ro <= 1.5 * rs + 1e-4, f"{name}: kernel further from fp32 than SDPA"
    assert rx < 1e-2, f"{name}: kernel and SDPA disagree beyond bf16 noise"


def bench(lengths, reps, dev, dt, Ll=36, H=32, Hkv=8, D=128, frac=0.1):
    print(f"\n=== speed on {torch.cuda.get_device_name()}: attention of the recompute call, {Ll} layers, H={H} Hkv={Hkv} D={D}, "
          f"sink 8 + {frac:.0%} of the middle + tail 64 (median of {reps}, ms)")
    g = torch.Generator(device="cpu").manual_seed(0)
    for T in lengths:
        pos = positions(T, frac, g, dev)
        Q = pos.shape[0]
        q = torch.randn(Q, H, D, device=dev, dtype=dt).transpose(0, 1)
        k = torch.randn(Hkv, T, D, device=dev, dtype=dt)
        v = torch.randn(Hkv, T, D, device=dev, dtype=dt)
        scale = D ** -0.5
        qc = q.contiguous()

        def run_sdpa():
            mask = causal_mask_for(pos, T, dt, dev, H // Hkv)[None, None]  # built once per call, as in extend_at
            for _ in range(Ll):
                sdpa_masked(qc, k, v, pos, scale, mask)

        def run_kernel():
            for _ in range(Ll):
                pos_attention(q, k, v, pos, scale)

        out = []
        for fn in (run_sdpa, run_kernel):
            ts = []
            for i in range(reps + 2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            out.append(sorted(ts[2:])[len(ts[2:]) // 2])
        visible = (pos.double() + 1).sum().item()  # (query, key) pairs that are not masked
        tflop = 4 * Ll * H * D * visible / 1e12
        print(f"  T={T:6d} Q={Q:5d}: sdpa {out[0]:8.1f}   kernel {out[1]:8.1f}   speedup {out[0] / out[1]:.2f}x   "
              f"({tflop:.1f} TFLOP visible -> kernel {tflop / out[1] * 1e3:.0f} TFLOP/s)", flush=True)
        del q, qc, k, v
        torch.cuda.empty_cache()
    print(f"  autotuned: {_pos_attn_kernel.cache}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.bfloat16
    g = torch.Generator(device="cpu").manual_seed(1)
    print("=== correctness")
    for T in (300, 4096, 16384):
        check("qwen3-8b heads, recompute 10 %", 32, 8, 128, T, positions(T, 0.1, g, dev), dev, dt)
    check("qwen3-8b heads, recompute 50 %", 32, 8, 128, 4096, positions(4096, 0.5, g, dev), dev, dt)
    check("qwen3-8b heads, sink + tail only", 32, 8, 128, 4096, positions(4096, 0.0, g, dev), dev, dt)
    check("qwen3-8b heads, contiguous q", 32, 8, 128, 2048, positions(2048, 0.1, g, dev), dev, dt, transposed_q=False)
    check("llama-1b-like D=64 (H=32, Hkv=8)", 32, 8, 64, 4096, positions(4096, 0.1, g, dev), dev, dt)
    check("qwen3-0.6b-like (H=16, Hkv=8)", 16, 8, 128, 4096, positions(4096, 0.1, g, dev), dev, dt)
    check("single query at the end", 32, 8, 128, 4096, torch.tensor([4095], device=dev), dev, dt)
    check("single query at position 0", 32, 8, 128, 4096, torch.tensor([0], device=dev), dev, dt)
    p = positions(4096, 0.1, g, dev)
    check("unsorted positions", 32, 8, 128, 4096, p[torch.randperm(p.shape[0], generator=g).to(dev)], dev, dt)
    # compact buffer (drop-* / multiq: rows 0..K-1 hold kept tokens, queries at rows K..K+n-1)
    K = 1000
    check("compact buffer, rows != positions", 32, 8, 128, 0, torch.arange(K, K + 80, device=dev), dev, dt, k_rows=K + 80)
    print("  ok")
    bench([int(x) for x in a.lengths.split(",")], a.reps, dev, dt)
    print("\nall done")


if __name__ == "__main__":
    main()
