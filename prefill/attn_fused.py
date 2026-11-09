"""
Attention of queries at arbitrary absolute positions against a KV buffer, causal by position, as one Triton kernel.

Used by recompute.extend_at (selective recompute: the chosen ~10 % of the prompt plus sink and tail attend to the whole
buffer of T rows, projected KV for the tokens not chosen) and FastBig.extend (sink + tail call). The eager path builds
an additive [rep*Q, T] mask and calls SDPA with it; an arbitrary mask rules out FlashAttention, so PyTorch falls back to
the mem-efficient kernel, which reads the mask and computes every (query, key) pair, including the ~half that are
masked out. Here each program owns BM queries of one head and walks the keys block by block only up to the largest
position among its queries: blocks below the smallest position need no mask at all, the rest are masked per element.
Online softmax in fp32 (FlashAttention-2 style), GQA by indexing the KV head, no mask tensor, no copies.

  pos_attention(q [H, Q, D], k [Hkv, T, D], v [Hkv, T, D], pos [Q], scale) -> o [H, Q, D]
  query j sees buffer rows <= pos[j]; pos need not be sorted (sorted is faster: tighter per-block key ranges).

Switch: PREFILL_ATTN=triton (default on CUDA) | sdpa (the original masked SDPA), read at call time (fast.attn_enabled).
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


def _configs():
    out = []
    for BM in (64, 128):
        for BN in (32, 64, 128):
            for w in (4, 8):
                for s in (2, 3):
                    out.append(triton.Config({"BM": BM, "BN": BN}, num_warps=w, num_stages=s))
    return out


@triton.autotune(configs=_configs(), key=["D", "QB"])  # QB: query-count bucket (tail call ~72 vs recompute ~T/10)
@triton.jit
def _pos_attn_kernel(Q, K, V, POS, O, n_q, qk_scale, QB,
                     sq_h, sq_m, sk_h, sk_n, sv_h, sv_n, so_h, so_m,
                     REP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // REP
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BN)
    mm = offs_m < n_q
    p = tl.load(POS + offs_m, mask=mm, other=0).to(tl.int32)  # padding rows look at key 0 only (no NaN), never stored
    p_hi = tl.max(p, 0)
    p_lo = tl.min(tl.where(mm, p, p_hi), 0)
    q = tl.load(Q + h.to(tl.int64) * sq_h + offs_m[:, None] * sq_m + offs_d[None, :], mask=mm[:, None], other=0.0)
    kb = K + kvh.to(tl.int64) * sk_h
    vb = V + kvh.to(tl.int64) * sv_h
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    n_full = (p_lo + 1) // BN * BN  # keys < n_full are visible to every query of the block
    for start in range(0, n_full, BN):
        kn = start + offs_n
        k = tl.load(kb + kn[:, None] * sk_n + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        a = tl.math.exp2(m_i - m_new)
        pr = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * a + tl.sum(pr, 1)
        v = tl.load(vb + kn[:, None] * sv_n + offs_d[None, :])
        acc = acc * a[:, None] + tl.dot(pr.to(v.dtype), v)
        m_i = m_new
    for start in range(n_full, p_hi + 1, BN):  # diagonal band: per-element mask; key 0..p_lo already seen -> m_i finite
        kn = start + offs_n
        km = kn <= p_hi
        k = tl.load(kb + kn[:, None] * sk_n + offs_d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(kn[None, :] <= p[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        a = tl.math.exp2(m_i - m_new)
        pr = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * a + tl.sum(pr, 1)
        v = tl.load(vb + kn[:, None] * sv_n + offs_d[None, :], mask=km[:, None], other=0.0)
        acc = acc * a[:, None] + tl.dot(pr.to(v.dtype), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(O + h.to(tl.int64) * so_h + offs_m[:, None] * so_m + offs_d[None, :], out.to(O.dtype.element_ty),
             mask=mm[:, None])


def pos_attention(q, k, v, pos, scale, out=None):
    """q [H, Q, D] RoPE'd queries (any head/row strides, unit last stride); k, v [Hkv, T, D]; pos [Q] int: query j
    sees rows <= pos[j] (pos[j] < T). Returns o [H, Q, D] (contiguous)."""
    H, Qn, D = q.shape
    Hkv, T, _ = k.shape
    assert H % Hkv == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, (q.stride(), k.stride())
    assert pos.shape == (Qn,) and D in (64, 128, 256)
    o = torch.empty(H, Qn, D, dtype=q.dtype, device=q.device) if out is None else out
    if Qn == 0:
        return o
    qb = 0 if Qn <= 256 else 1
    grid = lambda meta: (triton.cdiv(Qn, meta["BM"]), H)
    _pos_attn_kernel[grid](q, k, v, pos, o, Qn, scale * LOG2E, qb,
                           q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0), o.stride(1),
                           REP=H // Hkv, D=D)
    return o
