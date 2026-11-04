"""
Fused Triton version of FoldedProjectorKV.__call__ (prefill/fast.py).

The folded projector is one GEMM per large layer,  Y_l = a_l @ Wcat_l + b_l  with columns [K | V | norm | pad],
followed by an epilogue: RMSNorm scale from the norm columns, k_norm, RoPE, and a strided write into the KV buffer.
The eager version materialises Y ([Ll, n, ncol_pad] bf16, 1.25 GB at 4k for 8B) and then runs ~15 separate
elementwise kernels over slices of it, which costs more than the GEMM itself (REPORT_2026-10-28 §3.2).

Here one Triton program owns a block of BM tokens of one layer and computes the whole row:
  phase A  loop over the norm column tiles, accumulate the row sum of squares in registers -> inv = rsqrt(ss/d_l+eps)
  phase B  loop over the K heads, then the V heads; each head is two half-tiles (lo, hi) so that k_norm (RMS over the
           head) and RoPE (rotate_half pairs lo/hi) are computed in the epilogue in fp32, then stored once.
Nothing but the final K/V is written. The math is the same as FoldedProjectorKV; the intermediate roundings to bf16
(Y, the scaled K before k_norm, the bf16 RoPE products) are gone, so the result is slightly closer to the fp64 value.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .fast import FoldedProjectorKV
from .projector import rms_unit


def _configs():
    cfgs = []
    for bm, bk, w, s in ((64, 64, 4, 3), (128, 64, 4, 3), (128, 64, 8, 3), (128, 32, 4, 4), (64, 128, 4, 3), (128, 128, 8, 3)):
        cfgs.append(triton.Config({"BM": bm, "BK": bk}, num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_configs(), key=[])  # tuned once (first call); FusedFoldKV.warm() makes that a 4k-token call
@triton.jit
def _fold_kv_kernel(A, W, B, COS, SIN, KN, KOUT, VOUT,
                    n, inv_dl, eps,
                    sa_l, sa_m, sw_l, sw_k, sb_l, sc_m,
                    sk_l, sk_h, sk_m, sv_l, sv_h, sv_m,
                    HKV: tl.constexpr, HALF: tl.constexpr, DMID: tl.constexpr,
                    NORM_START: tl.constexpr, NORM_TILES: tl.constexpr, BN: tl.constexpr, HAS_KN: tl.constexpr,
                    BM: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)  # token block (fastest varying: all blocks of one layer run together, W_l stays in L2)
    l = tl.program_id(1).to(tl.int64)  # large layer (int64: layer offsets of a 64-layer / 32k KV buffer exceed 2^31)
    rm = pm * BM + tl.arange(0, BM)
    mask_m = rm < n
    rk = tl.arange(0, BK)
    a_ptr = A + l * sa_l + rm[:, None] * sa_m  # + k
    w_ptr = W + l * sw_l                        # + k * sw_k + col
    b_ptr = B + l * sb_l                        # + col

    # ---- phase A: ||x_l||^2 from the norm columns (a QR factor of the folded map, see FoldedProjectorKV)
    ss = tl.zeros([BM], dtype=tl.float32)
    for t in range(NORM_TILES):
        cols = NORM_START + t * BN + tl.arange(0, BN)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, DMID, BK):
            a = tl.load(a_ptr + (k0 + rk)[None, :], mask=mask_m[:, None], other=0.0)
            w = tl.load(w_ptr + (k0 + rk)[:, None] * sw_k + cols[None, :])
            acc = tl.dot(a, w, acc)
        acc += tl.load(b_ptr + cols)[None, :].to(tl.float32)
        ss += tl.sum(acc * acc, axis=1)
    inv = tl.math.rsqrt(ss * inv_dl + eps)  # [BM]  RMSNorm scale of the predicted residual stream

    rh = tl.arange(0, HALF)
    cs_ptr = rm[:, None] * sc_m + rh[None, :]
    # ---- phase B1: K heads  (scale, k_norm, RoPE)
    for h in range(HKV):
        c_lo = h * (2 * HALF) + rh
        c_hi = c_lo + HALF
        acc_lo = tl.zeros([BM, HALF], dtype=tl.float32)
        acc_hi = tl.zeros([BM, HALF], dtype=tl.float32)
        for k0 in range(0, DMID, BK):
            a = tl.load(a_ptr + (k0 + rk)[None, :], mask=mask_m[:, None], other=0.0)
            w_lo = tl.load(w_ptr + (k0 + rk)[:, None] * sw_k + c_lo[None, :])
            w_hi = tl.load(w_ptr + (k0 + rk)[:, None] * sw_k + c_hi[None, :])
            acc_lo = tl.dot(a, w_lo, acc_lo)
            acc_hi = tl.dot(a, w_hi, acc_hi)
        x_lo = (acc_lo + tl.load(b_ptr + c_lo)[None, :].to(tl.float32)) * inv[:, None]
        x_hi = (acc_hi + tl.load(b_ptr + c_hi)[None, :].to(tl.float32)) * inv[:, None]
        if HAS_KN:
            r = tl.math.rsqrt((tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)) / (2 * HALF) + eps)
            x_lo = x_lo * r[:, None] * tl.load(KN + l * (2 * HALF) + rh)[None, :].to(tl.float32)
            x_hi = x_hi * r[:, None] * tl.load(KN + l * (2 * HALF) + HALF + rh)[None, :].to(tl.float32)
        cos_lo = tl.load(COS + cs_ptr, mask=mask_m[:, None], other=0.0).to(tl.float32)
        cos_hi = tl.load(COS + cs_ptr + HALF, mask=mask_m[:, None], other=0.0).to(tl.float32)
        sin_lo = tl.load(SIN + cs_ptr, mask=mask_m[:, None], other=0.0).to(tl.float32)
        sin_hi = tl.load(SIN + cs_ptr + HALF, mask=mask_m[:, None], other=0.0).to(tl.float32)
        o_lo = x_lo * cos_lo - x_hi * sin_lo  # rotate_half(x) = [-x_hi, x_lo]
        o_hi = x_hi * cos_hi + x_lo * sin_hi
        out = KOUT + l * sk_l + h * sk_h + rm[:, None] * sk_m + rh[None, :]
        tl.store(out, o_lo.to(KOUT.dtype.element_ty), mask=mask_m[:, None])
        tl.store(out + HALF, o_hi.to(KOUT.dtype.element_ty), mask=mask_m[:, None])

    # ---- phase B2: V heads  (scale only)
    for h in range(HKV):
        c_lo = (HKV + h) * (2 * HALF) + rh
        c_hi = c_lo + HALF
        acc_lo = tl.zeros([BM, HALF], dtype=tl.float32)
        acc_hi = tl.zeros([BM, HALF], dtype=tl.float32)
        for k0 in range(0, DMID, BK):
            a = tl.load(a_ptr + (k0 + rk)[None, :], mask=mask_m[:, None], other=0.0)
            w_lo = tl.load(w_ptr + (k0 + rk)[:, None] * sw_k + c_lo[None, :])
            w_hi = tl.load(w_ptr + (k0 + rk)[:, None] * sw_k + c_hi[None, :])
            acc_lo = tl.dot(a, w_lo, acc_lo)
            acc_hi = tl.dot(a, w_hi, acc_hi)
        x_lo = (acc_lo + tl.load(b_ptr + c_lo)[None, :].to(tl.float32)) * inv[:, None]
        x_hi = (acc_hi + tl.load(b_ptr + c_hi)[None, :].to(tl.float32)) * inv[:, None]
        out = VOUT + l * sv_l + h * sv_h + rm[:, None] * sv_m + rh[None, :]
        tl.store(out, x_lo.to(VOUT.dtype.element_ty), mask=mask_m[:, None])
        tl.store(out + HALF, x_hi.to(VOUT.dtype.element_ty), mask=mask_m[:, None])


def _configs2():
    cfgs = []
    for bm, bk, bn, w, st in ((64, 64, 128, 4, 3), (64, 128, 128, 4, 3), (64, 128, 256, 4, 3), (128, 64, 128, 8, 3),
                              (128, 64, 256, 8, 3), (128, 128, 256, 8, 2), (128, 32, 256, 8, 4), (128, 64, 256, 4, 3)):
        cfgs.append(triton.Config({"BM": bm, "BK": bk, "BN": bn}, num_warps=w, num_stages=st))
    return cfgs


@triton.jit
def _k_epilogue(acc_lo, acc_hi, inv, b_ptr, c_lo, c_hi, KN, l, COS, SIN, cs_ptr, mask_m, eps,
                HALF: tl.constexpr, HAS_KN: tl.constexpr):
    x_lo = (acc_lo + tl.load(b_ptr + c_lo)[None, :].to(tl.float32)) * inv[:, None]
    x_hi = (acc_hi + tl.load(b_ptr + c_hi)[None, :].to(tl.float32)) * inv[:, None]
    if HAS_KN:
        r = tl.math.rsqrt((tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)) / (2 * HALF) + eps)
        x_lo = x_lo * r[:, None] * tl.load(KN + l * (2 * HALF) + tl.arange(0, HALF))[None, :].to(tl.float32)
        x_hi = x_hi * r[:, None] * tl.load(KN + l * (2 * HALF) + HALF + tl.arange(0, HALF))[None, :].to(tl.float32)
    cos_lo = tl.load(COS + cs_ptr, mask=mask_m[:, None], other=0.0).to(tl.float32)
    cos_hi = tl.load(COS + cs_ptr + HALF, mask=mask_m[:, None], other=0.0).to(tl.float32)
    sin_lo = tl.load(SIN + cs_ptr, mask=mask_m[:, None], other=0.0).to(tl.float32)
    sin_hi = tl.load(SIN + cs_ptr + HALF, mask=mask_m[:, None], other=0.0).to(tl.float32)
    return x_lo * cos_lo - x_hi * sin_lo, x_hi * cos_hi + x_lo * sin_hi


@triton.autotune(configs=_configs2(), key=[])
@triton.jit
def _fold_kv_kernel2(A, W, B, COS, SIN, KN, KOUT, VOUT,
                     n, inv_dl, eps,
                     sa_l, sa_m, sw_l, sw_k, sb_l, sc_m,
                     sk_l, sk_h, sk_m, sv_l, sv_h, sv_m,
                     HKV: tl.constexpr, HALF: tl.constexpr, DMID: tl.constexpr,
                     NORM_START: tl.constexpr, NORM_COLS: tl.constexpr, HAS_KN: tl.constexpr,
                     BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    """Variant 2: BN of the norm phase is tuned (NORM_COLS % BN == 0), and the head phases do two heads
    (four half tiles, 256 columns) per pass over a_l, halving the re-reads of the activation block."""
    pm = tl.program_id(0)
    l = tl.program_id(1).to(tl.int64)
    rm = pm * BM + tl.arange(0, BM)
    mask_m = rm < n
    rk = tl.arange(0, BK)
    a_ptr = A + l * sa_l + rm[:, None] * sa_m
    w_ptr = W + l * sw_l
    b_ptr = B + l * sb_l

    ss = tl.zeros([BM], dtype=tl.float32)
    for c in range(0, NORM_COLS, BN):
        cols = NORM_START + c + tl.arange(0, BN)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, DMID, BK):
            a = tl.load(a_ptr + (k0 + rk)[None, :], mask=mask_m[:, None], other=0.0)
            w = tl.load(w_ptr + (k0 + rk)[:, None] * sw_k + cols[None, :])
            acc = tl.dot(a, w, acc)
        acc += tl.load(b_ptr + cols)[None, :].to(tl.float32)
        ss += tl.sum(acc * acc, axis=1)
    inv = tl.math.rsqrt(ss * inv_dl + eps)

    rh = tl.arange(0, HALF)
    cs_ptr = rm[:, None] * sc_m + rh[None, :]
    for h in range(0, HKV, 2):
        c0l = h * (2 * HALF) + rh
        c0h = c0l + HALF
        c1l = c0h + HALF
        c1h = c1l + HALF
        acc0l = tl.zeros([BM, HALF], dtype=tl.float32)
        acc0h = tl.zeros([BM, HALF], dtype=tl.float32)
        acc1l = tl.zeros([BM, HALF], dtype=tl.float32)
        acc1h = tl.zeros([BM, HALF], dtype=tl.float32)
        for k0 in range(0, DMID, BK):
            a = tl.load(a_ptr + (k0 + rk)[None, :], mask=mask_m[:, None], other=0.0)
            wr = w_ptr + (k0 + rk)[:, None] * sw_k
            acc0l = tl.dot(a, tl.load(wr + c0l[None, :]), acc0l)
            acc0h = tl.dot(a, tl.load(wr + c0h[None, :]), acc0h)
            acc1l = tl.dot(a, tl.load(wr + c1l[None, :]), acc1l)
            acc1h = tl.dot(a, tl.load(wr + c1h[None, :]), acc1h)
        o_lo, o_hi = _k_epilogue(acc0l, acc0h, inv, b_ptr, c0l, c0h, KN, l, COS, SIN, cs_ptr, mask_m, eps, HALF, HAS_KN)
        out = KOUT + l * sk_l + h * sk_h + rm[:, None] * sk_m + rh[None, :]
        tl.store(out, o_lo.to(KOUT.dtype.element_ty), mask=mask_m[:, None])
        tl.store(out + HALF, o_hi.to(KOUT.dtype.element_ty), mask=mask_m[:, None])
        o_lo, o_hi = _k_epilogue(acc1l, acc1h, inv, b_ptr, c1l, c1h, KN, l, COS, SIN, cs_ptr, mask_m, eps, HALF, HAS_KN)
        out = out + sk_h
        tl.store(out, o_lo.to(KOUT.dtype.element_ty), mask=mask_m[:, None])
        tl.store(out + HALF, o_hi.to(KOUT.dtype.element_ty), mask=mask_m[:, None])

    for h in range(0, HKV, 2):
        c0l = (HKV + h) * (2 * HALF) + rh
        c0h = c0l + HALF
        c1l = c0h + HALF
        c1h = c1l + HALF
        acc0l = tl.zeros([BM, HALF], dtype=tl.float32)
        acc0h = tl.zeros([BM, HALF], dtype=tl.float32)
        acc1l = tl.zeros([BM, HALF], dtype=tl.float32)
        acc1h = tl.zeros([BM, HALF], dtype=tl.float32)
        for k0 in range(0, DMID, BK):
            a = tl.load(a_ptr + (k0 + rk)[None, :], mask=mask_m[:, None], other=0.0)
            wr = w_ptr + (k0 + rk)[:, None] * sw_k
            acc0l = tl.dot(a, tl.load(wr + c0l[None, :]), acc0l)
            acc0h = tl.dot(a, tl.load(wr + c0h[None, :]), acc0h)
            acc1l = tl.dot(a, tl.load(wr + c1l[None, :]), acc1l)
            acc1h = tl.dot(a, tl.load(wr + c1h[None, :]), acc1h)
        out = VOUT + l * sv_l + h * sv_h + rm[:, None] * sv_m + rh[None, :]
        x = (acc0l + tl.load(b_ptr + c0l)[None, :].to(tl.float32)) * inv[:, None]
        tl.store(out, x.to(VOUT.dtype.element_ty), mask=mask_m[:, None])
        x = (acc0h + tl.load(b_ptr + c0h)[None, :].to(tl.float32)) * inv[:, None]
        tl.store(out + HALF, x.to(VOUT.dtype.element_ty), mask=mask_m[:, None])
        out = out + sv_h
        x = (acc1l + tl.load(b_ptr + c1l)[None, :].to(tl.float32)) * inv[:, None]
        tl.store(out, x.to(VOUT.dtype.element_ty), mask=mask_m[:, None])
        x = (acc1h + tl.load(b_ptr + c1h)[None, :].to(tl.float32)) * inv[:, None]
        tl.store(out + HALF, x.to(VOUT.dtype.element_ty), mask=mask_m[:, None])


def _configs_mix():
    return [triton.Config({"BLOCK": b}, num_warps=w, num_stages=st) for b, w, st in ((128, 4, 2), (256, 4, 2), (256, 8, 2), (512, 8, 2))]


@triton.autotune(configs=_configs_mix(), key=[])
@triton.jit
def _mix_silu_kernel(H1, MIX, B1, OUT, total, s_h, s_o,
                     LS: tl.constexpr, LL: tl.constexpr, LS_PAD: tl.constexpr, LL_PAD: tl.constexpr,
                     DMID: tl.constexpr, BLOCK: tl.constexpr):
    """a[l, f] = silu(sum_s mix[l, s] H1[s, f] + b1[f % d_mid]) over the flat token*feature index f: every H1 element
    is read once (the eager K=28 matmul + bias + silu are three passes over [Ll, n, d_mid])."""
    f0 = tl.program_id(0) * BLOCK
    rf = f0 + tl.arange(0, BLOCK)
    rs = tl.arange(0, LS_PAD)
    rl = tl.arange(0, LL_PAD)
    mask_f = rf < total
    h = tl.load(H1 + rs[:, None] * s_h + rf[None, :], mask=(rs[:, None] < LS) & mask_f[None, :], other=0.0)
    mix = tl.load(MIX + rl[:, None] * LS + rs[None, :], mask=(rl[:, None] < LL) & (rs[None, :] < LS), other=0.0)
    x = tl.dot(mix, h) + tl.load(B1 + (f0 % DMID) + tl.arange(0, BLOCK)).to(tl.float32)[None, :]
    y = x * tl.sigmoid(x)
    tl.store(OUT + rl[:, None].to(tl.int64) * s_o + rf[None, :], y.to(OUT.dtype.element_ty),
             mask=(rl[:, None] < LL) & mask_f[None, :])


class FusedFoldKV(FoldedProjectorKV):
    """Drop-in replacement for FoldedProjectorKV: same constructor and __call__, fused Triton epilogue.
    chunk only bounds the materialised activations a_l [Ll, chunk, d_mid] (Y is never materialised)."""

    BN_NORM = 128

    def __init__(self, projector, large_fast, chunk=8192, warm=True, variant=2):
        super().__init__(projector, large_fast, chunk=chunk)
        self.chunk = chunk  # the parent may shrink it for the [Ll, chunk, ncol_pad] Y; irrelevant here
        self.variant = variant
        Ll, d_mid, ncol_pad = self.Wcat.shape
        if variant == 2:  # norm columns padded to a multiple of 256 (zero columns add nothing to the sum of squares)
            assert self.Hkv % 2 == 0
            norm_cols = (ncol_pad - 2 * self.KV + 255) // 256 * 256
            if 2 * self.KV + norm_cols != ncol_pad:
                W = torch.zeros(Ll, d_mid, 2 * self.KV + norm_cols, dtype=self.Wcat.dtype, device=self.Wcat.device)
                b = torch.zeros(Ll, 1, W.shape[2], dtype=self.bcat.dtype, device=self.bcat.device)
                W[..., :ncol_pad], b[..., :ncol_pad] = self.Wcat, self.bcat
                self.Wcat, self.bcat = W, b
                ncol_pad = W.shape[2]
        assert self.D % 2 == 0 and ncol_pad % self.BN_NORM == 0 and 2 * self.KV % self.BN_NORM == 0
        assert d_mid % 128 == 0, "BK up to 128 must divide d_mid"
        self.norm_cols = ncol_pad - 2 * self.KV
        self.norm_tiles = self.norm_cols // self.BN_NORM  # pad columns are zero weight / zero bias
        self.kn_flat = None if self.kn is None else self.kn.reshape(Ll, self.D).contiguous()
        if warm:
            self.warm()

    def warm(self, n=4096):
        """Run once on a representative length so that autotuning / compilation never happen inside a timed or
        CUDA-graph-captured call."""
        (_fold_kv_kernel if self.variant == 1 else _fold_kv_kernel2).cache.clear()  # re-tune at this length
        _mix_silu_kernel.cache.clear()
        Ll, Ls, d_s = self.Wcat.shape[0], self.mix.shape[1], self.W1.shape[1]
        dev = self.Wcat.device
        hs = torch.randn(Ls, n, d_s, device=dev, dtype=self.dt)
        cos = torch.ones(n, self.D, device=dev, dtype=self.dt)
        k = torch.empty(Ll, self.Hkv, n, self.D, device=dev, dtype=self.dt)
        v = torch.empty_like(k)
        self(hs, cos, cos, k, v)
        torch.cuda.synchronize()

    fuse_mix = True

    def activations_eager(self, hs):
        """a_l = silu(W1 · mix_l(h) + b1) for all layers: [Ll, n, d_mid] (same as the eager path)."""
        hc = rms_unit(hs) if self.in_norm else hs
        H1 = F.linear(hc, self.W1)  # [Ls, n, d_mid]
        Ll, n = self.mix.shape[0], hs.shape[1]
        return F.silu(torch.matmul(self.mix, H1.reshape(H1.shape[0], -1)).view(Ll, n, -1) + self.b1)

    def activations(self, hs):
        """Same as activations_eager, with the layer mix + bias + silu in one Triton pass over H1."""
        if not self.fuse_mix:
            return self.activations_eager(hs)
        hc = rms_unit(hs) if self.in_norm else hs
        H1 = F.linear(hc, self.W1).contiguous()  # [Ls, n, d_mid]
        Ls, n, d_mid = H1.shape
        Ll = self.mix.shape[0]
        a = torch.empty(Ll, n, d_mid, dtype=H1.dtype, device=H1.device)
        total = n * d_mid
        pad = lambda v: max(16, triton.next_power_of_2(v))
        assert d_mid % 512 == 0
        _mix_silu_kernel[lambda meta: (triton.cdiv(total, meta["BLOCK"]),)](
            H1, self.mix, self.b1, a, total, H1.stride(0), a.stride(0),
            LS=Ls, LL=Ll, LS_PAD=pad(Ls), LL_PAD=pad(Ll), DMID=d_mid)
        return a

    def __call__(self, hs, cos, sin, kdst, vdst, n_layers=None):
        n = hs.shape[1]
        Ll = self.Wcat.shape[0] if n_layers is None else n_layers
        assert kdst.stride(-1) == 1 and vdst.stride(-1) == 1 and cos.stride(-1) == 1 and sin.stride(-1) == 1
        for c0 in range(0, n, self.chunk):
            c1 = min(n, c0 + self.chunk)
            nc = c1 - c0
            a = self.activations(hs[:, c0:c1])[:Ll].contiguous()  # [Ll, nc, d_mid]
            kd, vd = kdst[:, :, c0:c1], vdst[:, :, c0:c1]
            cs, sn = cos[c0:c1], sin[c0:c1]
            grid = lambda meta: (triton.cdiv(nc, meta["BM"]), Ll)
            args = (a, self.Wcat, self.bcat, cs, sn, self.kn_flat if self.kn_flat is not None else a, kd, vd,
                    nc, 1.0 / self.d_l, self.eps,
                    a.stride(0), a.stride(1), self.Wcat.stride(0), self.Wcat.stride(1), self.bcat.stride(0), cs.stride(0),
                    kd.stride(0), kd.stride(1), kd.stride(2), vd.stride(0), vd.stride(1), vd.stride(2))
            kw = dict(HKV=self.Hkv, HALF=self.D // 2, DMID=self.Wcat.shape[1], NORM_START=2 * self.KV, HAS_KN=self.kn_flat is not None)
            if self.variant == 1:
                _fold_kv_kernel[grid](*args, NORM_TILES=self.norm_tiles, BN=self.BN_NORM, **kw)
            else:
                _fold_kv_kernel2[grid](*args, NORM_COLS=self.norm_cols, **kw)
