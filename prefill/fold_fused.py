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


class FusedFoldKV(FoldedProjectorKV):
    """Drop-in replacement for FoldedProjectorKV: same constructor and __call__, fused Triton epilogue.
    chunk only bounds the materialised activations a_l [Ll, chunk, d_mid] (Y is never materialised)."""

    BN_NORM = 128

    def __init__(self, projector, large_fast, chunk=8192, warm=True):
        super().__init__(projector, large_fast, chunk=chunk)
        self.chunk = chunk  # the parent may shrink it for the [Ll, chunk, ncol_pad] Y; irrelevant here
        Ll, d_mid, ncol_pad = self.Wcat.shape
        assert self.D % 2 == 0 and ncol_pad % self.BN_NORM == 0 and 2 * self.KV % self.BN_NORM == 0
        assert d_mid % 128 == 0, "BK up to 128 must divide d_mid"
        self.norm_tiles = (ncol_pad - 2 * self.KV) // self.BN_NORM  # pad columns are zero weight / zero bias
        self.kn_flat = None if self.kn is None else self.kn.reshape(Ll, self.D).contiguous()
        if warm:
            self.warm()

    def warm(self, n=4096):
        """Run once on a representative length so that autotuning / compilation never happen inside a timed or
        CUDA-graph-captured call."""
        _fold_kv_kernel.cache.clear()  # re-tune at this length even if an earlier (smaller) call already tuned
        Ll, Ls, d_s = self.Wcat.shape[0], self.mix.shape[1], self.W1.shape[1]
        dev = self.Wcat.device
        hs = torch.randn(Ls, n, d_s, device=dev, dtype=self.dt)
        cos = torch.ones(n, self.D, device=dev, dtype=self.dt)
        k = torch.empty(Ll, self.Hkv, n, self.D, device=dev, dtype=self.dt)
        v = torch.empty_like(k)
        self(hs, cos, cos, k, v)
        torch.cuda.synchronize()

    def activations(self, hs):
        """a_l = silu(W1 · mix_l(h) + b1) for all layers: [Ll, n, d_mid] (same as the eager path)."""
        hc = rms_unit(hs) if self.in_norm else hs
        H1 = F.linear(hc, self.W1)  # [Ls, n, d_mid]
        Ll, n = self.mix.shape[0], hs.shape[1]
        return F.silu(torch.matmul(self.mix, H1.reshape(H1.shape[0], -1)).view(Ll, n, -1) + self.b1)

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
            _fold_kv_kernel[grid](
                a, self.Wcat, self.bcat, cs, sn, self.kn_flat if self.kn_flat is not None else a, kd, vd,
                nc, 1.0 / self.d_l, self.eps,
                a.stride(0), a.stride(1), self.Wcat.stride(0), self.Wcat.stride(1), self.bcat.stride(0), cs.stride(0),
                kd.stride(0), kd.stride(1), kd.stride(2), vd.stride(0), vd.stride(1), vd.stride(2),
                HKV=self.Hkv, HALF=self.D // 2, DMID=self.Wcat.shape[1],
                NORM_START=2 * self.KV, NORM_TILES=self.norm_tiles, BN=self.BN_NORM, HAS_KN=self.kn_flat is not None)
