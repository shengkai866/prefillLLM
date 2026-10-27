"""
RaReCache baseline (Sarkar, Mitra, Huang, Kundu, Beerel, arXiv 2610.11358, 2026): cross-model KV transfer with
closed-form ridge maps plus rank-disagreement selective recompute. Reimplemented from the paper (no code released).

  map     for every target layer l and sigma in {key, value}:  y = (x - xbar) W + ybar.  x = the source-cache rows of the
          K source layers with the best single-source R^2 for (l, sigma), concatenated (K = 8 for Qwen3); keys are
          de-rotated (RoPE removed) before mapping and the mapped target keys re-rotated. W = (Xc^T Xc + lam I)^-1 Xc^T Yc
          with column-centred calibration matrices, lam = 0.01, fitted once (RaReFitter, scripts/fit_rarecache.py)
  score   rank disagreement: sum over l, sigma of ||z||^2 - ||z V_r||^2, z = (x - xbar) W, V_r = top-r (128) eigenvectors
          of the covariance of the map's predictions on the calibration tokens (= the squared gap between the full map
          and its rank-r restriction W V_r V_r^T with the bias refitted to the calibration mean). Source prefill only.
  select  the ceil(rho T) highest-scoring prompt positions, one set shared by all layers; the target recomputes them at
          their own positions against the full mapped cache (recompute.extend_at, causal mask)
Choices the paper leaves open: the last prompt token is always run by the target (it has to produce the first-token
logits; at rho = 0 it is the only one), no sink / tail positions are forced, calibration data is chosen by the caller.
"""
import math

import torch
import torch.nn.functional as F

from .fast import rope
from .recompute import extend_at


def derotate(k, cos, sin):
    """Inverse of fast.rope (rotation by -theta)."""
    return rope(k, cos, -sin)


@torch.no_grad()
def cache_rows(Fm, ids):
    """Prefill ids [T] with the lean model Fm; returns the cache as rows: keys de-rotated, both [L, T, Hkv*D]."""
    T = ids.shape[0]
    kb = torch.empty(Fm.n_layers, Fm.Hkv, T, Fm.D, dtype=Fm.embed.dtype, device=ids.device)
    vb = torch.empty_like(kb)
    Fm.prefill(ids, kb, vb, logits=False)
    kb = derotate(kb, Fm.cos[:T], Fm.sin[:T])
    rows = lambda b: b.transpose(1, 2).reshape(Fm.n_layers, T, Fm.Hkv * Fm.D)
    return rows(kb), rows(vb)


class RaReFitter:
    """Accumulates calibration statistics for ONE sigma (keys or values): source rows x [n, Ls*d_s] -> target rows
    y [n, Lt*d_t]. fp32 sums of data centred at the first batch's mean; solve() works in fp64 on slices."""

    def __init__(self, n_src, d_src, n_tgt, d_tgt, device):
        self.n_src, self.d_src, self.n_tgt, self.d_tgt = n_src, d_src, n_tgt, d_tgt
        Ds, Dt = n_src * d_src, n_tgt * d_tgt
        f = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)
        self.Sxx, self.Sxy, self.Sx, self.Sy, self.Syy = f(Ds, Ds), f(Ds, Dt), f(Ds), f(Dt), f(Dt)
        self.x0 = self.y0 = None
        self.N = 0

    def add(self, x, y):
        x, y = x.float(), y.float()
        if self.x0 is None:
            self.x0, self.y0 = x.mean(0), y.mean(0)
        xc, yc = x - self.x0, y - self.y0
        self.Sxx.addmm_(xc.T, xc)
        self.Sxy.addmm_(xc.T, yc)
        self.Sx += xc.sum(0)
        self.Sy += yc.sum(0)
        self.Syy += yc.pow(2).sum(0)
        self.N += x.shape[0]

    def _cxx(self, a, b):
        """Centred (sum, not mean) source covariance block for index tensors a, b, fp64."""
        mx = self.Sx.double() / self.N
        return self.Sxx[a][:, b].double() - self.N * mx[a, None] * mx[None, b]

    @torch.no_grad()
    def solve(self, K=8, lam=0.01, rank=128):
        """-> list over target layers of dict(src [K] source layers, W [K*d_s, d_t], xbar [K*d_s], ybar [d_t],
        Vr [d_t, rank], r2_single [n_src], r2 joint fit R^2 on the calibration data)."""
        dev = self.Sxx.device
        ds, dt = self.d_src, self.d_tgt
        mx, my = self.Sx.double() / self.N, self.Sy.double() / self.N
        blk = lambda s: torch.arange(s * ds, (s + 1) * ds, device=dev)
        out = []
        for t in range(self.n_tgt):
            cols = torch.arange(t * dt, (t + 1) * dt, device=dev)
            sxy = self.Sxy[:, cols]  # [Ds, d_t] slice first, then rows per source set
            cxy = lambda a: sxy[a].double() - self.N * mx[a, None] * my[None, cols]
            sst = float((self.Syy[cols].double() - self.N * my[cols] ** 2).sum())
            r2s = []
            for s in range(self.n_src):
                b = blk(s)
                A0, B = self._cxx(b, b), cxy(b)
                W = torch.linalg.solve(A0 + lam * torch.eye(ds, dtype=A0.dtype, device=dev), B)
                sse = sst - 2 * float((W * B).sum()) + float((W * (A0 @ W)).sum())
                r2s.append(1 - sse / sst)
            src = sorted(torch.tensor(r2s).topk(min(K, self.n_src)).indices.tolist())
            idx = torch.cat([blk(s) for s in src])
            A0, B = self._cxx(idx, idx), cxy(idx)
            W = torch.linalg.solve(A0 + lam * torch.eye(idx.numel(), dtype=A0.dtype, device=dev), B)
            sse = sst - 2 * float((W * B).sum()) + float((W * (A0 @ W)).sum())
            cov = W.T @ A0 @ W / self.N  # covariance of the predictions on the calibration tokens
            Vr = torch.linalg.eigh(cov)[1][:, -rank:]  # eigenvalues ascending -> top-rank eigenvectors
            out.append(dict(src=src, W=W.float().cpu(), xbar=(self.x0[idx].double() + mx[idx]).float().cpu(),
                            ybar=(self.y0[cols].double() + my[cols]).float().cpu(), Vr=Vr.float().cpu(),
                            r2_single=r2s, r2=1 - sse / sst))
        return out


@torch.no_grad()
def calibrate(S, L, prompts, sigma, stride=1, seed=0, log_every=0):
    """One pass over calibration prompts (iterable of id tensors) for sigma in {"k", "v"}: every `stride`-th position
    (random offset per prompt) is added to a RaReFitter. Returns the fitter."""
    g = torch.Generator().manual_seed(seed)
    fit = RaReFitter(S.n_layers, S.Hkv * S.D, L.n_layers, L.Hkv * L.D, L.embed.device)
    flat = lambda r: r.transpose(0, 1).reshape(r.shape[1], -1)  # [L, T, d] -> [T, L*d], layer-major like RaReFitter
    for i, ids in enumerate(prompts):
        T = ids.shape[0]
        pos = torch.arange(int(torch.randint(stride, (1,), generator=g)), T, stride, device=ids.device)
        xs, xt = cache_rows(S, ids), cache_rows(L, ids)
        j = 0 if sigma == "k" else 1
        fit.add(flat(xs[j])[pos], flat(xt[j])[pos])
        if log_every and (i + 1) % log_every == 0:
            print(f"[calib {sigma}] {i + 1} prompts, {fit.N} tokens", flush=True)
    return fit


class RaReCache:
    """Inference with a fitted map: source prefill -> mapped target cache + rank-disagreement scores -> recompute."""

    def __init__(self, S, L, fit, chunk=8192):
        dev, dt = L.embed.device, L.embed.dtype
        self.S, self.L, self.chunk = S, L, chunk
        self.meta = fit.get("cfg", {})
        load = lambda m: [dict(src=torch.tensor(e["src"], device=dev), W=e["W"].to(dev, dt), xbar=e["xbar"].to(dev, dt),
                               ybar=e["ybar"].to(dev, dt), Vr=e["Vr"].to(dev, torch.float32)) for e in m]
        self.k, self.v = load(fit["k"]), load(fit["v"])

    def _map(self, e, rows, c0, c1):
        """rows [Ls, T, d_s] -> (prediction [n, d_t], rank-disagreement score [n] fp32) for positions c0:c1."""
        x = rows[e["src"], c0:c1].transpose(0, 1).reshape(c1 - c0, -1)
        z = (x - e["xbar"]) @ e["W"]
        zf = z.float()
        return z + e["ybar"], zf.pow(2).sum(-1) - (zf @ e["Vr"]).pow(2).sum(-1)

    @torch.no_grad()
    def map_cache(self, ids):
        """-> (kbuf, vbuf [Lt, Hkv, T, D] mapped target cache, keys RoPE'd at their positions; score [T] fp32)."""
        L = self.L
        T = ids.shape[0]
        ks, vs = cache_rows(self.S, ids)
        kbuf = torch.empty(L.n_layers, L.Hkv, T, L.D, dtype=L.embed.dtype, device=ids.device)
        vbuf = torch.empty_like(kbuf)
        score = torch.zeros(T, dtype=torch.float32, device=ids.device)
        for c0 in range(0, T, self.chunk):
            c1 = min(T, c0 + self.chunk)
            for l in range(L.n_layers):
                yk, sk = self._map(self.k[l], ks, c0, c1)
                yv, sv = self._map(self.v[l], vs, c0, c1)
                kbuf[l, :, c0:c1] = rope(yk.view(c1 - c0, L.Hkv, L.D).transpose(0, 1), L.cos[c0:c1], L.sin[c0:c1])
                vbuf[l, :, c0:c1] = yv.view(c1 - c0, L.Hkv, L.D).transpose(0, 1)
                score[c0:c1] += sk + sv
        return kbuf, vbuf, score

    @torch.no_grad()
    def prefill(self, ids, rho):
        """-> (kbuf, vbuf, first-token logits, number of prompt positions run by the target incl. the last token)."""
        T = ids.shape[0]
        kbuf, vbuf, score = self.map_cache(ids)
        sel = torch.zeros(T, dtype=torch.bool, device=ids.device)
        n = min(T, math.ceil(rho * T))
        if n:
            sel[torch.topk(score, n).indices] = True
        sel[T - 1] = True
        pos = torch.nonzero(sel).flatten()
        logits, _ = extend_at(self.L, ids[pos], pos, kbuf, vbuf)
        return kbuf, vbuf, logits, int(pos.numel())
