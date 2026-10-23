"""
Deep handoff: the small model + projector stand in for the large model's FIRST layers only; the large model itself runs
its last `k` layers on every prompt token.

  layers 0 .. split-1     middle-token K/V from the folded projector (as in FastHybrid); sink + tail tokens run through
                          these layers in the large model (attending to the projected middle K/V)
  layer split             residual-stream input of the middle tokens = projector output `split` (the projector already
                          predicts the input residual stream of every large layer); sink/tail keep their true residuals
  layers split .. Ll-1    the large model runs ALL prompt tokens causally (ordinary prefill over these layers), so the
                          deep K/V of the middle tokens is the large model's own computation on the projected input

split = Ll - k.  k = 0 reproduces FastHybrid exactly; k = Ll reproduces the large model's own prefill exactly.
Cost ~ small prefill + k/Ll of the large prefill (+ the first split layers for the sink/tail tokens only).
"""
import torch
import torch.nn.functional as F

from .fast import FastBig, FastHybrid, causal_mask_for, rms
from .kv import split_prompt


def run_layers_at(L, h, lo, hi, pos, kbuf, vbuf):
    """Tokens with residual streams h [Q, d] at sorted absolute positions pos [Q] through large layers lo..hi-1 against
    the KV buffer [Ll, Hkv, T, D]; their K/V are written to rows `pos` of each layer before attention. Returns h."""
    Q, T = h.shape[0], kbuf.shape[2]
    cos, sin = L.cos[pos], L.sin[pos]
    mask = causal_mask_for(pos, T, L.embed.dtype, h.device, L.rep)[None, None]
    for i in range(lo, hi):
        lw = L.layers[i]
        q, k, v = L._qkv(lw, rms(h, lw["ln1"], L.eps), cos, sin)
        kbuf[i].index_copy_(1, pos, k)
        vbuf[i].index_copy_(1, pos, v)
        qg = q.reshape(L.Hkv, L.rep * Q, L.D)
        o = F.scaled_dot_product_attention(qg[None], kbuf[i][None], vbuf[i][None], attn_mask=mask, scale=L.scaling)[0]
        h = L._mlp(lw, L._out(lw, h, o.reshape(L.H, Q, L.D)))
    return h


def run_layers_causal(L, h, lo, hi, kbuf, vbuf):
    """All T tokens (positions 0..T-1, residual streams h [T, d]) through large layers lo..hi-1 with plain causal
    attention among themselves (same code as FastQwen3.prefill); writes their K/V. Returns h."""
    T = h.shape[0]
    cos, sin = L.cos[:T], L.sin[:T]
    for i in range(lo, hi):
        lw = L.layers[i]
        q, k, v = L._qkv(lw, rms(h, lw["ln1"], L.eps), cos, sin)
        kbuf[i, :, :T] = k
        vbuf[i, :, :T] = v
        o = F.scaled_dot_product_attention(q[None], k.repeat_interleave(L.rep, 0)[None], v.repeat_interleave(L.rep, 0)[None],
                                           is_causal=True, scale=L.scaling)[0]
        h = L._mlp(lw, L._out(lw, h, o))
    return h


class DeepHybrid(FastHybrid):
    """FastHybrid where the large model takes over at layer Ll - k for every token."""

    def __init__(self, small_fast, large_fast, folded, projector, n_sink=8, n_tail=64):
        super().__init__(small_fast, large_fast, folded, n_sink=n_sink, n_tail=n_tail)
        self.p = projector
        self._resid = {}  # split -> (M [d_mid, d_l], c [d_l]) : projector output `split` = silu(mix W1 hs + b1) M + c

    def resid_map(self, l):
        if l not in self._resid:
            p, dev, dt = self.p, self.L.embed.device, self.L.embed.dtype
            f64 = lambda t: t.detach().to(device=dev, dtype=torch.float64)
            with torch.no_grad():
                W2T, b2 = f64(p.trunk[2].weight).T, f64(p.trunk[2].bias)
                Dl, Ul = f64(p.down[l]), f64(p.up[l])
                M = W2T + (W2T @ Dl) @ Ul
                c = b2 + (b2 @ Dl) @ Ul + f64(p.bias[l].reshape(-1))
            self._resid[l] = (M.to(dt), c.to(dt))
        return self._resid[l]

    def project_resid(self, hs_mid, l):
        """Projector's predicted input residual stream of large layer l for the middle tokens: [n, d_l]."""
        fo = self.fold
        M, c = self.resid_map(l)
        a = F.silu(torch.einsum("s,snd->nd", fo.mix[l], F.linear(hs_mid, fo.W1)) + fo.b1)
        return torch.addmm(c, a, M)

    @torch.no_grad()
    def prefill_deep(self, ids, k, return_h=False):
        """ids [T]; k = number of final large layers run on all tokens. Returns (kbuf, vbuf, first-token logits) and,
        with return_h, also (projected residual of the middle tokens at layer split [m, d] or None,
        final normed hidden states of all tokens [T, d] or None)."""
        L, dev = self.L, ids.device
        T, Ll = ids.shape[0], L.n_layers
        s, m, _ = split_prompt(T, self.n_sink, self.n_tail)
        k = max(0, min(k, Ll))
        if k == 0 or m == 0:  # projector everywhere (plain hybrid) / nothing to project
            out = self.prefill(ids) if k == 0 else FastBig(L).prefill(ids)
            return (*out, None, None) if return_h else out
        split = Ll - k
        kbuf = torch.empty(Ll, L.Hkv, T, L.D, dtype=L.embed.dtype, device=dev)
        vbuf = torch.empty_like(kbuf)
        h = F.embedding(ids, L.embed)
        h_mid = None
        if split > 0:
            hs = self.stage_small(ids[:s + m])[:, s:]
            self.fold(hs, L.cos[s:s + m], L.sin[s:s + m], kbuf[:split, :, s:s + m], vbuf[:split, :, s:s + m], n_layers=split)
            pos = torch.cat([torch.arange(0, s, device=dev), torch.arange(s + m, T, device=dev)])
            h_st = run_layers_at(L, h[pos], 0, split, pos, kbuf, vbuf)
            h_mid = self.project_resid(hs, split)
            h = torch.cat([h_st[:s], h_mid, h_st[s:]])
        h = run_layers_causal(L, h, split, Ll, kbuf, vbuf)
        hn = rms(h, L.norm, L.eps)
        logits = F.linear(hn[-1:], L.lm_head)
        return (kbuf, vbuf, logits, h_mid, hn) if return_h else (kbuf, vbuf, logits)
