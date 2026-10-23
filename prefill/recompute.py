"""
Selective recompute on top of the fast hybrid prefill.

After the small model + projector have filled the middle of the KV buffer, the large model recomputes the KV of a
chosen subset of middle tokens. Those tokens run in the SAME large-model call as the sink and tail tokens, each at its
own position, attending to the whole buffer (projected KV for the tokens not chosen); in every layer their K/V
overwrite the projected rows before attention, so later chosen tokens and the tail already see the corrected values.
Choosing every middle token reproduces the large model's own prefill exactly.

Token scorers (all training-free):
  random      uniform
  recent      the middle tokens right before the tail (equivalent to a longer tail)
  attn_small  attention mass that the small model's tail queries (question + chat template) put on each middle token
  attn_big    attention mass that the large model's tail queries put on each projected middle key; costs one extra
              tail pass of the large model before the recompute pass
  oracle      positions where an answer string occurs in the prompt (diagnostic upper bound; needs the answers,
              ignores the budget)
  spec        SpecPrefill token importance (spec_importance) of the small model; with seg=None it selects SpecPrefill's
              32-token chunks by mean score
Selection unit (`seg`):
  None        single tokens; attention scores are max-pooled over a small window so the neighbours of a hit come with it
  int B       fixed blocks of B consecutive tokens, block score = max token score, whole blocks are recomputed
  "sent"      sentence/line segments (a segment ends after a token containing . ! ? or a newline; capped at `cap` tokens),
              segment score = max token score, whole segments are recomputed
Block/segment selection takes units in score order until the token budget is reached (the last unit may overshoot).

drop=True (controlled SpecPrefill-style ablation): same sink / tail / selection, but the middle tokens NOT chosen are
dropped instead of filled with projected KV; the large model prefills only the kept tokens at their original positions
and the returned cache is compact (decoding must continue at position T).

SpecPrefill itself (Liu, Chen, Zhang, ICML 2025; official code github.com/Jingyu6/speculative_prefill), as the external
baseline: RecomputeHybrid.prefill_specprefill — whole-prompt chunk selection, no sink / tail, no projector.
"""
import math

import torch
import torch.nn.functional as F

from .fast import FastHybrid, causal_mask_for, rms
from .kv import split_prompt

SCORERS = ("random", "recent", "attn_small", "attn_big", "oracle", "spec")


def attn_mass(q, k, qpos, lo, hi, scale):
    """q [H, n, D] RoPE'd queries at absolute positions qpos [n]; k [Hkv, T, D] keys of buffer positions 0..T-1.
    Causal softmax attention probabilities, summed over keys [lo, hi) per key and averaged over heads and queries -> [hi-lo]."""
    Hkv, T, D = k.shape
    H, n, _ = q.shape
    rep = H // Hkv
    qg = q.reshape(Hkv, rep * n, D).float()  # row r*n + j <-> head g*rep + r, query j (same grouping as fast.extend)
    logits = torch.matmul(qg, k.float().transpose(1, 2)) * scale  # [Hkv, rep*n, T]
    keys = torch.arange(T, device=k.device)
    logits.masked_fill_(keys[None, None, :] > qpos.repeat(rep)[None, :, None], float("-inf"))
    return logits.softmax(-1)[:, :, lo:hi].mean((0, 1))


@torch.no_grad()
def small_prefill_scored(S, ids, n_q, lo, hi):
    """Like FastQwen3.prefill(collect=True) on ids [T], plus attn_mass of the last n_q queries on keys [lo, hi),
    averaged over layers. Returns (hidden states [L+1, T, d], score [hi-lo] fp32)."""
    T = ids.shape[0]
    cos, sin = S.cos[:T], S.sin[:T]
    h = F.embedding(ids, S.embed)
    hs = [h]
    score = torch.zeros(hi - lo, dtype=torch.float32, device=ids.device)
    qpos = torch.arange(T - n_q, T, device=ids.device)
    for lw in S.layers:
        q, k, v = S._qkv(lw, rms(h, lw["ln1"], S.eps), cos, sin)
        o = F.scaled_dot_product_attention(q[None], k.repeat_interleave(S.rep, 0)[None],
                                           v.repeat_interleave(S.rep, 0)[None], is_causal=True, scale=S.scaling)[0]
        score += attn_mass(q[:, T - n_q:], k, qpos, lo, hi, S.scaling)
        h = S._mlp(lw, S._out(lw, h, o))
        hs.append(h)
    hs[-1] = rms(h, S.norm, S.eps)
    return torch.stack(hs, 0), score / len(S.layers)


@torch.no_grad()
def spec_importance(S, ids, n_lah=8, eos=(), kernel=13, collect=False):
    """SpecPrefill token importance, as in the official implementation: the small model prefills ids [T] and greedily
    decodes look-ahead tokens until it has n_lah queries (last prompt token + n_lah-1 look-ahead tokens; it stops early
    after producing EOS). In every layer and head each of these queries attends to the T prompt keys (softmax over the
    prompt only); scores are average-pooled along the prompt (kernel, stride 1), maxed over layers x heads and averaged
    over the queries. Returns (hidden states [L+1, T, d] if collect else None, importance [T] fp32)."""
    T = ids.shape[0]
    kb = torch.empty(S.n_layers, S.Hkv, T + n_lah, S.D, dtype=S.embed.dtype, device=ids.device)
    vb = torch.empty_like(kb)
    qs = []
    logits, hs = S.prefill(ids, kb, vb, collect=collect, last_q=qs)
    qs = [[q] for q in qs]
    for p in range(T, T + n_lah - 1):
        nxt = logits.argmax(-1)
        if int(nxt) in eos:
            break
        h = F.embedding(nxt, S.embed)
        for i, lw in enumerate(S.layers):
            q, k, v = S._qkv(lw, rms(h, lw["ln1"], S.eps), S.cos[p:p + 1], S.sin[p:p + 1])
            kb[i, :, p:p + 1], vb[i, :, p:p + 1] = k, v
            qs[i].append(q)
            o = F.scaled_dot_product_attention(q.reshape(S.Hkv, S.rep, S.D)[None], kb[i, :, :p + 1][None],
                                               vb[i, :, :p + 1][None], scale=S.scaling)[0]
            h = S._mlp(lw, S._out(lw, h, o.reshape(S.H, 1, S.D)))
        logits = F.linear(rms(h, S.norm, S.eps), S.lm_head)
    imp = None
    for i, ql in enumerate(qs):
        q = torch.cat(ql, 1)  # [H, n, D]
        n = q.shape[1]
        a = torch.matmul(q.reshape(S.Hkv, S.rep * n, S.D).float(), kb[i, :, :T].float().transpose(1, 2)) * S.scaling
        a = a.softmax(-1).reshape(S.H, n, T)
        if kernel:
            a = F.avg_pool1d(a, kernel, 1, kernel // 2)
        a = a.amax(0)  # [n, T], max over heads; max over layers below
        imp = a if imp is None else torch.maximum(imp, a)
    return hs, imp.mean(0)


def prefill_kept(L, ids, pos):
    """Large-model prefill of only the tokens at sorted absolute positions pos [K] (the others are dropped).
    Returns compact (kbuf, vbuf [L, Hkv, K, D], last-token logits [1, V])."""
    K = pos.shape[0]
    kbuf = torch.empty(L.n_layers, L.Hkv, K, L.D, dtype=L.embed.dtype, device=ids.device)
    vbuf = torch.empty_like(kbuf)
    logits, _ = L.prefill(ids[pos], kbuf, vbuf, pos=pos)
    return kbuf, vbuf, logits


@torch.no_grad()
def extend_at(L, ids, pos, kbuf, vbuf, n_q=0, lo=0, hi=0):
    """Run tokens ids [Q] at sorted absolute positions pos [Q] through the large model against the KV buffer
    kbuf/vbuf [L, Hkv, T, D]; in every layer their K/V are written to rows `pos` before attention.
    Returns (last-token logits [1, V], attention mass of the last n_q queries on keys [lo, hi) or None)."""
    Q = ids.shape[0]
    T = kbuf.shape[2]
    cos, sin = L.cos[pos], L.sin[pos]
    mask = causal_mask_for(pos, T, L.embed.dtype, ids.device, L.rep)[None, None]
    h = F.embedding(ids, L.embed)
    score = torch.zeros(hi - lo, dtype=torch.float32, device=ids.device) if n_q else None
    for i, lw in enumerate(L.layers):
        q, k, v = L._qkv(lw, rms(h, lw["ln1"], L.eps), cos, sin)
        kbuf[i].index_copy_(1, pos, k)
        vbuf[i].index_copy_(1, pos, v)
        qg = q.reshape(L.Hkv, L.rep * Q, L.D)
        o = F.scaled_dot_product_attention(qg[None], kbuf[i][None], vbuf[i][None], attn_mask=mask, scale=L.scaling)[0]
        if n_q:
            score += attn_mass(q[:, Q - n_q:], kbuf[i], pos[Q - n_q:], lo, hi, L.scaling)
        h = L._mlp(lw, L._out(lw, h, o.reshape(L.H, Q, L.D)))
    logits = F.linear(rms(h[-1:], L.norm, L.eps), L.lm_head)
    return logits, (score / len(L.layers) if n_q else None)


def find_spans(mid_ids, needles, pad=2):
    """Boolean mask [m] of positions covered by any occurrence of any token sequence in `needles`, dilated by `pad`."""
    m = mid_ids.shape[0]
    hit = torch.zeros(m, dtype=torch.bool, device=mid_ids.device)
    for nd in needles:
        n = len(nd)
        if n == 0 or n > m:
            continue
        w = mid_ids.unfold(0, n, 1) == torch.as_tensor(nd, device=mid_ids.device)[None]
        for st in torch.nonzero(w.all(-1)).flatten().tolist():
            hit[max(0, st - pad):min(m, st + n + pad)] = True
    return hit


def segment_ids(mid_ids, seg, boundary=None, cap=64):
    """Unit index [m] of each middle token for fixed blocks (seg=int) or sentence/line segments (seg='sent')."""
    m = mid_ids.shape[0]
    pos = torch.arange(m, device=mid_ids.device)
    if seg != "sent":
        return pos // int(seg)
    b = boundary[mid_ids].long()
    sid = torch.cumsum(torch.cat([b.new_zeros(1), b[:-1]]), 0)  # a new segment starts after every boundary token
    first = torch.full((int(sid[-1]) + 1,), m, device=mid_ids.device, dtype=torch.long).scatter_reduce(0, sid, pos, "amin")
    sub = (pos - first[sid]) // cap  # split segments longer than cap
    return torch.unique(sid * (m + 1) + sub, return_inverse=True)[1]


def choose_units(score, unit, k, reduce="amax"):
    """Sorted token indices of the highest-scoring units (unit score = max, or with reduce='mean' mean, token score)
    until >= k tokens are taken."""
    nu = int(unit.max()) + 1
    us = torch.zeros(nu, device=score.device).scatter_reduce(0, unit, score, reduce, include_self=False)
    ul = torch.bincount(unit, minlength=nu)
    order = torch.argsort(us, descending=True)
    before = torch.cumsum(ul[order], 0) - ul[order]  # tokens already taken before each unit
    keep = torch.zeros(nu, dtype=torch.bool, device=score.device)
    keep[order[before < k]] = True
    return torch.nonzero(keep[unit]).flatten()


def choose(score, k, pool):
    """Indices (sorted) of the k highest max-pooled scores."""
    if pool > 1:
        score = F.max_pool1d(score[None, None], pool, 1, pool // 2)[0, 0, :score.shape[0]]
    return torch.topk(score, k).indices.sort().values


class RecomputeHybrid(FastHybrid):
    """FastHybrid + selective recompute of a fraction `frac` of the middle tokens chosen by `scorer`."""

    def __init__(self, *a, pool=7, seed=0, boundary=None, cap=64, eos=(), n_lah=8, spec_kernel=13, spec_chunk=32, **kw):
        super().__init__(*a, **kw)
        self.pool = pool
        self.eos, self.n_lah, self.spec_kernel, self.spec_chunk = set(eos), n_lah, spec_kernel, spec_chunk
        self.boundary, self.cap = boundary, cap  # boundary: bool [vocab], token contains . ! ? or a newline
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def prefill_sel(self, ids, scorer="attn_big", frac=0.1, needles=None, seg=None, drop=False):
        """ids [T]. Returns (kbuf, vbuf, first-token logits, number of middle tokens recomputed). With drop=True the
        unchosen middle tokens are dropped and kbuf/vbuf hold only sink + chosen + tail (compact)."""
        assert scorer in SCORERS, scorer
        L, dev = self.L, ids.device
        T = ids.shape[0]
        s, m, t = split_prompt(T, self.n_sink, self.n_tail)
        k = min(m, int(round(frac * m)))
        if m == 0 or (k == 0 and scorer != "oracle" and not drop):
            return (*self.prefill(ids), 0)
        fill = not drop or scorer == "attn_big"  # attn_big scores the projected keys, so it needs them even when dropping
        hs = None
        if scorer == "attn_small":
            hs, score = small_prefill_scored(self.S, ids, t, s, s + m)
            hs = hs[:, s:s + m]
        elif scorer == "spec":
            hs, score = spec_importance(self.S, ids, self.n_lah, self.eos, self.spec_kernel, collect=fill)
            hs, score = (hs[:, s:s + m] if fill else None), score[s:s + m]
        elif fill:
            hs = self.stage_small(ids[:s + m])[:, s:]
        if fill:
            kbuf, vbuf = self.stage_fold(hs, s, T)
        ar = lambda a, b: torch.arange(a, b, device=dev)
        if scorer == "attn_big":
            p0 = torch.cat([ar(0, s), ar(s + m, T)])
            _, score = extend_at(L, ids[p0], p0, kbuf, vbuf, n_q=t, lo=s, hi=s + m)
        if k == 0 and scorer != "oracle":
            idx = ar(0, 0)
        elif scorer == "spec" and seg is None:
            idx = choose_units(score, ar(0, m) // self.spec_chunk, k, "mean")
        elif scorer in ("attn_small", "attn_big", "spec") and seg is not None:
            idx = choose_units(score, segment_ids(ids[s:s + m], seg, self.boundary, self.cap), k)
        elif scorer in ("attn_small", "attn_big"):
            idx = choose(score, k, self.pool)
        elif scorer == "random":
            idx = torch.randperm(m, generator=self.gen)[:k].sort().values.to(dev)
        elif scorer == "recent":
            idx = ar(m - k, m)
        else:  # oracle
            idx = torch.nonzero(find_spans(ids[s:s + m], needles or [])).flatten()
        pos = torch.cat([ar(0, s), s + idx, ar(s + m, T)])
        if drop:
            return (*prefill_kept(L, ids, pos), int(idx.numel()))
        logits, _ = extend_at(L, ids[pos], pos, kbuf, vbuf)
        return kbuf, vbuf, logits, int(idx.numel())

    @torch.no_grad()
    def prefill_specprefill(self, ids, frac=0.1, n_lah=None):
        """SpecPrefill baseline (official defaults: n_lah 8, average-pool kernel 13, 32-token chunks scored by their mean
        importance, keep ceil(frac x number of chunks) chunks of the WHOLE prompt; no sink / tail are forced, no
        projector). Returns (compact kbuf, vbuf, first-token logits, number of prompt tokens kept)."""
        T = ids.shape[0]
        _, imp = spec_importance(self.S, ids, n_lah or self.n_lah, self.eos, self.spec_kernel)
        c = self.spec_chunk
        nc = (T + c - 1) // c
        unit = torch.arange(T, device=ids.device) // c
        cs = torch.zeros(nc, device=ids.device).scatter_reduce(0, unit, imp, "mean", include_self=False)
        keep = torch.zeros(nc, dtype=torch.bool, device=ids.device)
        keep[torch.topk(cs, math.ceil(nc * frac)).indices] = True
        pos = torch.nonzero(keep[unit]).flatten()
        return (*prefill_kept(self.L, ids, pos), int(pos.numel()))
