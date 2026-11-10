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

keep=<f> (branch proj-drop): recompute + project + drop. Of the middle tokens not recomputed, only the share `keep`
(next in small-model attention order, or random units) gets projected KV; the rest is dropped (compact cache).

drop=True (controlled SpecPrefill-style ablation): same sink / tail / selection, but the middle tokens NOT chosen are
dropped instead of filled with projected KV; the large model prefills only the kept tokens at their original positions
and the returned cache is compact (decoding must continue at position T).

SpecPrefill itself (Liu, Chen, Zhang, ICML 2025; official code github.com/Jingyu6/speculative_prefill), as the external
baseline: RecomputeHybrid.prefill_specprefill — whole-prompt chunk selection, no sink / tail, no projector.
"""
import math

import torch
import torch.nn.functional as F

from .fast import FastHybrid, Graphed, causal_mask_for, rms
from .fast import attn_enabled, pos_attention
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
    h = F.embedding(S.ids_in(ids, first=True), S.embed)
    hs = torch.empty(len(S.layers) + 1, T, h.shape[-1], dtype=h.dtype, device=h.device)  # in place, see FastQwen3.prefill
    hs[0] = h
    score = torch.zeros(hi - lo, dtype=torch.float32, device=ids.device)
    qpos = torch.arange(T - n_q, T, device=ids.device)
    for i, lw in enumerate(S.layers):
        q, k, v = S._qkv(lw, rms(h, lw["ln1"], S.eps), cos, sin)
        o = F.scaled_dot_product_attention(q[None], k.repeat_interleave(S.rep, 0)[None],
                                           v.repeat_interleave(S.rep, 0)[None], is_causal=True, scale=S.scaling)[0]
        score += attn_mass(q[:, T - n_q:], k, qpos, lo, hi, S.scaling)
        h = S._mlp(lw, S._out(lw, h, o))
        hs[i + 1] = h
    hs[-1] = rms(h, S.norm, S.eps)
    return hs, score / len(S.layers)


class LookaheadGraph:
    """CUDA graph of one SpecPrefill look-ahead decode step of the small model for a fixed prompt length T (the eager
    loop is launch-bound: ~20 ms per step for 0.6B on an L40S; vLLM, used by the official code, also replays graphs).
    The token at absolute position p (a device tensor) writes its K/V into row p of the static buffer kb/vb
    [L, Hkv, T + n_lah, D], attends to it under a mask over keys <= p (same math as the eager slice :p+1) and stores its
    queries in qb [L, H, n_lah + 1, D] at slot p - T + 1. Row T + n_lah - 1 and slot n_lah are scratch for capture."""

    def __init__(self, S, T, n_lah):
        dev, dt = S.embed.device, S.embed.dtype
        self.S, self.T, self.n_lah = S, T, n_lah
        self.kb = torch.zeros(S.n_layers, S.Hkv, T + n_lah, S.D, dtype=dt, device=dev)  # rows > p are masked, but
        self.vb = torch.zeros_like(self.kb)                                               # must not hold NaN patterns
        self.qb = torch.zeros(S.n_layers, S.H, n_lah + 1, S.D, dtype=dt, device=dev)
        self.g = Graphed(self._step, torch.zeros(1, dtype=torch.long, device=dev),
                         torch.tensor([T + n_lah - 1], dtype=torch.long, device=dev))

    def _step(self, tok, p):
        S = self.S
        cos, sin = S.cos[p], S.sin[p]
        mask = causal_mask_for(p, self.kb.shape[2], S.embed.dtype, p.device, S.rep)[None, None]
        slot = p - self.T + 1
        h = F.embedding(tok, S.embed)
        for i, lw in enumerate(S.layers):
            q, k, v = S._qkv(lw, rms(h, lw["ln1"], S.eps), cos, sin)
            self.kb[i].index_copy_(1, p, k)
            self.vb[i].index_copy_(1, p, v)
            self.qb[i].index_copy_(1, slot, q)
            o = F.scaled_dot_product_attention(q.reshape(S.Hkv, S.rep, S.D)[None], self.kb[i][None], self.vb[i][None],
                                               attn_mask=mask, scale=S.scaling)[0]
            h = S._mlp(lw, S._out(lw, h, o.reshape(S.H, 1, S.D)))
        return F.linear(rms(h, S.norm, S.eps), S.lm_head)

    def step(self, tok, p):
        """tok [1] long on the device, p int -> logits [1, V] (graph-owned, overwritten by the next step)."""
        return self.g(tok, torch.tensor([p], dtype=torch.long, device=tok.device))


@torch.no_grad()
def spec_importance(S, ids, n_lah=8, eos=(), kernel=13, collect=False, graph=None):
    """SpecPrefill token importance, as in the official implementation: the small model prefills ids [T] and greedily
    decodes look-ahead tokens until it has n_lah queries (last prompt token + n_lah-1 look-ahead tokens; it stops early
    after producing EOS). In every layer and head each of these queries attends to the T prompt keys (softmax over the
    prompt only); scores are average-pooled along the prompt (kernel, stride 1), maxed over layers x heads and averaged
    over the queries. Returns (hidden states [L+1, T, d] if collect else None, importance [T] fp32).
    graph: a LookaheadGraph for (T, n_lah) to replay the look-ahead steps instead of running them eagerly."""
    T = ids.shape[0]
    eos = S.own_eos(eos)  # the look-ahead tokens are the small model's own
    if graph is not None:
        kb, vb = graph.kb, graph.vb
    else:
        kb = torch.empty(S.n_layers, S.Hkv, T + n_lah, S.D, dtype=S.embed.dtype, device=ids.device)
        vb = torch.empty_like(kb)
    qs = []
    logits, hs = S.prefill(ids, kb, vb, collect=collect, last_q=qs)
    if graph is not None:
        for i, q in enumerate(qs):
            graph.qb[i, :, :1] = q
        n = 1
        for p in range(T, T + n_lah - 1):
            nxt = logits.argmax(-1)
            if int(nxt) in eos:
                break
            logits = graph.step(nxt, p)
            n += 1
        qs = [[graph.qb[i, :, :n]] for i in range(S.n_layers)]
    else:
        qs = [[q] for q in qs]
    for p in range(T, T + n_lah - 1) if graph is None else ():
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
    return hs, spec_scores(S, [torch.cat(ql, 1) for ql in qs], kb, T, kernel)


def spec_scores(S, qs, kb, T, kernel=13):
    """SpecPrefill importance [T] fp32 from the per-layer look-ahead queries qs [L] x [H, n, D] against the prompt keys
    kb[:, :, :T] (see spec_importance)."""
    imp = None
    for i, q in enumerate(qs):
        n = q.shape[1]
        a = torch.matmul(q.reshape(S.Hkv, S.rep * n, S.D).float(), kb[i, :, :T].float().transpose(1, 2)) * S.scaling
        a = a.softmax(-1).reshape(S.H, n, T)
        if kernel:
            a = F.avg_pool1d(a, kernel, 1, kernel // 2)
        a = a.amax(0)  # [n, T], max over heads; max over layers below
        imp = a if imp is None else torch.maximum(imp, a)
    return imp.mean(0)


def prefill_kept(L, ids, pos, extra=0):
    """Large-model prefill of only the tokens at sorted absolute positions pos [K] (the others are dropped).
    Returns compact (kbuf, vbuf [L, Hkv, K + extra, D], last-token logits [1, V]); the `extra` trailing rows are
    left free for tokens appended later (extend_at with rows=)."""
    K = pos.shape[0]
    kbuf = torch.zeros(L.n_layers, L.Hkv, K + extra, L.D, dtype=L.embed.dtype, device=ids.device)  # free rows are
    vbuf = torch.zeros_like(kbuf)                                                                  # masked, not NaN
    logits, _ = L.prefill(ids[pos], kbuf, vbuf, pos=pos)
    return kbuf, vbuf, logits


def spec_select(imp, frac, chunk=32, T=None):
    """SpecPrefill chunk selection: positions (sorted) of the ceil(frac x number of chunks) `chunk`-token chunks of
    imp[:T] with the highest mean importance."""
    T = imp.shape[0] if T is None else T
    nc = (T + chunk - 1) // chunk
    unit = torch.arange(T, device=imp.device) // chunk
    cs = torch.zeros(nc, device=imp.device).scatter_reduce(0, unit, imp[:T], "mean", include_self=False)
    keep = torch.zeros(nc, dtype=torch.bool, device=imp.device)
    keep[torch.topk(cs, math.ceil(nc * frac)).indices] = True
    return torch.nonzero(keep[unit]).flatten()


@torch.no_grad()
def extend_at(L, ids, pos, kbuf, vbuf, n_q=0, lo=0, hi=0, rows=None):
    """Run tokens ids [Q] at sorted absolute positions pos [Q] through the model L against the KV buffer
    kbuf/vbuf [L, Hkv, T, D]; in every layer their K/V are written to buffer rows `rows` (default: pos) before
    attention, and a query at row r sees buffer rows <= r. rows differs from pos for compact caches (dropped tokens),
    whose rows 0..K-1 hold earlier positions.
    Returns (last-token logits [1, V], attention mass of the last n_q queries on buffer rows [lo, hi) or None)."""
    Q = ids.shape[0]
    T = kbuf.shape[2]
    rows = pos if rows is None else rows
    cos, sin = L.cos[pos], L.sin[pos]
    fused = attn_enabled(ids.device)  # Triton position-causal attention (prefill/attn_fused.py) instead of masked SDPA
    mask = None if fused else causal_mask_for(rows, T, L.embed.dtype, ids.device, L.rep)[None, None]
    h = F.embedding(L.ids_in(ids), L.embed)
    score = torch.zeros(hi - lo, dtype=torch.float32, device=ids.device) if n_q else None
    for i, lw in enumerate(L.layers):
        q, k, v = L._qkv(lw, rms(h, lw["ln1"], L.eps), cos, sin)
        kbuf[i].index_copy_(1, rows, k)
        vbuf[i].index_copy_(1, rows, v)
        if fused:
            o = pos_attention(q, kbuf[i], vbuf[i], rows, L.scaling)
        else:
            qg = q.reshape(L.Hkv, L.rep * Q, L.D)
            o = F.scaled_dot_product_attention(qg[None], kbuf[i][None], vbuf[i][None], attn_mask=mask, scale=L.scaling)[0]
        if n_q:
            score += attn_mass(q[:, Q - n_q:], kbuf[i], rows[Q - n_q:], lo, hi, L.scaling)
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
        self.lah_graph, self._lah = False, None  # lah_graph=True: replay SpecPrefill look-ahead steps as a CUDA graph

    def lah(self, T, n_lah):
        """The LookaheadGraph for (T, n_lah) if graphs are on (one cached at a time), else None."""
        if not self.lah_graph:
            self._lah = None
            return None
        if self._lah is None or (self._lah.T, self._lah.n_lah) != (T, n_lah):
            self._lah = None
            torch.cuda.empty_cache()
            self._lah = LookaheadGraph(self.S, T, n_lah)
        return self._lah

    @torch.no_grad()
    def prefill_sel(self, ids, scorer="attn_big", frac=0.1, needles=None, seg=None, drop=False, keep=None, keep_mode="attn"):
        """ids [T]. Returns (kbuf, vbuf, first-token logits, number of middle tokens recomputed). With drop=True the
        unchosen middle tokens are dropped and kbuf/vbuf hold only sink + chosen + tail (compact).
        keep (branch proj-drop, recompute + project + drop): of the middle tokens NOT recomputed, only a fraction `keep`
        gets projected KV, the rest is dropped (compact cache). keep_mode 'attn': the kept units are the next ones in
        the same score order (the least attended are dropped); 'random': random units. keep=1 == normal, keep=0 == drop."""
        if keep is not None:
            return self._prefill_mix(ids, scorer, frac, seg, keep, keep_mode)
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
            hs, score = spec_importance(self.S, ids, self.n_lah, self.eos, self.spec_kernel, collect=fill,
                                        graph=self.lah(T, self.n_lah))
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
    def _prefill_mix(self, ids, scorer, frac, seg, keep, keep_mode):
        """Three-way split of the middle: recompute (top `frac` units by small-model attention) | project (the next
        `keep` share of the remaining tokens, or random units) | drop (the rest). Only the kept tokens are folded.
        Returns compact (kbuf, vbuf, first-token logits, number recomputed); decoding continues at position T."""
        assert scorer == "attn_small" and seg is not None, "mix configs: attn_small with units"
        L, dev = self.L, ids.device
        T = ids.shape[0]
        s, m, t = split_prompt(T, self.n_sink, self.n_tail)
        ar = lambda a, b: torch.arange(a, b, device=dev)
        if m == 0:
            return (*self.prefill(ids), 0)
        k = min(m, int(round(frac * m)))
        hs, score = small_prefill_scored(self.S, ids, t, s, s + m)
        hs = hs[:, s:s + m]
        unit = segment_ids(ids[s:s + m], seg, self.boundary, self.cap)
        idx = choose_units(score, unit, k) if k > 0 else ar(0, 0)
        chosen = torch.zeros(m, dtype=torch.bool, device=dev)
        chosen[idx] = True
        rest = int(m - idx.numel())
        nk = int(round(keep * rest))
        if nk >= rest:
            pidx = torch.nonzero(~chosen).flatten()
        elif nk <= 0:
            pidx = ar(0, 0)
        else:
            nu = int(unit.max()) + 1
            if keep_mode == "random":
                us = torch.rand(nu, generator=self.gen).to(dev)
            else:
                us = torch.zeros(nu, device=dev).scatter_reduce(0, unit, score, "amax", include_self=False)
            ul = torch.bincount(unit, minlength=nu)
            uch = torch.zeros(nu, dtype=torch.bool, device=dev)
            uch[unit[idx]] = True  # recomputed units are not candidates
            order = torch.argsort(us, descending=True)
            order = order[~uch[order]]
            before = torch.cumsum(ul[order], 0) - ul[order]
            kp = torch.zeros(nu, dtype=torch.bool, device=dev)
            kp[order[before < nk]] = True
            pidx = torch.nonzero(kp[unit] & ~chosen).flatten()
        q = torch.cat([ar(0, s), s + idx, ar(s + m, T)])  # computed by the large model
        allpos = torch.cat([q, s + pidx]).sort().values    # positions held by the compact cache
        K = allpos.shape[0]
        kbuf = torch.zeros(L.n_layers, L.Hkv, K, L.D, dtype=L.embed.dtype, device=dev)
        vbuf = torch.zeros_like(kbuf)
        if pidx.numel():
            kt = torch.empty(L.n_layers, L.Hkv, pidx.numel(), L.D, dtype=L.embed.dtype, device=dev)
            vt = torch.empty_like(kt)
            pp = s + pidx
            self.fold(hs[:, pidx].contiguous(), L.cos[pp].contiguous(), L.sin[pp].contiguous(), kt, vt)
            rp = torch.searchsorted(allpos, pp)
            kbuf.index_copy_(2, rp, kt)
            vbuf.index_copy_(2, rp, vt)
        logits, _ = extend_at(L, ids[q], q, kbuf, vbuf, rows=torch.searchsorted(allpos, q))
        return kbuf, vbuf, logits, int(idx.numel())

    @torch.no_grad()
    def prefill_specprefill(self, ids, frac=0.1, n_lah=None):
        """SpecPrefill baseline (official defaults: n_lah 8, average-pool kernel 13, 32-token chunks scored by their mean
        importance, keep ceil(frac x number of chunks) chunks of the WHOLE prompt; no sink / tail are forced, no
        projector). Returns (compact kbuf, vbuf, first-token logits, number of prompt tokens kept)."""
        T = ids.shape[0]
        n_lah = n_lah or self.n_lah
        _, imp = spec_importance(self.S, ids, n_lah, self.eos, self.spec_kernel, graph=self.lah(T, n_lah))
        pos = spec_select(imp, frac, self.spec_chunk)
        return (*prefill_kept(self.L, ids, pos), int(pos.numel()))
