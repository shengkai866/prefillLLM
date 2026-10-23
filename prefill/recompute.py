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
Selection unit (`seg`):
  None        single tokens; attention scores are max-pooled over a small window so the neighbours of a hit come with it
  int B       fixed blocks of B consecutive tokens, block score = max token score, whole blocks are recomputed
  "sent"      sentence/line segments (a segment ends after a token containing . ! ? or a newline; capped at `cap` tokens),
              segment score = max token score, whole segments are recomputed
Block/segment selection takes units in score order until the token budget is reached (the last unit may overshoot).
"""
import torch
import torch.nn.functional as F

from .fast import FastHybrid, causal_mask_for, rms
from .kv import split_prompt

SCORERS = ("random", "recent", "attn_small", "attn_big", "oracle")


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


def choose_units(score, unit, k):
    """Sorted token indices of the highest-scoring units (unit score = max token score) until >= k tokens are taken."""
    nu = int(unit.max()) + 1
    us = torch.full((nu,), float("-inf"), device=score.device).scatter_reduce(0, unit, score, "amax")
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

    def __init__(self, *a, pool=7, seed=0, boundary=None, cap=64, **kw):
        super().__init__(*a, **kw)
        self.pool = pool
        self.boundary, self.cap = boundary, cap  # boundary: bool [vocab], token contains . ! ? or a newline
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def prefill_sel(self, ids, scorer="attn_big", frac=0.1, needles=None, seg=None):
        """ids [T]. Returns (kbuf, vbuf, first-token logits, number of middle tokens recomputed)."""
        assert scorer in SCORERS, scorer
        L, dev = self.L, ids.device
        T = ids.shape[0]
        s, m, t = split_prompt(T, self.n_sink, self.n_tail)
        k = min(m, int(round(frac * m)))
        if m == 0 or (k == 0 and scorer != "oracle"):
            return (*self.prefill(ids), 0)
        if scorer == "attn_small":
            hs, score = small_prefill_scored(self.S, ids, t, s, s + m)
            hs = hs[:, s:s + m]
        else:
            hs = self.stage_small(ids[:s + m])[:, s:]
        kbuf, vbuf = self.stage_fold(hs, s, T)
        ar = lambda a, b: torch.arange(a, b, device=dev)
        if scorer == "attn_big":
            p0 = torch.cat([ar(0, s), ar(s + m, T)])
            _, score = extend_at(L, ids[p0], p0, kbuf, vbuf, n_q=t, lo=s, hi=s + m)
        if scorer in ("attn_small", "attn_big") and seg is not None:
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
        logits, _ = extend_at(L, ids[pos], pos, kbuf, vbuf)
        return kbuf, vbuf, logits, int(idx.numel())
