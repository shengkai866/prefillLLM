"""
Serving-side costs beyond batch-1 TTFT (scripts/bench_e2e.py):

  Decoder              greedy decoding of FastQwen3 against a static KV buffer, one step = one CUDA-graph replay
                       (token, RoPE position and buffer row live on the device, so N steps need no host sync). The
                       step attends over the whole buffer under a mask (keys > row are masked), so its cost is that
                       of the final cache length: an upper bound by at most N / (cache + N) of the attention part.
  dense_prefill_b      batched causal prefill of B equal-length prompts [B, T] (large or small model alone), KV
                       written to kb [B, L, Hkv, T, D] (same math as FastQwen3.prefill per row).
  OursBatched          batched "ours": small-model prefill + tail attention scores for B prompts in one call, eager
                       sentence selection per prompt, projector fold per prompt, then ONE large-model call over
                       the sink + chosen + tail tokens of all B prompts (extend_b). For equal shapes every prompt
                       recomputes exactly k = round(frac * m) middle tokens: the selected sentences are trimmed /
                       topped up with the next-best tokens (timing only; quality runs use RecomputeHybrid).
Everything is batch-major: kb[b] is one prompt's buffer [L, Hkv, T, D], laid out like the batch-1 code.
"""
import torch
import torch.nn.functional as F

from .fast import causal_mask_for, rms
from .kv import split_prompt
from .recompute import attn_mass, choose_units, segment_ids


class Decoder:
    """Greedy decode steps of model M (FastQwen3) against kb/vb [L, Hkv, Cp, D] (Cp = capacity rounded up to 16).
    reset(n_cached, pos0, tok) marks rows < n_cached as the prompt cache; the next token goes to row n_cached with
    RoPE position pos0 (pos0 != n_cached for compact caches, e.g. SpecPrefill keeps original positions)."""

    def __init__(self, M, capacity, kb=None, vb=None, graph=True):
        self.M = M
        dev, dt = M.embed.device, M.embed.dtype
        Cp = (capacity + 15) // 16 * 16
        shape = (M.n_layers, M.Hkv, Cp, M.D)
        if kb is None:  # timing only: random cache contents (decode cost does not depend on them)
            kb, vb = torch.randn(shape, dtype=dt, device=dev) * 0.5, torch.randn(shape, dtype=dt, device=dev) * 0.5
        assert kb.shape[2] == Cp, (kb.shape, Cp)
        self.kb, self.vb = kb, vb
        self.mask = torch.zeros(Cp, dtype=torch.float32, device=dev)  # additive, shared by every head
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.row = torch.zeros(1, dtype=torch.long, device=dev)
        self.g = None
        if graph:
            from .fast import Graphed
            self.reset(Cp - 4, Cp - 4, 0)  # Graphed runs 2 warm-up steps (rows Cp-4, Cp-3) before the capture
            self.g = Graphed(self._step)
        self.reset(0, 0, 0)

    @staticmethod
    def capacity_rows(capacity):
        return (capacity + 15) // 16 * 16

    def reset(self, n_cached, pos0, tok):
        self.mask.fill_(float("-inf"))
        self.mask[:n_cached] = 0
        self.row.fill_(n_cached)
        self.pos.fill_(pos0)
        self.tok.fill_(int(tok))

    def _step(self):
        M = self.M
        self.mask.index_fill_(0, self.row, 0.0)
        cos, sin = M.cos[self.pos], M.sin[self.pos]
        h = F.embedding(M.ids_in(self.tok), M.embed)
        for i, lw in enumerate(M.layers):
            q, k, v = M._qkv(lw, rms(h, lw["ln1"], M.eps), cos, sin)
            self.kb[i].index_copy_(1, self.row, k)
            self.vb[i].index_copy_(1, self.row, v)
            # single-token attention as two batched GEMMs over the whole buffer (bandwidth-bound, like flash-decoding;
            # SDPA with a mask runs one block per KV head here and leaves the GPU mostly idle on long caches)
            qg = q.reshape(M.Hkv, M.rep, M.D)
            s = torch.matmul(qg, self.kb[i].transpose(1, 2)).float() * M.scaling + self.mask  # [Hkv, rep, Cp]
            o = torch.matmul(torch.softmax(s, -1).to(qg.dtype), self.vb[i])  # [Hkv, rep, D]
            h = M._mlp(lw, M._out(lw, h, o.reshape(M.H, 1, M.D)))
        logits = F.linear(rms(h, M.norm, M.eps), M.lm_head)
        self.tok.copy_(logits.argmax(-1))
        self.pos.add_(1)
        self.row.add_(1)
        return logits

    def step(self):
        """One decode step: feeds self.tok, leaves the next greedy token in self.tok. Returns logits [1, V]."""
        if self.g is not None:
            self.g.graph.replay()
            return self.g.out
        return self._step()


def _qkv_b(M, lw, h, cos, sin, B, n):
    """h [B*n, d] -> q [B, H, n, D], k / v [B, Hkv, n, D]."""
    q, k, v = M._qkv(lw, rms(h, lw["ln1"], M.eps), cos, sin)  # [H, B*n, D] (rows are b-major)
    return (q.view(M.H, B, n, M.D).transpose(0, 1), k.view(M.Hkv, B, n, M.D).transpose(0, 1),
            v.view(M.Hkv, B, n, M.D).transpose(0, 1))


def _out_b(M, lw, h, o, B, n):
    """o [B, H, n, D] -> residual update of h [B*n, d] (o_proj + MLP)."""
    o = o.transpose(0, 1).reshape(M.H, B * n, M.D)
    return M._mlp(lw, M._out(lw, h, o))


def dense_prefill_b(M, ids, kb=None, vb=None, collect=False, score=None):
    """ids [B, T], prefill from position 0. kb/vb [B, L, Hkv, >=T, D] receive K/V if given.
    collect: also return hidden states [L+1, B, T, d] (HF convention, last one normed).
    score=(n_q, lo, hi): also return the attention mass of every prompt's last n_q queries on keys [lo, hi),
    averaged over layers -> [B, hi-lo] (as small_prefill_scored).
    Returns (last-token logits [B, V], hidden states or None, scores or None)."""
    B, T = ids.shape
    cos, sin = M.cos[:T].repeat(B, 1), M.sin[:T].repeat(B, 1)
    h = F.embedding(M.ids_in(ids.reshape(-1), first=False), M.embed)  # same-tokenizer pairs only (no BOS swap)
    hs = None
    if collect:
        hs = torch.empty(M.n_layers + 1, B, T, h.shape[-1], dtype=h.dtype, device=h.device)
        hs[0] = h.view(B, T, -1)
    sc = None
    if score is not None:
        n_q, lo, hi = score
        sc = torch.zeros(B, hi - lo, dtype=torch.float32, device=ids.device)
        qpos = torch.arange(T - n_q, T, device=ids.device)
    for i, lw in enumerate(M.layers):
        q, k, v = _qkv_b(M, lw, h, cos, sin, B, T)
        if kb is not None:
            kb[:, i, :, :T] = k
            vb[:, i, :, :T] = v
        o = F.scaled_dot_product_attention(q, k.repeat_interleave(M.rep, 1), v.repeat_interleave(M.rep, 1),
                                           is_causal=True, scale=M.scaling)
        if sc is not None:
            for b in range(B):
                sc[b] += attn_mass(q[b, :, T - n_q:], k[b], qpos, lo, hi, M.scaling)
        h = _out_b(M, lw, h, o, B, T)
        if collect:
            hs[i + 1] = h.view(B, T, -1)
    hT = h.view(B, T, -1)
    if collect:
        hs[-1] = rms(hT, M.norm, M.eps)
    logits = F.linear(rms(hT[:, -1], M.norm, M.eps), M.lm_head)
    return logits, hs, (None if sc is None else sc / M.n_layers)


def extend_b(L, ids, pos, kb, vb):
    """Batched extend_at: tokens ids [B, Q] at sorted absolute positions pos [B, Q] (rows = positions) through the
    large model against kb/vb [B, L, Hkv, T, D]; each prompt's K/V rows are written before attention and a query sees
    rows <= its position. Returns last-token logits [B, V]."""
    B, Q = ids.shape
    T = kb.shape[3]
    cos, sin = L.cos[pos.reshape(-1)], L.sin[pos.reshape(-1)]
    mask = torch.stack([causal_mask_for(pos[b], T, L.embed.dtype, ids.device, L.rep) for b in range(B)])[:, None]
    h = F.embedding(L.ids_in(ids.reshape(-1)), L.embed)
    for i, lw in enumerate(L.layers):
        q, k, v = _qkv_b(L, lw, h, cos, sin, B, Q)
        for b in range(B):
            kb[b, i].index_copy_(1, pos[b], k[b])
            vb[b, i].index_copy_(1, pos[b], v[b])
        qg = q.reshape(B, L.Hkv, L.rep * Q, L.D)
        o = F.scaled_dot_product_attention(qg, kb[:, i], vb[:, i], attn_mask=mask, scale=L.scaling)
        h = _out_b(L, lw, h, o.reshape(B, L.H, Q, L.D), B, Q)
    return F.linear(rms(h.view(B, Q, -1)[:, -1], L.norm, L.eps), L.lm_head)


class OursBatched:
    """Batched ours (or hyb with frac=0) on top of a RecomputeHybrid rh (its S, L, fold, n_sink, n_tail, boundary).
    stage1(ids) -> (hidden states, scores); select(ids, scores) -> pos [B, Q] (eager);
    stage2(hs, ids, pos) -> logits (fold per prompt + one batched large call). run(ids) chains them."""

    def __init__(self, rh, T, frac):
        self.rh, self.T, self.frac = rh, T, frac
        self.s, self.m, self.t = split_prompt(T, rh.n_sink, rh.n_tail)
        self.k = min(self.m, int(round(frac * self.m)))

    def stage1(self, ids):
        _, hs, sc = dense_prefill_b(self.rh.S, ids[:, :self.s + self.m + self.t], collect=True,
                                    score=(self.t, self.s, self.s + self.m) if self.k else None)
        return hs, sc

    def select(self, ids, sc):
        s, m, T, k = self.s, self.m, self.T, self.k
        dev = ids.device
        ar = lambda a, b: torch.arange(a, b, device=dev)
        out = []
        for b in range(ids.shape[0]):
            if k == 0:
                idx = ar(0, 0)
            else:
                idx = choose_units(sc[b], segment_ids(ids[b, s:s + m], "sent", self.rh.boundary, self.rh.cap), k)
                if idx.numel() != k:  # equal shapes across the batch: trim / top up with the best remaining tokens
                    keep = torch.zeros(m, dtype=torch.bool, device=dev)
                    keep[idx] = True
                    pri = sc[b] + 1e3 * keep.float()
                    idx = torch.topk(pri, k).indices.sort().values
            out.append(torch.cat([ar(0, s), s + idx, ar(s + m, T)]))
        return torch.stack(out)

    def stage2(self, hs, ids, pos, kb, vb):
        rh, s, m = self.rh, self.s, self.m
        L = rh.L
        for b in range(ids.shape[0]):
            rh.fold(hs[:, b, s:s + m], L.cos[s:s + m], L.sin[s:s + m], kb[b, :, :, s:s + m], vb[b, :, :, s:s + m])
        sel = torch.gather(ids, 1, pos)
        return extend_b(L, sel, pos, kb, vb)

    def new_buffers(self, B):
        L = self.rh.L
        kb = torch.empty(B, L.n_layers, L.Hkv, self.T, L.D, dtype=L.embed.dtype, device=L.embed.device)
        return kb, torch.empty_like(kb)
