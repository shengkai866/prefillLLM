"""
Piecewise CUDA graphs of every prefill method, for a TTFT comparison in which no method is penalised by kernel-launch
overhead (scripts/eval_recompute.py task speed_allgraph). Every fixed-shape stage is captured once for the prompt
being timed and replayed; only the data-dependent selection between stages runs eagerly (top-k / nonzero over the
scores, and SpecPrefill's per-step EOS check of the look-ahead), as in a serving engine. The selected token sets have a
fixed size for a fixed prompt, so the later stages are captured for exactly that size.

  big, small, hyb             one graph for the whole prefill
  ours   (attn_small:f:sent)  g1 small prefill + tail attention scores -> eager sentence selection
                              -> g2 projector fold + large-model recompute of sink, chosen sentences and tail
  drop   (drop-attn_small)    g1 -> eager selection -> g2 large-model prefill of the kept tokens only
  spec   (specprefill:f)      g1 small prefill into the look-ahead buffer -> look-ahead steps (LookaheadGraph)
                              -> g2 importance -> eager chunk selection -> g3 large-model prefill of the kept tokens
"""
import torch

from .fast import FastBig, Graphed
from .kv import split_prompt
from .recompute import (LookaheadGraph, choose_units, extend_at, prefill_kept, segment_ids, small_prefill_scored,
                        spec_scores, spec_select)


def build(rh, kind, frac, ids):
    """rh: RecomputeHybrid. Captures the graphs of method `kind` for the prompt ids [T] and returns run(ids) ->
    first-token logits [1, V] (graph-owned). run must be called with this same ids tensor."""
    S, L = rh.S, rh.L
    dev = ids.device
    T = ids.shape[0]
    s, m, t = split_prompt(T, rh.n_sink, rh.n_tail)
    ar = lambda a, b: torch.arange(a, b, device=dev)
    nocopy = dict(copy_inputs=False)

    if kind == "big":
        fb = FastBig(L)
        g = Graphed(fb.prefill, ids, **nocopy)
        return lambda x: g(x)[2]
    if kind == "small":
        g = Graphed(lambda x: S.prefill(x)[0], ids, **nocopy)
        return g
    if kind == "hyb":
        g = Graphed(rh.prefill, ids, **nocopy)
        return lambda x: g(x)[2]

    if kind in ("ours", "drop"):
        k = min(m, int(round(frac * m)))
        g1 = Graphed(lambda x: small_prefill_scored(S, x, t, s, s + m), ids, **nocopy)

        def select(x):
            hs, score = g1(x)
            idx = choose_units(score, segment_ids(x[s:s + m], "sent", rh.boundary, rh.cap), k)
            return hs, torch.cat([ar(0, s), s + idx, ar(s + m, T)])

        hs0, pos0 = select(ids)
        if kind == "ours":
            def tail(hs, x, pos):
                kb, vb = rh.stage_fold(hs[:, s:s + m], s, T)
                return extend_at(L, x[pos], pos, kb, vb)[0]
            g2 = Graphed(tail, hs0, ids, pos0, **nocopy)
        else:
            g2 = Graphed(lambda x, pos: prefill_kept(L, x, pos)[2], ids, pos0, **nocopy)

        def run(x):
            hs, pos = select(x)
            assert pos.shape == pos0.shape, "selection size changed; graphs are per prompt"
            return g2(hs, x, pos) if kind == "ours" else g2(x, pos)
        return run

    if kind == "spec":
        n_lah = rh.n_lah
        lah = LookaheadGraph(S, T, n_lah)

        def head(x):
            qs = []
            logits, _ = S.prefill(x, lah.kb, lah.vb, last_q=qs)
            for i, q in enumerate(qs):
                lah.qb[i, :, :1] = q
            return logits
        g1 = Graphed(head, ids, **nocopy)
        imp_graphs = {}
        eos = S.own_eos(rh.eos)

        def importance(x):
            logits = g1(x)
            n = 1
            for p in range(T, T + n_lah - 1):
                nxt = logits.argmax(-1)
                if int(nxt) in eos:
                    break
                logits = lah.step(nxt, p)
                n += 1
            if n not in imp_graphs:
                imp_graphs[n] = Graphed(lambda: spec_scores(S, [lah.qb[i, :, :n] for i in range(S.n_layers)], lah.kb, T,
                                                            rh.spec_kernel))
            return imp_graphs[n]()

        pos0 = spec_select(importance(ids), frac, rh.spec_chunk)
        g3 = Graphed(lambda x, pos: prefill_kept(L, x, pos)[2], ids, pos0, **nocopy)

        def run(x):
            pos = spec_select(importance(x), frac, rh.spec_chunk)
            assert pos.shape == pos0.shape, "selection size changed; graphs are per prompt"
            return g3(x, pos)
        return run

    raise ValueError(kind)
