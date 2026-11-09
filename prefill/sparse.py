"""
Sparse-attention prefill baseline: FlexPrefill (Lai et al., ICLR 2025; official Triton op vendored in
prefill/third_party/flex_prefill_attention.py). The large model prefills the WHOLE prompt itself, but every layer's
causal attention is replaced by FlexPrefill's block-sparse attention: per head, the last query block decides between a
query-aware block pattern and a vertical-slash pattern (the MInference pattern), and keeps the fewest blocks whose
estimated attention mass reaches gamma (min_budget / max_budget tokens). Everything else (projections, MLP, RoPE) is the
dense lean forward of FastQwen3, so the result is a complete, question-independent cache like ours.
Defaults are the official ones (README): block 128, gamma 0.9, tau 0.1, min_budget 512, max_budget None.
FlexPrefill's index selection runs per layer and per head on the host (torch.unique, .item()), so it is not
CUDA-graphable; it is timed eagerly, exactly as released.
"""
import torch
import torch.nn.functional as F

from .fast import rms

DEFAULTS = dict(gamma=0.9, tau=0.1, min_budget=512, max_budget=None, block_size=128)


def flex_attention_layer(q, k, v, scale, gamma, tau, min_budget, max_budget, block_size):
    """q [H, T, D], k / v [Hkv, T, D] (RoPE applied) -> (o [H, T, D], fraction of causal blocks computed)."""
    from .third_party.flex_prefill_attention import flex_prefill_attention
    qb, kb, vb = (x.transpose(0, 1).contiguous()[None] for x in (q, k, v))  # [1, T, heads, D]
    o, ratio = flex_prefill_attention(qb, kb, vb, gamma, tau, min_budget=min_budget, max_budget=max_budget,
                                      softmax_scale=scale, block_size=block_size, return_computational_ratio=True)
    return o[0].transpose(0, 1), float(ratio)


@torch.no_grad()
def flex_prefill(M, ids, kbuf=None, vbuf=None, logits=True, **kw):
    """FastQwen3.prefill with FlexPrefill attention. ids [T] from position 0; K/V go to kbuf/vbuf [L, Hkv, >=T, D].
    Returns (last-token logits [1, V] or None, mean fraction of causal attention blocks computed over layers).
    The block ratio counts all (query block, key block) pairs FlexPrefill activates over the full causal grid; a dense
    causal prefill would be ~0.5 + 0.5 / n_blocks in the same units, so 0.5 = no saving."""
    o = {**DEFAULTS, **kw}
    T = ids.shape[0]
    cos, sin = M.cos[:T], M.sin[:T]
    h = F.embedding(M.ids_in(ids, first=True), M.embed)
    ratios = []
    for i, lw in enumerate(M.layers):
        q, k, v = M._qkv(lw, rms(h, lw["ln1"], M.eps), cos, sin)
        if kbuf is not None:
            kbuf[i, :, :T] = k
            vbuf[i, :, :T] = v
        a, r = flex_attention_layer(q, k, v, M.scaling, o["gamma"], o["tau"], o["min_budget"], o["max_budget"],
                                    o["block_size"])
        ratios.append(r)
        h = M._mlp(lw, M._out(lw, h, a.to(h.dtype)))
    lg = F.linear(rms(h[-1:], M.norm, M.eps), M.lm_head) if logits else None
    return lg, sum(ratios) / len(ratios)


def parse_flex(cfg):
    """'flexprefill' | 'flexprefill:<gamma>' | 'flexprefill:<gamma>:t<tau>' -> kwargs for flex_prefill."""
    parts = cfg.split(":")
    assert parts[0] == "flexprefill", cfg
    kw = {}
    if len(parts) > 1 and parts[1]:
        kw["gamma"] = float(parts[1])
    for p in parts[2:]:
        if p.startswith("t"):
            kw["tau"] = float(p[1:])
        elif p.startswith("m"):
            kw["min_budget"] = int(p[1:])
    return kw
