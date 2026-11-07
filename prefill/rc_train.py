"""
Recompute-aware stage 2 (branch moe-router): the stage-2 student forward with selective recompute INSIDE it.

Inference ("ours") = sink + tail + a fraction of the middle tokens (chosen by the small model's attention from the tail,
sentence units) are computed by the large model; the other middle tokens keep projected K/V. Plain stage 2 trains the
projector as if nothing were recomputed. Here the training forward does exactly what inference does:

  1. small model (no grad) -> hidden states of the middle tokens + attention scores of its tail queries
  2. choose k = round(frac * m) middle tokens (choose_units over sentence segments, as RecomputeHybrid.prefill_sel)
  3. projector -> predicted residual streams of every large layer -> projected K/V of all middle tokens (with grad)
  4. ONE large-model pass over [chosen middle | tail | continuation[:-1]] at their own positions; in every layer their
     K/V overwrite the buffer rows before attention (index_copy, out of place), so the projector receives no gradient
     through the K/V of recomputed tokens and learns only the cells the large model does NOT compute.

frac = 0 reproduces kv.student_forward (up to the fast-path numerics). The large-model pass is gradient-checkpointed per
layer by default (the chosen tokens make it up to ~6x longer than tail + continuation).
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .fast import causal_mask_for, rms, rope
from .kv import split_prompt
from .recompute import choose_units, segment_ids, small_prefill_scored


def perturb_score(score, unit, k, mode, miss_frac=0.5, gen=None):
    """Training-time imperfect selection (rcn runs). score [m] token scores, unit [m] sentence ids, k token budget.
    exact : unchanged (= inference)
    miss  : the highest-scoring sentences making up ~miss_frac of the budget are EXCLUDED (score -inf), the budget is
            filled from the next ones -> the most attended sentences stay projected, as when inference misses the answer
    random: random sentences (unit score uniform), budget filled the same way."""
    if mode == "exact" or k == 0:
        return score
    nu = int(unit.max()) + 1
    if mode == "random":
        r = torch.rand(nu, generator=gen, device="cpu").to(score.device)
        return r[unit]
    if mode == "miss":
        us = torch.zeros(nu, device=score.device).scatter_reduce(0, unit, score, "amax", include_self=False)
        ul = torch.bincount(unit, minlength=nu)
        order = torch.argsort(us, descending=True)
        before = torch.cumsum(ul[order], 0) - ul[order]
        drop = order[before < max(1, int(round(miss_frac * k)))]
        if int(ul[drop].sum()) >= unit.numel():  # never drop everything
            return score
        bad = torch.zeros(nu, dtype=torch.bool, device=score.device)
        bad[drop] = True
        return torch.where(bad[unit], torch.full_like(score, float("-inf")), score)
    raise ValueError(mode)


@torch.no_grad()
def select_middle(S, ids, P, n_sink, n_tail, frac, boundary, cap=64, mode="exact", miss_frac=0.5):
    """ids [T]. Returns (hs_mid [Ls, m, d_s], idx [k] sorted middle indices to recompute, (s, m, t)).
    mode: exact / miss / random (perturb_score)."""
    s, m, t = split_prompt(P, n_sink, n_tail)
    k = min(m, int(round(frac * m)))
    if m == 0:
        return None, ids.new_zeros(0), (s, m, t)
    if k > 0:
        hs, score = small_prefill_scored(S, ids[:P], t, s, s + m)
        unit = segment_ids(ids[s:s + m], "sent", boundary, cap)
        idx = choose_units(perturb_score(score, unit, k, mode, miss_frac), unit, k)
    else:
        hs = S.prefill(ids[:s + m], collect=True, logits=False)[1]
        idx = ids.new_zeros(0)
    return hs[:, s:s + m], idx, (s, m, t)


def student_forward_rc(L, S, projector, ids, P, n_sink=8, n_tail=64, frac=0.1, boundary=None, cap=64, ckpt=True,
                       return_hidden=False, sel=None):
    """
    L, S: FastQwen3 wrappers of the large / small model. ids [1, T] (B = 1), prompt = ids[:, :P].
    Returns logits [1, T-P, V] predicting ids[:, P:], the recomputed middle indices idx [k] (relative to the middle),
    and with return_hidden the projected middle residual streams h_hat [Ll, 1, m, d_l] (None if no middle).
    sel: precomputed select_middle(...) result (lets the caller reuse it).
    """
    ids1 = ids[0]
    T, dev = ids1.shape[0], ids1.device
    hs_mid, idx, (s, m, t) = sel if sel is not None else select_middle(S, ids1, P, n_sink, n_tail, frac, boundary, cap)
    Ll, Hkv, D, H, rep = L.n_layers, L.Hkv, L.D, L.H, L.rep
    dt = L.embed.dtype
    Tb = T - 1  # buffer positions 0 .. T-2 (the last continuation token is never an input)
    ar = lambda a, b: torch.arange(a, b, device=dev)

    with torch.no_grad():  # true sink K/V
        kS = torch.empty(Ll, Hkv, s, D, dtype=dt, device=dev)
        vS = torch.empty_like(kS)
        if s > 0:
            L.prefill(ids1[:s], kS, vS, logits=False)
    h_hat = None
    if m > 0:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h_hat = projector(hs_mid[:, None])  # [Ll, 1, m, d_l]
    cos_m, sin_m = L.cos[s:s + m], L.sin[s:s + m]

    qpos = torch.cat([s + idx, ar(s + m, Tb)])  # chosen middle | tail | continuation[:-1], sorted
    Q = qpos.shape[0]
    cos_q, sin_q = L.cos[qpos], L.sin[qpos]
    mask = causal_mask_for(qpos, Tb, dt, dev, rep)[None, None]
    n_after = Tb - s - m  # tail + continuation rows, filled by index_copy

    def layer(i, h, hh, ks, vs):
        lw = L.layers[i]
        parts_k, parts_v = [ks], [vs]
        if m > 0:  # projected K/V of the middle tokens from the predicted residual stream hh [m, d_l]
            x = rms(hh.to(dt), lw["ln1"], L.eps)
            kv = F.linear(x, lw["wqkv"][H * D:])
            km, vm = kv.split([Hkv * D, Hkv * D], -1)
            km = km.view(m, Hkv, D)
            if lw["kn"] is not None:
                km = rms(km, lw["kn"], L.eps)
            parts_k.append(rope(km.transpose(0, 1), cos_m, sin_m))
            parts_v.append(vm.view(m, Hkv, D).transpose(0, 1))
        z = torch.zeros(Hkv, n_after, D, dtype=dt, device=dev)
        K = torch.cat(parts_k + [z], 1)
        V = torch.cat(parts_v + [z], 1)
        q, k, v = L._qkv(lw, rms(h, lw["ln1"], L.eps), cos_q, sin_q)
        K = K.index_copy(1, qpos, k)  # recomputed / tail / continuation rows replace the projected ones
        V = V.index_copy(1, qpos, v)
        qg = q.reshape(Hkv, rep * Q, D)
        o = F.scaled_dot_product_attention(qg[None], K[None], V[None], attn_mask=mask, scale=L.scaling)[0]
        return L._mlp(lw, L._out(lw, h, o.reshape(H, Q, D)))

    h = F.embedding(L.ids_in(ids1[qpos]), L.embed)
    for i in range(Ll):
        hh = h_hat[i, 0] if h_hat is not None else None
        if ckpt and torch.is_grad_enabled():
            h = checkpoint(layer, i, h, hh, kS[i], vS[i], use_reentrant=False)
        else:
            h = layer(i, h, hh, kS[i], vS[i])
    n_pred = T - P
    logits = F.linear(rms(h[-n_pred:], L.norm, L.eps), L.lm_head)[None]  # positions P-1 .. T-2
    return (logits, idx, h_hat) if return_hidden else (logits, idx)
