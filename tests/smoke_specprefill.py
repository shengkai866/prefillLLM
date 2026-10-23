#!/usr/bin/env python
"""CPU test of the SpecPrefill baseline and drop-* configs (prefill/recompute.py) with tiny random Qwen3 models."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.nn.functional as F

from prefill.fast import FastBig, FastHybrid, FastQwen3, FoldedProjectorKV, kv_to_dynamic_cache
from prefill.kv import greedy_decode
from prefill.models import new_projector
from prefill.recompute import RecomputeHybrid, prefill_kept, spec_importance
from smoke_recompute import V, close, tiny  # noqa: E402  (also re-runs the recompute tests)

torch.manual_seed(1)
small, large = tiny(4, 64, 4, 2, True), tiny(6, 96, 6, 2, False)
T, s, t, n_lah = 300, 8, 64, 5
m = T - s - t
ids = torch.randint(0, V, (T,))
S, L = FastQwen3(small), FastQwen3(large)
proj = new_projector(small, large, d_mid=32, rank=4)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
fold = FoldedProjectorKV(proj, L, chunk=100)
fh = FastHybrid(S, L, fold, s, t)
rh = RecomputeHybrid(S, L, fold, n_sink=s, n_tail=t, n_lah=n_lah)
kb_h, vb_h, lg_h = fh.prefill(ids)
kb_b, vb_b, lg_b = FastBig(L).prefill(ids)

# 1) SpecPrefill importance == reference from HF eager attentions: greedy look-ahead tokens, queries at T-1 .. T+n_lah-2,
#    softmax restricted to the prompt keys (= renormalised full softmax), avg-pool 13, max over layers x heads, mean
with torch.no_grad():
    seq = ids[None]
    for _ in range(n_lah - 1):
        seq = torch.cat([seq, small(input_ids=seq).logits[:, -1:].argmax(-1)], 1)
    small.config._attn_implementation = "eager"
    att = small(input_ids=seq, output_attentions=True).attentions  # L x [1, H, T+n-1, T+n-1]
    small.config._attn_implementation = "sdpa"
a = torch.stack([x[0, :, T - 1:, :T] for x in att]).float()  # [L, H, n, T]
a = a / a.sum(-1, keepdim=True)
a = F.avg_pool1d(a.flatten(0, 1), 13, 1, 6).amax(0).mean(0)
hs, imp = spec_importance(S, ids, n_lah, collect=True)
close(imp, a, "spec_importance == HF eager reference", 1e-3)
_, hs_ref = S.prefill(ids, collect=True, logits=False)
close(hs, hs_ref, "spec_importance hidden states")
first = int(small(input_ids=ids[None]).logits[0, -1].argmax())
_, imp1 = spec_importance(S, ids, n_lah, eos={first})
close(imp1, F.avg_pool1d(torch.stack([x[0, :, T - 1:T, :T] for x in att]).float().div(
    torch.stack([x[0, :, T - 1:T, :T] for x in att]).float().sum(-1, keepdim=True)).flatten(0, 1), 13, 1, 6).amax(0).mean(0),
    "EOS as first look-ahead token -> last prompt query only", 1e-3)

# 2) prefill_kept == HF large model on the kept tokens at their original positions, and decoding continues at T
pos = torch.cat([torch.arange(0, 20), torch.arange(100, 140), torch.arange(T - 30, T)])
kb, vb, lg = prefill_kept(L, ids, pos)
with torch.no_grad():
    ref = large(input_ids=ids[pos][None], position_ids=pos[None], use_cache=True)
close(lg[0], ref.logits[0, -1], "prefill_kept logits == HF with position_ids")
close(kb[0], ref.past_key_values.layers[0].keys[0] if hasattr(ref.past_key_values, "layers") else ref.past_key_values.key_cache[0][0],
      "prefill_kept layer-0 K == HF")
out = greedy_decode(large, kv_to_dynamic_cache(kb, vb), lg, 6, pos0=T)
with torch.no_grad():
    full = torch.cat([ids[pos], torch.tensor(out[:-1])])
    fpos = torch.cat([pos, torch.arange(T, T + len(out) - 1)])
    ref_out = large(input_ids=full[None], position_ids=fpos[None]).logits[0, len(pos) - 1:].argmax(-1).tolist()
assert ref_out == out, (ref_out, out)
print("[ok] compact-cache decoding at position T == HF full forward")
kb, vb, lg = prefill_kept(L, ids, torch.arange(T))
close(lg[0], lg_b[0], "prefill_kept(all) == big")

# 3) drop-*: same selection as fill, compact cache; frac=1 -> big; frac=0 -> sink + tail only
for sc in ("attn_small", "spec", "random"):
    kb, vb, lg, n = rh.prefill_sel(ids, sc, 1.0, drop=True)
    assert n == m and kb.shape[2] == T
    close(lg[0], lg_b[0], f"drop-{sc} frac=1 == big")
    kb, vb, lg, n = rh.prefill_sel(ids, sc, 0.0, drop=True)
    assert n == 0 and kb.shape[2] == s + t
    close(lg[0], prefill_kept(L, ids, torch.cat([torch.arange(s), torch.arange(s + m, T)]))[2][0], f"drop-{sc} frac=0 == sink+tail only")
kb_f, _, _, n_f = rh.prefill_sel(ids, "attn_small", 0.1, seg=16)
# (attn_small runs the small model on the whole prompt, the plain hybrid on the prefix -> float noise in unchanged rows)
changed = torch.nonzero((kb_f[:, :, s:s + m] - kb_h[:, :, s:s + m]).abs().amax((0, 1, 3)) > 1e-3).flatten()
kb, vb, lg, n = rh.prefill_sel(ids, "attn_small", 0.1, seg=16, drop=True)
assert n == n_f == changed.numel() and kb.shape[2] == s + n + t
keep = torch.cat([torch.arange(s), s + changed, torch.arange(s + m, T)])
close(lg[0], prefill_kept(L, ids, keep)[2][0], "drop-attn_small 10% keeps exactly the tokens fill recomputes")
print("[ok] drop configs")

# 4) spec scorer with projected fill: frac 0 -> hybrid, 1 -> big; 32-token chunks by mean score
_, _, lg, n = rh.prefill_sel(ids, "spec", 0.0)
close(lg[0], lg_h[0], "spec frac=0 == hybrid")
_, _, lg, n = rh.prefill_sel(ids, "spec", 1.0)
close(lg[0], lg_b[0], "spec frac=1 == big")
kb, _, _, n = rh.prefill_sel(ids, "spec", 0.2)
rows = torch.nonzero((kb[:, :, s:s + m] - kb_h[:, :, s:s + m]).abs().amax((0, 1, 3)) > 1e-3).flatten()
assert rows.numel() == n and set((rows // 32).tolist()) and all(((rows // 32) == c).sum() == min(32, m - 32 * c) for c in set((rows // 32).tolist()))
print(f"[ok] spec 20%: {n} rows in whole 32-token chunks")

# 5) SpecPrefill baseline: ceil(frac x chunks) whole chunks of the prompt by mean pooled importance
for frac in (0.1, 0.5):
    kb, vb, lg, n = rh.prefill_specprefill(ids, frac)
    nc = math.ceil(T / 32)
    cs = torch.zeros(nc).scatter_reduce(0, torch.arange(T) // 32, imp, "mean", include_self=False)
    want = torch.nonzero(torch.isin(torch.arange(T) // 32, torch.topk(cs, math.ceil(nc * frac)).indices)).flatten()
    assert n == want.numel() == kb.shape[2], (n, want.numel())
    close(lg[0], prefill_kept(L, ids, want)[2][0], f"specprefill {frac:.0%} == prefill_kept(top chunks)")
kb, vb, lg, n = rh.prefill_specprefill(ids, 1.0)
assert n == T
close(lg[0], lg_b[0], "specprefill 100% == big")
print("ALL SPECPREFILL TESTS PASSED")
