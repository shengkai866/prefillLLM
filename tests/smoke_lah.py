#!/usr/bin/env python
"""CPU test of the draft-answer selection scorer (prefill/recompute.py: small_prefill_into, lookahead, lah_mass,
small_prefill_scored_lah; scorers attn_lah / attn_lah32 / attn_lahonly) with tiny random Qwen3 models."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from smoke_serving import close, tiny  # noqa: E402  (tiny random models, relative-error check)
from prefill.fast import FastQwen3, FoldedProjectorKV  # noqa: E402
from prefill.models import new_projector  # noqa: E402
from prefill.recompute import (LookaheadGraph, RecomputeHybrid, lah_mass, lah_scorer, lookahead, small_prefill_into,  # noqa: E402
                               small_prefill_scored, small_prefill_scored_lah)

torch.manual_seed(1)
V = 500
small, large = tiny(4, 64, 4, 2, True), tiny(6, 96, 6, 2, False)
S, L = FastQwen3(small), FastQwen3(large)
T, s, t, n_lah = 300, 8, 64, 8
m = T - s - t
ids = torch.randint(0, V, (T,))

# 1) small_prefill_into == small_prefill_scored (hidden states, tail score), K/V == a plain prefill's
lah = LookaheadGraph(S, T, n_lah, capture=False)
hs, sc, lg = small_prefill_into(S, ids, lah.kb, lah.vb, t, s, s + m, qb=lah.qb)
hs1, sc1 = small_prefill_scored(S, ids, t, s, s + m)
close(hs, hs1, "small_prefill_into hidden states")
close(sc, sc1, "small_prefill_into tail score")
kb = torch.zeros(S.n_layers, S.Hkv, T, S.D)
vb = torch.zeros_like(kb)
lg1, _ = S.prefill(ids, kb, vb)
close(lah.kb[:, :, :T], kb, "small_prefill_into K")
close(lg, lg1, "small_prefill_into logits")

# 2) look-ahead + lah_mass == prefilling prompt + draft tokens and scoring the draft tokens as the tail
toks = []
step0 = lah.step
lah.step = lambda tok, p: (toks.append(int(tok)), step0(tok, p))[1]
n = lookahead(S, lah, lg, T, n_lah, eos=())
assert n == n_lah and len(toks) == n_lah - 1, (n, toks)
full = torch.cat([ids, torch.tensor(toks)])
ref_draft = small_prefill_scored(S, full, n - 1, s, s + m)[1]
close(lah_mass(S, lah.qb, lah.kb, T, n, s, s + m), ref_draft, "lah_mass (draft queries) == full prefill tail score")
ref_last = small_prefill_scored(S, full, n, s, s + m)[1]  # last prompt token + draft
close(lah_mass(S, lah.qb, lah.kb, T, n, s, s + m, with_last=True), ref_last, "lah_mass with_last == full prefill tail score")

# 3) small_prefill_scored_lah = tail score + draft score; only=True = draft incl. last prompt token
hs2, sc2 = small_prefill_scored_lah(S, ids, t, s, s + m, n_lah, eos=())
close(sc2, sc1 + ref_draft, "small_prefill_scored_lah = tail + draft")
close(hs2, hs1, "small_prefill_scored_lah hidden states")
_, sc3 = small_prefill_scored_lah(S, ids, t, s, s + m, n_lah, eos=(), only=True)
close(sc3, ref_last, "small_prefill_scored_lah only=True")
assert lah_scorer("attn_lah") == (8, False) and lah_scorer("attn_lah32") == (32, False) and lah_scorer("attn_lahonly") == (8, True)

# 4) early EOS: the draft stops, n counts the filled slots
eos_tok = toks[2]
lah2 = LookaheadGraph(S, T, n_lah, capture=False)
_, _, lg2 = small_prefill_into(S, ids, lah2.kb, lah2.vb, t, s, s + m, qb=lah2.qb)
n2 = lookahead(S, lah2, lg2, T, n_lah, eos=(eos_tok,))
assert n2 == toks.index(eos_tok) + 1, (n2, toks)  # greedy drafts of a random model repeat tokens: first occurrence stops
print(f"[ok] early EOS stops the draft (n = {n2})")

# 5) prefill_sel runs end to end with the new scorers (projected cache + recompute of the chosen sentences)
proj = new_projector(small, large, d_mid=32, rank=4)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
rh = RecomputeHybrid(S, L, FoldedProjectorKV(proj, L, chunk=100), n_sink=s, n_tail=t, boundary=torch.rand(V) < 0.1)
for sc_name in ("attn_lah", "attn_lah32", "attn_lahonly"):
    kbr, vbr, lgr, nrec = rh.prefill_sel(ids, sc_name, 0.1, seg="sent")
    assert torch.isfinite(lgr).all() and nrec >= round(0.1 * m), (sc_name, nrec)
    print(f"[ok] prefill_sel {sc_name}: {nrec} middle tokens recomputed")
print("all lah tests passed")
