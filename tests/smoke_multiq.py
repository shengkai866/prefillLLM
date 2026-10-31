#!/usr/bin/env python
"""CPU test of prefill/multiq.py (shared document, several questions) with tiny random Qwen3 models."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from transformers import DynamicCache

from prefill.fast import FastBig, FastQwen3, FoldedProjectorKV, kv_to_dynamic_cache
from prefill.models import new_projector
from prefill.multiq import MultiQ
from prefill.recompute import RecomputeHybrid, extend_at
from smoke_recompute import V, close, tiny  # noqa: E402  (also re-runs the recompute tests)

# Uninitialised buffer rows may hold NaN bit patterns on a real GPU; a masked NaN key or value still poisons the
# attention output (0 x NaN = NaN). Make every torch.empty / empty_like return NaN so such rows cannot go unnoticed.
_empty, _empty_like = torch.empty, torch.empty_like
torch.empty = lambda *s, **k: _empty(*s, **k).fill_(float("nan")) if (k.get("dtype") or torch.float32).is_floating_point else _empty(*s, **k)
torch.empty_like = lambda x, **k: _empty_like(x, **k).fill_(float("nan")) if x.is_floating_point() else _empty_like(x, **k)

torch.manual_seed(2)
small, large = tiny(4, 64, 4, 2, True), tiny(6, 96, 6, 2, False)
T, s, t = 300, 8, 64
m = T - s - t
doc = torch.randint(0, V, (T,))
qs = [torch.randint(0, V, (n,)) for n in (20, 33, 27)]
S, L = FastQwen3(small), FastQwen3(large)
proj = new_projector(small, large, d_mid=32, rank=4)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
boundary = torch.rand(V) < 0.08
rh = RecomputeHybrid(S, L, FoldedProjectorKV(proj, L, chunk=100), n_sink=s, n_tail=t, boundary=boundary, n_lah=4)
mq = MultiQ(rh, large, small, eos=(), q_max=40)


def hf_logits(model, ids, pos, cache):
    with torch.no_grad():
        out = model(input_ids=ids[None], position_ids=pos[None], past_key_values=cache, use_cache=True)
    return out.logits[0, -1]


# 1) big: document cache + question extension == the large model's own prefill of document + question
st = mq.doc("big", doc)
for q in qs:
    kb, vb, lg, frac, pos0 = mq.ask_prefill(st, q)
    kb_ref, vb_ref, lg_ref = FastBig(L).prefill(torch.cat([doc, q]))
    close(lg[0], lg_ref[0], f"big: extension logits (Q={q.shape[0]})")
    close(kb, kb_ref, "big: cache rows")
    assert pos0 is None and frac == 1.0 and kb.shape[2] == T + q.shape[0]

# 2) hyb: == HF large model fed the question on top of the projected cache
st = mq.doc("hyb", doc)
kb_h, vb_h, _ = rh.prefill(doc)
for q in qs[:2]:
    _, _, lg, _, _ = mq.ask_prefill(st, q)
    ref = hf_logits(large, q, torch.arange(T, T + q.shape[0]), kv_to_dynamic_cache(kb_h, vb_h))
    close(lg[0], ref, "hyb: extension logits == HF on projected cache")

# 3) ours with everything recomputed: middle rows become exact (they only see true sink rows and each other);
#    the shared cache must stay untouched (fresh) or keep the recomputed rows (acc)
kb_big, vb_big, _ = FastBig(L).prefill(doc)
st = mq.doc("ours:1.0", doc)
before = st["kb"].clone()
kb, vb, lg, frac, _ = mq.ask_prefill(st, qs[0])
close(kb[:, :, s:s + m], kb_big[:, :, s:s + m], "ours 100%: recomputed middle keys == large model")
close(vb[:, :, s:s + m], vb_big[:, :, s:s + m], "ours 100%: recomputed middle values == large model")
assert frac == 1.0 and torch.equal(before, st["kb"]), "fresh variant must not modify the shared cache"
st = mq.doc("ours-acc:1.0", doc)
mq.ask_prefill(st, qs[0])
close(st["kb"][:, :, s:s + m], kb_big[:, :, s:s + m], "ours-acc 100%: shared cache holds the recomputed rows")
assert bool(st["done"].all())
_, _, _, frac2, _ = mq.ask_prefill(st, qs[1])
assert frac2 == 0.0, "nothing left to recompute"
# accumulate: 2 questions at 40% with sentence units cover ~80% without overlap
st = mq.doc("ours-acc:0.4:sent", doc)
_, _, _, f1, _ = mq.ask_prefill(st, qs[0])
d1 = st["done"].clone()
_, _, _, f2, _ = mq.ask_prefill(st, qs[1])
assert 0.35 <= f1 <= 0.6 and 0.3 <= f2 <= 0.6 and int(st["done"].sum()) > int(d1.sum()) * 1.5, (f1, f2, int(d1.sum()), int(st["done"].sum()))
print(f"[ok] ours-acc 40% sentences: q1 {f1:.2f}, q2 {f2:.2f}, done {int(st['done'].sum())}/{m}")
# fresh with a budget: selection is a subset of the middle, logits finite
st = mq.doc("ours:0.2:sent", doc)
for q in qs:
    _, _, lg, frac, _ = mq.ask_prefill(st, q)
    assert 0.15 <= frac <= 0.4 and torch.isfinite(lg).all(), frac
print("[ok] ours 20% sentences: 3 questions")

# 3b) pre: document-time recompute. 100% -> exact large-model cache; the question extension then equals big
st = mq.doc("pre:1.0", doc)
close(st["kb"][:, :, :T], kb_big, "pre 100%: document cache keys == large model")
_, _, lg, frac, _ = mq.ask_prefill(st, qs[0])
_, _, lg_ref = FastBig(L).prefill(torch.cat([doc, qs[0]]))
close(lg[0], lg_ref[0], "pre 100%: question logits == large model")
assert frac == 0.0
# 20% sentences, no per-question part == the single-prompt pipeline on the document alone (same tail scoring)
st = mq.doc("pre:0.2:sent", doc)
kb_r, vb_r, _, n_r = rh.prefill_sel(doc, "attn_small", 0.2, seg="sent")
close(st["kb"][:, :, :T], kb_r, "pre 20% sentences: document cache == prefill_sel(document)")
assert int(st["done"].sum()) == n_r, (int(st["done"].sum()), n_r)
# + 20% per question: new units only, they persist
st = mq.doc("pre:0.2:sent:0.2", doc)
d0 = int(st["done"].sum())
for q in qs[:2]:
    _, _, lg, frac, _ = mq.ask_prefill(st, q)
    assert 0.1 <= frac <= 0.4 and torch.isfinite(lg).all(), frac
assert int(st["done"].sum()) > d0 * 2, (d0, int(st["done"].sum()))
print(f"[ok] pre 20% + 20%/question: done {d0} -> {int(st['done'].sum())}/{m}")

# 4) spec-agn: compact cache + extension == HF with the kept positions, then the question at T..T+Q-1
st = mq.doc("spec-agn:0.3", doc)
K = st["pos"].shape[0]
for q in qs[:2]:
    kb, vb, lg, frac, pos0 = mq.ask_prefill(st, q)
    cache = DynamicCache()
    hf_logits(large, doc[st["pos"]], st["pos"], cache)
    ref = hf_logits(large, q, torch.arange(T, T + q.shape[0]), cache)
    close(lg[0], ref, f"spec-agn: compact extension logits (K={K}, Q={q.shape[0]})")
    assert pos0 == T + q.shape[0] and kb.shape[2] == K + q.shape[0] and abs(frac - K / T) < 1e-9

# 5) spec-q1: built on the first question from document chunks only, reused afterwards
st = mq.doc("spec-q1:0.3", doc)
assert "kb" not in st
kb, vb, lg1, frac, pos0 = mq.ask_prefill(st, qs[0])
K = st["pos"].shape[0]
assert int(st["pos"].max()) < T and abs(K / T - 0.3) < 0.15, (K, T)
_, _, lg2, _, _ = mq.ask_prefill(st, qs[1])
assert torch.isfinite(lg1).all() and torch.isfinite(lg2).all()
print(f"[ok] spec-q1: {K}/{T} document tokens kept, 2 questions")

# 6) *-each == the single-prompt pipelines on document + question
st = mq.doc("ours-each:0.2:sent", doc)
kb, vb, lg, frac, pos0 = mq.ask_prefill(st, qs[0])
full = torch.cat([doc, qs[0]])
kb_r, vb_r, lg_r, n = rh.prefill_sel(full, "attn_small", 0.2, seg="sent")
close(lg[0], lg_r[0], "ours-each == prefill_sel")
st = mq.doc("spec-each:0.3", doc)
kb, vb, lg, frac, pos0 = mq.ask_prefill(st, qs[0])
kb_r, vb_r, lg_r, n = rh.prefill_specprefill(full, 0.3)
close(lg[0], lg_r[0], "spec-each == prefill_specprefill")
assert pos0 == full.shape[0]

# 7) extend_at with rows != positions on a buffer with free (zero) rows
kb_z = torch.zeros(L.n_layers, L.Hkv, T + 50, L.D)
vb_z = torch.zeros_like(kb_z)
L.prefill(doc, kb_z, vb_z, logits=False)
lg, _ = extend_at(L, qs[0], torch.arange(T, T + 20), kb_z, vb_z, rows=torch.arange(T, T + 20))
kb_ref, vb_ref, lg_ref = FastBig(L).prefill(torch.cat([doc, qs[0]]))
close(lg[0], lg_ref[0], "extend_at rows= on a padded buffer")

# 8) ask(): decoding runs for every method
for cfg in ("big", "small", "hyb", "ours:0.2:sent", "ours-acc:0.2:sent", "spec-agn:0.3", "spec-q1:0.3", "spec-each:0.3", "ours-each:0.2:sent"):
    st = mq.doc(cfg, doc)
    for q in qs[:2]:
        kb, vb, lg, _, _ = mq.ask_prefill(st, q)
        assert torch.isfinite(lg).all() and torch.isfinite(kb).all() and torch.isfinite(vb).all(), f"{cfg}: NaN in cache or logits"
        out, frac = mq.ask(st, q, 4)
        assert len(out) == 4, (cfg, out)
print("[ok] ask() decodes for all 9 methods, caches and logits finite under NaN-filled torch.empty")
print("MULTIQ SMOKE TEST PASSED")
