#!/usr/bin/env python
"""CPU test of prefill/recompute.py with tiny random Qwen3 models."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

# SMOKE_ARCH=llama: same tests on tiny Llama 3 models (no q/k norm, llama3 RoPE scaling)
LLAMA = os.environ.get("SMOKE_ARCH", "qwen3") == "llama"
if LLAMA:
    _LR = dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=512)
    Qwen3Config = lambda **kw: LlamaConfig(rope_scaling=_LR, **kw)  # noqa: E731
    Qwen3ForCausalLM = LlamaForCausalLM

from prefill.fast import FastBig, FastHybrid, FastQwen3, FoldedProjectorKV
from prefill.models import new_projector
from prefill.recompute import RecomputeHybrid, attn_mass, extend_at, find_spans, small_prefill_scored

torch.manual_seed(0)
V = 500


def tiny(n_layers, hidden, heads, kv_heads, tie):
    cfg = Qwen3Config(vocab_size=V, hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=n_layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=tie, attn_implementation="sdpa", rope_theta=1e6)
    m = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    for p in m.parameters():
        if p.dim() == 1:
            p.data = 1 + 0.2 * torch.randn_like(p)
    return m


def close(a, b, name, tol=1e-3):
    err = (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-9)
    assert err < tol, f"{name}: rel err {err:.2e}"
    print(f"[ok] {name}: rel err {err:.1e}")


small, large = tiny(4, 64, 4, 2, True), tiny(6, 96, 6, 2, False)
T, s, t = 300, 8, 64
m = T - s - t
ids = torch.randint(0, V, (T,))
S, L = FastQwen3(small), FastQwen3(large)
proj = new_projector(small, large, d_mid=32, rank=4)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
fold = FoldedProjectorKV(proj, L, chunk=100)
fh = FastHybrid(S, L, fold, s, t)
rh = RecomputeHybrid(S, L, fold, n_sink=s, n_tail=t)
kb_h, vb_h, lg_h = fh.prefill(ids)
kb_b, vb_b, lg_b = FastBig(L).prefill(ids)

# 1) extend_at with sink+tail positions == the existing fused sink+tail call
kb, vb = fh.stage_fold(fh.stage_small(ids[:s + m])[:, s:], s, T)
p0 = torch.cat([torch.arange(s), torch.arange(s + m, T)])
lg, sc = extend_at(L, ids[p0], p0, kb, vb, n_q=t, lo=s, hi=s + m)
close(lg[0], lg_h[0], "extend_at(sink+tail) == FastHybrid")
assert sc.shape == (m,) and 0 < sc.sum() <= 1 + 1e-5, sc.sum()

# 2) attention mass == explicit per-head computation on HF eager attention weights (small model)
hs, sc_small = small_prefill_scored(S, ids, t, s, s + m)
_, hs_ref = S.prefill(ids, collect=True, logits=False)
close(hs, hs_ref, "scored small prefill hidden states")
small.config._attn_implementation = "eager"
att = small(input_ids=ids[None], output_attentions=True).attentions  # L x [1, H, T, T]
small.config._attn_implementation = "sdpa"
ref = torch.stack([a[0, :, T - t:, s:s + m].mean((0, 1)) for a in att]).mean(0)
close(sc_small, ref, "attn_small score == HF eager attentions", 1e-3)

# 3) frac = 0 -> plain hybrid ; frac = 1 -> exactly the large model's own prefill
for sc_name in ("random", "recent", "attn_small", "attn_big"):
    kb, vb, lg, n = rh.prefill_sel(ids, sc_name, 0.0)
    assert n == 0
    close(lg[0], lg_h[0], f"{sc_name} frac=0 == hybrid")
    kb, vb, lg, n = rh.prefill_sel(ids, sc_name, 1.0)
    assert n == m
    close(lg[0], lg_b[0], f"{sc_name} frac=1 logits == big")
    close(kb, kb_b, f"{sc_name} frac=1 K == big")
    close(vb, vb_b, f"{sc_name} frac=1 V == big")

# 4) partial recompute: chosen rows changed, others keep projected values; sizes as requested
kb, vb, lg, n = rh.prefill_sel(ids, "attn_big", 0.1)
assert n == round(0.1 * m), n
changed = (kb[:, :, s:s + m] - kb_h[:, :, s:s + m]).abs().amax((0, 1, 3)) > 0
assert changed.sum().item() == n, (changed.sum().item(), n)
print(f"[ok] attn_big 10%: {n} middle rows recomputed, the rest untouched")

# 5) oracle spans
needle = ids[s + 50:s + 54].tolist()
mask = find_spans(ids[s:s + m], [needle], pad=2)
assert mask[48:56].all() and mask.sum() >= 8
kb, vb, lg, n = rh.prefill_sel(ids, "oracle", needles=[needle])
assert n == mask.sum().item()
kb, vb, lg, n = rh.prefill_sel(ids, "oracle", needles=[[V + 1]])
assert n == 0
print("[ok] oracle spans")

# 6) short prompts
for T2 in (5, 40, 72, 73, 80):
    _, _, l1 = fh.prefill(ids[:T2])
    _, _, l2, _ = rh.prefill_sel(ids[:T2], "attn_big", 0.5)
    if T2 <= s + t:
        close(l2[0], l1[0], f"short T={T2}")
# 7) block / sentence units
from prefill.recompute import choose_units, segment_ids
mid = ids[s:s + m]
u = segment_ids(mid, 16)
assert (u == torch.arange(m) // 16).all()
boundary = torch.zeros(V, dtype=torch.bool); boundary[[3, 7]] = True
toy = torch.tensor([1, 3, 5, 5, 7, 2, 2, 2, 2, 2, 2, 9])
assert segment_ids(toy, "sent", boundary, cap=3).tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
sc = torch.zeros(12); sc[6] = 5; sc[0] = 1
assert choose_units(sc, segment_ids(toy, "sent", boundary, cap=3), 3).tolist() == [5, 6, 7]
assert choose_units(sc, segment_ids(toy, "sent", boundary, cap=3), 4).tolist() == [0, 1, 5, 6, 7]
rb = RecomputeHybrid(S, L, fold, n_sink=s, n_tail=t, boundary=boundary)
for seg in (8, 32, "sent"):
    kb, vb, lg, n = rb.prefill_sel(ids, "attn_big", 0.1, seg=seg)
    changed = (kb[:, :, s:s + m] - kb_h[:, :, s:s + m]).abs().amax((0, 1, 3)) > 0
    assert changed.sum().item() == n and n >= round(0.1 * m), (seg, n)
    kb, vb, lg, n = rb.prefill_sel(ids, "attn_big", 1.0, seg=seg)
    close(lg[0], lg_b[0], f"seg={seg} frac=1 == big")
print("[ok] block / sentence selection")
print("ALL RECOMPUTE TESTS PASSED")
