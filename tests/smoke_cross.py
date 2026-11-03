#!/usr/bin/env python
"""CPU test of the cross-tokenizer path (prefill/tokalign.py lookup table on the small model) with tiny random models
whose vocabularies differ: fast path == reference path == plain small model on the mapped ids; scoring / SpecPrefill /
multi-question run on large-model ids; the real Qwen3 <-> Llama 3 tables map the chat markers and digits as intended."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from prefill.fast import FastBig, FastHybrid, FastQwen3, FoldedProjectorKV
from prefill.kv import hybrid_prefill, small_hidden_states
from prefill.models import new_projector
from prefill.multiq import MultiQ
from prefill.recompute import RecomputeHybrid

torch.manual_seed(0)
VS, VL = 500, 700  # different vocabularies


def tiny(n_layers, hidden, heads, kv_heads, tie, V):
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


small, large = tiny(4, 64, 4, 2, True, VS), tiny(6, 96, 6, 2, False, VL)
S_plain = FastQwen3(small)  # before the table is attached: a plain small model
table = torch.randint(0, VS, (VL,))
BOS, EOS = 7, 1
small.prefill_id_map, small.prefill_bos, small.prefill_small_eos = table, BOS, (EOS,)
S, L = FastQwen3(small), FastQwen3(large)
T, s, t = 300, 8, 64
m = T - s - t
ids = torch.randint(0, VL, (T,))
mapped = table[ids].clone()
mapped[0] = BOS

# 1) the small model sees the mapped ids (+ BOS at 0): fast path and HF reference path
_, hs = S.prefill(ids, collect=True, logits=False)
_, hs_ref = S_plain.prefill(mapped, collect=True, logits=False)
close(hs, hs_ref, "FastQwen3.prefill(large ids) == plain prefill(mapped ids)")
close(small_hidden_states(small, ids[None])[:, 0], hs_ref, "small_hidden_states(large ids) == mapped", 2e-3)

# 2) hybrid prefill: fast == reference (both through the table), and differs from the unmapped small model
proj = new_projector(small, large, d_mid=32, rank=4)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
fold = FoldedProjectorKV(proj, L, chunk=100)
fh = FastHybrid(S, L, fold, s, t)
kb, vb, lg = fh.prefill(ids)
_, lg_ref = hybrid_prefill(large, small, proj, ids[None], s, t, autocast=False)
close(lg[0], lg_ref[0], "FastHybrid == kv.hybrid_prefill (cross)", 2e-3)

# 3) selective recompute, drop and SpecPrefill (look-ahead EOS is the small model's own) run on large ids
boundary = torch.zeros(VL, dtype=torch.bool)
boundary[::7] = True
rh = RecomputeHybrid(S, L, fold, n_sink=s, n_tail=t, boundary=boundary, eos=(VL - 1,))
for scorer, seg, drop in (("attn_small", "sent", False), ("attn_small", None, True), ("spec", "sent", False)):
    kb2, vb2, lg2, n = rh.prefill_sel(ids, scorer, 0.2, seg=seg, drop=drop)
    assert torch.isfinite(lg2).all() and n > 0, (scorer, n)
    print(f"[ok] prefill_sel {scorer} seg={seg} drop={drop}: {n} tokens")
kb3, vb3, lg3, n3 = rh.prefill_specprefill(ids, 0.3)
assert torch.isfinite(lg3).all() and 0 < n3 < T
print(f"[ok] SpecPrefill baseline: kept {n3}/{T}")
assert S.own_eos(rh.eos) == (EOS,) and L.own_eos(rh.eos) == rh.eos
print("[ok] own_eos: small -> its own EOS, large -> the prompt EOS")

# 4) multi-question (small cache extended by question ids through the table)
mq = MultiQ(rh, large, small, (VL - 1,), q_max=32)
st = mq.doc("ours-acc:0.1:sent", ids)
for _ in range(2):
    q = torch.randint(0, VL, (20,))
    kq, vq, lq, frac, _ = mq.ask_prefill(st, q)
    assert torch.isfinite(lq).all() and frac > 0
print(f"[ok] MultiQ ours-acc: 2 questions, last fraction {frac:.3f}")

# 5) real tokenizers (if the models are on scratch): chat markers, BOS, digits
try:
    from transformers import AutoTokenizer
    from prefill import config as C
    from prefill.tokalign import build_table
    tq = AutoTokenizer.from_pretrained(os.path.join(C.MODELS_DIR, "Qwen3-8B"))
    tl = AutoTokenizer.from_pretrained(os.path.join(C.MODELS_DIR, "Llama-3.1-8B-Instruct"))
except Exception as e:  # noqa: BLE001
    print(f"[skip] real tokenizers: {e}")
    sys.exit(0)
tab, st_, _ = build_table(tq, tl)  # Llama small <- Qwen large
v = tl.get_vocab()
assert tab[tq.convert_tokens_to_ids("<|im_start|>")] == v["<|start_header_id|>"]
assert tab[tq.convert_tokens_to_ids("<|im_end|>")] == v["<|eot_id|>"]
assert all(tl.decode([int(tab[i])]) == tq.decode([i]) for i in tq("Hello world, 42!", add_special_tokens=False)["input_ids"])
print(f"[ok] Qwen -> Llama table: {st_}")
tab2, st2, _ = build_table(tl, tq)  # Qwen small <- Llama large
assert tab2[tl.bos_token_id] == tq.convert_tokens_to_ids("<|endoftext|>")
assert tab2[tl.convert_tokens_to_ids("<|eot_id|>")] == tq.convert_tokens_to_ids("<|im_end|>")
i123 = tl("123", add_special_tokens=False)["input_ids"]
assert len(i123) == 1 and tq.decode([int(tab2[i123[0]])]) == "1", "Llama 3-digit token -> first digit (longest = tie -> first)"
print(f"[ok] Llama -> Qwen table: {st2}")
print("ALL PASSED")
