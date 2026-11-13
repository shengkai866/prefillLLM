#!/usr/bin/env python
"""
CPU end-to-end check of the layer-routed MoE projector with tiny random Qwen3 models (no data, no GPU):
  1. stage-1 losses (alignment + attention output) and the stage-2 KL back-propagate into BOTH experts
  2. the fast inference path (FastHybrid with the folded MoE projector) equals the reference hybrid prefill
     (projector -> hidden_to_kv) in the K/V cache and the first-token logits, and decodes the same tokens
  3. a few optimisation steps of the stage-1 loss reduce it, save / load keeps the outputs

Run:  python tests/smoke_moe.py
"""
import os
import sys
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill.fast import FastHybrid, FastQwen3, FoldedProjectorKV, kv_to_dynamic_cache  # noqa: E402
from prefill.kv import (greedy_decode, hybrid_prefill, kl_teacher_student, large_hidden_states,  # noqa: E402
                        small_hidden_states, student_forward, teacher_forward)
from prefill.losses import attention_output_loss, hidden_alignment_loss  # noqa: E402
from prefill.models import new_projector  # noqa: E402
from prefill.projector import Projector, ProjectorMoE  # noqa: E402

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
ids = torch.randint(0, V, (1, 300))
for in_norm in (False, True):
    proj = new_projector(small, large, d_mid=32, rank=4, in_norm=in_norm, moe_spans="2-3")
    assert isinstance(proj, ProjectorMoE) and [ex.cfg["Ll"] for ex in proj.experts] == [4, 2]
    with torch.no_grad():
        for ex in proj.experts:
            ex.up.normal_(0, 0.05); ex.bias.normal_(0, 0.1); ex.layer_mix.normal_(0, 1)

    # 1. both training losses reach both experts
    h_true = large_hidden_states(large, ids)
    hs = small_hidden_states(small, ids)
    mask = torch.ones(ids.shape, dtype=torch.bool); mask[:, :4] = False
    loss, info = hidden_alignment_loss(proj(hs), h_true, mask)
    (loss + attention_output_loss(large, proj(hs), h_true, 4, 2)).backward()
    assert all(q.grad is not None and q.grad.abs().sum() > 0 for q in proj.parameters()), "stage 1: missing grads"
    proj.zero_grad()
    t = teacher_forward(large, ids, 200)
    kl = kl_teacher_student(t, student_forward(large, small, proj, ids, 200, 8, 16)).mean()
    kl.backward()
    for e, ex in enumerate(proj.experts):
        assert ex.trunk[0].weight.grad.abs().sum() > 0 and ex.up.grad.abs().sum() > 0, f"stage 2: expert {e} without grads"
    proj.zero_grad()
    print(f"[ok] in_norm={in_norm}: stage-1 loss {loss.item():.3f} and stage-2 KL {kl.item():.3f} give gradients to both experts")

    # 2. fast path == reference
    S, L = FastQwen3(small), FastQwen3(large)
    fold = FoldedProjectorKV(proj, L, chunk=100)
    assert fold.grouped
    with torch.no_grad():
        cache_r, lg_r = hybrid_prefill(large, small, proj, ids, 8, 64)
        fh = FastHybrid(S, L, fold, 8, 64)
        kb, vb, lg_f = fh.prefill(ids[0])
        close(lg_f[0], lg_r[0], "fast hybrid first-token logits")
        close(kb, torch.stack([cache_r.key_cache[l][0] for l in range(6)]), "fast hybrid K cache")
        close(vb, torch.stack([cache_r.value_cache[l][0] for l in range(6)]), "fast hybrid V cache")
        out = greedy_decode(large, kv_to_dynamic_cache(kb, vb), lg_f, 5)
        assert out == greedy_decode(large, cache_r, lg_r, 5), "decode mismatch"
    print("[ok] decode from the fast cache matches the reference:", out)

# 3. optimisation + save / load
proj.train()
opt = torch.optim.AdamW(proj.parameters(), lr=2e-3)
first = None
for i in range(30):
    loss, _ = hidden_alignment_loss(proj(hs), h_true, mask)
    first = first or loss.item()
    loss.backward(); opt.step(); opt.zero_grad()
assert loss.item() < 0.9 * first, (first, loss.item())
with tempfile.TemporaryDirectory() as d:
    proj.save(os.path.join(d, "p.pt"), step=30)
    p2, ck = Projector.load(os.path.join(d, "p.pt"))
    assert isinstance(p2, ProjectorMoE) and ck["step"] == 30
    close(p2(hs), proj(hs), "save / load", 1e-6)
print(f"[ok] 30 steps: stage-1 loss {first:.3f} -> {loss.item():.3f}")
print("ALL MOE SMOKE TESTS PASSED")
