#!/usr/bin/env python
"""CPU test for round 5: projector input RMSNorm (reference vs folded fast path, checkpoint compatibility) and the
stage-2 attention-output matching path (student_forward(return_hidden=True) + attention_output_loss)."""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from prefill.fast import FastHybrid, FastQwen3, FoldedProjectorKV
from prefill.kv import hidden_to_kv, hybrid_prefill, kl_teacher_student, large_hidden_states, small_hidden_states, split_prompt, student_forward, teacher_forward
from prefill.losses import attention_output_loss
from prefill.models import new_projector
from prefill.projector import Projector

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
ids = torch.randint(0, V, (300,))
S, L = FastQwen3(small), FastQwen3(large)
hs = small_hidden_states(small, ids[None])[:, 0] * 30  # large-magnitude inputs, like deep layers of bigger models

# 1) in_norm projector: folded fast path == reference projector -> hidden_to_kv; output invariant to input scale
proj = new_projector(small, large, d_mid=32, rank=4, in_norm=True)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
s, m = 8, 300 - 8 - 64
h_hat = proj(hs[:, None, s:s + m])
close(proj(hs[:, None, s:s + m] * 7.0), h_hat, "in_norm output invariant to input scale")
ks, vs = hidden_to_kv(large, h_hat, torch.arange(s, s + m)[None])
fold = FoldedProjectorKV(proj, L, chunk=100)
assert fold.in_norm
kd = torch.empty(6, 2, m, 16); vd = torch.empty_like(kd)
fold(hs[:, s:s + m], L.cos[s:s + m], L.sin[s:s + m], kd, vd)
close(kd, torch.stack([k[0] for k in ks]), "folded K with in_norm", 1e-3)
close(vd, torch.stack([v[0] for v in vs]), "folded V with in_norm", 1e-3)
cache_r, lg_r = hybrid_prefill(large, small, proj, ids[None], 8, 64)
_, _, lg_f = FastHybrid(S, L, fold, 8, 64).prefill(ids)
close(lg_f[0], lg_r[0], "fast hybrid == reference hybrid with in_norm")

# 2) checkpoints: in_norm round-trips; an old checkpoint (no key in cfg) loads with in_norm off
with tempfile.TemporaryDirectory() as d:
    proj.save(os.path.join(d, "p.pt"), step=1)
    p2, _ = Projector.load(os.path.join(d, "p.pt"))
    assert p2.cfg["in_norm"] is True
    close(p2(hs[:, None, s:s + m]), h_hat, "reloaded in_norm projector")
    old = new_projector(small, large, d_mid=32, rank=4)
    ck = {"state_dict": old.state_dict(), "cfg": {k: v for k, v in old.cfg.items() if k != "in_norm"}}
    torch.save(ck, os.path.join(d, "old.pt"))
    p3, _ = Projector.load(os.path.join(d, "old.pt"))
    assert not p3.cfg.get("in_norm")
    assert not FoldedProjectorKV(p3, L).in_norm
print("[ok] checkpoint compatibility")

# 3) stage-2 path with the attention-output loss (mirrors scripts/train_stage2.py)
proj.train()
P = 220
seq = ids[None]
with torch.no_grad():
    out = large(input_ids=seq, output_hidden_states=True, use_cache=False, logits_to_keep=seq.shape[1] - P + 1)
t_logits = out.logits[:, :-1]
close(t_logits, teacher_forward(large, seq, P), "teacher logits from the hidden-states pass")
Ll = len(large.model.layers)
h_true = torch.stack([h[:, :P] for h in out.hidden_states[:Ll]], 0)
close(h_true, large_hidden_states(large, seq[:, :P]), "teacher residual streams on the prompt")
s_logits, h_hat2 = student_forward(large, small, proj, seq, P, 8, 64, return_hidden=True)
close(s_logits, student_forward(large, small, proj, seq, P, 8, 64), "student logits unchanged by return_hidden")
n_s, n_m, _ = split_prompt(P, 8, 64)
assert h_hat2.shape == (Ll, 1, n_m, 96), h_hat2.shape
h_mix = torch.cat([h_true[:, :, :n_s], h_hat2.to(h_true.dtype), h_true[:, :, n_s + n_m:]], 2)
loss_a = attention_output_loss(large, h_mix, h_true, n_s, layers=random.sample(range(Ll // 3, Ll), 3))
loss_true = attention_output_loss(large, h_true.clone(), h_true, n_s, layers=[2, 4])
assert loss_true.item() < 1e-10, loss_true.item()
loss = kl_teacher_student(t_logits, s_logits).mean() + 0.1 * F.cross_entropy(s_logits.flatten(0, 1), seq[:, P:].flatten()) + loss_a
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in proj.parameters() if p.grad is not None)
assert proj.trunk[0].weight.grad.abs().sum() > 0
print(f"[ok] stage-2 loss with attention matching: attn {loss_a.item():.3f} (0 for the true streams), grads finite")
print("ALL ROUND-5 TESTS PASSED")
