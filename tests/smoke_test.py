#!/usr/bin/env python
"""CPU smoke test with tiny random Qwen3 models. Usage: PREFILL_ROOT=<tmp> python tests/smoke_test.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from prefill import config as C
from prefill.data import save_store
from prefill.kv import (greedy_decode, hidden_to_kv, hybrid_prefill, large_hidden_states, small_hidden_states,
                        student_forward, teacher_forward, true_prefill, kl_teacher_student)
from prefill.losses import attention_output_loss, hidden_alignment_loss
from prefill.models import new_projector

torch.manual_seed(0)
root = C.SCRATCH_ROOT
tok_dir = os.path.join(root, "tok")
tok = AutoTokenizer.from_pretrained(tok_dir)
V = len(tok)


def tiny(n_layers, hidden, heads, kv_heads):
    cfg = Qwen3Config(vocab_size=V, hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=n_layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=16, max_position_embeddings=8192,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    m = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    return m


small = tiny(4, 64, 4, 2)
large = tiny(6, 96, 6, 2)
for name, m in (("Qwen3-0.6B", small), ("Qwen3-8B", large)):
    d = os.path.join(C.MODELS_DIR, name)
    m.save_pretrained(d)
    tok.save_pretrained(d)
ids = torch.randint(0, V, (1, 300))

# 1) KV from true hidden states must reproduce the true cache
h_true = large_hidden_states(large, ids)
pos = torch.arange(ids.shape[1]).unsqueeze(0)
ks, vs = hidden_to_kv(large, h_true, pos)
cache, logits_true = true_prefill(large, ids)
for l in range(large.config.num_hidden_layers):
    assert torch.allclose(ks[l], cache.key_cache[l], atol=1e-5), f"K mismatch layer {l}"
    assert torch.allclose(vs[l], cache.value_cache[l], atol=1e-5), f"V mismatch layer {l}"
print("[ok] hidden_to_kv reproduces the true KV cache")


# 2) hybrid prefill with an oracle projector equals true prefill
class Oracle(torch.nn.Module):
    def forward(self, hs):  # ignore small hidden; return true large hidden for the middle positions
        return h_true[:, :, 8 : 8 + hs.shape[2]]


c2, logits_h = hybrid_prefill(large, small, Oracle(), ids, n_sink=8, n_tail=16)
assert torch.allclose(logits_h, logits_true, atol=1e-4), (logits_h - logits_true).abs().max()
assert c2.get_seq_length() == ids.shape[1]
print("[ok] hybrid prefill with oracle projector == true prefill")

# 3) student_forward with oracle == teacher_forward
class Oracle2(torch.nn.Module):
    def forward(self, hs):
        return h_true[:, :, 8 : 8 + hs.shape[2]]
s = student_forward(large, small, Oracle2(), ids, 200, 8, 16)
t = teacher_forward(large, ids, 200)
assert s.shape == t.shape == (1, 100, V)
assert torch.allclose(s, t, atol=1e-4)
print("[ok] student_forward with oracle == teacher_forward; KL=", kl_teacher_student(t, s).mean().item())

# 4) real projector: stage-1 loss + backward, stage-2 loss + backward
proj = new_projector(small, large, d_mid=32, rank=4)
hs = small_hidden_states(small, ids)
h_hat = proj(hs)
mask = torch.ones(ids.shape, dtype=torch.bool); mask[:, :4] = False
loss, info = hidden_alignment_loss(h_hat, h_true, mask)
loss = loss + attention_output_loss(large, h_hat, h_true, 4, 2)
loss.backward()
assert all(p.grad is not None for p in proj.parameters()), "missing grads"
print(f"[ok] stage1 loss={loss.item():.3f} cos={info['cos'].mean():.3f} nmse={info['nmse'].mean():.3f}")
proj.zero_grad()
s = student_forward(large, small, proj, ids, 200, 8, 16)
l2 = kl_teacher_student(t, s).mean()
l2.backward()
assert proj.trunk[0].weight.grad.abs().sum() > 0
print(f"[ok] stage2 KL={l2.item():.3f}, grads flow through the frozen large model")

# 5) greedy decode
with torch.no_grad():
    cache, lg = hybrid_prefill(large, small, proj, ids, 8, 16)
    out = greedy_decode(large, cache, lg, 5, eos_ids=())
assert len(out) == 5 and cache.get_seq_length() == ids.shape[1] + 4
print("[ok] greedy decode", out)

# 6) tiny data stores + evaluate.py end-to-end (speed, ppl, oracle, niah; gsm8k needs the real dataset)
rng = np.random.default_rng(0)
for name in ("fineweb_edu", "slimpajama"):
    seqs = [rng.integers(0, V, size=int(rng.integers(900, 1300))).astype(np.uint32) for _ in range(6)]
    save_store(name, "test", seqs)
    save_store(name, "train", seqs)
proj.save(os.path.join(root, "proj.pt"), step=1)
print("[ok] stores + projector saved; now run evaluate.py")
