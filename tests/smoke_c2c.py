#!/usr/bin/env python
"""CPU test of prefill/c2c.py against the official C2C RosettaModel (github.com/thu-nics/C2C) with tiny random Qwen3s:
  1. c2c_student_forward == RosettaModel.forward with C2C's own section layout (prompt-1 fused | rest receiver-only)
  2. c2c_prefill last-token logits == the same with the continuation empty
  3. a C2C SFT_train.py-style checkpoint dir loads through Projector.load (from_rosetta_dir) and gives the same output
  4. our own save/load round trip; gradients reach the fuser in train mode (with activation checkpointing)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from prefill.c2c import C2CFuser, c2c_prefill, c2c_student_forward
from prefill.kv import student_forward
from prefill.projector import Projector
from rosetta.model.wrapper import RosettaModel

torch.manual_seed(0)
V = 500


def tiny(n_layers, hidden, heads, kv_heads):
    cfg = Qwen3Config(vocab_size=V, hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=n_layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=False, attn_implementation="sdpa", rope_theta=1e6)
    m = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    for p in m.parameters():
        if p.dim() == 1:
            p.data = 1 + 0.2 * torch.randn_like(p)
    return m


def close(a, b, name, tol=1e-4):
    err = (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-9)
    assert err < tol, f"{name}: rel err {err:.2e}"
    print(f"[ok] {name}: rel err {err:.1e}")


dec, src = tiny(4, 64, 4, 2), tiny(6, 96, 6, 3)  # decoder (receiver) 4 layers, source (sharer) 6 layers
fuser = C2CFuser.new(src, dec, hidden_dim=32, intermediate_dim=48).eval()
assert fuser.offset == 2
with torch.no_grad():
    for p in fuser.projs:  # open the gates (logit > 0 at inference) so the fusion actually changes the cache
        p.key_gate_logit.fill_(0.7)
        p.value_gate_logit.fill_(1.3)
for p in fuser.projs:
    p.use_gumbel = True

rm = RosettaModel([dec, src], base_model_idx=0, projector_list=list(fuser.projs), multi_source_fusion_mode="sequential").eval()
for l in range(4):
    rm.set_projector_config(source_model_idx=1, source_model_layer_idx=l + 2, target_model_idx=0, target_model_layer_idx=l,
                            projector_idx=l)

T, P = 40, 29
ids = torch.randint(0, V, (1, T))


def official(ids, P):
    n = ids.shape[1]
    kvi = [torch.tensor([[1, 0]] * (P - 1))[None], torch.tensor([[-1, 0]] * (n - P + 1))[None]]
    with torch.no_grad():
        out = rm(kv_cache_index=kvi, input_ids=ids, attention_mask=torch.ones_like(ids),
                 position_ids=torch.arange(n)[None], use_cache=True)
    return out.logits


with torch.no_grad():
    ref = official(ids, P)[:, :-1]
    ours = c2c_student_forward(dec, src, fuser, ids, P)
    plain = dec(input_ids=ids).logits[:, P - 1:-1]
close(ours, ref, "c2c_student_forward == official RosettaModel")
assert (ours - plain).abs().max() > 1e-2, "fusion had no effect (gates closed?)"
with torch.no_grad():
    close(student_forward(dec, src, fuser, ids, P), ref, "kv.student_forward dispatch")
    _, lg = c2c_prefill(dec, src, fuser, ids[0, :P])
    close(lg, official(ids[:, :P], P)[:, -1], "c2c_prefill last-token logits == official")

with tempfile.TemporaryDirectory() as d:
    # 3. C2C SFT_train.py checkpoint layout
    src_dir = os.path.join(d, "src")
    src.config.save_pretrained(src_dir)
    for i, p in enumerate(fuser.projs):
        torch.save(p.state_dict(), os.path.join(d, f"projector_{i}.pt"))
    rm.save_projector_config(os.path.join(d, "projector_config.json"))
    json.dump({"model": {"teacher_model": src_dir, "mapping": "last_aligned",
                         "projector": {"type": "C2CProjector", "params": dict(fuser.cfg["hp"])}}},
              open(os.path.join(d, "config.json"), "w"))
    fin = os.path.join(d, "final")
    os.makedirs(fin)
    for i in range(4):
        os.link(os.path.join(d, f"projector_{i}.pt"), os.path.join(fin, f"projector_{i}.pt"))
    os.link(os.path.join(d, "projector_config.json"), os.path.join(fin, "projector_config.json"))
    f2b, _ = Projector.load(fin)
    f2b.eval()
    with torch.no_grad():
        close(c2c_student_forward(dec, src, f2b, ids, P), ref, "SFT_train.py layout (config in the parent dir)")
    f2, _ = Projector.load(d)
    f2.eval()
    assert f2.offset == 2 and len(f2.projs) == 4
    with torch.no_grad():
        close(c2c_student_forward(dec, src, f2, ids, P), ref, "from_rosetta_dir checkpoint")
    # 4. own format
    fuser.save(os.path.join(d, "fuser.pt"), step=7)
    f3, ck = Projector.load(os.path.join(d, "fuser.pt"))
    f3.eval()
    assert ck["step"] == 7
    with torch.no_grad():
        close(c2c_student_forward(dec, src, f3, ids, P), ref, "save/load round trip")

fuser.train()
fuser.update_temperature(10)
loss = c2c_student_forward(dec, src, fuser, ids, P).logsumexp(-1).mean()
loss.backward()
g = [p.key_in.weight.grad for p in fuser.projs]
assert all(x is not None and x.abs().sum() > 0 for x in g), "no gradient into the fuser"
assert fuser.projs[0].key_gate_logit.grad is not None
print("[ok] gradients reach every layer's fuser and the gates (train mode, Gumbel gates, per-layer activation checkpointing)")
print("ALL C2C TESTS PASSED")
