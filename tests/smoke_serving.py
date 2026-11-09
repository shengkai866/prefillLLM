#!/usr/bin/env python
"""CPU test of prefill/serving.py (eager, no CUDA graphs) with tiny random Qwen3 models:
batched prefill == per-prompt prefill, batched ours == per-prompt fold + extend_at, Decoder == HF greedy decoding."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from prefill.fast import FastBig, FastQwen3, FoldedProjectorKV, kv_to_dynamic_cache
from prefill.kv import greedy_decode
from prefill.models import new_projector
from prefill.recompute import RecomputeHybrid, extend_at, prefill_kept, small_prefill_scored
from prefill.serving import Decoder, OursBatched, dense_prefill_b

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
S, L = FastQwen3(small), FastQwen3(large)
B, T, s, t = 3, 300, 8, 64
m = T - s - t
ids = torch.randint(0, V, (B, T))

# 1) batched dense prefill == per-prompt FastQwen3.prefill / small_prefill_scored
kb = torch.zeros(B, L.n_layers, L.Hkv, T, L.D)
vb = torch.zeros_like(kb)
lg, _, _ = dense_prefill_b(L, ids, kb, vb)
for b in range(B):
    kb1, vb1, lg1 = FastBig(L).prefill(ids[b])
    close(lg[b], lg1[0], f"dense_prefill_b logits [{b}]")
    close(kb[b], kb1, f"dense_prefill_b K [{b}]")
    close(vb[b], vb1, f"dense_prefill_b V [{b}]")
_, hs, sc = dense_prefill_b(S, ids, collect=True, score=(t, s, s + m))
for b in range(B):
    hs1, sc1 = small_prefill_scored(S, ids[b], t, s, s + m)
    close(hs[:, b], hs1, f"dense_prefill_b hidden states [{b}]")
    close(sc[b], sc1, f"dense_prefill_b scores [{b}]")

# 2) batched ours == per-prompt fold + extend_at at the same positions
proj = new_projector(small, large, d_mid=32, rank=4)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
fold = FoldedProjectorKV(proj, L, chunk=100)
boundary = torch.rand(V) < 0.1
rh = RecomputeHybrid(S, L, fold, n_sink=s, n_tail=t, boundary=boundary)
for frac in (0.0, 0.1, 0.3):
    ob = OursBatched(rh, T, frac)
    hs, sc = ob.stage1(ids)
    pos = ob.select(ids, sc)
    assert pos.shape == (B, s + ob.k + t), pos.shape
    kbo, vbo = ob.new_buffers(B)
    lgo = ob.stage2(hs, ids, pos, kbo, vbo)
    for b in range(B):
        kb1, vb1 = rh.stage_fold(rh.stage_small(ids[b, :s + m])[:, s:], s, T)
        lg1, _ = extend_at(L, ids[b][pos[b]], pos[b], kb1, vb1)
        close(lgo[b], lg1[0], f"OursBatched frac={frac} logits [{b}]")
        close(kbo[b][:, :, pos[b]], kb1[:, :, pos[b]], f"OursBatched frac={frac} recomputed K [{b}]")
    if frac == 0.0:
        for b in range(B):
            close(lgo[b], rh.prefill(ids[b])[2][0], f"OursBatched frac=0 == hyb [{b}]")

# 3) Decoder == HF greedy decoding (full cache and compact SpecPrefill-style cache)
N = 12
x = ids[0]
kb1, vb1, lg1 = FastBig(L).prefill(x)
ref = greedy_decode(large, kv_to_dynamic_cache(kb1, vb1), lg1, N)
Cp = Decoder.capacity_rows(T + N)
kbd = torch.zeros(L.n_layers, L.Hkv, Cp, L.D)
vbd = torch.zeros_like(kbd)
kbd[:, :, :T], vbd[:, :, :T] = kb1, vb1
dec = Decoder(L, T + N, kbd, vbd, graph=False)
dec.reset(T, T, int(lg1.argmax(-1)))
out = [int(dec.tok)]
for _ in range(N - 1):
    dec.step()
    out.append(int(dec.tok))
assert out == ref, (out, ref)
print(f"[ok] Decoder full cache: {N} greedy tokens == HF")
pos = torch.cat([torch.arange(0, 40), torch.arange(150, 200), torch.arange(T - 30, T)])
K = pos.numel()
kbc, vbc, lgc = prefill_kept(L, x, pos)
ref = greedy_decode(large, kv_to_dynamic_cache(kbc, vbc), lgc, N, pos0=T)
Cp = Decoder.capacity_rows(K + N)
kbd = torch.zeros(L.n_layers, L.Hkv, Cp, L.D)
vbd = torch.zeros_like(kbd)
kbd[:, :, :K], vbd[:, :, :K] = kbc, vbc
dec = Decoder(L, K + N, kbd, vbd, graph=False)
dec.reset(K, T, int(lgc.argmax(-1)))
out = [int(dec.tok)]
for _ in range(N - 1):
    dec.step()
    out.append(int(dec.tok))
assert out == ref, (out, ref)
print(f"[ok] Decoder compact cache (rows {K}, positions from {T}): {N} greedy tokens == HF")
print("all serving tests passed")
