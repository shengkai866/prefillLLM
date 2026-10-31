#!/usr/bin/env python
"""CPU equivalence test of prefill/fast.py against the reference path in prefill/kv.py (tiny random Qwen3 models)."""
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

from prefill.fast import FastHybrid, FastQwen3, FoldedProjectorKV, FastBig, kv_to_dynamic_cache
from prefill.kv import greedy_decode, hidden_to_kv, hybrid_prefill, large_hidden_states, small_hidden_states, true_prefill
from prefill.models import new_projector

torch.manual_seed(0)
V = 500


def tiny(n_layers, hidden, heads, kv_heads, tie):
    cfg = Qwen3Config(vocab_size=V, hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=n_layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=tie, attn_implementation="sdpa", rope_theta=1e6)
    m = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    for p in m.parameters():  # non-trivial norm weights
        if p.dim() == 1:
            p.data = 1 + 0.2 * torch.randn_like(p)
    return m


def close(a, b, name, tol=2e-4):
    err = (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-9)
    assert err < tol, f"{name}: rel err {err:.2e}"
    print(f"[ok] {name}: rel err {err:.1e}")


small, large = tiny(4, 64, 4, 2, True), tiny(6, 96, 6, 2, False)
ids = torch.randint(0, V, (300,))
ref_logits_large = large(input_ids=ids[None]).logits[0, -1]
ref_hs_small = small_hidden_states(small, ids[None])[:, 0]
S, L = FastQwen3(small), FastQwen3(large)

# 1) HF model unchanged after weight re-pointing; lean forward == HF
close(large(input_ids=ids[None]).logits[0, -1], ref_logits_large, "HF large unchanged after fusing")
kb, vb, lg = FastBig(L).prefill(ids)
close(lg[0], ref_logits_large, "lean large logits")
cache, _ = true_prefill(large, ids[None])
close(kb, torch.stack([cache.key_cache[l][0] for l in range(6)]), "lean large K")
close(vb, torch.stack([cache.value_cache[l][0] for l in range(6)]), "lean large V")
_, hs = S.prefill(ids, collect=True, logits=False)
close(hs, ref_hs_small, "lean small hidden states (incl. final norm)")

# 2) folded projector+KV == projector -> hidden_to_kv
proj = new_projector(small, large, d_mid=32, rank=4)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
fold = FoldedProjectorKV(proj, L, chunk=100)  # chunk < n to exercise chunking
s, m = 8, 300 - 8 - 64
h_hat = proj(ref_hs_small[:, None, s:s + m])
pos = torch.arange(s, s + m)[None]
ks, vs = hidden_to_kv(large, h_hat, pos)
kd = torch.empty(6, 2, m, 16); vd = torch.empty_like(kd)
fold(ref_hs_small[:, s:s + m], L.cos[s:s + m], L.sin[s:s + m], kd, vd)
close(kd, torch.stack([k[0] for k in ks]), "folded K", 1e-3)
close(vd, torch.stack([v[0] for v in vs]), "folded V", 1e-3)

# 3) fast hybrid (sink+tail fused into one call) == reference hybrid
cache_r, lg_r = hybrid_prefill(large, small, proj, ids[None], 8, 64)
fh = FastHybrid(S, L, fold, 8, 64)
kb, vb, lg_f = fh.prefill(ids)
close(lg_f[0], lg_r[0], "fast hybrid first-token logits", 1e-3)
close(kb, torch.stack([cache_r.key_cache[l][0] for l in range(6)]), "fast hybrid K cache", 1e-3)
close(vb, torch.stack([cache_r.value_cache[l][0] for l in range(6)]), "fast hybrid V cache", 1e-3)

# 4) patching all middle KV with the truth must reproduce the full-model prefill
kt, vt, _ = FastBig(L).prefill(ids)
def patch(kbuf, vbuf, s, m):
    kbuf[:, :, s:s + m] = kt[:, :, s:s + m]; vbuf[:, :, s:s + m] = vt[:, :, s:s + m]
_, _, lg_p = fh.prefill(ids, patch=patch)
close(lg_p[0], ref_logits_large, "hybrid with true middle KV == big")

# 5) short prompts (no middle) and decode from a fast cache
for T in (5, 40, 72, 73):
    x = ids[:T]
    _, lr = hybrid_prefill(large, small, proj, x[None], 8, 64)
    _, _, lf = fh.prefill(x)
    close(lf[0], lr[0], f"fast hybrid T={T}", 1e-3)
kb, vb, lg = fh.prefill(ids)
out = greedy_decode(large, kv_to_dynamic_cache(kb, vb), lg, 5)
cache_r, lg_r = hybrid_prefill(large, small, proj, ids[None], 8, 64)
assert out == greedy_decode(large, cache_r, lg_r, 5), "decode mismatch"
print("[ok] decode from fast cache matches reference:", out)
print("ALL FAST-PATH TESTS PASSED")
