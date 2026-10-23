#!/usr/bin/env python
"""CPU test of prefill/deep.py with tiny random Qwen3 models."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from prefill.deep import DeepHybrid  # noqa: E402
from prefill.fast import FastBig, FastHybrid, FastQwen3, FoldedProjectorKV  # noqa: E402
from prefill.models import new_projector  # noqa: E402

torch.manual_seed(1)
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
Ll = 6
ids = torch.randint(0, V, (T,))
S, L = FastQwen3(small), FastQwen3(large)
proj = new_projector(small, large, d_mid=32, rank=4)
with torch.no_grad():
    proj.up.normal_(0, 0.05); proj.bias.normal_(0, 0.1); proj.layer_mix.normal_(0, 1)
fold = FoldedProjectorKV(proj, L, chunk=100)
fh = FastHybrid(S, L, fold, s, t)
dh = DeepHybrid(S, L, fold, proj, s, t)
kb_h, vb_h, lg_h = fh.prefill(ids)
kb_b, vb_b, lg_b = FastBig(L).prefill(ids)
_, hs_true = L.prefill(ids, collect=True, logits=False)  # [Ll+1, T, d] inputs of every layer (+ normed final)

# 1) k = 0 -> plain hybrid, k = Ll -> large model alone
_, _, lg = dh.prefill_deep(ids, 0)
close(lg[0], lg_h[0], "deep k=0 == hybrid")
kb, vb, lg = dh.prefill_deep(ids, Ll)
close(lg[0], lg_b[0], "deep k=Ll == big")

# 2) projected residual == unfolded projector output; partial-layer fold == full fold rows
hs = fh.stage_small(ids[:s + m])[:, s:]
with torch.no_grad():
    ref = proj(hs[:, None])[:, 0]  # [Ll, m, d]
for l in range(Ll):
    close(dh.project_resid(hs, l), ref[l], f"project_resid layer {l}")
for k in range(1, Ll):
    kb, vb, lg, h_mid, hn = dh.prefill_deep(ids, k, return_h=True)
    sp = Ll - k
    close(kb[:sp, :, s:s + m], kb_h[:sp, :, s:s + m], f"k={k}: projected K of layers < split")
    close(vb[:sp, :, s:s + m], vb_h[:sp, :, s:s + m], f"k={k}: projected V of layers < split")
    close(h_mid, ref[sp], f"k={k}: handoff residual")
    assert hn.shape == (T, L.d)
    close(lg[0], torch.nn.functional.linear(hn[-1:], L.lm_head)[0], f"k={k}: logits from returned hidden")

# 3) oracle inputs (true middle K/V below the split, true residual at the split) -> exactly the large model
class TrueFold:
    def __init__(self, kb, vb):
        self.kb, self.vb, self.mix, self.W1, self.b1 = kb, vb, fold.mix, fold.W1, fold.b1

    def __call__(self, hs, cos, sin, kdst, vdst, n_layers=None):
        kdst.copy_(self.kb[:n_layers, :, s:s + m])
        vdst.copy_(self.vb[:n_layers, :, s:s + m])


dh_or = DeepHybrid(S, L, TrueFold(kb_b, vb_b), proj, s, t)
for k in (1, 3, 5):
    dh_or.project_resid = lambda hs_, l: hs_true[l, s:s + m]
    kb, vb, lg = dh_or.prefill_deep(ids, k)
    close(lg[0], lg_b[0], f"oracle k={k} logits == big")
    close(kb, kb_b, f"oracle k={k} K == big")
    close(vb, vb_b, f"oracle k={k} V == big")
print("[smoke_deep] all ok")
