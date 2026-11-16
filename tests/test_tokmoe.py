#!/usr/bin/env python
"""CPU checks for the token-routed MoE projector (prefill/projector.ProjectorTokMoE, branch tok-moe) on tiny random
Qwen3 models, reverse-pair shapes (source wider and deeper than the decoder):
  1. from_projector is exact (any gate), and stays exact through save / Projector.load
  2. grouped sparse forward == dense reference (every expert on every cell, masked top-k weights)
  3. band prior: layers start on the neighbouring experts; no prior: routing depends on the token
  4. fast.TokMoEKV (via build_fold) == projector -> hidden_to_kv; FastHybrid == reference hybrid_prefill
  5. stage-2 loss (KL + NLL + load balance): finite gradients on every parameter group, experts diverge after a step
  6. top_k = n_experts = 1 is the plain Projector
Usage: python tests/test_tokmoe.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from prefill.fast import FastHybrid, FastQwen3, TokMoEKV, build_fold
from prefill.kv import hidden_to_kv, hybrid_prefill, kl_teacher_student, small_hidden_states, student_forward, teacher_forward
from prefill.models import new_projector
from prefill.projector import Projector, ProjectorTokMoE, rms_unit

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


def close(a, b, name, tol=1e-4):
    err = (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-9)
    assert err < tol, f"{name}: rel err {err:.2e}"
    print(f"[ok] {name}: rel err {err:.1e}")


def dense_reference(p, hs):
    """Every expert on every cell, weights = renormalised softmax over the top-k gate logits, zero elsewhere."""
    u = rms_unit(hs) if p.cfg["in_norm"] else hs
    mixed = torch.einsum("ls,sbtd->lbtd", p.layer_mix.softmax(-1), u)
    logits = torch.einsum("lbtd,led->lbte", mixed, p.gate_w) + p.gate_b[:, None, None]
    top, sel = logits.topk(p.cfg["top_k"], -1)
    g = torch.zeros_like(logits).scatter_(-1, sel, top.softmax(-1))
    a = F.silu(torch.einsum("lbtd,emd->lbtem", mixed, p.W1) + p.b1)
    y = torch.einsum("lbtem,eom->lbteo", a, p.W2) + p.b2
    base = (y * g[..., None]).sum(-2)
    corr = torch.einsum("lbtr,lre->lbte", torch.einsum("lbtd,ldr->lbtr", base, p.down), p.up)
    return base + corr + p.bias, sel


source, decoder = tiny(7, 96, 6, 2, False), tiny(4, 64, 4, 2, True)  # code names: small = source, large = decoder
ids = torch.randint(0, V, (300,))
S, L = FastQwen3(source), FastQwen3(decoder)
hs = small_hidden_states(source, ids[None]) * 30  # [Ls, 1, T, d_s]
s, m = 8, 300 - 8 - 64
hm = hs[:, :, s:s + m]

# 1) exact conversion
base = new_projector(source, decoder, d_mid=32, rank=4, in_norm=True)
with torch.no_grad():
    base.up.normal_(0, 0.05); base.bias.normal_(0, 0.1); base.layer_mix.normal_(0, 1)
ref = base(hm)
for E, k, band in ((8, 2, 0.0), (4, 1, 0.0), (6, 3, 2.0)):
    p = ProjectorTokMoE.from_projector(base, E, k, band).eval()
    close(p(hm), ref, f"from_projector E={E} top-{k} band={band} == Projector", 1e-5)
with tempfile.TemporaryDirectory() as d:
    p.save(os.path.join(d, "p.pt"), step=3)
    p2, ck = Projector.load(os.path.join(d, "p.pt"))
    assert isinstance(p2, ProjectorTokMoE) and ck["step"] == 3 and p2.cfg == p.cfg
    close(p2(hm), ref, "save / Projector.load round trip", 1e-5)
    base.save(os.path.join(d, "b.pt"))
    assert type(Projector.load(os.path.join(d, "b.pt"))[0]) is Projector
print("[ok] plain checkpoints still load as Projector")

# 2) sparse grouped forward == dense reference, with experts that differ
moe = new_projector(source, decoder, d_mid=32, rank=4, in_norm=True, tmoe=8, tmoe_topk=2)
with torch.no_grad():
    moe.up.normal_(0, 0.05); moe.bias.normal_(0, 0.1); moe.layer_mix.normal_(0, 1); moe.gate_w.normal_(0, 0.3)
moe.eval()
out = moe(hm)
dref, sel = dense_reference(moe, hm)
close(out, dref, "grouped top-2 forward == dense reference", 1e-5)
close(moe(hm * 7.0), out, "in_norm: output invariant to input scale", 1e-4)
assert (out - ProjectorTokMoE.from_projector(base, 8, 2)(hm)).abs().max() > 1e-2  # really a different function
moe.train()
close(moe(hm), out, "train mode == eval mode (deterministic routing)", 1e-6)
assert moe.aux is not None and moe.aux.requires_grad and 0.5 < moe.stats["gate_ent"] <= 1.0
print(f"[ok] aux {moe.aux.item():+.4f}, stats {{{', '.join(f'{k} {v:.3f}' for k, v in moe.stats.items())}}}")

# 3) routing: token-dependent without a prior, layer bands with one
used = torch.stack([torch.bincount(sel[l].flatten(), minlength=8) for l in range(sel.shape[0])]).float()
assert (used > 0).sum(1).min() >= 3, used  # a layer spreads its tokens over several experts
banded = new_projector(source, decoder, d_mid=32, rank=4, in_norm=True, tmoe=8, tmoe_topk=2, tmoe_band=50.0).train()
banded(hm)
_, bsel = dense_reference(banded, hm)
for l in range(4):
    c = l * 7 / 3
    assert set(bsel[l].flatten().tolist()) <= {int(c), int(c) + 1, max(int(c) - 1, 0)}, (l, set(bsel[l].flatten().tolist()))
assert banded.stats["gate_layer_top"] > 0.49
print(f"[ok] band prior routes by layer (per-layer top share {banded.stats['gate_layer_top']:.2f}); no prior: by token")

# 4) inference path
moe.eval()
fold = build_fold(moe, L)
assert isinstance(fold, TokMoEKV) and fold.in_norm
fold.chunk = 100  # < m: exercise chunking
ks, vs = hidden_to_kv(decoder, out, torch.arange(s, s + m)[None])
kd = torch.empty(4, 2, m, 16); vd = torch.empty_like(kd)
fold(hm[:, 0], L.cos[s:s + m], L.sin[s:s + m], kd, vd)
close(kd, torch.stack([k[0] for k in ks]), "TokMoEKV K == projector -> hidden_to_kv", 1e-4)
close(vd, torch.stack([v[0] for v in vs]), "TokMoEKV V == projector -> hidden_to_kv", 1e-4)
kd2 = torch.empty(2, 2, m, 16); vd2 = torch.empty_like(kd2)
fold(hm[:, 0], L.cos[s:s + m], L.sin[s:s + m], kd2, vd2, n_layers=2)
close(kd2, kd[:2], "n_layers=2", 1e-6)
_, lg_r = hybrid_prefill(decoder, source, moe, ids[None], 8, 64)
_, _, lg_f = FastHybrid(S, L, fold, 8, 64).prefill(ids)
close(lg_f[0], lg_r[0], "FastHybrid(TokMoEKV) == reference hybrid_prefill", 1e-3)

# 5) stage-2 loss as in scripts/train_stage2.py (teacher = the source model, PREFILL_TEACHER=small), from a conversion
p = ProjectorTokMoE.from_projector(base, 8, 2).train()
opt = torch.optim.AdamW(p.parameters(), lr=1e-2)
P, seq = 220, ids[None]
t_logits = teacher_forward(source, seq, P)
s_logits = student_forward(decoder, source, p, seq, P, 8, 64)
loss = kl_teacher_student(t_logits, s_logits).mean() + 0.1 * F.cross_entropy(s_logits.flatten(0, 1), seq[:, P:].flatten()) + 0.01 * p.aux
loss.backward()
for name, q in p.named_parameters():
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0, name
gW1 = p.W1.grad.flatten(1)
assert (gW1[0] - gW1[1]).abs().max() > 0, "identical experts received identical gradients"
opt.step()
assert (p.W1[0] - p.W1[1]).abs().max() > 0
print(f"[ok] stage-2 loss {loss.item():.3f}: gradients on all {len(list(p.parameters()))} parameter tensors, experts diverge after one step")

# 6) one expert = plain Projector
close(ProjectorTokMoE.from_projector(base, 1, 1).eval()(hm), ref, "E=1 top-1 == Projector", 1e-5)
print("ALL TOK-MOE TESTS PASSED")
