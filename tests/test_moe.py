#!/usr/bin/env python
"""CPU checks for the Qwen3-MoE source model (reverse pair Qwen3-30B-A3B -> Qwen3-0.6B) on a tiny random Qwen3-MoE:
  1. prefill/moe.py grouped forward == HF Qwen3MoeSparseMoeBlock forward (fp32 and bf16)
  2. FastQwen3 (MoE layers) hidden states + last logits == HF output_hidden_states / logits
  3. kv.small_hidden_states == HF hidden states
Multi-GPU splitting cannot be exercised on CPU; the GPU smoke stage of slurm/run_a3b.slurm covers it.
Usage: python tests/test_moe.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from prefill import moe  # noqa: E402
from prefill.fast import FastQwen3  # noqa: E402
from prefill.kv import small_hidden_states  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def main():
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=16, num_experts_per_tok=4,
                         norm_topk_prob=True, max_position_embeddings=256, rms_norm_eps=1e-6, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    ok = True
    for dt in (torch.float32, torch.bfloat16):
        m = Qwen3MoeForCausalLM(cfg).to(dt).eval().requires_grad_(False)
        blk = m.model.layers[1].mlp
        x = torch.randn(1, 97, 64, dtype=dt)
        ref, rl = type(blk).forward(blk, x)  # HF (class not patched yet on the first pass)
        if hasattr(type(blk), "_hf_forward"):
            ref, rl = type(blk)._hf_forward(blk, x)
        got, gl = moe.moe_forward(blk, x)
        e = rel(got, ref)
        print(f"[{dt}] moe block rel err {e:.2e}, router logits equal {torch.equal(rl, gl)}")
        ok &= e < (1e-6 if dt == torch.float32 else 1e-2)
        ids = torch.randint(0, 512, (1, 80))
        with torch.no_grad():
            out = m(input_ids=ids, output_hidden_states=True, use_cache=False)
            moe.patch_qwen3_moe()
            out2 = m(input_ids=ids, output_hidden_states=True, use_cache=False)
            e2 = rel(out2.logits, out.logits)
            print(f"[{dt}] patched HF model vs HF logits rel err {e2:.2e}")
            ok &= e2 < (1e-5 if dt == torch.float32 else 2e-2)
            hs_ref = torch.stack(out.hidden_states, 0)
            hs_kv = small_hidden_states(m, ids)
            e3 = rel(hs_kv, hs_ref)
            print(f"[{dt}] kv.small_hidden_states rel err {e3:.2e}")
            ok &= e3 < (1e-5 if dt == torch.float32 else 2e-2)
            F = FastQwen3(m)
            assert not F.multi_gpu and F.layers[0]["moe"] is not None
            lg, hs = F.prefill(ids[0], collect=True)
            e4, e5 = rel(hs, hs_ref[:, 0]), rel(lg, out.logits[0, -1:])
            print(f"[{dt}] FastQwen3 hidden states rel err {e4:.2e}, last logits rel err {e5:.2e}")
            ok &= e4 < (1e-5 if dt == torch.float32 else 2e-2) and e5 < (1e-5 if dt == torch.float32 else 2e-2)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
