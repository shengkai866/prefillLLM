"""Faster forward for HF's Qwen3MoeSparseMoeBlock (Qwen3-30B-A3B: 48 layers x 128 experts, top-8).

HF 4.52 loops over all 128 experts and calls torch.where per expert: 128 host syncs per layer, empty experts included.
Here the (token, slot) pairs are sorted by expert once, the per-expert counts come to the host in ONE sync, and only the
experts that received tokens run. Per expert the math is HF's: expert_layer(x) * routing weight (bf16), then index_add_
into a zero buffer in the input dtype, experts in ascending order. A token occurs at most once per expert, so every
token's sum is accumulated in the same order as HF; only the row order inside an expert's GEMM differs (tests/test_moe.py
checks the match). Frozen-model use (no_grad); autograd also works.
"""
import torch
import torch.nn.functional as F


def moe_forward(self, hidden_states):
    B, T, d = hidden_states.shape
    x = hidden_states.view(-1, d)
    router_logits = self.gate(x)
    w = F.softmax(router_logits, dim=1, dtype=torch.float)
    w, sel = torch.topk(w, self.top_k, dim=-1)
    if self.norm_topk_prob:
        w /= w.sum(dim=-1, keepdim=True)
    w = w.to(x.dtype)
    out = torch.zeros_like(x)
    flat = sel.reshape(-1)                                    # [N*k] expert of each (token, slot)
    order = torch.argsort(flat, stable=True)                  # stable: tokens in ascending order within an expert, as HF
    tok = order // self.top_k
    wt = w.reshape(-1)[order]
    counts = torch.bincount(flat, minlength=self.num_experts).tolist()  # the one host sync
    start = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        t = tok[start:start + n]
        y = self.experts[e](x[t]) * wt[start:start + n, None]
        out.index_add_(0, t, y.to(x.dtype))
        start += n
    return out.view(B, T, d), router_logits


_PATCHED = False


def patch_qwen3_moe():
    global _PATCHED
    if _PATCHED:
        return
    from transformers.models.qwen3_moe import modeling_qwen3_moe as M
    M.Qwen3MoeSparseMoeBlock._hf_forward = M.Qwen3MoeSparseMoeBlock.forward
    M.Qwen3MoeSparseMoeBlock.forward = moe_forward
    _PATCHED = True
    print("[moe] Qwen3MoeSparseMoeBlock.forward -> prefill.moe.moe_forward (grouped, one sync per layer)", flush=True)
