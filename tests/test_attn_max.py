"""
CPU check of recompute.attn_max_layer (the attn_max scorer): for queries that may see every prompt key (positions >= T-1)
it must equal SpecPrefill's spec_scores aggregation for one layer, and the causal mask must zero keys after a query.

Run:  python tests/test_attn_max.py
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill.recompute import attn_max_layer, spec_scores  # noqa: E402


def main():
    torch.manual_seed(0)
    H, Hkv, D, T, n = 16, 8, 64, 500, 3
    k = torch.randn(Hkv, T, D)
    q = torch.randn(H, n, D) * 2
    scale = D ** -0.5
    # 1. all queries see all T keys -> one layer of spec_scores (max over a single layer = that layer)
    a = attn_max_layer(q, k, torch.full((n,), T - 1), 0, T, scale, kernel=13)  # [n, T]
    S = types.SimpleNamespace(H=H, Hkv=Hkv, rep=H // Hkv, D=D, scaling=scale)
    ref = spec_scores(S, [q], k[None], T, kernel=13)  # [T], mean over queries
    err = (a.mean(0) - ref).abs().max().item()
    print(f"attn_max_layer vs spec_scores (1 layer, full visibility): max abs diff {err:.2e}")
    assert err < 1e-6
    # 2. causal: a query at position p puts no mass on keys > p (before pooling; check with kernel=0)
    p = torch.tensor([100, 250, 499])
    a0 = attn_max_layer(q, k, p, 0, T, scale, kernel=0)
    for j in range(n):
        assert a0[j, p[j] + 1:].abs().max() == 0 if p[j] + 1 < T else True
        assert a0[j, :p[j] + 1].sum() > 0
    # 3. slicing [lo, hi) equals slicing the full result
    a1 = attn_max_layer(q, k, p, 40, 300, scale, kernel=13)
    full = attn_max_layer(q, k, p, 0, T, scale, kernel=13)
    assert torch.equal(a1, full[:, 40:300])
    print("ok")


if __name__ == "__main__":
    main()
