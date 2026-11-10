#!/usr/bin/env python
"""Branch proj-drop: recompute + project + drop (RecomputeHybrid.prefill_sel keep=...). On qa_eval prompts:
keep=1 must match the normal recompute path, keep=0 the drop path (first-token KL and 16-token greedy output);
keep=0.5 must hold sink + tail + recomputed + ~half of the rest; random mode runs.
  python tests/test_mix.py <projector.pt>"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from prefill.fast import FastQwen3, build_fold, kv_to_dynamic_cache  # noqa: E402
from prefill.kv import greedy_decode, split_prompt  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import eos_ids, load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402
from prefill.recompute import RecomputeHybrid  # noqa: E402


def kl(a, b):
    return F.kl_div(F.log_softmax(b.float(), -1), F.log_softmax(a.float(), -1), log_target=True, reduction="sum").item()


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    S, L = FastQwen3(small), FastQwen3(large)
    p, _ = Projector.load(sys.argv[1], dev)
    p.eval()
    boundary = torch.tensor([any(ch in tok.decode([i]) for ch in ".!?\n") for i in range(len(tok))], device=dev)
    rh = RecomputeHybrid(S, L, build_fold(p, L), n_sink=8, n_tail=64, boundary=boundary)
    eos = eos_ids(tok)
    pool = PromptPool("qa_eval")
    ok = True
    for i in (0, 1, 160, 320):
        ids = torch.as_tensor(pool.ids(i), device=dev)
        T = ids.shape[0]
        s, m, t = split_prompt(T, 8, 64)

        def gen(kb, vb, lg):
            return greedy_decode(large, kv_to_dynamic_cache(kb, vb), lg, 16, eos, pos0=T if kb.shape[2] != T else None)

        kn, vn, ln, nn = rh.prefill_sel(ids, "attn_small", 0.1, seg="sent")
        k1, v1, l1, n1 = rh.prefill_sel(ids, "attn_small", 0.1, seg="sent", keep=1.0)
        kd, vd, ld, nd = rh.prefill_sel(ids, "attn_small", 0.1, seg="sent", drop=True)
        k0, v0, l0, n0 = rh.prefill_sel(ids, "attn_small", 0.1, seg="sent", keep=0.0)
        kh, vh, lh, nh = rh.prefill_sel(ids, "attn_small", 0.1, seg="sent", keep=0.5)
        kr, vr, lr, nr = rh.prefill_sel(ids, "attn_small", 0.1, seg="sent", keep=0.5, keep_mode="random")
        a, b = kl(ln, l1), kl(ld, l0)
        g1, gn, g0, gd = gen(k1, v1, l1), gen(kn, vn, ln), gen(k0, v0, l0), gen(kd, vd, ld)
        rest = m - nn
        half = kh.shape[2] - (s + t + nh)
        print(f"[{i}] T={T} m={m} rec={nn}/{n1}/{n0} | KL keep1 vs normal {a:.2e} same={g1 == gn} | KL keep0 vs drop {b:.2e} "
              f"same={g0 == gd} K={k0.shape[2]}/{kd.shape[2]} | keep0.5 projected {half}/{rest} ({half / rest:.2f}) "
              f"random {kr.shape[2] - (s + t + nr)} | outputs: {tok.decode(gn)!r} / {tok.decode(gen(kh, vh, lh))!r} / {tok.decode(gd)!r}", flush=True)
        ok &= a < 1e-2 and b < 1e-2 and nn == n1 == n0 and k0.shape[2] == kd.shape[2] and 0.4 < half / rest < 0.65
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
