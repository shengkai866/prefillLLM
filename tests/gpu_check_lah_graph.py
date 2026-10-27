#!/usr/bin/env python
"""
GPU check (real small model) that the CUDA-graphed SpecPrefill look-ahead (prefill/recompute.LookaheadGraph) gives the
same importance and the same kept chunks as the eager loop, and how much time it saves. Exit code 1 on mismatch.
Run with PREFILL_SMALL_DIR set like the eval jobs.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.fast import FastQwen3, time_call  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import eos_ids, load_frozen  # noqa: E402
from prefill.recompute import LookaheadGraph, spec_importance  # noqa: E402


def kept(imp, frac=0.1, c=32):
    T = imp.shape[0]
    unit = torch.arange(T, device=imp.device) // c
    cs = torch.zeros((T + c - 1) // c, device=imp.device).scatter_reduce(0, unit, imp, "mean", include_self=False)
    return set(torch.topk(cs, math.ceil(cs.numel() * frac)).indices.tolist())


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(C.LARGE_DIR)
    S = FastQwen3(load_frozen(C.SMALL_DIR, dev))
    eos = set(eos_ids(tok))
    pool = PromptPool("qa_eval")
    ok = True
    for i in (0, 120, 250):  # real QA prompts (~3.5k and ~7.5k tokens)
        ids = torch.as_tensor(pool.ids(i), device=dev)
        T = ids.shape[0]
        g = LookaheadGraph(S, T, 8)
        _, ie = spec_importance(S, ids, 8, eos)
        _, ig = spec_importance(S, ids, 8, eos, graph=g)
        rel = ((ie - ig).abs().max() / ie.abs().max()).item()
        ke, kg = kept(ie), kept(ig)
        jac = len(ke & kg) / len(ke | kg)
        te = time_call(lambda: spec_importance(S, ids, 8, eos))
        tg = time_call(lambda: spec_importance(S, ids, 8, eos, graph=g))
        print(f"[lah graph] T={T}: importance rel err {rel:.1e}, kept-chunk Jaccard {jac:.3f}, "
              f"eager {te*1e3:.1f} ms, graph {tg*1e3:.1f} ms", flush=True)
        ok &= rel < 5e-2 and jac >= 0.9
        del g
        torch.cuda.empty_cache()
    print("LAH GRAPH CHECK " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
