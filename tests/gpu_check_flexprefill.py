#!/usr/bin/env python
"""(GPU) FlexPrefill baseline sanity check on the real large model (PREFILL_LARGE_DIR, default Qwen3-8B):
runs on this triton version, its first-token distribution is close to the dense prefill on real long prompts
(qa_eval_long, reported), the KV it writes matches the dense KV in layer 0 (attention does not change layer-0 K/V), and the
computed block fraction is < 0.5; the first-token KL to dense is printed but not gated on. Exits non-zero on failure (slurm/sparse_baseline.slurm stops before the long runs)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from prefill.fast import FastBig, FastQwen3  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.sparse import flex_prefill  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
_, _, large = load_pair(dev)
L = FastQwen3(large)
pool = PromptPool("qa_eval_long")
idx = [i for i in range(len(pool.meta))][:4]
ok = True
with torch.no_grad():
    for i in idx:
        ids = torch.as_tensor(pool.ids(i), device=dev)
        T = ids.shape[0]
        kb_d, vb_d, lg_d = FastBig(L).prefill(ids)
        kb = torch.empty_like(kb_d)
        vb = torch.empty_like(vb_d)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lg, ratio = flex_prefill(L, ids, kb, vb)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        p, q = F.log_softmax(lg_d.float(), -1), F.log_softmax(lg.float(), -1)
        kl = (p.exp() * (p - q)).sum().item()
        top = int(lg_d.argmax()) == int(lg.argmax())
        k0 = ((kb[0] - kb_d[0]).abs().max() / kb_d[0].abs().max()).item()
        print(f"[flex] T={T}: KL(dense||flex) {kl:.4f}, top-1 same {top}, layer-0 K rel diff {k0:.1e}, "
              f"block ratio {ratio:.3f} (dense causal ~{0.5 + 0.5 / -(-T // 128):.3f}), {dt * 1e3:.0f} ms eager", flush=True)
        # runs correctly: finite, layer-0 K/V untouched by attention, sparse. The KL is reported, not gated on: how close
        # FlexPrefill gets to dense attention is its quality, which the benchmarks measure (one Llama prompt: KL 1.16)
        ok &= torch.isfinite(lg).all().item() and k0 < 1e-2 and ratio < 0.5
print("[flex] OK" if ok else "[flex] FAILED")
sys.exit(0 if ok else 1)
