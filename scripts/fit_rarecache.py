#!/usr/bin/env python
"""
Fit the RaReCache baseline maps (prefill/rarecache.py) for the current small model (PREFILL_SMALL_DIR) -> Qwen3-8B.
Calibration: prompts from our own stage-2 training pools (mix4k_train + mixlong_train: web, QA, code, chat, math,
retrieval), i.e. the same data the projector was trained on, every `stride`-th position until n_tokens positions.
The paper calibrates per task on each benchmark's train split; we use one task-agnostic fit like our projector.
Paper hyper-parameters: K = 8 source layers per target layer, ridge lambda = 0.01, rank r = 128.
Output: CKPT_DIR/rarecache_<small>/fit.pt  ({"cfg", "k": [per target layer], "v": [...]}) + fit_log.json.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.fast import FastQwen3  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.rarecache import calibrate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(C.CKPT_DIR, f"rarecache_{C.SMALL_NAME}", "fit.pt"))
    ap.add_argument("--pools", default="mix4k_train,mixlong_train")
    ap.add_argument("--n_tokens", type=int, default=300000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"[fit] {a.out} exists, skipping")
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dev = torch.device("cuda")
    _, small, large = load_pair(dev)
    S, L = FastQwen3(small), FastQwen3(large)

    # same calibration prompts for the key and the value pass: shuffle both pools together, take prompts until
    # ~n_tokens positions at the given stride
    rng = random.Random(a.seed)
    cands = [(p, i) for p in a.pools.split(",") for i in range(len(PromptPool(p)))]
    rng.shuffle(cands)
    pools = {p: PromptPool(p) for p in a.pools.split(",")}
    chosen, n = [], 0
    for p, i in cands:
        T = int(pools[p].offsets[i + 1] - pools[p].offsets[i])
        chosen.append((p, i))
        n += T // a.stride
        if n >= a.n_tokens:
            break
    n_prompt_tok = sum(int(pools[p].offsets[i + 1] - pools[p].offsets[i]) for p, i in chosen)
    print(f"[fit] {len(chosen)} calibration prompts, {n_prompt_tok} tokens, ~{n} positions at stride {a.stride}", flush=True)
    prompts = lambda: (torch.as_tensor(pools[p].ids(i), device=dev) for p, i in chosen)

    fit, log = {}, {"args": vars(a), "n_prompts": len(chosen), "n_prompt_tokens": n_prompt_tok}
    for sig in ("k", "v"):
        t0 = time.time()
        f = calibrate(S, L, prompts(), sig, a.stride, a.seed, log_every=200)
        t1 = time.time()
        sol = f.solve(K=a.K, lam=a.lam, rank=a.rank)
        print(f"[fit {sig}] {f.N} positions, calibration {(t1-t0)/60:.1f} min, solve {(time.time()-t1)/60:.1f} min", flush=True)
        for t, e in enumerate(sol):
            print(f"[fit {sig}] target layer {t:2d}: R^2 {e['r2']:.3f}  sources {e['src']}  best single {max(e['r2_single']):.3f}", flush=True)
        log[sig] = [{"src": e["src"], "r2": e["r2"], "r2_single": e["r2_single"]} for e in sol]
        log[f"n_{sig}"] = f.N
        fit[sig] = [dict(e, W=e["W"].bfloat16()) for e in sol]
        del f
        torch.cuda.empty_cache()
    fit["cfg"] = {"small": C.SMALL_NAME, "large": C.LARGE_NAME, **vars(a)}
    torch.save(fit, a.out)
    json.dump(log, open(os.path.join(os.path.dirname(a.out), "fit_log.json"), "w"), indent=1)
    print(f"[fit] saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
