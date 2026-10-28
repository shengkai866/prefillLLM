#!/usr/bin/env python
"""
Diagnostics for the RaReCache baseline fit (prefill/rarecache.py): is the low quality a property of the ridge map or of
our implementation choices? On held-out prompts (pool mixtest by default) this measures, per sigma (keys / values):
  1. held-out R^2 of the SAVED map (bf16 W) applied the way inference applies it (bf16 centring + matmul) vs in fp32
     -> gap = bf16 rounding of (x - xbar) @ W at inference
  2. the same split by position: position 0 (attention sink) / positions 1-7 / the rest
     -> a bad sink row would disturb every later query of the target
  3. a fresh fit on --n_refit positions (fp64 solve) and its held-out R^2 with fp32 W vs bf16-cast W
     -> gap = quantising W itself; also prints the ridge's effective regularisation (lam vs. the smallest eigenvalue of
        the centred source covariance SUM, which is what the solve actually regularises)
Output: a JSON + a markdown table in --out (default results/rare_diag_<small>/fit_diag.{json,md}).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.fast import FastQwen3  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.rarecache import RaReFitter, cache_rows  # noqa: E402


def flat(r):
    return r.transpose(0, 1).reshape(r.shape[1], -1)  # [L, T, d] -> [T, L*d]


@torch.no_grad()
def heldout(S, L, prompts, stride):
    """-> dict sigma -> (X [n, Ls*d_s] fp32 source rows, Y [n, Lt*d_t] fp32 target rows, pos [n] positions)."""
    out = {"k": [[], [], []], "v": [[], [], []]}
    for ids in prompts:
        T = ids.shape[0]
        pos = torch.arange(0, T, stride, device=ids.device)
        xs, xt = cache_rows(S, ids), cache_rows(L, ids)
        for j, sig in enumerate("kv"):
            out[sig][0].append(flat(xs[j])[pos].float().cpu())
            out[sig][1].append(flat(xt[j])[pos].float().cpu())
            out[sig][2].append(pos.cpu())
    return {s: tuple(torch.cat(v) for v in lists) for s, lists in out.items()}


def r2(pred, y, mask=None):
    if mask is not None:
        pred, y = pred[mask], y[mask]
    if y.shape[0] == 0:
        return float("nan")
    sse = (pred - y).pow(2).sum()
    sst = (y - y.mean(0)).pow(2).sum()
    return float(1 - sse / sst)


def apply(entry, X, d_s, mode, dev):
    """Predict target rows for ONE target layer from source rows X [n, Ls*d_s] (fp32 cpu). mode: 'bf16' = inference path
    (bf16 x, xbar, W, matmul), 'fp32' = everything fp32 (W upcast from whatever it is stored as)."""
    idx = torch.cat([torch.arange(s * d_s, (s + 1) * d_s) for s in entry["src"]])
    x = X[:, idx].to(dev)
    W, xb, yb = entry["W"].to(dev), entry["xbar"].to(dev).float(), entry["ybar"].to(dev).float()
    if mode == "bf16":
        z = (x.bfloat16() - xb.bfloat16()) @ W.bfloat16()
        return (z.float() + yb).cpu()
    return ((x - xb) @ W.float() + yb).cpu()


def eval_map(entries, X, Y, pos, d_s, d_t, dev):
    """-> dict of R^2 (mean over target layers, SSE-weighted) for both modes and the position splits."""
    splits = {"all": torch.ones_like(pos, dtype=torch.bool), "pos0": pos == 0, "pos1-7": (pos >= 1) & (pos < 8), "pos>=8": pos >= 8}
    acc = {m: {k: [0.0, 0.0] for k in splits} for m in ("bf16", "fp32")}
    per_layer = {m: [] for m in acc}
    for t, e in enumerate(entries):
        y = Y[:, t * d_t:(t + 1) * d_t]
        for m in acc:
            p = apply(e, X, d_s, m, dev)
            per_layer[m].append(r2(p, y))
            for k, msk in splits.items():
                if msk.any():
                    yy = y[msk]
                    acc[m][k][0] += float((p[msk] - yy).pow(2).sum())
                    acc[m][k][1] += float((yy - yy.mean(0)).pow(2).sum())
    out = {m: {k: (1 - a / b if b else float("nan")) for k, (a, b) in d.items()} for m, d in acc.items()}
    for m in acc:
        out[m]["per_layer"] = per_layer[m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=os.path.join(C.CKPT_DIR, f"rarecache_{C.SMALL_NAME}", "fit.pt"))
    ap.add_argument("--pool", default="mixtest")
    ap.add_argument("--n_prompts", type=int, default=24)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n_refit", type=int, default=60000, help="positions for the fresh fp64 fit (0 = skip)")
    ap.add_argument("--refit_pools", default="mix4k_train,mixlong_train")
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, f"rare_diag_{C.SMALL_NAME}"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    _, small, large = load_pair(dev)
    S, L = FastQwen3(small), FastQwen3(large)
    d_s, d_t = S.Hkv * S.D, L.Hkv * L.D
    fit = torch.load(a.fit, map_location="cpu")

    pool = PromptPool(a.pool)
    step = max(1, len(pool) // a.n_prompts)
    held_ids = list(range(0, len(pool), step))[:a.n_prompts]
    t0 = time.time()
    H = heldout(S, L, (torch.as_tensor(pool.ids(i), device=dev) for i in held_ids), a.stride)
    print(f"[diag] held-out {a.pool}: {len(held_ids)} prompts, {H['k'][0].shape[0]} positions at stride {a.stride} ({time.time()-t0:.0f}s)", flush=True)

    res = {"fit": a.fit, "pool": a.pool, "n_prompts": len(held_ids), "n_positions": int(H["k"][0].shape[0]), "saved": {}, "refit": {}}
    for sig in "kv":
        X, Y, pos = H[sig]
        res["saved"][sig] = eval_map(fit[sig], X, Y, pos, d_s, d_t, dev)
        calib = sum(e["r2"] for e in fit[sig]) / len(fit[sig])
        res["saved"][sig]["calibration_r2_mean"] = calib
        r = res["saved"][sig]
        print(f"[saved {sig}] calibration R^2 {calib:.3f} | held-out bf16: all {r['bf16']['all']:.3f} pos0 {r['bf16']['pos0']:.3f} "
              f"pos1-7 {r['bf16']['pos1-7']:.3f} rest {r['bf16']['pos>=8']:.3f} | fp32: all {r['fp32']['all']:.3f} pos0 {r['fp32']['pos0']:.3f}", flush=True)

    if a.n_refit:
        import random
        rng = random.Random(0)
        pools = {p: PromptPool(p) for p in a.refit_pools.split(",")}
        cands = [(p, i) for p in pools for i in range(len(pools[p]))]
        rng.shuffle(cands)
        chosen, n = [], 0
        for p, i in cands:
            chosen.append((p, i))
            n += int(pools[p].offsets[i + 1] - pools[p].offsets[i]) // 8
            if n >= a.n_refit:
                break
        print(f"[refit] {len(chosen)} prompts, ~{n} positions at stride 8", flush=True)
        for sig in "kv":
            fitter = RaReFitter(S.n_layers, d_s, L.n_layers, d_t, dev)
            g = torch.Generator().manual_seed(0)
            j = 0 if sig == "k" else 1
            for p, i in chosen:
                ids = torch.as_tensor(pools[p].ids(i), device=dev)
                pos = torch.arange(int(torch.randint(8, (1,), generator=g)), ids.shape[0], 8, device=dev)
                xs, xt = cache_rows(S, ids), cache_rows(L, ids)
                fitter.add(flat(xs[j])[pos], flat(xt[j])[pos])
            # effective regularisation: lam = 0.01 against the eigenvalues of the centred covariance SUM of one source
            # layer block (what solve() adds lam*I to)
            A0 = fitter._cxx(torch.arange(d_s, device=dev), torch.arange(d_s, device=dev))
            ev = torch.linalg.eigvalsh(A0)
            sol = fitter.solve(K=fit["cfg"].get("K", 8), lam=fit["cfg"].get("lam", 0.01), rank=fit["cfg"].get("rank", 128))
            X, Y, pos = H[sig]
            fp = eval_map(sol, X, Y, pos, d_s, d_t, dev)
            bf = eval_map([dict(e, W=e["W"].bfloat16()) for e in sol], X, Y, pos, d_s, d_t, dev)
            res["refit"][sig] = {"n_positions": fitter.N, "calibration_r2_mean": sum(e["r2"] for e in sol) / len(sol),
                                 "fp32W": fp["fp32"], "bf16W_fp32apply": bf["fp32"], "bf16W_bf16apply": bf["bf16"],
                                 "cov_sum_eig_min": float(ev[0]), "cov_sum_eig_max": float(ev[-1]), "cov_sum_eig_med": float(ev[d_s // 2]),
                                 "lam": fit["cfg"].get("lam", 0.01)}
            r = res["refit"][sig]
            print(f"[refit {sig}] N {fitter.N} calibration R^2 {r['calibration_r2_mean']:.3f} | held-out fp32 W {fp['fp32']['all']:.3f} "
                  f"| bf16 W, fp32 apply {bf['fp32']['all']:.3f} | bf16 W, bf16 apply {bf['bf16']['all']:.3f} "
                  f"| cov-sum eig min/med/max {ev[0]:.3g}/{ev[d_s//2]:.3g}/{ev[-1]:.3g} vs lam {r['lam']}", flush=True)
            del fitter
            torch.cuda.empty_cache()

    json.dump(res, open(os.path.join(a.out, "fit_diag.json"), "w"), indent=1)
    rows = [f"# RaReCache 映射诊断（{C.SMALL_NAME} -> {C.LARGE_NAME}，held-out {a.pool}，{len(held_ids)} 条、{res['n_positions']} 个位置）", "",
            "| 量 | 键 R² | 值 R² |", "|---|---|---|"]
    s = res["saved"]
    rows.append(f"| 保存的映射：校准集 R²（fit 日志） | {s['k']['calibration_r2_mean']:.3f} | {s['v']['calibration_r2_mean']:.3f} |")
    for lab, m, k in [("保存的映射：held-out，推理路径（bf16）", "bf16", "all"), ("保存的映射：held-out，fp32 应用", "fp32", "all"),
                      ("保存的映射：held-out，位置 0（bf16）", "bf16", "pos0"), ("保存的映射：held-out，位置 1-7（bf16）", "bf16", "pos1-7"),
                      ("保存的映射：held-out，位置 ≥ 8（bf16）", "bf16", "pos>=8")]:
        rows.append(f"| {lab} | {s['k'][m][k]:.3f} | {s['v'][m][k]:.3f} |")
    if res["refit"]:
        f_ = res["refit"]
        for lab, k in [("重新拟合：校准集 R²", "calibration_r2_mean")]:
            rows.append(f"| {lab}（{f_['k']['n_positions']} 个位置） | {f_['k'][k]:.3f} | {f_['v'][k]:.3f} |")
        for lab, k in [("重新拟合：held-out，fp32 W", "fp32W"), ("重新拟合：held-out，bf16 W + fp32 应用", "bf16W_fp32apply"),
                       ("重新拟合：held-out，bf16 W + bf16 应用", "bf16W_bf16apply")]:
            rows.append(f"| {lab} | {f_['k'][k]['all']:.3f} | {f_['v'][k]['all']:.3f} |")
        rows.append(f"| 源协方差（和）特征值 min / 中位 / max，对比 λ = {f_['k']['lam']} | "
                    f"{f_['k']['cov_sum_eig_min']:.3g} / {f_['k']['cov_sum_eig_med']:.3g} / {f_['k']['cov_sum_eig_max']:.3g} | "
                    f"{f_['v']['cov_sum_eig_min']:.3g} / {f_['v']['cov_sum_eig_med']:.3g} / {f_['v']['cov_sum_eig_max']:.3g} |")
    open(os.path.join(a.out, "fit_diag.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
