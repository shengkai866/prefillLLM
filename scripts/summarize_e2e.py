#!/usr/bin/env python
"""
results/e2e_{06,17,4B}/results.json (scripts/bench_e2e.py, slurm/bench_e2e.slurm) -> results/e2e_summary.md:
end-to-end latency vs output length, break-even output lengths, memory, batched prefill throughput.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

# (label, setup, config)
METHODS = [("8B 单独", "06", "big"), ("4B 单独", "4B", "small"), ("1.7B 单独", "17", "small"), ("0.6B 单独", "06", "small"),
           ("投影缓存不重算（0.6B）", "06", "hyb"), ("我们 0.6B→8B", "06", "attn_small:0.1:sent"),
           ("我们 1.7B→8B", "17", "attn_small:0.1:sent"), ("SpecPrefill 10%（0.6B）", "06", "specprefill:0.1"),
           ("SpecPrefill 10%（1.7B）", "17", "specprefill:0.1")]
G = 2 ** 30


def load(root, setup):
    p = os.path.join(root, f"e2e_{setup}", "results.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def find(rows, T, cfg):
    return next((r for r in rows if r["prompt_len"] == T and r["config"] == cfg and not r.get("oom")), None)


def break_even(a, b):
    """Smallest output length N at which method b's end-to-end latency is <= a's (a faster TTFT, slower decode)."""
    ta, tb = a["ttft"], b["ttft"]
    sa, sb = a["ms_per_token"] / 1e3, b["ms_per_token"] / 1e3
    if ta >= tb:
        return "从第 1 个 token 起就不快"
    if sa <= sb:
        return "始终更快"
    return f"N ≥ {1 + (tb - ta) / (sa - sb):.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULTS_DIR)
    a = ap.parse_args()
    R = {s: load(a.results, s) for s in ("06", "17", "4B")}
    meta = next((r["meta"] for r in R.values() if r.get("meta")), {})
    out = ["# 端到端延迟、显存、batch 吞吐（L40S，目标模型 Qwen3-8B）", "",
           f"生成：{time.strftime('%F %T')}，`scripts/summarize_e2e.py`；数据 `results/e2e_<setup>`（`scripts/bench_e2e.py`，"
           f"GPU {meta.get('gpu')}）。首 token 延迟：所有方法 CUDA graph（同 speed_allgraph）。解码：CUDA graph 的贪心解码，"
           "每步一次 graph replay；我们 / 投影缓存 / SpecPrefill 都由 8B 解码（SpecPrefill 在它压缩后的 cache 上），“X 单独”由 X 自己解码。"
           "解码每步对整个预分配 buffer（提示 + N）做注意力并用 mask 屏蔽未来位置，所以解码时间是略偏大的上界。", ""]
    if any(r.get("check") for r in R.values()):
        out += [f"GPU 正确性检查：{R['06'].get('check')}", ""]

    lens = sorted({r["prompt_len"] for s in R.values() for r in s.get("decode", [])})
    gens = sorted({int(k) for s in R.values() for r in s.get("decode", []) for k in r.get("e2e", {})})
    out += ["## 1. 端到端延迟（毫秒）= 首 token 延迟 + 再生成 N−1 个 token", ""]
    for T in lens:
        out += [f"### 提示 {T // 1024}k", "", "| 方法 | 解码模型 | 每 token（ms） | " + " | ".join(f"N={n}" for n in gens) + " |",
                "|---|---|---|" + "---|" * len(gens)]
        for lab, st, cfg in METHODS:
            r = find(R[st].get("decode", []), T, cfg)
            if r is None:
                continue
            out.append(f"| {lab} | {'8B' if r['decoder'] == 'large' else lab.split(' ')[0]}（{r['cache_rows']} 行 cache） | "
                       f"{r['ms_per_token']:.1f} | " + " | ".join(f"{r['e2e'][str(n)] * 1e3:.0f}" for n in gens) + " |")
        out.append("")
    out += ["## 2. 输出多长时对手追平我们（端到端延迟相等的输出长度）", "",
            "我们的首 token 更快，但用 8B 解码；对手若解码更快，输出足够长时会追上。", "",
            "| 提示 | 我们 | 对手 | 追平点 |", "|---|---|---|---|"]
    for T in lens:
        for ol, ost, ocfg in METHODS[5:7]:
            o = find(R[ost].get("decode", []), T, ocfg)
            for xl, xst, xcfg in [METHODS[0], METHODS[1], METHODS[2], METHODS[7], METHODS[8]]:
                x = find(R[xst].get("decode", []), T, xcfg)
                if o and x:
                    out.append(f"| {T // 1024}k | {ol} | {xl} | {break_even(o, x)} |")
    out.append("")

    out += ["## 3. 显存（GB）", "",
            "权重 = 这个方法要放在显存里的模型（我们：小模型 + 8B + projector 折叠后的张量）；prefill 峰值 = 一次 eager prefill 在权重之外的峰值"
            "（含它写出的 KV）；KV = 解码 N=1024 时保留的 cache；合计 ≈ 权重 + max(prefill 峰值, KV)。", ""]
    mlens = sorted({r["prompt_len"] for s in R.values() for r in s.get("memory", [])})
    out += ["| 方法 | 权重 | " + " | ".join(f"{T // 1024}k prefill 峰值 / KV / 合计" for T in mlens) + " |", "|---|---|" + "---|" * len(mlens)]
    for lab, st, cfg in METHODS:
        rows = [find(R[st].get("memory", []), T, cfg) for T in mlens]
        if not any(rows):
            continue
        w = next(r["weights"] for r in rows if r)
        cells = []
        for r in rows:
            if r is None:
                cells.append("-")
                continue
            kv = r["kv_cache"][max(r["kv_cache"], key=int)]
            cells.append(f"{r['prefill_peak'] / G:.1f} / {kv / G:.1f} / {(w + max(r['prefill_peak'], kv)) / G:.1f}")
        out.append(f"| {lab} | {w / G:.1f} | " + " | ".join(cells) + " |")
    out.append("")

    out += ["## 4. batch prefill 吞吐（千 token/秒；括号内 = 相对同一 setup、同一 batch 的 8B 单独）", "",
            "B 个等长提示一次 batched 调用。我们：小模型一次 batched 调用 + 每条提示单独选句子和 fold + 8B 一次 batched 调用；"
            "为了形状一致，每条提示正好重算 round(10% × 中间长度) 个 token。SpecPrefill 没有 batched 版本，不在表里。", ""]
    bm = [(l, s, c) for l, s, c in METHODS if not c.startswith("specprefill")]
    blens = sorted({r["prompt_len"] for s in R.values() for r in s.get("batch", [])})
    bs = sorted({r["batch"] for s in R.values() for r in s.get("batch", [])})
    for T in blens:
        out += [f"### 提示 {T // 1024}k", "", "| 方法 | " + " | ".join(f"B={b}" for b in bs) + " |", "|---|" + "---|" * len(bs)]
        for lab, st, cfg in bm:
            cells = []
            for b in bs:
                r = next((x for x in R[st].get("batch", []) if x["prompt_len"] == T and x["batch"] == b), None)
                v = None if r is None else r.get(cfg)
                big = None if r is None else r.get("big")
                cells.append("-" if v is None else f"{b * T / v / 1e3:.1f}" + (f"（{big / v:.2f}×）" if big else ""))
            if any(c != "-" for c in cells):
                out.append(f"| {lab} | " + " | ".join(cells) + " |")
        out.append("")
    path = os.path.join(a.results, "e2e_summary.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("[summarize_e2e]", path)


if __name__ == "__main__":
    main()
