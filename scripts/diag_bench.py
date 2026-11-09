"""
Diagnostics of where ours loses to the large model on the long-context bench (slurm/diag_bench.slurm):
  1. ContractNLI (short contracts, 1-2k tokens): does a floor on the number of recomputed tokens close the gap?
     results/diag_nli_<pair>: attn_small:0.1m256:sent / 0.1m512:sent (10 % but at least N tokens), 0.2 / 0.3.
  2. Multi-hop QA (LastingBench, MuSiQue): sensitivity to the recompute budget. results/diag_hop_<pair>: 0.2 / 0.3.
  3. Llama-3.1-8B target degrading with length (LooGLE / LooGLE-long): projection-only (hyb), selection-only
     (drop-attn_small:0.1:sent) and 20 % recompute, bucketed by prompt length; Qwen3-8B target as the contrast.
     results/diag_long_<pair>.
Baselines (big, attn_small:0.1:sent, specprefill:0.1/0.2) come from results/bench_<pair>.  -> results/diag_summary.md
"""
import collections
import json
import os
import random

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PAIRS = {"06": "Qwen3-0.6B → Qwen3-8B", "17": "Qwen3-1.7B → Qwen3-8B", "ll": "Llama-3.2-1B → Llama-3.1-8B",
         "xq17ll8": "Qwen3-1.7B → Llama-3.1-8B", "xll1q8": "Llama-3.2-1B → Qwen3-8B"}
NAME = {"big": "大模型单独", "hyb": "只投影，不重算", "attn_small:0.1:sent": "我们 10%", "attn_small:0.2:sent": "我们 20%",
        "attn_small:0.3:sent": "我们 30%", "attn_small:0.1m256:sent": "我们 10%（至少 256 token）",
        "attn_small:0.1m512:sent": "我们 10%（至少 512 token）", "drop-attn_small:0.1:sent": "只选不投影（丢弃其余）10%",
        "specprefill:0.1": "SpecPrefill 10%", "specprefill:0.2": "SpecPrefill 20%"}
BUCKETS = [(0, 20480, "<20k"), (20480, 34816, "20k–34k"), (34816, 49152, "34k–48k"), (49152, 1 << 30, "≥48k")]


def rows(pair, kind):
    """results.json rows of bench_<pair> and diag_<kind>_<pair>, merged: {(task, config, subset): row}"""
    out = {}
    for d in (f"bench_{pair}", f"diag_{kind}_{pair}"):
        p = os.path.join(R, d, "results.json")
        if not os.path.exists(p):
            continue
        for t, rs in json.load(open(p)).items():
            if t != "meta":
                for r in rs:
                    out[(t, r["config"], r["subset"])] = r
    return out


def gens(pair, kind, tasks):
    """per-item scores {config: {item key: (score, prompt_len)}} from the generation logs (last record wins)."""
    out = collections.defaultdict(dict)
    for d in (f"bench_{pair}", f"diag_{kind}_{pair}"):
        p = os.path.join(R, d, "generations.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            g = json.loads(line)
            if g["task"] in tasks:
                out[g["config"]][(g["task"], g["id"], str(g.get("answers"))[:60])] = (g["score"], g["prompt_len"])
    return out


def ci(diff, n_boot=2000, seed=0):
    rnd = random.Random(seed)
    bs = sorted(sum(rnd.choice(diff) for _ in diff) / len(diff) for _ in range(n_boot))
    return bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    L = ["# 诊断：我们的方法在哪些基准上不如大模型、为什么", "",
         "由 `slurm/diag_bench.slurm` 产生，`scripts/diag_bench.py` 汇总。基线（大模型、我们 10%、SpecPrefill）来自 `results/bench_<pair>`。"
         "括号内是与大模型逐题配对的差值的 95% bootstrap 置信区间。", ""]

    L += ["## 1. ContractNLI：给重算加最少 token 数", "",
          "合同只有 1–2k token，10% 只有 100–200 个 token。“至少 N token”= 重算比例取 max(10%, N / 中间 token 数)。", ""]
    cfgs = ["big", "attn_small:0.1:sent", "attn_small:0.1m256:sent", "attn_small:0.1m512:sent", "attn_small:0.2:sent",
            "attn_small:0.3:sent", "specprefill:0.1"]
    for pair, pname in PAIRS.items():
        rs = rows(pair, "nli")
        if not any(("contractnli", c, "all") in rs for c in cfgs[2:5]):
            continue
        g = gens(pair, "nli", {"contractnli"})
        L += [f"### {pname}（`{pair}`）", "", "| 配置 | 实际重算比例 | 整体 | 与大模型差（95% CI） | 蕴含 | 矛盾 | 未提及 |", "|---|---|---|---|---|---|---|"]
        for c in cfgs:
            r = rs.get(("contractnli", c, "all"))
            if not r:
                continue
            d = ""
            if c != "big" and g.get("big") and g.get(c):
                ks = [k for k in g[c] if k in g["big"]]
                diff = [g[c][k][0] - g["big"][k][0] for k in ks]
                lo, hi = ci(diff)
                d = f"{sum(diff) / len(diff):+.3f}（{lo:+.3f}, {hi:+.3f}）"
            sub = [f3(rs.get(("contractnli", c, s), {}).get("score")) for s in ("entailment", "contradiction", "neutral")]
            L.append(f"| {NAME.get(c, c)} | {r['frac']:.1%} | {f3(r['score'])} | {d} | " + " | ".join(sub) + " |")
        L.append("")

    L += ["## 2. 多跳问答：重算比例 10% → 20% → 30%", ""]
    cfgs = ["big", "attn_small:0.1:sent", "attn_small:0.2:sent", "attn_small:0.3:sent", "specprefill:0.1", "specprefill:0.2"]
    head = False
    for pair, pname in PAIRS.items():
        rs = rows(pair, "hop")
        if not any((t, "attn_small:0.2:sent", "all") in rs for t in ("lastingbench", "musique")):
            continue
        if not head:
            L += ["| 模型对 | 任务 | " + " | ".join(NAME[c] for c in cfgs) + " |", "|---|---|" + "---|" * len(cfgs)]
            head = True
        for t, tn in (("lastingbench", "LastingBench"), ("musique", "MuSiQue")):
            L.append(f"| {pname} | {tn} | " + " | ".join(f3(rs.get((t, c, "all"), {}).get("score")) for c in cfgs) + " |")
    L.append("")

    L += ["## 3. LooGLE + LooGLE-long 按 prompt 长度分段：投影 vs 选择", "",
          "“只投影”= 中间全部用投影 KV、一个都不重算；“只选不投影”= 和我们 10% 选同样的 token，但没选中的直接丢掉。"
          "如果随长度变差的是“只投影”，问题在 projector（只在 ≤8k 上训练）；如果是“只选不投影”，问题在小模型选 token。"
          "每格：平均 F1（题数）。", ""]
    cfgs = ["big", "attn_small:0.1:sent", "attn_small:0.2:sent", "hyb", "drop-attn_small:0.1:sent", "specprefill:0.1"]
    for pair, pname in PAIRS.items():
        g = gens(pair, "long", {"loogle", "loogle_long"})
        if not any(c in g for c in ("hyb", "drop-attn_small:0.1:sent", "attn_small:0.2:sent")):
            continue
        L += [f"### {pname}（`{pair}`）", "", "| 配置 | " + " | ".join(b[2] for b in BUCKETS) + " | 全部 | 与大模型差（95% CI） |",
              "|---|" + "---|" * (len(BUCKETS) + 2)]
        for c in cfgs:
            if c not in g:
                continue
            cells = []
            for lo, hi, _ in BUCKETS:
                v = [s for s, n in g[c].values() if lo <= n < hi]
                cells.append(f"{sum(v) / len(v):.3f}（{len(v)}）" if v else "-")
            allv = [s for s, _ in g[c].values()]
            d = ""
            if c != "big" and "big" in g:
                ks = [k for k in g[c] if k in g["big"]]
                diff = [g[c][k][0] - g["big"][k][0] for k in ks]
                lo_, hi_ = ci(diff)
                d = f"{sum(diff) / len(diff):+.3f}（{lo_:+.3f}, {hi_:+.3f}）"
            L.append(f"| {NAME.get(c, c)} | " + " | ".join(cells) + f" | {sum(allv) / len(allv):.3f}（{len(allv)}） | {d} |")
        L.append("")

    path = os.path.join(R, "diag_summary.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", os.path.abspath(path))


if __name__ == "__main__":
    main()
