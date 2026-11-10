#!/usr/bin/env python
"""
Draft-answer selection scorers (slurm/lah_select.slurm -> results/lah_<tag>_<step>) vs the current one, per pair:
paired bootstrap CIs of  new scorer − attn_small (same run, same items), plus each config's mean, the oracle (answer
positions, where the task has answer strings) and TTFT. -> results/lah_summary.md

  python scripts/summarize_lah.py [--results DIR] [--pairs 06,17]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import items  # noqa: E402
from paired_ci import paired  # noqa: E402
from prefill import config as C  # noqa: E402

BASE = "attn_small:0.1:sent"
NEW = [("attn_lah:0.1:sent", "问题 + 草稿 7 token"), ("attn_lah32:0.1:sent", "问题 + 草稿 31 token"),
       ("attn_lahonly:0.1:sent", "只用草稿（含最后一个提示 token）")]
PAIRS = {"06": "Qwen3-0.6B → Qwen3-8B", "17": "Qwen3-1.7B → Qwen3-8B", "ll": "Llama-3.2-1B → Llama-3.1-8B"}
BENCHES = [("main", "ruler@4096", "RULER 4k"), ("main", "ruler@8192", "RULER 8k"), ("main", "qa@4k", "抽取式 QA 4k"),
           ("main", "qa@8k", "抽取式 QA 8k"), ("long", "qa_long@16k", "长 QA 16k"), ("long", "qa_long@32k", "长 QA 32k"),
           ("quality", "quality", "QuALITY"), ("bench", "lastingbench", "LastingBench"), ("bench", "musique", "MuSiQue"),
           ("bench", "qasper", "Qasper"), ("bench", "longhealth", "LongHealth"), ("bench", "contractnli", "ContractNLI")]


def mean_of(A, cfg, field):
    v = [x[cfg][field] for x in A.values() if cfg in x and x[cfg][field] is not None]
    return (float(np.mean(v)), len(v)) if v else None


def fmt_ci(r):
    if r is None:
        return "-"
    star = " **✓**" if r[3] > 0 else " **✗**" if r[4] < 0 else ""
    return f"{r[2]:+.3f} [{r[3]:+.3f}, {r[4]:+.3f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULTS_DIR)
    ap.add_argument("--pairs", default="06,17")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = ["# 选择打分加入草稿答案（look-ahead）：和现有打分的对比", "",
           f"生成：{time.strftime('%F %T')}，`scripts/summarize_lah.py`；数据 `results/lah_<pair>_<step>`（`slurm/lah_select.slurm`）。"
           f"所有配置都重算 10% 中间 token（按句子），其余用投影 KV；只有选哪些句子的打分不同。现有打分 = 最后 64 个提示 token（问题）"
           "对中间 token 的注意力；新打分再加上小模型贪心生成的草稿答案 token 的注意力（借鉴 Prefilling-dLLM 的伪标签、SpecPrefill 的 look-ahead）。"
           "差值 = 新打分 − 现有打分，同一次运行、同一批题逐题配对，bootstrap 95% 置信区间；**✓** / **✗** = 显著更好 / 更差。"
           "理想 = 直接重算答案出现的位置（只在有答案文本的任务上），是选择能做到的参考上限。", ""]
    for pair in [p for p in a.pairs.split(",") if p]:
        tag = f"lah_{pair}"
        cache = {}

        def get(step):
            if step not in cache:
                cache[step] = items.load(os.path.join(a.results, f"{tag}_{step}"), step)
            return cache[step]
        try:
            sp = json.load(open(os.path.join(a.results, f"{tag}_speed", "results.json")))["speed_allgraph"]["rows"]
        except (OSError, KeyError, json.JSONDecodeError):
            sp = []
        out += [f"## {PAIRS.get(pair, pair)}（`{pair}`）", ""]
        if sp:
            cfgs = ["big", BASE] + [c for c, _ in NEW]
            out += ["首 token 延迟（ms，L40S，CUDA graph）：", "", "| 配置 | " + " | ".join(f"{r['prompt_len'] // 1024}k" for r in sp) + " |",
                    "|---|" + "---|" * len(sp)]
            for c in cfgs:
                lab = {"big": "大模型单独", BASE: "现有打分"}.get(c) or dict(NEW)[c]
                out.append(f"| {lab} | " + " | ".join("-" if r.get(c) is None else f"{r[c] * 1e3:.0f}" for r in sp) + " |")
            out.append("")
        head = ["基准", "指标", "n", "现有打分", "理想"] + [f"{lab}：均值 / 差值" for _, lab in NEW]
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for step, bench, name in BENCHES:
            A = get(step).get(bench, {})
            if not A:
                continue
            for field, fname in (("score", "分数"), ("contains", "包含答案")):
                base = mean_of(A, BASE, field)
                if base is None:
                    continue
                orc = mean_of(A, "oracle", field)
                cells = [name, fname, str(base[1]), f"{base[0]:.3f}", "-" if orc is None else f"{orc[0]:.3f}"]
                for c, _ in NEW:
                    mv = mean_of(A, c, field)
                    r = paired(A, A, c, BASE, field, False, rng, a.boot)
                    cells.append("-" if mv is None else f"{mv[0]:.3f} / {fmt_ci(r)}")
                out.append("| " + " | ".join(cells) + " |")
        out.append("")
    path = os.path.join(a.results, "lah_summary.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("[summarize_lah]", path)


if __name__ == "__main__":
    main()
