#!/usr/bin/env python
"""
Paired bootstrap 95 % confidence intervals of every headline difference, item by item (scripts/items.py):
  ours (recompute 10 %) − large model alone, ours − SpecPrefill 10 %, ours − a mid-size model alone
  (Qwen3-8B targets: Qwen3-4B alone, results/mid_4B_*; the 32B target: Qwen3-8B alone from the 0.6B -> 8B runs).
Items are paired by their key (same prompt pool); for the long-context suite across runs with different prompt caps
only items with the same prompt length are paired. Both metrics where they exist: the benchmark's own score (F1 of the
first line / accuracy / ROUGE / edit similarity) and `contains` (a normalised gold answer occurs in the output), which
does not penalise longer answers and is the fair cross-model metric for free-form QA.
-> results/ci_summary.md, results/ci.json.  GSM8K / code completion have no per-item log and are not covered.

  python scripts/paired_ci.py [--results DIR] [--boot 10000]
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
from prefill import config as C  # noqa: E402

STEPS = ("main", "long", "multiq", "longbench", "quality", "quality_mq", "bench")


def runs(prefix, pair, fallback=None):
    r = {s: f"{prefix}_{s}" for s in ("main", "long", "multiq", "longbench")}
    r.update(quality=f"quality_{pair}", quality_mq=f"quality_mq_{pair}", bench=f"bench_{pair}")
    return r, fallback


MID4B = {s: f"mid_4B_{s}" for s in STEPS}
BIG8 = runs("fused_06", "06")[0]
PAIRS = [  # (pair, label, runs, mid label, mid runs, mid config)
    ("06", "Qwen3-0.6B → Qwen3-8B", runs("fused_06", "06")[0], "4B 单独", MID4B, "small"),
    ("17", "Qwen3-1.7B → Qwen3-8B", runs("fused_17", "17")[0], "4B 单独", MID4B, "small"),
    ("xll1q8", "Llama-3.2-1B → Qwen3-8B", runs("fused_ll1toq8", "xll1q8")[0], "4B 单独", MID4B, "small"),
    ("ll", "Llama-3.2-1B → Llama-3.1-8B", runs("fused_ll1to8", "ll")[0], None, None, None),
    ("xq17ll8", "Qwen3-1.7B → Llama-3.1-8B", runs("fused_q17toll8", "xq17ll8")[0], None, None, None),
    ("32", "Qwen3-1.7B → Qwen3-32B", {**runs("e32", "32")[0]}, "8B 单独", BIG8, "big"),
]
FALLBACK = {"fused_ll1toq8": "ll1toq8"}
BENCHES = [("ruler@4096", "RULER 4k"), ("ruler@8192", "RULER 8k"), ("qa@4k", "抽取式 QA 4k"), ("qa@8k", "抽取式 QA 8k"),
           ("qa_long@16k", "长 QA 16k"), ("qa_long@32k", "长 QA 32k"), ("multiq@4k", "同一文档 5 问 4k"),
           ("multiq@8k", "同一文档 5 问 8k"), ("gov_report", "gov_report"), ("qmsum", "QMSum"), ("lcc", "LCC"),
           ("repobench-p", "RepoBench-P"), ("quality", "QuALITY"), ("quality_mq@4k", "QuALITY 多问题 4k"),
           ("quality_mq@8k", "QuALITY 多问题 8k"), ("lastingbench", "LastingBench"), ("musique", "MuSiQue"),
           ("qasper", "Qasper"), ("loogle", "LooGLE"), ("loogle_long", "LooGLE-long"), ("clipper", "CLIPPER"),
           ("longhealth", "LongHealth"), ("contractnli", "ContractNLI")]
STEP_OF_BENCH = {"ruler": "main", "qa": "main", "qa_long": "long", "multiq": "multiq", "quality": "quality",
                 "quality_mq": "quality_mq", "gov_report": "longbench", "qmsum": "longbench", "lcc": "longbench",
                 "repobench-p": "longbench"}


def step_of(bench):
    return STEP_OF_BENCH.get(bench.split("@")[0], "bench")


def ours(step):
    return "ours:0.1:sent" if step in ("multiq", "quality_mq") else "attn_small:0.1:sent"


def spec(step):
    return "spec-each:0.1" if step in ("multiq", "quality_mq") else "specprefill:0.1"


class Loader:
    def __init__(self, root):
        self.root, self.cache = root, {}

    def get(self, run, step):
        if run is None:
            return {}
        if run not in self.cache:
            d = os.path.join(self.root, run)
            if not os.path.exists(os.path.join(d, "generations.jsonl")):
                pre = next((p for p in FALLBACK if run.startswith(p)), None)
                if pre:
                    d = os.path.join(self.root, run.replace(pre, FALLBACK[pre], 1))
            got = items.load(d, step)
            try:  # only benchmarks the run finished (an interrupted run can leave partial generations behind)
                done = set(json.load(open(os.path.join(d, "results.json"))))
            except (OSError, json.JSONDecodeError):
                done = set()
            task = lambda b: {"multiq": "qa", "quality_mq": "qa"}.get(step, b.split("@")[0])
            self.cache[run] = {b: v for b, v in got.items() if task(b) in done}
        return self.cache[run]


def paired(A, B, cfg_a, cfg_b, field, same_len, rng, boot):
    """A, B: {item: {config: rec}} -> (mean a, mean b, delta, lo, hi, n) or None."""
    xs, ys = [], []
    for k, ra in A.items():
        rb = B.get(k)
        if rb is None or cfg_a not in ra or cfg_b not in rb:
            continue
        a, b = ra[cfg_a], rb[cfg_b]
        if a[field] is None or b[field] is None:
            continue
        if same_len and a.get("prompt_len") != b.get("prompt_len"):
            continue
        xs.append(a[field])
        ys.append(b[field])
    if len(xs) < 5:
        return None
    x, y = np.array(xs), np.array(ys)
    d = x - y
    idx = rng.integers(0, len(d), (boot, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(x.mean()), float(y.mean()), float(d.mean()), float(lo), float(hi), len(d)


def fmt(r):
    if r is None:
        return "-"
    star = " **✓**" if r[3] > 0 else " **✗**" if r[4] < 0 else ""
    return f"{r[2]:+.3f} [{r[3]:+.3f}, {r[4]:+.3f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULTS_DIR)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    L = Loader(a.results)
    rng = np.random.default_rng(a.seed)
    allres = {}
    out = ["# 主要差值的配对 bootstrap 95% 置信区间", "",
           f"生成：{time.strftime('%F %T')}，`scripts/paired_ci.py`（逐题配对，{a.boot} 次重采样）。每格：差值 [95% 置信区间]；"
           "**✓** = 区间整体 > 0（我们显著更好），**✗** = 区间整体 < 0（我们显著更差），没有标记 = 不显著。"
           "我们 = 重算 10%；SpecPrefill = 单问题保留 10%，多问题设置下每个问题重新 prefill（spec-each）。"
           "“分数”是该基准自己的指标（生成类 QA：第一行的 F1；选择题：准确率；摘要：ROUGE；代码：编辑相似度）；"
           "“包含答案”= 归一化后的标准答案出现在输出里，不惩罚更长的回答，是跨模型比较自由回答 QA 时更公平的指标。"
           "不同 prompt 上限的跨运行比较（32B 的 16k 对 8B 的 64k）只配对 prompt 长度相同的题。"
           "GSM8K、代码补全没有逐题记录，不在表内。32B 的 LongBench LCC / RepoBench-P 用原始 prompt 时 32B 会用文字回答"
           "（见 REPORT_2026-11-02 §2.3），所以不和 8B 单独比。每个格子是独立的检验，没有做多重比较校正：几十项里有一两项“显著”可能是偶然。", ""]
    for pair, label, R, mid_label, MR, mid_cfg in PAIRS:
        rows = []
        head = ["基准", "指标", "n", f"我们 − 大模型单独", "我们 − SpecPrefill 10%"] + ([f"我们 − {mid_label}", "n（中等模型配对）"] if mid_label else [])
        for bench, name in BENCHES:
            st = step_of(bench)
            A = L.get(R.get(st), st).get(bench, {})
            if not A:
                continue
            M = L.get(MR.get(st), st).get(bench, {}) if mid_label else {}
            for field, fname in (("score", "分数"), ("contains", "包含答案")):
                r_big = paired(A, A, ours(st), "big", field, False, rng, a.boot)
                r_spec = paired(A, A, ours(st), spec(st), field, False, rng, a.boot)
                # 32B LongBench code: raw prompts make 32B (hence ours, decoded by 32B) answer in prose -> not comparable with 8B
                skip_mid = pair == "32" and bench in ("lcc", "repobench-p")
                r_mid = paired(A, M, ours(st), mid_cfg, field, st == "bench", rng, a.boot) if mid_label and not skip_mid else None
                if r_big is None and r_spec is None and r_mid is None:
                    continue
                n = (r_big or r_spec or r_mid)[5]
                cells = [name, fname, str(n), fmt(r_big), fmt(r_spec)] + ([fmt(r_mid), str(r_mid[5]) if r_mid else "-"] if mid_label else [])
                rows.append("| " + " | ".join(cells) + " |")
                allres.setdefault(pair, {})[f"{bench}|{field}"] = {"vs_big": r_big, "vs_spec": r_spec, "vs_mid": r_mid}
        if not rows:
            continue
        tally = []
        for key, lab in (("vs_big", "大模型"), ("vs_spec", "SpecPrefill 10%"), ("vs_mid", mid_label)):
            v = [x[key] for k, x in allres[pair].items() if x.get(key) and k.endswith("|score")]
            if v:
                tally.append(f"对{lab}：显著好 {sum(r[3] > 0 for r in v)}，显著差 {sum(r[4] < 0 for r in v)}，不显著 "
                             f"{sum(r[3] <= 0 <= r[4] for r in v)}（共 {len(v)} 项，按“分数”）")
        out += [f"## {label}（`{pair}`）", "", "；".join(tally) + "。", "", "| " + " | ".join(head) + " |",
                "|" + "---|" * len(head)] + rows + [""]
    open(os.path.join(a.results, "ci_summary.md"), "w").write("\n".join(out) + "\n")
    json.dump(allres, open(os.path.join(a.results, "ci.json"), "w"), indent=1)
    print("[paired_ci]", os.path.join(a.results, "ci_summary.md"))


if __name__ == "__main__":
    main()
