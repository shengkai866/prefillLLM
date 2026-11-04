#!/usr/bin/env python
"""
Cross-pair table of the long-context benchmark suite: results/bench_<pair>/results.json (scripts/eval_bench.py) plus
QuALITY from results/quality_<pair>/results.json (scripts/eval_recompute.py --tasks quality). For every pair and task it
lists the large model alone, ours (attn_small:0.1:sent) and SpecPrefill (10 % / 20 %). Writes results/bench_summary.md.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

PAIRS = {"06": "Qwen3-0.6B → Qwen3-8B", "17": "Qwen3-1.7B → Qwen3-8B", "32": "Qwen3-1.7B → Qwen3-32B",
         "ll": "Llama-3.2-1B → Llama-3.1-8B", "xq17ll8": "Qwen3-1.7B → Llama-3.1-8B", "xll1q8": "Llama-3.2-1B → Qwen3-8B"}
COLS = [("big", "大模型单独"), ("attn_small:0.1:sent", "我们（重算 10%）"), ("specprefill:0.1", "SpecPrefill 10%"), ("specprefill:0.2", "SpecPrefill 20%")]
TASKS = [("lastingbench", "LastingBench", "F1"), ("musique", "MuSiQue", "F1"), ("loogle", "LooGLE", "F1"),
         ("loogle_long", "LooGLE-long", "F1"), ("qasper", "Qasper", "F1"), ("clipper", "CLIPPER", "acc"),
         ("quality", "QuALITY", "acc"), ("longhealth", "LongHealth", "acc"), ("contractnli", "ContractNLI", "acc")]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def cell(res, task, cfg):
    if res is None or task not in res:
        return None
    if task == "quality":
        r = next((r for r in res["quality"] if r["config"] == cfg and r["bucket"] == "all" and r["part"] == "all"), None)
        return (r["accuracy"], r["n"]) if r else None
    r = next((r for r in res[task] if r["config"] == cfg and r["subset"] == "all"), None)
    return (r["score"], r["n"]) if r else None


def f(x):
    return "-" if x is None else f"{x[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "bench_summary.md"))
    a = ap.parse_args()
    pairs = [p for p in a.pairs.split(",") if p]
    R = {p: {"bench": load(os.path.join(C.RESULTS_DIR, f"bench_{p}", "results.json")),
             "quality": load(os.path.join(C.RESULTS_DIR, f"quality_{p}", "results.json"))} for p in pairs}
    L = ["# 长文本基准汇总：大模型单独 vs 我们 vs SpecPrefill", "",
         "数据来自 `results/bench_<pair>/results.json`（scripts/eval_bench.py）和 `results/quality_<pair>/results.json`。"
         "F1 任务：贪心生成第一行与参考答案的 token F1；acc 任务：首 token 字母 logits 判分。我们 = 小模型注意力按句子选 10% 中间 token 重算，"
         "其余用投影 KV；SpecPrefill = 官方方法保留整段 prompt 的 10% / 20%。各 pair 的 prompt 上限、题数见各自的 summary.md。", "",
         "数据来源与和原始表格的差异（scripts/prepare_bench.py）：",
         "- LastingBench：github Seriousss/LastingBench 的 hotpotqa_antifact.jsonl（反事实改写的 HotpotQA，LongBench 格式），前 150 题。",
         "- MuSiQue：LongBench 的 musique.jsonl 前 120 题。Qasper：allenai/qasper test 集，只取第一位标注者给出片段或是/否答案的问题，前 304 题。",
         "- LooGLE：bigai-nlco/LooGLE 的 shortdep_qa（每篇 ≤2 题）和 shortdep_cloze（每篇 1 题）各 150 题；LooGLE-long 只取 ≥48k token 的文档"
         "（符合的只有 3 篇问答 + 16 篇完形文档，所以以完形为主）。",
         "- CLIPPER：chtmp223/CLIPPER test 集按 id 取前 504 行，奇偶行交替问该行的真/假断言（各半）；整本书 56k–125k token，统一截断到 prompt 上限。",
         "- LongHealth：github kbressem/LongHealth benchmark_v5.json，fold 0 = patient_01–04 的 80 题（5 选 1）。"
         "ContractNLI：kiddothe2b/contract-nli 的整份合同版（contract_nli_long）test 集，seed 0 抽 300 个（合同，假设）对，3 分类。",
         "- QuALITY：全部 2086 题（results/quality_<pair>），不是原表的 1057 题。Shakespeare-CF 没有找到公开数据，未包含。",
         "- 原表的 “clean” 子集筛选规则未知，这里没有做筛选；prompt 超过上限（8B 对 64k，32B 对 16k）从中间截断（LongBench 规则）。", ""]
    for p in pairs:
        rb, rq = R[p]["bench"], R[p]["quality"]
        if rb is None and rq is None:
            continue
        meta = (rb or {}).get("meta", {})
        L += [f"## {PAIRS.get(p, p)}（`{p}`）", "",
              (f"GPU {meta.get('gpu')}；prompt 上限 {meta.get('args', {}).get('max_len')} token" + (f"，YaRN ×{meta['yarn']}" if meta.get("yarn") else "") + "。") if meta else "",
              "| 任务 | 指标 | 题数 | " + " | ".join(c[1] for c in COLS) + " | 我们 − 大模型 | 我们 − Spec 10% |", "|---|---|---|" + "---|" * (len(COLS) + 2)]
        for t, name, metric in TASKS:
            src = rq if t == "quality" else rb
            cells = [cell(src, t, c) for c, _ in COLS]
            if all(x is None for x in cells):
                continue
            n = next(x[1] for x in cells if x)
            d1 = f"{cells[1][0] - cells[0][0]:+.3f}" if cells[0] and cells[1] else "-"
            d2 = f"{cells[1][0] - cells[2][0]:+.3f}" if cells[1] and cells[2] else "-"
            L.append(f"| {name} | {metric} | {n} | " + " | ".join(f(x) for x in cells) + f" | {d1} | {d2} |")
        L.append("")
    L += ["## 各 pair 对比（我们 / SpecPrefill 10% / 大模型）", "", "| 任务 | " + " | ".join(PAIRS.get(p, p) for p in pairs) + " |", "|---|" + "---|" * len(pairs)]
    for t, name, metric in TASKS:
        v = []
        for p in pairs:
            src = R[p]["quality" if t == "quality" else "bench"]
            o, s, b = cell(src, t, "attn_small:0.1:sent"), cell(src, t, "specprefill:0.1"), cell(src, t, "big")
            v.append("-" if o is None and b is None else f"{f(o)} / {f(s)} / {f(b)}")
        L.append(f"| {name}（{metric}） | " + " | ".join(v) + " |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("[written]", a.out)


if __name__ == "__main__":
    main()
