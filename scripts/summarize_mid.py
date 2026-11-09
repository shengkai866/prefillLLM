#!/usr/bin/env python
"""
Iso-latency comparison: a mid-size model alone (slurm/eval_mid.slurm, default Qwen3-4B -> results/mid_4B_<step>) vs
the Qwen3-8B target alone, the small models alone, ours (recompute 10 %) and SpecPrefill 10 % for the pairs whose
target is Qwen3-8B, on every benchmark. -> results/mid_summary.md

Sources (missing files / configs are shown as "-"):
  pair runs  results/fused_<tag>_{speed,main,long,multiq,longbench} (falls back to <tag>_<step>), quality_<pair>,
             quality_mq_<pair>, bench_<pair>
  mid run    results/mid_<name>_{speed,main,long,multiq,longbench,quality,quality_mq,bench}, config `small`
TTFT cells: ms (speedup vs the 8B alone of the same run).

  python scripts/summarize_mid.py [--results DIR] [--mid 4B]
"""
import argparse
import json
import os
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

PAIRS = {"06": "fused_06", "17": "fused_17", "xll1q8": "fused_ll1toq8"}  # pair -> run prefix (target Qwen3-8B)
FALLBACK = {"xll1q8": "ll1toq8"}
SINGLE = {"ours": "attn_small:0.1:sent", "spec": "specprefill:0.1"}
MULTI = {"ours": "ours:0.1:sent", "spec": "spec-each:0.1"}  # spec-each = SpecPrefill re-prefilled for every question


def columns(mid):
    # (label, source, config kind): source "mid" or a pair; kind big / small / ours / spec
    return [("8B 单独", "06", "big"), (f"{mid} 单独", "mid", "small"), ("1.7B 单独", "17", "small"),
            ("我们 0.6B→8B", "06", "ours"), ("我们 1.7B→8B", "17", "ours"), ("我们 Llama-1B→8B", "xll1q8", "ours"),
            ("SpecPrefill 10%（0.6B 打分）", "06", "spec"), ("SpecPrefill 10%（1.7B 打分）", "17", "spec")]


def load(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


class Runs:
    def __init__(self, root, mid):
        self.root, self.mid, self.cache = root, mid, {}

    def get(self, src, step):
        key = (src, step)
        if key not in self.cache:
            if src == "mid":
                d = f"mid_{self.mid}_{step}"
            elif step in ("quality", "quality_mq", "bench"):
                d = f"{step}_{src}"
            else:
                d = f"{PAIRS[src]}_{step}"
                if not os.path.exists(os.path.join(self.root, d, "results.json")) and src in FALLBACK:
                    d = f"{FALLBACK[src]}_{step}"
            self.cache[key] = load(os.path.join(self.root, d, "results.json"))
        return self.cache[key]


def avg(rows, cfg, val, **match):
    v = [r[val] for r in rows if r.get("config") == cfg and all(str(r.get(k)) == str(x) for k, x in match.items()) and r.get(val) is not None]
    return mean(v) if v else None


def row_specs():
    """(group, label, step, multi?, fn(res, cfg) -> value)"""
    S = []
    for L in (4096, 8192, 16384, 32768):
        S.append(("首 token 延迟（L40S，CUDA graph，ms）", f"{L // 1024}k", "speed", False, ("ttft", L)))
    for L in (2048, 3584, 7680):
        S.append(("NIAH（准确率）", f"{L}", "main", False, lambda r, c, L=L: avg(r.get("niah", []), c, "accuracy", length=L)))
    for L in (4096, 8192):
        S.append(("RULER（各子任务平均）", f"{L // 1024}k", "main", False, lambda r, c, L=L: avg(r.get("ruler", []), c, "score", length=L)))
    for b in ("4k", "8k"):
        S.append(("抽取式 QA（F1，三个来源平均）", b, "main", False, lambda r, c, b=b: avg(r.get("qa", []), c, "f1", bucket=b)))
    S.append(("GSM8K few-shot", "准确率", "main", False, lambda r, c: avg(r.get("gsm8k", []), c, "accuracy")))
    for b in ("4k", "8k"):
        S.append(("代码补全（编辑相似度）", b, "main", False, lambda r, c, b=b: avg(r.get("code", []), c, "similarity", bucket=b)))
    for L in (15872, 32256):
        S.append(("长 NIAH（准确率）", f"{L}", "long", False, lambda r, c, L=L: avg(r.get("niah_long", []), c, "accuracy", length=L)))
    for b in ("16k", "32k"):
        S.append(("长 QA（F1）", b, "long", False, lambda r, c, b=b: avg(r.get("qa_long", []), c, "f1", bucket=b)))
    for t, name in (("gov_report", "gov_report"), ("qmsum", "QMSum"), ("lcc", "LCC"), ("repobench-p", "RepoBench-P")):
        S.append(("LongBench", name, "longbench", False, lambda r, c, t=t: avg(r.get(t, []), c, "score")))
    S.append(("LongBench", "GSM8K 32-shot", "longbench", False, lambda r, c: avg(r.get("gsm32", []), c, "accuracy")))
    for b in ("4k", "8k"):
        S.append(("同一文档 5 个问题（F1，全部问题）", b, "multiq", True, lambda r, c, b=b: avg(r.get("qa", []), c, "f1", bucket=b, part="all")))
    S.append(("QuALITY", "单问题 准确率", "quality", False, lambda r, c: avg(r.get("quality", []), c, "accuracy", bucket="all", part="all")))
    for b in ("4k", "8k"):
        S.append(("QuALITY", f"同一文档多问题 {b}", "quality_mq", True, lambda r, c, b=b: avg(r.get("qa", []), c, "f1", bucket=b, part="all")))
    for t, name in (("lastingbench", "LastingBench（F1）"), ("musique", "MuSiQue（F1）"), ("loogle", "LooGLE（F1）"),
                    ("loogle_long", "LooGLE-long（F1）"), ("qasper", "Qasper（F1）"), ("clipper", "CLIPPER（acc）"),
                    ("longhealth", "LongHealth（acc）"), ("contractnli", "ContractNLI（acc）")):
        S.append(("长文本基准（64k 上限）", name, "bench", False, lambda r, c, t=t: avg(r.get(t, []), c, "score", subset="all")))
    return S


def ttft(res, cfg, L):
    rows = (res or {}).get("speed_allgraph", {}).get("rows", [])
    row = next((r for r in rows if r["prompt_len"] == L), None)
    if not row or row.get(cfg) is None:
        return None
    return row[cfg] * 1e3, (row["big"] / row[cfg] if row.get("big") else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULTS_DIR)
    ap.add_argument("--mid", default="4B", help="results/mid_<mid>_<step>")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = Runs(a.results, a.mid)
    cols = columns(a.mid)
    head = [c[0] for c in cols] + ["我们 0.6B − " + a.mid, "我们 1.7B − " + a.mid]
    out = [f"# 同等延迟对比：Qwen3-{a.mid} 单独 vs Qwen3-8B 单独 vs 我们 vs SpecPrefill", "",
           f"生成：{time.strftime('%F %T')}，`scripts/summarize_mid.py`。{a.mid} 单独来自 `slurm/eval_mid.slurm`（`results/mid_{a.mid}_<step>`，"
           "配置 `small`：它自己 prefill 并自己解码）；其余列来自各 pair 的运行（`results/fused_<tag>_<step>`、`quality_<pair>`、"
           "`quality_mq_<pair>`、`bench_<pair>`）。我们 = 重算 10% 中间 token（按句子，小模型注意力打分），其余用投影 KV，8B 解码。"
           "同一文档多问题的 SpecPrefill 是每个问题重新 prefill（spec-each），单问题是官方 SpecPrefill 保留 10%。"
           "首 token 延迟一格 = 毫秒（相对同一次运行里 8B 单独的加速比）。“-” = 没有这一项。", "",
           "| 基准 | 设置 | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    for group, label, step, multi, fn in row_specs():
        vals = []
        for _, src, kind in cols:
            res = R.get(src, step)
            cfg = {"big": "big", "small": "small"}.get(kind) or (MULTI if multi else SINGLE)[kind]
            if res is None:
                vals.append(None)
            elif isinstance(fn, tuple):
                vals.append(ttft(res, cfg, fn[1]))
            else:
                vals.append(fn(res, cfg))
        if all(v is None for v in vals):
            continue
        if step == "speed":
            cells = ["-" if v is None else f"{v[0]:.0f}" + (f"（{v[1]:.2f}×）" if v[1] else "") for v in vals]
            deltas = ["", ""]
        else:
            cells = ["-" if v is None else f"{v:.3f}" for v in vals]
            m = vals[1]
            deltas = ["-" if m is None or vals[i] is None else f"{vals[i] - m:+.3f}" for i in (3, 4)]
        out.append(f"| {group} | {label} | " + " | ".join(cells + deltas) + " |")
    path = a.out or os.path.join(a.results, "mid_summary.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("[summarize_mid]", path)


if __name__ == "__main__":
    main()
