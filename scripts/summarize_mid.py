#!/usr/bin/env python
"""
Iso-latency comparison: smaller models alone vs the large model alone, ours (recompute 10 %) and SpecPrefill 10 %, on
every benchmark. -> results/mid_summary.md
  §1 Qwen3-8B target: 8B / 4B (slurm/eval_mid.slurm -> results/mid_<name>_<step>) / 1.7B alone, ours 0.6B, 1.7B,
     Llama-1B -> 8B, SpecPrefill 10 %
  §2 Qwen3-32B target: 32B / ours 1.7B -> 32B / SpecPrefill 10 % / 1.7B alone (results/e32_*, quality_32, bench_32),
     8B alone (the 0.6B -> 8B runs) and 4B alone. No TTFT rows: the 32B pair was timed on an H100, everything else on
     an L40S. The long-context suite of 32B is capped at 16k (8B / 4B: 64k): only tasks whose prompts are identical
     under both caps (Qasper, LongHealth, ContractNLI) get 8B / 4B values. LongBench LCC / RepoBench-P with raw prompts
     make 32B answer in prose, so 8B / 4B values are not comparable there and are left out.
Free-form QA rows come twice: the benchmark's own score (F1 of the first line) and `contains` (a normalised gold answer
occurs in the output; scripts/items.py), which does not penalise longer answers — larger models answer more verbosely
(32B: QA 4k F1 0.638 vs 8B 0.772 at the same contains 0.773), so `contains` is the metric to compare models with.

  python scripts/summarize_mid.py [--results DIR] [--mid 4B]
"""
import argparse
import json
import os
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import items  # noqa: E402
from prefill import config as C  # noqa: E402

RUNS = {"06": "fused_06", "17": "fused_17", "xll1q8": "fused_ll1toq8", "32": "e32"}  # pair -> run prefix
FALLBACK = {"xll1q8": "ll1toq8"}
SINGLE = {"ours": "attn_small:0.1:sent", "spec": "specprefill:0.1"}
MULTI = {"ours": "ours:0.1:sent", "spec": "spec-each:0.1"}  # spec-each = SpecPrefill re-prefilled for every question
SAME_PROMPTS_16K_64K = {"qasper", "longhealth", "contractnli"}


def sections(mid):
    # (title, columns [(label, source, kind)], delta columns [(label, i_ours, i_ref)], with TTFT rows, cross-cap sources)
    s1 = ("1. 目标 Qwen3-8B（L40S）",
          [("8B 单独", "06", "big"), (f"{mid} 单独", "mid", "small"), ("1.7B 单独", "17", "small"),
           ("我们 0.6B→8B", "06", "ours"), ("我们 1.7B→8B", "17", "ours"), ("我们 Llama-1B→8B", "xll1q8", "ours"),
           ("SpecPrefill 10%（0.6B 打分）", "06", "spec"), ("SpecPrefill 10%（1.7B 打分）", "17", "spec")],
          [(f"我们 0.6B − {mid}", 3, 1), (f"我们 1.7B − {mid}", 4, 1)], True, set())
    s2 = ("2. 目标 Qwen3-32B（32B 的结果在 H100 上，质量与 GPU 无关）",
          [("32B 单独", "32", "big"), ("我们 1.7B→32B", "32", "ours"), ("SpecPrefill 10%（1.7B 打分）", "32", "spec"),
           ("1.7B 单独", "32", "small"), ("8B 单独", "06", "big"), (f"{mid} 单独", "mid", "small")],
          [("我们 − 8B", 1, 4), (f"我们 − {mid}", 1, 5), ("我们 − 32B", 1, 0)], False, {"06", "mid"})
    return [s1, s2]


class Runs:
    def __init__(self, root, mid):
        self.root, self.mid, self.cache, self.icache = root, mid, {}, {}

    def dir(self, src, step):
        if src == "mid":
            return f"mid_{self.mid}_{step}"
        if step in ("quality", "quality_mq", "bench"):
            return f"{step}_{src}"
        d = f"{RUNS[src]}_{step}"
        if not os.path.exists(os.path.join(self.root, d, "results.json")) and src in FALLBACK:
            d = f"{FALLBACK[src]}_{step}"
        return d

    def get(self, src, step):
        key = (src, step)
        if key not in self.cache:
            try:
                self.cache[key] = json.load(open(os.path.join(self.root, self.dir(src, step), "results.json")))
            except (OSError, json.JSONDecodeError):
                self.cache[key] = None
        return self.cache[key]

    def items(self, src, step):
        key = (src, step)
        if key not in self.icache:
            self.icache[key] = items.load(os.path.join(self.root, self.dir(src, step)), step)
        return self.icache[key]


def avg(rows, cfg, val, **match):
    """Mean of `val` over the rows of `cfg` matching `match`. Rows whose task starts with "_" are bookkeeping, not scores
    (RULER stores the recomputed fraction as task "_frac", score ~0.1) and are skipped."""
    v = [r[val] for r in rows if r.get("config") == cfg and not str(r.get("task", "")).startswith("_")
         and all(str(r.get(k)) == str(x) for k, x in match.items()) and r.get(val) is not None]
    return mean(v) if v else None


def item_contains(R, src, step, task, cfg):
    d = R.items(src, step).get(task, {})
    v = [x[cfg]["contains"] for x in d.values() if cfg in x and x[cfg]["contains"] is not None]
    return mean(v) if v else None


def row_specs():
    """(group, label, step, multi?, kind, fn(R, src, res, cfg) -> value); kind: ttft / score / contains."""
    S = []
    for L in (4096, 8192, 16384, 32768):
        S.append(("首 token 延迟（L40S，CUDA graph，ms）", f"{L // 1024}k", "speed", False, "ttft", L))
    for L in (2048, 3584, 7680):
        S.append(("NIAH（准确率）", f"{L}", "main", False, "score", lambda R, s, r, c, L=L: avg(r.get("niah", []), c, "accuracy", length=L)))
    for L in (4096, 8192):
        S.append(("RULER（各子任务平均）", f"{L // 1024}k", "main", False, "score", lambda R, s, r, c, L=L: avg(r.get("ruler", []), c, "score", length=L)))
    for b in ("4k", "8k"):
        S.append(("抽取式 QA（三个来源平均）", f"{b} F1", "main", False, "score", lambda R, s, r, c, b=b: avg(r.get("qa", []), c, "f1", bucket=b)))
        S.append(("抽取式 QA（三个来源平均）", f"{b} 包含答案", "main", False, "contains", lambda R, s, r, c, b=b: avg(r.get("qa", []), c, "contains", bucket=b)))
    S.append(("GSM8K few-shot", "准确率", "main", False, "score", lambda R, s, r, c: avg(r.get("gsm8k", []), c, "accuracy")))
    for b in ("4k", "8k"):
        S.append(("代码补全（编辑相似度）", b, "main", False, "score", lambda R, s, r, c, b=b: avg(r.get("code", []), c, "similarity", bucket=b)))
    for L in (15872, 32256):
        S.append(("长 NIAH（准确率）", f"{L}", "long", False, "score", lambda R, s, r, c, L=L: avg(r.get("niah_long", []), c, "accuracy", length=L)))
    for b in ("16k", "32k"):
        S.append(("长 QA", f"{b} F1", "long", False, "score", lambda R, s, r, c, b=b: avg(r.get("qa_long", []), c, "f1", bucket=b)))
        S.append(("长 QA", f"{b} 包含答案", "long", False, "contains", lambda R, s, r, c, b=b: avg(r.get("qa_long", []), c, "contains", bucket=b)))
    for t, name in (("gov_report", "gov_report"), ("qmsum", "QMSum"), ("lcc", "LCC"), ("repobench-p", "RepoBench-P")):
        S.append(("LongBench", name, "longbench", False, "score", lambda R, s, r, c, t=t: avg(r.get(t, []), c, "score")))
    S.append(("LongBench", "GSM8K 32-shot", "longbench", False, "score", lambda R, s, r, c: avg(r.get("gsm32", []), c, "accuracy")))
    for b in ("4k", "8k"):
        S.append(("同一文档 5 个问题（全部问题）", f"{b} F1", "multiq", True, "score", lambda R, s, r, c, b=b: avg(r.get("qa", []), c, "f1", bucket=b, part="all")))
        S.append(("同一文档 5 个问题（全部问题）", f"{b} 包含答案", "multiq", True, "contains", lambda R, s, r, c, b=b: avg(r.get("qa", []), c, "contains", bucket=b, part="all")))
    S.append(("QuALITY", "单问题 准确率", "quality", False, "score", lambda R, s, r, c: avg(r.get("quality", []), c, "accuracy", bucket="all", part="all")))
    for b in ("4k", "8k"):
        S.append(("QuALITY", f"同一文档多问题 {b}", "quality_mq", True, "score", lambda R, s, r, c, b=b: avg(r.get("qa", []), c, "f1", bucket=b, part="all")))
    for t, name, free in (("lastingbench", "LastingBench", True), ("musique", "MuSiQue", True), ("loogle", "LooGLE", True),
                          ("loogle_long", "LooGLE-long", True), ("qasper", "Qasper", True), ("clipper", "CLIPPER", False),
                          ("longhealth", "LongHealth", False), ("contractnli", "ContractNLI", False)):
        S.append(("长文本基准", f"{name}（{'F1' if free else 'acc'}）", "bench", False, "score",
                  lambda R, s, r, c, t=t: avg(r.get(t, []), c, "score", subset="all") if t in r else None, t))
        if free:
            S.append(("长文本基准", f"{name}（包含答案{'，只算问答题' if t.startswith('loogle') else ''}）", "bench", False, "contains",
                      lambda R, s, r, c, t=t: item_contains(R, s, "bench", t, c) if t in r else None, t))
    return S


def ttft(res, cfg, L):
    rows = (res or {}).get("speed_allgraph", {}).get("rows", [])
    row = next((r for r in rows if r["prompt_len"] == L), None)
    if not row or row.get(cfg) is None:
        return None
    return row[cfg] * 1e3, (row["big"] / row[cfg] if row.get("big") else None)


def table(R, title, cols, deltas, with_ttft, cross_cap):
    head = [c[0] for c in cols] + [d[0] for d in deltas]
    out = [f"## {title}", "", "| 基准 | 设置 | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    for spec in row_specs():
        group, label, step, multi, kind, fn = spec[:6]
        task = spec[6] if len(spec) > 6 else None
        if kind == "ttft" and not with_ttft:
            continue
        vals = []
        for _, src, k in cols:
            res = R.get(src, step)
            cfg = {"big": "big", "small": "small"}.get(k) or (MULTI if multi else SINGLE)[k]
            if res is None:
                vals.append(None)
            elif kind == "ttft":
                vals.append(ttft(res, cfg, fn))
            elif src in cross_cap and step == "bench" and task not in SAME_PROMPTS_16K_64K:
                vals.append(None)  # 32B suite is capped at 16k, 8B / 4B at 64k: different prompts
            elif src in cross_cap and step == "longbench" and label in ("LCC", "RepoBench-P"):
                vals.append(None)
            else:
                vals.append(fn(R, src, res, cfg))
        if all(v is None for (_, src, _), v in zip(cols, vals) if src not in cross_cap):
            continue  # nothing from the section's own target (e.g. 32B at 32k does not fit one H100)
        if kind == "ttft":
            cells = ["-" if v is None else f"{v[0]:.0f}" + (f"（{v[1]:.2f}×）" if v[1] else "") for v in vals]
            dcells = [""] * len(deltas)
        else:
            cells = ["-" if v is None else f"{v:.3f}" for v in vals]
            dcells = ["-" if vals[i] is None or vals[j] is None else f"{vals[i] - vals[j]:+.3f}" for _, i, j in deltas]
        out.append(f"| {group} | {label} | " + " | ".join(cells + dcells) + " |")
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULTS_DIR)
    ap.add_argument("--mid", default="4B", help="results/mid_<mid>_<step>")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = Runs(a.results, a.mid)
    out = [f"# 同等延迟对比：中等模型单独 vs 大模型单独 vs 我们 vs SpecPrefill", "",
           f"生成：{time.strftime('%F %T')}，`scripts/summarize_mid.py`。{a.mid} 单独来自 `slurm/eval_mid.slurm`（`results/mid_{a.mid}_<step>`，"
           "配置 `small`：它自己 prefill 并自己解码）；其余列来自各 pair 的运行（`results/fused_<tag>_<step>`、`e32_<step>`、`quality_<pair>`、"
           "`quality_mq_<pair>`、`bench_<pair>`）。我们 = 重算 10% 中间 token（按句子，小模型注意力打分），其余用投影 KV，大模型解码。"
           "同一文档多问题的 SpecPrefill 是每个问题重新 prefill（spec-each），单问题是官方 SpecPrefill 保留 10%。"
           "首 token 延迟一格 = 毫秒（相对同一次运行里大模型单独的加速比）。“-” = 没有这一项或不可比。", "",
           "**指标**：自由回答的 QA 给两行。“F1”= 输出第一行和标准答案的 token F1（各基准的原始指标）；“包含答案”= 归一化后的标准答案出现在输出里"
           "（`scripts/items.py`）。更大的模型回答更啰嗦，F1 会被压低：32B 在抽取式 QA 4k 上 F1 0.638、8B 0.772，但包含答案都是 0.773。"
           "**跨模型比较看“包含答案”**；同一个大模型解码的方法之间（大模型单独 / 我们 / SpecPrefill）两种指标都可用。"
           "差值的置信区间见 `results/ci_summary.md`（`scripts/paired_ci.py`）。", ""]
    for sec in sections(a.mid):
        out += table(R, *sec)
    out += ["§2 的说明：32B 的长文本基准上限 16k、8B / 4B 是 64k，只有 Qasper、LongHealth、ContractNLI 的 prompt 在两种上限下完全相同，"
            "其余长文本任务不给 8B / 4B 的值；LongBench 的 LCC / RepoBench-P 在原始 prompt 下 32B 会用文字回答，"
            "8B / 4B 的值不可比，也不给。32B 那组的首 token 延迟是在 H100 上测的，8B / 4B 在 H100 上还没测，所以这一节没有延迟行。", ""]
    path = a.out or os.path.join(a.results, "mid_summary.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("[summarize_mid]", path)


if __name__ == "__main__":
    main()
