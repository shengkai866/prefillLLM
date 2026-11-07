#!/usr/bin/env python
"""
Token-selection scorers compared inside our pipeline (slurm/scorer_max.slurm):
  results/scorer_<tag>/results.json        RULER 4k / 8k, QA 4k / 8k, QA 16k / 32k per config
  results/scorer_<tag>_speed/results.json  TTFT (CUDA graph) of big / attn_small / attn_max / SpecPrefill 10 %
-> results/scorer_max_summary.md. Missing files / configs are shown as "-".
"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

PAIRS = [("06", "Qwen3-0.6B → Qwen3-8B"), ("17", "Qwen3-1.7B → Qwen3-8B")]
CFGS = [("attn_small:0.1:sent", "现在的打分：平均，按句"), ("attn_max:0.1:sent", "SpecPrefill 式汇总，按句"),
        ("attn_max:0.1", "SpecPrefill 式汇总，32 token 块"), ("spec:0.1", "SpecPrefill 完整打分（含 look-ahead），32 token 块"),
        ("attn_small:0.05:sent", "现在的打分，5%"), ("attn_max:0.05:sent", "SpecPrefill 式汇总，按句，5%")]
SPEED = [("big", "大模型单独"), ("attn_small:0.1:sent", "我们 10%，现在的打分"), ("attn_max:0.1:sent", "我们 10%，SpecPrefill 式汇总"),
         ("specprefill:0.1", "SpecPrefill 10%")]


def load(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {}


def mean(x):
    return sum(x) / len(x) if x else None


def cell(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


def main():
    out = ["# 挑 token 的打分方式：SpecPrefill 式汇总（不做 look-ahead）", "",
           f"生成：{time.strftime('%F %T')}。所有配置都用我们的流程：没挑中的中间 token 用 projector 补 KV，开头 8 个、结尾 64 个由大模型算，"
           "只是挑哪些中间 token 交给大模型重算不同。“SpecPrefill 式汇总”= 末尾 64 个 token 的 attention 沿 prompt 宽度 13 平滑，"
           "对 head 和层取最大，对 query 取平均（`attn_max`）；“现在的打分”= 对 head、query、层取平均（`attn_small`）。"
           "RULER 为各子任务平均分，QA 为各数据集 F1 平均。数据：`results/scorer_<tag>`（`slurm/scorer_max.slurm`）。", ""]
    for tag, name in PAIRS:
        r = load(os.path.join(C.RESULTS_DIR, f"scorer_{tag}", "results.json"))
        sp = load(os.path.join(C.RESULTS_DIR, f"scorer_{tag}_speed", "results.json"))
        if not r and not sp:
            continue
        cols = [("ruler", 4096, "RULER 4k"), ("ruler", 8192, "RULER 8k"), ("qa", "4k", "问答 4k"), ("qa", "8k", "问答 8k"),
                ("qa_long", "16k", "问答 16k"), ("qa_long", "32k", "问答 32k")]
        agg = collections.defaultdict(list)
        for row in r.get("ruler", []):
            agg[(row["config"], "ruler", row["length"])].append(row["score"])
        for t in ("qa", "qa_long"):
            for row in r.get(t, []):
                agg[(row["config"], t, row["bucket"])].append(row["f1"])
        out += [f"## {name}（`{tag}`）", "", "| 配置 | " + " | ".join(c[2] for c in cols) + " | 平均 |", "|---|" + "---|" * (len(cols) + 1)]
        for cfg, label in CFGS:
            vals = [mean(agg.get((cfg, t, k), [])) for t, k, _ in cols]
            if all(v is None for v in vals):
                continue
            ok = [v for v in vals if v is not None]
            out.append(f"| {label} | " + " | ".join(cell(v) for v in vals) + f" | {cell(mean(ok))} |")
        rows = {x["prompt_len"]: x for x in sp.get("speed_allgraph", {}).get("rows", [])}
        if rows:
            lens = sorted(rows)
            out += ["", "首 token 延迟（毫秒，CUDA graph，括号内为相对大模型单独的加速比）：", "",
                    "| 配置 | " + " | ".join(f"{L // 1024}k" for L in lens) + " |", "|---|" + "---|" * len(lens)]
            for cfg, label in SPEED:
                cs = []
                for L in lens:
                    v, b = rows[L].get(cfg), rows[L].get("big")
                    cs.append("-" if v is None else f"{v * 1e3:.0f}（{b / v:.2f}）" if b else f"{v * 1e3:.0f}")
                out.append(f"| {label} | " + " | ".join(cs) + " |")
        out.append("")
    path = os.path.join(C.RESULTS_DIR, "scorer_max_summary.md")
    open(path, "w").write("\n".join(out))
    print("[summarize_scorer]", path)


if __name__ == "__main__":
    main()
