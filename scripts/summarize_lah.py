#!/usr/bin/env python
"""
Look-ahead length of the SpecPrefill scorer inside our pipeline (slurm/scorer_lah.slurm):
  results/scorer_lah_<tag>        spec:0.1:lah2 / lah3 / lah4 (RULER 4k / 8k, QA 4k / 8k, QA 16k / 32k)
  results/scorer_<tag>            baselines from job 28091569: attn_small:0.1:sent (ours today), spec:0.1 (look-ahead 8)
  results/scorer_lah_<tag>_speed  TTFT (CUDA graph) of all of them + large alone + SpecPrefill 10 %
-> results/scorer_lah_summary.md. Missing files / configs are shown as "-".
"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

PAIRS = [("06", "Qwen3-0.6B → Qwen3-8B"), ("17", "Qwen3-1.7B → Qwen3-8B")]
CFGS = [("attn_small:0.1:sent", "现在的打分（不生成）"), ("spec:0.1:lah2", "look-ahead 2（生成 1 个 token）"),
        ("spec:0.1:lah3", "look-ahead 3（生成 2 个）"), ("spec:0.1:lah4", "look-ahead 4（生成 3 个）"),
        ("spec:0.1", "look-ahead 8（生成 7 个，SpecPrefill 默认）")]
SPEED = [("big", "大模型单独")] + [(c, f"我们 10%，{l}") for c, l in CFGS] + [("specprefill:0.1", "SpecPrefill 10%")]
COLS = [("ruler", 4096, "RULER 4k"), ("ruler", 8192, "RULER 8k"), ("qa", "4k", "问答 4k"), ("qa", "8k", "问答 8k"),
        ("qa_long", "16k", "问答 16k"), ("qa_long", "32k", "问答 32k")]


def load(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {}


def mean(x):
    return sum(x) / len(x) if x else None


def collect(res, agg):
    for row in res.get("ruler", []):
        agg[(row["config"], "ruler", row["length"])].append(row["score"])
    for t in ("qa", "qa_long"):
        for row in res.get(t, []):
            agg[(row["config"], t, row["bucket"])].append(row["f1"])


def main():
    out = ["# 我们的流程 + SpecPrefill 打分：look-ahead 步数", "",
           f"生成：{time.strftime('%F %T')}。look-ahead N = N 个 query：prompt 最后一个 token，加上小模型往后生成的 N-1 个 token；"
           "按 32 token 一块取平均分挑中间部分的 10%，其余用 projector 补 KV。RULER 为各子任务平均分，QA 为各数据集 F1 平均。"
           "“现在的打分”和 look-ahead 8 来自同一代码、同一批题目的上一个任务（`results/scorer_<tag>`），其余来自 `results/scorer_lah_<tag>`。", ""]
    for tag, name in PAIRS:
        agg = collections.defaultdict(list)
        collect(load(os.path.join(C.RESULTS_DIR, f"scorer_{tag}", "results.json")), agg)
        collect(load(os.path.join(C.RESULTS_DIR, f"scorer_lah_{tag}", "results.json")), agg)
        sp = load(os.path.join(C.RESULTS_DIR, f"scorer_lah_{tag}_speed", "results.json"))
        if not agg and not sp:
            continue
        out += [f"## {name}（`{tag}`）", "", "| 打分 | " + " | ".join(c[2] for c in COLS) + " | 平均 |", "|---|" + "---|" * (len(COLS) + 1)]
        for cfg, label in CFGS:
            vals = [mean(agg.get((cfg, t, k), [])) for t, k, _ in COLS]
            if all(v is None for v in vals):
                continue
            ok = [v for v in vals if v is not None]
            out.append(f"| {label} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + f" | {mean(ok):.3f} |")
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
            chk = sp.get("speed_allgraph", {}).get("check", {})
            if chk:
                out += ["", "graph 与 eager 首 token logits 的最大相对差：" + "，".join(f"`{c}` {v:.1e}" for c, v in chk.items())]
        out.append("")
    path = os.path.join(C.RESULTS_DIR, "scorer_lah_summary.md")
    open(path, "w").write("\n".join(out))
    print("[summarize_lah]", path)


if __name__ == "__main__":
    main()
