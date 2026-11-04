#!/usr/bin/env python
"""
First-token latency of every method, per pair, from the fused re-run (slurm/rerun_fused_all.slurm):
  results/fused_<tag>_speed/results.json            PREFILL_FOLD=fused: large alone, small alone, projected cache (hyb),
                                                    ours (recompute 10 / 20 %), drop-rest, SpecPrefill 10 / 20 / 30 %
  results/fused_<tag>_speed_eagerfold/results.json  PREFILL_FOLD=eager: large alone, hyb, ours -> the "without the fused
                                                    kernel" rows (only the projector -> KV stage differs)
-> results/speed_summary.md. Missing files / configs are shown as "-".
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

PAIRS = [("06", "Qwen3-0.6B → Qwen3-8B"), ("17", "Qwen3-1.7B → Qwen3-8B"), ("ll1to8", "Llama-3.2-1B → Llama-3.1-8B"),
         ("q17toll8", "Qwen3-1.7B → Llama-3.1-8B（跨家族）"), ("ll1toq8", "Llama-3.2-1B → Qwen3-8B（跨家族）")]
ROWS = [  # (label, config, source)  source: f = fused run, e = eager-fold run
    ("大模型单独", "big", "f"), ("小模型单独", "small", "f"),
    ("投影缓存，不重算，原始 fold", "hyb", "e"), ("投影缓存，不重算，融合 kernel", "hyb", "f"),
    ("我们（重算 10%），原始 fold", "attn_small:0.1:sent", "e"), ("我们（重算 10%），融合 kernel", "attn_small:0.1:sent", "f"),
    ("我们（重算 20%），原始 fold", "attn_small:0.2:sent", "e"), ("我们（重算 20%），融合 kernel", "attn_small:0.2:sent", "f"),
    ("丢弃其余 20%（不投影）", "drop-attn_small:0.2:sent", "f"),
    ("SpecPrefill 10%", "specprefill:0.1", "f"), ("SpecPrefill 20%", "specprefill:0.2", "f"), ("SpecPrefill 30%", "specprefill:0.3", "f"),
]


def load(path):
    try:
        r = json.load(open(path))
        return r["meta"].get("gpu", "?"), {row["prompt_len"]: row for row in r["speed_allgraph"]["rows"]}
    except (OSError, KeyError, json.JSONDecodeError):
        return None, {}


def main():
    out = ["# 首 token 延迟汇总（所有方法 CUDA graph，batch 1）", "",
           f"生成：{time.strftime('%F %T')}。毫秒，括号内是相对同一次运行的大模型单独的加速比。“原始 fold”= `FoldedProjectorKV`（`PREFILL_FOLD=eager`），"
           "“融合 kernel”= `FusedFoldKV`（`prefill/fold_fused.py`，默认）；两者只在 projector → KV 阶段不同，大模型 / 小模型 / SpecPrefill 不经过这一阶段。"
           "数据：`results/fused_<tag>_speed`（融合）和 `results/fused_<tag>_speed_eagerfold`（原始），由 `slurm/rerun_fused_all.slurm` 产生。", ""]
    for tag, name in PAIRS:
        gf, F = load(os.path.join(C.RESULTS_DIR, f"fused_{tag}_speed", "results.json"))
        ge, E = load(os.path.join(C.RESULTS_DIR, f"fused_{tag}_speed_eagerfold", "results.json"))
        if not F and not E:
            continue
        lens = sorted(set(F) | set(E))
        out += [f"## {name}（`{tag}`，{gf or ge}）", "", "| 配置 | " + " | ".join(f"{L // 1024}k" for L in lens) + " |", "|---|" + "---|" * len(lens)]
        for label, cfg, src in ROWS:
            rows = F if src == "f" else E
            cells = []
            for L in lens:
                row = rows.get(L, {})
                v = row.get(cfg)
                cells.append("-" if v is None else f"{v * 1e3:.0f}（{row['big'] / v:.2f}）" if row.get("big") else f"{v * 1e3:.0f}")
            if all(c == "-" for c in cells):
                continue
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    path = os.path.join(C.RESULTS_DIR, "speed_summary.md")
    open(path, "w").write("\n".join(out))
    print("[summarize_speed]", path)


if __name__ == "__main__":
    main()
