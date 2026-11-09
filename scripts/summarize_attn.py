#!/usr/bin/env python
"""
First-token latency with the position-causal attention kernel (prefill/attn_fused.py), from slurm/speed_attn.slurm:
  results/attn_<tag>_speed/results.json        PREFILL_ATTN=triton: speed_allgraph (large / small / SpecPrefill / projected
                                               cache / ours), breakdown (per stage), attn_check (sdpa vs triton logits)
  results/attn_<tag>_speed_sdpa/results.json   PREFILL_ATTN=sdpa: projected cache / ours, large alone, breakdown
  results/fused_<tag>_speed_eagerfold          (earlier run, main) ours with the original fold and SDPA, for reference
  results/attn_06_qa vs <main>/results/fused_06_main   QA F1 of ours 10 % with the kernel vs masked SDPA
-> results/attn_speed_summary.md. Missing files / configs are shown as "-".
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

MAIN_RESULTS = "/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/results"
PAIRS = [("06", "Qwen3-0.6B → Qwen3-8B"), ("17", "Qwen3-1.7B → Qwen3-8B"), ("ll1to8", "Llama-3.2-1B → Llama-3.1-8B"),
         ("q17toll8", "Qwen3-1.7B → Llama-3.1-8B（跨家族）"), ("ll1toq8", "Llama-3.2-1B → Qwen3-8B（跨家族）")]
ROWS = [  # (label, config, source)  t = attention-kernel run, s = sdpa run (same job), e = earlier eager-fold run
    ("大模型单独", "big", "t"), ("小模型单独", "small", "t"),
    ("SpecPrefill 10%", "specprefill:0.1", "t"), ("SpecPrefill 20%", "specprefill:0.2", "t"), ("SpecPrefill 30%", "specprefill:0.3", "t"),
    ("投影缓存，不重算：融合 fold + SDPA", "hyb", "s"), ("投影缓存，不重算：融合 fold + attention kernel", "hyb", "t"),
    ("我们（重算 10%）：原始 fold + SDPA（早先一次运行）", "attn_small:0.1:sent", "e"),
    ("我们（重算 10%）：融合 fold + SDPA", "attn_small:0.1:sent", "s"),
    ("**我们（重算 10%）：融合 fold + attention kernel**", "attn_small:0.1:sent", "t"),
    ("我们（重算 20%）：融合 fold + SDPA", "attn_small:0.2:sent", "s"),
    ("**我们（重算 20%）：融合 fold + attention kernel**", "attn_small:0.2:sent", "t"),
]
STAGES = [("small_scored", "小模型 prefill + 尾部打分"), ("fold", "projector → KV（融合）"), ("tail", "大模型 开头 + 尾部（不重算时）"),
          ("recompute", "大模型 重算调用（10% 句子 + 开头 + 尾部）"), ("big", "大模型单独")]


def load(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {}


def by_len(res, key):
    try:
        rows = res[key]["rows"] if key == "speed_allgraph" else res[key]
        return {r["prompt_len"]: r for r in rows}
    except (KeyError, TypeError):
        return {}


def main():
    out = ["# 首 token 延迟：attention kernel（位置因果，`prefill/attn_fused.py`）", "",
           f"生成：{time.strftime('%F %T')}。毫秒，括号内是相对同一次运行的大模型单独的加速比；所有方法 CUDA graph，batch 1，单张 L40S。"
           "“SDPA”= 原实现（稠密 mask + `scaled_dot_product_attention`，`PREFILL_ATTN=sdpa`），“attention kernel”= 新 kernel（`PREFILL_ATTN=triton`）；"
           "两者只在重算调用和开头+尾部调用的 attention 上不同，大模型单独 / 小模型 / SpecPrefill 不经过它们。"
           "数据：`results/attn_<tag>_speed`、`results/attn_<tag>_speed_sdpa`（`slurm/speed_attn.slurm`），“原始 fold”一行来自 main 的 "
           "`results/fused_<tag>_speed_eagerfold`（另一次运行）。", ""]
    for tag, name in PAIRS:
        rt = load(os.path.join(C.RESULTS_DIR, f"attn_{tag}_speed", "results.json"))
        rs = load(os.path.join(C.RESULTS_DIR, f"attn_{tag}_speed_sdpa", "results.json"))
        re_ = load(os.path.join(MAIN_RESULTS, f"fused_{tag}_speed_eagerfold", "results.json"))
        src = {"t": by_len(rt, "speed_allgraph"), "s": by_len(rs, "speed_allgraph"), "e": by_len(re_, "speed_allgraph")}
        if not src["t"] and not src["s"]:
            continue
        gpu = (rt or rs).get("meta", {}).get("gpu", "?")
        lens = sorted(set(src["t"]) | set(src["s"]))
        out += [f"## {name}（`{tag}`，{gpu}）", "", "| 配置 | " + " | ".join(f"{L // 1024}k" for L in lens) + " |", "|---|" + "---|" * len(lens)]
        for label, cfg, s in ROWS:
            cells = []
            for L in lens:
                row = src[s].get(L, {})
                v = row.get(cfg)
                cells.append("-" if v is None else f"{v * 1e3:.0f}（{row['big'] / v:.2f}）" if row.get("big") else f"{v * 1e3:.0f}")
            if all(c == "-" for c in cells):
                continue
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        bt, bs = by_len(rt, "breakdown"), by_len(rs, "breakdown")
        if bt or bs:
            out += ["", "分阶段（eager 单独计时，毫秒；SDPA → attention kernel）：", "",
                    "| 阶段 | " + " | ".join(f"{L // 1024}k" for L in lens) + " |", "|---|" + "---|" * len(lens)]
            for k, label in STAGES:
                cells = []
                for L in lens:
                    a, b = bs.get(L, {}).get(k), bt.get(L, {}).get(k)
                    if k in ("tail", "recompute"):
                        cells.append(f"{a * 1e3:.0f} → **{b * 1e3:.0f}**" if a and b else "-" if not (a or b) else f"{(a or b) * 1e3:.0f}")
                    else:
                        v = b or a
                        cells.append("-" if v is None else f"{v * 1e3:.0f}")
                out.append(f"| {label} | " + " | ".join(cells) + " |")
            nq = [str(bt.get(L, bs.get(L, {})).get("n_recomputed", "-")) for L in lens]
            out.append("| 重算的中间 token 数 | " + " | ".join(nq) + " |")
        ck = rt.get("attn_check")
        if ck:
            out += ["", "真实 prompt 上 SDPA 与 attention kernel 的首 token logits 对比（同一 prompt、同一配置）：", "",
                    "| 数据 | 长度 | 配置 | n | KL 均值 | KL 最大 | 最大相对差 | top-1 一致 |", "|---|---|---|---|---|---|---|---|"]
            for r in ck:
                out.append(f"| {r['pool']} | {r['bucket']} | {r['config']} | {r['n']} | {r['kl_mean']:.1e} | {r['kl_max']:.1e} | "
                           f"{r['rel_max']:.1e} | {r['top1']:.2f} |")
        out.append("")
    qa_t = load(os.path.join(C.RESULTS_DIR, "attn_06_qa", "results.json")).get("qa", [])
    qa_s = load(os.path.join(MAIN_RESULTS, "fused_06_main", "results.json")).get("qa", [])
    if qa_t:
        ref = {(r["config"], r["source"], r["bucket"]): r for r in qa_s}
        out += ["## QA 质量对照（Qwen3-0.6B → Qwen3-8B，我们 重算 10%）", "",
                "F1：SDPA 来自 main 的 `results/fused_06_main`，attention kernel 来自 `results/attn_06_qa`。", "",
                "| 数据 | 长度 | SDPA | attention kernel |", "|---|---|---|---|"]
        for r in qa_t:
            s = ref.get((r["config"], r["source"], r["bucket"]))
            out.append(f"| {r['source']} | {r['bucket']} | {s['f1']:.3f} | {r['f1']:.3f} |" if s else
                       f"| {r['source']} | {r['bucket']} | - | {r['f1']:.3f} |")
        out.append("")
    path = os.path.join(C.RESULTS_DIR, "attn_speed_summary.md")
    open(path, "w").write("\n".join(out))
    print("[summarize_attn]", path)


if __name__ == "__main__":
    main()
