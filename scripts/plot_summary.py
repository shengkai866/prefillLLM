#!/usr/bin/env python
"""
Cross-experiment figures for the 2026-10-15 summary report -> results/summary_2026-10-15/figures/
  training_click.png : stage-2 validation KL (retrieval, QA, mean) vs step for the three model pairs
  pareto_8k.png      : RULER 8k mean vs TTFT speedup at 8k (H100, eager) for every measured configuration
Training curves are parsed from the job logs; quality/speed numbers come from the results/*/results.json files.
"""
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "summary_2026-10-15", "figures")
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
PAIR = {"0.6B→8B": "#2a78d6", "1.7B→8B": "#eb6834", "1.7B→32B": "#1baf7a"}  # categorical slots 1-3
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "font.family": "DejaVu Sans",
    "legend.frameon": False})


def stage2_curve(log, start_marker):
    """[(step, {type: kl})] from 'step=N val_kl_web=...' lines after the stage-2 section header."""
    lines = open(os.path.join(ROOT, "logs", log)).read().split("\n")
    i = next(k for k, l in enumerate(lines) if start_marker in l)
    pts = []
    for l in lines[i:]:
        if l.startswith("step=") and "val_kl_web=" in l:
            kv = dict(x.split("=", 1) for x in l.split())
            pts.append((int(kv["step"]), {k[7:]: float(v) for k, v in kv.items() if k.startswith("val_kl")}))
        if l.startswith("[done] stage2"):
            break
    return pts


def ruler_mean(path, cfg, L):
    r = json.load(open(os.path.join(ROOT, "results", path, "results.json")))
    v = [x["score"] for x in r["ruler"] if x["config"] == cfg and x["length"] == L and x["task"] != "_frac"]
    return statistics.mean(v) if v else None


def speed(path, cfg, L):
    r = json.load(open(os.path.join(ROOT, "results", path, "results.json")))
    row = next(x for x in r["speed"] if x["prompt_len"] == L)
    return row["big"] / row[cfg] if row.get(cfg) else None


def fig_training():
    curves = {"0.6B→8B": stage2_curve("pf_round2-27891410.out", "train: mixed data, 8k"),
              "1.7B→8B": stage2_curve("pf_r3_s17-27903225.out", "stage2: mixed data"),
              "1.7B→32B": stage2_curve("pf_r3_32b-27903420.out", "stage2: mixed data")}
    panels = [("retrieval", "Synthetic retrieval"), ("qa", "Document QA"), ("", "Mean over 6 data types")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (key, title) in zip(axes, panels):
        for name, pts in curves.items():
            ax.plot([s for s, _ in pts], [d[key] for _, d in pts], color=PAIR[name], lw=2, label=name, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("stage-2 training step")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("validation KL to the large model")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Stage-2 validation KL (fixed held-out set, 2 sequences per type, lower is better)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_click.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# label offsets (points) where labels would collide
OFFSET = {("0.6B→8B", "8B attn 5%"): (6, -12, "left"), ("0.6B→8B", "0.6B attn 5% sent"): (6, 6, "left"),
          ("1.7B→8B", "1.7B attn 5% sent"): (-7, 4, "right")}


def fig_pareto():
    # (pair, label, speedup at 8k on H100, RULER 8k). 0.6B->8B quality: H100 run (results/recompute) where it exists,
    # sentence-unit configs from the L40S run (results/recompute_blocks); speed always from H100 runs.
    pts = [
        ("0.6B→8B", "no recompute", speed("recompute_speed", "hyb", 8192), ruler_mean("recompute", "hyb", 8192)),
        ("0.6B→8B", "0.6B attn 5% sent", speed("recompute_speed", "attn_small:0.05:sent", 8192), ruler_mean("recompute_blocks", "attn_small:0.05:sent", 8192)),
        ("0.6B→8B", "8B attn 5%", speed("recompute_speed", "attn_big:0.05", 8192), ruler_mean("recompute", "attn_big:0.05", 8192)),
        ("1.7B→8B", "no recompute", speed("r3_small17", "hyb", 8192), ruler_mean("r3_small17", "hyb", 8192)),
        ("1.7B→8B", "1.7B attn 5% sent", speed("r3_small17", "attn_small:0.05:sent", 8192), ruler_mean("r3_small17", "attn_small:0.05:sent", 8192)),
        ("1.7B→8B", "1.7B attn 10% sent", speed("r3_small17", "attn_small:0.1:sent", 8192), ruler_mean("r3_small17", "attn_small:0.1:sent", 8192)),
        ("1.7B→8B", "8B attn 5%", speed("r3_small17", "attn_big:0.05", 8192), ruler_mean("r3_small17", "attn_big:0.05", 8192)),
        ("1.7B→32B", "no recompute", speed("r3_s17_to_32b", "hyb", 8192), ruler_mean("r3_s17_to_32b", "hyb", 8192)),
        ("1.7B→32B", "1.7B attn 5% sent", speed("r3_s17_to_32b", "attn_small:0.05:sent", 8192), ruler_mean("r3_s17_to_32b", "attn_small:0.05:sent", 8192)),
        ("1.7B→32B", "1.7B attn 10% sent", speed("r3_s17_to_32b", "attn_small:0.1:sent", 8192), ruler_mean("r3_s17_to_32b", "attn_small:0.1:sent", 8192)),
        ("1.7B→32B", "32B attn 5%", speed("r3_s17_to_32b", "attn_big:0.05", 8192), ruler_mean("r3_s17_to_32b", "attn_big:0.05", 8192)),
    ]
    refs = [("8B alone", 1.0, ruler_mean("r3_small17", "big", 8192), "8B"),
            ("1.7B alone (vs 8B)", speed("r3_small17", "small", 8192), ruler_mean("r3_small17", "small", 8192), "8B"),
            ("32B alone", 1.0, ruler_mean("r3_s17_to_32b", "big", 8192), "32B"),
            ("1.7B alone (vs 32B)", speed("r3_s17_to_32b", "small", 8192), ruler_mean("r3_s17_to_32b", "small", 8192), "32B")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, target in zip(axes, ("8B", "32B")):
        for pair in PAIR:
            if not pair.endswith(target):
                continue
            P = [p for p in pts if p[0] == pair]
            ax.scatter([p[2] for p in P], [p[3] for p in P], s=70, color=PAIR[pair], edgecolor=SURFACE, linewidth=1.5,
                       label=pair, zorder=4)
            for _, lab, x, y in P:
                dx, dy, ha = OFFSET.get((pair, lab), (6, 4, "left"))
                ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=TEXT2, ha=ha)
        for lab, x, y, t in refs:
            if t != target:
                continue
            ax.scatter([x], [y], s=70, marker="D", color=TEXT2, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            ax.annotate(lab, (x, y), xytext=(6, -11), textcoords="offset points", fontsize=7.5, color=TEXT2)
        ax.set_xscale("log")
        ax.set_xticks([1, 1.5, 2, 3, 5, 8, 12])
        ax.set_xticklabels(["1x", "1.5x", "2x", "3x", "5x", "8x", "12x"])
        ax.set_xlim(0.85, 16)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Target: Qwen3-{target}", loc="left")
        ax.set_xlabel("TTFT speedup over the large model alone at 8k (H100, eager, log scale)")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("RULER 8k mean (13 tasks)")
    fig.suptitle("Quality vs. speed at 8k prompts: every measured configuration (up and right is better)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pareto_8k.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_training()
    fig_pareto()
    print("[plot] wrote", sorted(os.listdir(OUT)))
