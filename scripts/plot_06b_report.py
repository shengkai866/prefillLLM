#!/usr/bin/env python
"""
Figure for REPORT_2026-10-21_06B_to_8B.md -> results/report_06b/figures/ruler_subtasks.png
RULER per subtask at 4k and 8k: 8B alone vs 0.6B alone vs hybrid (no recompute) vs hybrid + 10% recompute
(0.6B attention, sentence units). Hybrid rows come from the L40S run results/recompute_blocks (8B re-run there too);
0.6B alone from results/round2 (H100; same examples).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "report_06b", "figures")
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.family": "DejaVu Sans", "legend.frameon": False})

TASKS = [("niah_single_1", "single needle 1"), ("niah_single_2", "single needle 2"), ("niah_single_3", "single needle 3 (uuid)"),
         ("niah_multikey_1", "multi-key 1"), ("niah_multikey_2", "multi-key 2"), ("niah_multikey_3", "multi-key 3 (uuid)"),
         ("niah_multiquery", "multi-query"), ("niah_multivalue", "multi-value"), ("vt", "variable tracking"),
         ("cwe", "common words (cwe)"), ("fwe", "frequent words (fwe)"), ("qa_1", "QA 1 (single doc)"), ("qa_2", "QA 2 (multi-hop)")]
# categorical slots 1-4 of the validated palette, fixed per configuration
SERIES = [("8B alone", "recompute_blocks", "big", "#2a78d6"), ("0.6B alone", "round2", "small", "#eb6834"),
          ("hybrid, no recompute", "recompute_blocks", "hyb", "#1baf7a"),
          ("hybrid + 10% recompute (0.6B attn, sentences)", "recompute_blocks", "attn_small:0.1:sent", "#eda100")]


def scores(run, cfg, L):
    r = json.load(open(os.path.join(ROOT, "results", run, "results.json")))
    return {x["task"]: x["score"] for x in r["ruler"] if x["config"] == cfg and x["length"] == L}


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True)
    n, h = len(SERIES), 0.2
    for ax, L in zip(axes, (4096, 8192)):
        for k, (lab, run, cfg, col) in enumerate(SERIES):
            sc = scores(run, cfg, L)
            ys = [i + (k - (n - 1) / 2) * h for i in range(len(TASKS))]
            ax.barh(ys, [sc[t] for t, _ in TASKS], height=h, color=col, edgecolor=SURFACE, linewidth=1, label=lab, zorder=3)
        ax.set_yticks(range(len(TASKS)))
        ax.set_yticklabels([name for _, name in TASKS])
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("score (15 examples per subtask, about ±0.1)")
        ax.set_title(f"RULER {L // 1024}k", loc="left")
    axes[0].invert_yaxis()  # shared y axis: invert once, so the first subtask and the 8B bar are on top
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("RULER per subtask: Qwen3-0.6B prefill -> Qwen3-8B vs the two models alone", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ruler_subtasks.png"), dpi=150, bbox_inches="tight")
    print("[plot] wrote", OUT)


if __name__ == "__main__":
    main()
