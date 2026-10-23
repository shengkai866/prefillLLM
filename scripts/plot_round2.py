#!/usr/bin/env python
"""
Figures for the round-2 experiment (static PNG, embedded in results/round2/summary.md).

Colour = configuration, fixed per entity across every figure (validated categorical palette, light surface):
  8B blue | 0.6B orange | round-1 hybrid aqua | control yellow | mix-4k magenta | mix-8k green
Every chart keeps one y-scale; quantities with different units go to separate panels.
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from prefill import config as C  # noqa: E402

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
COLOR = {"big": "#2a78d6", "small": "#eb6834", "r1": "#1baf7a", "ctrl4k": "#eda100", "mix4k": "#e87ba4", "mix8k": "#008300"}
LABEL = {"big": "8B alone", "small": "0.6B alone", "r1": "hybrid: round 1", "ctrl4k": "hybrid: control (web, 4k)",
         "mix4k": "hybrid: mixed data (4k)", "mix8k": "hybrid: mixed data (8k)"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue 100..700

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.family": "DejaVu Sans", "legend.frameon": False,
})


def mean(x):
    return statistics.mean(x) if x else float("nan")


def grouped_bars(ax, groups, cfgs, val, ylabel, ylim=None, fmt="{:.2f}", labels=True):
    """val(cfg, group) -> number or None. Thin bars, surface-coloured gaps, value labels in text ink."""
    n = len(cfgs)
    width = 0.8 / max(n, 1)
    rotate = n * len(groups) > 12
    top = 0.0
    for k, c in enumerate(cfgs):
        xs, ys = [], []
        for g_i, g in enumerate(groups):
            v = val(c, g)
            if v is None or v != v:
                continue
            xs.append(g_i - 0.4 + width * (k + 0.5))
            ys.append(v)
        bars = ax.bar(xs, ys, width=width, color=COLOR[c], edgecolor=SURFACE, linewidth=1.2, label=LABEL[c], zorder=3)
        top = max([top] + ys)
        if labels:
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y + (top * 0.01 if rotate else 0), fmt.format(y), ha="center", va="bottom",
                        fontsize=6.3, color=TEXT2, rotation=90 if rotate else 0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    head = 1.28 if rotate else 1.1  # room for the value labels above the tallest bar
    ax.set_ylim(ylim[0] if ylim else 0, max(ylim[1] if ylim else 0, top * head) or 1)
    ax.tick_params(axis="x", length=0)


def legend_below(fig, cfgs, ncol=3):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in cfgs]
    fig.legend(handles, [LABEL[c] for c in cfgs], loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.01))


def fig_kl(res, cfgs, out):
    hy = [c for c in cfgs if c not in ("big", "small")]
    types = ["web", "qa", "code", "chat", "math", "retrieval"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, b in zip(axes, ("4k", "8k")):
        d = {(r["config"], r["type"]): r["kl"] for r in res["kl"] if r["bucket"] == b}
        grouped_bars(ax, types, hy, lambda c, g: d.get((c, g)), "KL to 8B (lower is better)", fmt="{:.3f}")
        ax.set_title(f"{b} prompts")
    small = mean([r["kl"] for r in res["kl"] if r["config"] == "small"])
    fig.suptitle(f"Divergence from the 8B on held-out data of each type  (0.6B alone, not drawn: mean KL {small:.2f})", fontsize=11.5, color=TEXT)
    legend_below(fig, hy, ncol=len(hy))
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_tasks(res, cfgs, out):
    panels = []
    if "qa" in res:
        d = {}
        for r in res["qa"]:
            d.setdefault((r["config"], r["bucket"]), []).append(r["f1"])
        panels.append(("Document QA (F1, mean of 3 sets)", ["4k", "8k"], lambda c, g, d=d: mean(d.get((c, g), []))))
    if "ruler" in res:
        d = {}
        for r in res["ruler"]:
            d.setdefault((r["config"], str(r["length"])), []).append(r["score"])
        panels.append(("RULER (mean of 13 tasks)", ["4096", "8192"], lambda c, g, d=d: mean(d.get((c, g), []))))
    if "niah" in res:
        d = {(r["config"], str(r["length"])): r["accuracy"] for r in res["niah"]}
        panels.append(("Needle in a haystack (round-1 test)", sorted({str(r["length"]) for r in res["niah"]}, key=int), lambda c, g, d=d: d.get((c, g))))
    if "code" in res:
        d = {(r["config"], r["bucket"]): r["exact_match"] for r in res["code"]}
        panels.append(("Code next-line exact match", ["4k", "8k"], lambda c, g, d=d: d.get((c, g))))
    if "gsm8k" in res:
        d = {r["config"]: r for r in res["gsm8k"]}
        panels.append(("GSM8K 8-shot", ["accuracy", "uses #### format"],
                       lambda c, g, d=d: None if c not in d else d[c]["accuracy" if g == "accuracy" else "format_follow"]))
    if not panels:
        return False
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.3 * nrow), squeeze=False)
    for ax, (title, groups, fn) in zip(axes.flat, panels):
        grouped_bars(ax, groups, cfgs, fn, "score", ylim=(0, 1.0))
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Task accuracy on held-out data (higher is better)", fontsize=12, color=TEXT)
    legend_below(fig, cfgs, ncol=len(cfgs))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_ruler(res, cfgs, out):
    tasks = sorted({r["task"] for r in res["ruler"]})
    lens = sorted({r["length"] for r in res["ruler"]})
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, axes = plt.subplots(1, len(lens), figsize=(7.2 * len(lens), 0.45 * len(cfgs) + 2.6), squeeze=False)
    for ax, L in zip(axes.flat, lens):
        d = {(r["config"], r["task"]): r["score"] for r in res["ruler"] if r["length"] == L}
        M = [[d.get((c, t), float("nan")) for t in tasks] for c in cfgs]
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(len(cfgs)):
            for j in range(len(tasks)):
                v = M[i][j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.55 else TEXT)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels([LABEL[c] for c in cfgs])
        ax.set_title(f"RULER {L}: score per task (0 to 1, darker is better)")
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_curve(res, out):
    rows = res["curve"]
    runs = [r for r in ("mix4k", "ctrl4k") if any(x["run"] == r for x in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, key, title in ((axes[0], "retrieval_acc", "Retrieval accuracy (higher is better)"), (axes[1], "kl_4k", "KL to 8B, 4k held-out (lower is better)")):
        for r in runs:
            pts = sorted((x["step"], x[key]) for x in rows if x["run"] == r)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[r], linewidth=2, marker="o", markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[r], zorder=3)
            ax.annotate(f"{pts[-1][1]:.2f}" if key == "retrieval_acc" else f"{pts[-1][1]:.3f}", pts[-1], textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=8, color=TEXT2)
        ax.set_title(title)
        ax.set_xlabel("stage-2 training step")
        ax.grid(axis="both")
    axes[0].set_ylim(0, 1.05)
    fig.suptitle("Is it the amount of training or the data? Same steps, different data", fontsize=11.5, color=TEXT)
    legend_below(fig, runs, ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_train(out):
    logs = {}
    for r in ("ctrl4k", "mix4k", "mix8k"):
        p = os.path.join(C.CKPT_DIR, f"r2_{r}", "train_log.jsonl")
        if os.path.exists(p):
            logs[r] = [json.loads(l) for l in open(p) if '"val_kl"' in l]
    logs = {k: v for k, v in logs.items() if v}
    if not logs:
        return False
    keys = sorted({k for v in logs.values() for row in v for k in row if k.startswith("val_kl_")})
    keys = ["val_kl"] + keys
    ncol = 4
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.1 * nrow), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        for r, rows in logs.items():
            pts = [(x["step"], x[k]) for x in rows if k in x]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[r], linewidth=2, label=LABEL[r])
        ax.set_title("all types" if k == "val_kl" else k.replace("val_kl_", ""))
        ax.grid(axis="both")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle("Validation KL to 8B during stage-2 training (fixed held-out set, lower is better)", fontsize=11.5, color=TEXT)
    legend_below(fig, list(logs), ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_speed(out):
    p = os.path.join(C.RESULTS_DIR, "speed_v2", "results.json")
    if not os.path.exists(p):
        return False
    rows = [r for r in json.load(open(p))["speed"] if r.get("big_graph") and r.get("hybrid_graph")]
    L = [r["prompt_len"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for c, key in (("big", "big_graph"), ("mix4k", "hybrid_graph")):
        ys = [r[key] * 1000 for r in rows]
        ax.plot(L, ys, color=COLOR[c], linewidth=2, marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label="8B alone" if c == "big" else "hybrid (0.6B prefill)")
        ax.annotate(f"{ys[-1]:.0f} ms", (L[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color=TEXT2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(L)
    ax.set_xticklabels([f"{x // 1024}k" if x >= 1024 else str(x) for x in L])
    ax.set_xlabel("prompt length (tokens)")
    ax.set_title("Time to first token, ms (log scale)")
    ax.legend(loc="upper left")
    ax.grid(axis="both")
    ax = axes[1]
    sp = [r["big_graph"] / r["hybrid_graph"] for r in rows]
    ax.plot(L, sp, color=COLOR["mix4k"], linewidth=2, marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    for x, y in zip(L, sp):
        ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=TEXT2)
    ax.axhline(1, color=TEXT2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xscale("log", base=2)
    ax.set_xticks(L)
    ax.set_xticklabels([f"{x // 1024}k" if x >= 1024 else str(x) for x in L])
    ax.set_xlabel("prompt length (tokens)")
    ax.set_title("Speed-up of hybrid over 8B (TTFT ratio)")
    ax.set_ylim(0, max(sp) * 1.25)
    ax.grid(axis="both")
    fig.suptitle("Prefill speed on one H100 (lean code + CUDA graphs for both)", fontsize=11.5, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main():
    from evaluate_v2 import write_summary
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.RESULTS_DIR, "round2")
    res_path = os.path.join(out_dir, "results.json")
    res = json.load(open(res_path))
    cfgs = [c for c in res["meta"]["configs"] if c in COLOR]
    fdir = os.path.join(out_dir, "figures")
    os.makedirs(fdir, exist_ok=True)
    figs = []
    if "kl" in res:
        fig_kl(res, cfgs, os.path.join(fdir, "kl_by_type.png"))
        figs.append(("kl_by_type.png", "与 8B 的 KL，按数据类型"))
    if fig_tasks(res, cfgs, os.path.join(fdir, "task_accuracy.png")):
        figs.append(("task_accuracy.png", "各任务准确率"))
    if "ruler" in res:
        fig_ruler(res, cfgs, os.path.join(fdir, "ruler_tasks.png"))
        figs.append(("ruler_tasks.png", "RULER 各子任务"))
    if "curve" in res and res["curve"]:
        fig_curve(res, os.path.join(fdir, "steps_vs_data.png"))
        figs.append(("steps_vs_data.png", "训练步数 vs 数据"))
    if fig_train(os.path.join(fdir, "training_curves.png")):
        figs.append(("training_curves.png", "训练曲线"))
    if fig_speed(os.path.join(fdir, "speed.png")):
        figs.append(("speed.png", "速度"))
    res["figures"] = figs
    json.dump(res, open(res_path, "w"), indent=1)
    write_summary(res, os.path.join(out_dir, "summary.md"))
    print("[plot] wrote", [f for f, _ in figs])


if __name__ == "__main__":
    main()
