#!/usr/bin/env python
"""
Figures for REPORT_2026-10-27_specprefill.md and REPORT_2026-10-27_06B_vs_17B.md -> results/report_1027/figures/
  spec_pareto.png   quality vs TTFT speedup, all configs of results/spec_06 and spec_17 (L40S): ours, same selection
                    with the rest dropped, SpecPrefill, our fill with SpecPrefill's scores
  fabrication.png   share of QA answers with content not in the document vs fraction (needs qa_errors.json from
                    scripts/analyze_qa_errors.py)
  cmp_pareto.png    0.6B vs 1.7B projector on the same axes (spec_* runs, attn_big from cmp_*)
"""
import collections
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "report_1027", "figures")
SURFACE, TEXT, TEXT2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#3d3d3a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "legend.frameon": False})
# categorical slots 1-4 of the validated palette, fixed per method (same as the auto figures of eval_recompute.py)
METHODS = [("ours: projected KV + recompute (small-model attention, sentences)", "attn_small:{}:sent", (0.05, 0.1, 0.2), "#2a78d6", "o", (5, 4)),
           ("same selection, rest DROPPED", "drop-attn_small:{}:sent", (0.05, 0.1, 0.2), "#eb6834", "s", (5, 4)),
           ("SpecPrefill (fraction of whole prompt kept)", "specprefill:{}", (0.1, 0.2, 0.3, 0.5), "#1baf7a", "^", (-6, -11)),
           ("ours with SpecPrefill's look-ahead scores", "spec:{}", (0.1, 0.2), "#eda100", "D", None)]
YLIM = {"ruler": (0.6, 1.0), "qa": (0.4, 0.85)}
PANELS = [("RULER 4k", "ruler", 4096, 4096), ("RULER 8k", "ruler", 8192, 8192),
          ("Document QA F1, ~3.5k tokens", "qa", "4k", 4096), ("Document QA F1, ~7.5k tokens", "qa", "8k", 8192)]


def load(run):
    r = json.load(open(os.path.join(ROOT, "results", run, "results.json")))
    q = collections.defaultdict(list)
    for x in r["ruler"]:
        if x["task"] != "_frac":
            q[("ruler", x["config"], x["length"])].append(x["score"])
    for x in r["qa"]:
        q[("qa", x["config"], x["bucket"])].append(x["f1"])
    q = {k: sum(v) / len(v) for k, v in q.items()}
    sp = {(L["prompt_len"], c): L["big"] / t for L in r["speed"] for c, t in L.items() if c != "prompt_len"}
    return q, sp


def point(ax, x, y, label, marker, color=REF, size=70, dx=6, dy=-10):
    ax.scatter([x], [y], s=size, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=TEXT2)


def pareto(fname, rows, title):
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 4.1 * len(rows) + 0.6), squeeze=False)
    for (rowname, run, xmax), axrow in zip(rows, axes):
        q, sp = load(run)
        small = rowname.split()[0]
        for ax, (pt, task, key, L) in zip(axrow, PANELS):
            for lab, pat, fr, col, mk, off in METHODS:
                cs = [pat.format(f) for f in fr]
                xs, ys = [sp[(L, c)] for c in cs], [q[(task, c, key)] for c in cs]
                ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=7, mec=SURFACE, mew=1.2, label=lab, zorder=3)
                for f, x, y in zip(fr, xs, ys) if off else ():
                    ax.annotate(f"{f:.0%}", (x, y), xytext=off, textcoords="offset points", fontsize=7, color=TEXT2,
                                ha="right" if off[0] < 0 else "left")
            c = "specprefill:0.1:lah1"
            ax.scatter([sp[(L, c)]], [q[(task, c, key)]], s=55, marker="^", facecolor=SURFACE, edgecolor="#1baf7a", lw=1.8, zorder=4)
            ax.annotate("SpecPrefill 10%,\nlook-ahead 1", (sp[(L, c)], q[(task, c, key)]), xytext=(5, -18),
                        textcoords="offset points", fontsize=7, color=TEXT2)
            point(ax, 1.0, q[(task, "big", key)], "8B", "*", size=130, dx=-4, dy=6)
            point(ax, sp[(L, "hyb")], q[(task, "hyb", key)], "no recompute", "P")
            ax.text(0.98, 0.03, f"{small} alone: {sp[(L, 'small')]:.1f}x, {q[(task, 'small', key)]:.3f}", transform=ax.transAxes,
                    ha="right", fontsize=7.5, color=TEXT2)
            ax.set_title(f"{rowname} · {pt}", loc="left")
            ax.set_xlabel(f"TTFT speedup over 8B at {L} tokens (L40S, eager)")
            ax.set_ylim(*YLIM[task])
            ax.set_xlim(0.8, xmax)
        axrow[0].set_ylabel("score")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fabrication():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (name, run) in zip(axes, (("0.6B -> 8B", "spec_06"), ("1.7B -> 8B", "spec_17"))):
        e = json.load(open(os.path.join(ROOT, "results", run, "qa_errors.json")))
        for lab, pat, fr, col, mk, _ in METHODS[:3]:
            ax.plot([f * 100 for f in fr], [e[pat.format(f)]["not_in_doc"] * 100 for f in fr], color=col, lw=2, marker=mk,
                    ms=7, mec=SURFACE, mew=1.2, label=lab, zorder=3)
        for c, lab in (("big", "8B alone"), ("hyb", "no recompute")):
            ax.axhline(e[c]["not_in_doc"] * 100, color=REF, lw=1, ls="--" if c == "big" else ":")
            ax.annotate(f"{lab} {e[c]['not_in_doc']:.1%}", (49, e[c]["not_in_doc"] * 100), xytext=(0, 3),
                        textcoords="offset points", ha="right", fontsize=7.5, color=TEXT2)
        ax.set_title(name, loc="left")
        ax.set_xlabel("fraction recomputed / kept (%)")
        ax.set_xlim(0, 52)
    axes[0].set_ylabel("QA answers with content not in the document (%)\n300 questions, ±2-3 points")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Fabrication on document QA: dropping the unselected tokens makes the 8B invent answers", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "fabrication.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def cmp():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for name, spec, cmp_run, col, mk, off in (("0.6B", "spec_06", "cmp_06", "#2a78d6", "o", (5, -11)),
                                              ("1.7B", "spec_17", "cmp_17", "#eb6834", "s", (-5, 5))):
        q, sp = load(spec)
        qc, spc = load(cmp_run)
        cs = ["hyb", "attn_small:0.05:sent", "attn_small:0.1:sent", "attn_small:0.2:sent"]
        for ax, (pt, task, key, L) in zip(axes, PANELS):
            xs, ys = [sp[(L, c)] for c in cs], [q[(task, c, key)] for c in cs]
            ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=7, mec=SURFACE, mew=1.2, label=f"{name} projector: 0% / 5% / 10% / 20% recomputed", zorder=3)
            for t, x, y in zip(("0%", "5%", "10%", "20%"), xs, ys):
                ax.annotate(t, (x, y), xytext=off, textcoords="offset points", fontsize=7, color=TEXT2, ha="right" if off[0] < 0 else "left")
            if name == "1.7B":
                ax.scatter([sp[(L, "small")]], [q[(task, "small", key)]], s=80, marker="X", color=col, edgecolor=SURFACE, lw=1.2, zorder=4)
                ax.annotate("1.7B alone", (sp[(L, "small")], q[(task, "small", key)]), xytext=(-5, 6), textcoords="offset points",
                            fontsize=7.5, color=TEXT2, ha="right")
            c = "attn_big:0.05"
            ax.scatter([spc[(L, c)]], [qc[(task, c, key)]], s=60, marker=mk, facecolor=SURFACE, edgecolor=col, lw=1.8, zorder=4)
            if name == "0.6B":
                point(ax, 1.0, q[(task, "big", key)], "8B", "*", size=130, dx=-4, dy=6)
                ax.text(0.98, 0.03, f"0.6B alone: {sp[(L, 'small')]:.1f}x, {q[(task, 'small', key)]:.3f}", transform=ax.transAxes,
                        ha="right", fontsize=7.5, color=TEXT2)
                ax.set_title(pt, loc="left")
                ax.set_xlabel(f"TTFT speedup over 8B at {L} tokens (L40S, eager)")
                ax.set_ylim(*YLIM[task])
                ax.set_xlim(0.8, 4.4)
    axes[0].set_ylabel("score")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("0.6B vs 1.7B as the small model for an 8B target (filled = small-model attention, sentences; hollow = 8B attention 5%)", y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "cmp_pareto.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    pareto("spec_pareto.png", [("0.6B -> 8B", "spec_06", 5.0), ("1.7B -> 8B", "spec_17", 3.3)],
           "Quality vs TTFT speedup: our projected fill vs SpecPrefill (labels = fraction recomputed / kept)")
    fabrication()
    cmp()
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
