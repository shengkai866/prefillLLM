#!/usr/bin/env python
"""
Speed-quality Pareto plots (Qwen3-8B target, L40S, every method CUDA-graphed, fused fold) instead of single-point
comparisons at equal recompute fractions. One panel per benchmark with a fixed prompt length; x = TTFT at that length
(ms, log scale), y = quality. Points:
  ours        projected cache with 0 / 10 / 20 % of the middle recomputed (one curve per small model)
  SpecPrefill keep 10 / 20 / 30 % of the prompt (scored by the same small model)
  alone       0.6B / 1.7B / 4B / 8B, each prefilling and decoding by itself
  drop        same selection as ours 20 %, the rest dropped instead of projected (ablation, grey)
Free-form QA panels use `contains` (scripts/items.py: larger models answer more verbosely, F1 punishes that).
-> results/figures/pareto_<pair>.png / .pdf and results/pareto_summary.md (which points are Pareto-optimal per panel).

  python scripts/plot_pareto.py [--results DIR] [--pairs 06,17]
"""
import argparse
import json
import os
import sys
import time
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

# reference palette (dataviz skill, references/palette.md): first three categorical slots validate all-pairs
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
PAIRS = {"06": ("fused_06", "0.6B"), "17": ("fused_17", "1.7B")}
ALONE = [("0.6B", "fused_06", "small"), ("1.7B", "fused_17", "small"), ("4B", "mid_4B", "small"), ("8B", "fused_06", "big")]
OURS = [("0%", "hyb"), ("10%", "attn_small:0.1:sent"), ("20%", "attn_small:0.2:sent")]
SPEC = [("10%", "specprefill:0.1"), ("20%", "specprefill:0.2"), ("30%", "specprefill:0.3")]
DROP = [("drop", "drop-attn_small:0.2:sent")]


def avg(rows, cfg, val, **m):
    v = [r[val] for r in rows if r.get("config") == cfg and not str(r.get("task", "")).startswith("_")
         and all(str(r.get(k)) == str(x) for k, x in m.items()) and r.get(val) is not None]
    return mean(v) if v else None


# (panel title, step, prompt length for TTFT, fn(res, cfg))
PANELS = [
    ("RULER 4k", "main", 4096, lambda r, c: avg(r.get("ruler", []), c, "score", length=4096)),
    ("RULER 8k", "main", 8192, lambda r, c: avg(r.get("ruler", []), c, "score", length=8192)),
    ("Extractive QA 4k (contains)", "main", 4096, lambda r, c: avg(r.get("qa", []), c, "contains", bucket="4k")),
    ("Extractive QA 8k (contains)", "main", 8192, lambda r, c: avg(r.get("qa", []), c, "contains", bucket="8k")),
    ("Code completion 4k (edit sim.)", "main", 4096, lambda r, c: avg(r.get("code", []), c, "similarity", bucket="4k")),
    ("Code completion 8k (edit sim.)", "main", 8192, lambda r, c: avg(r.get("code", []), c, "similarity", bucket="8k")),
    ("QuALITY 4k (accuracy)", "quality", 4096, lambda r, c: avg(r.get("quality", []), c, "accuracy", bucket="4k", part="all")),
    ("QuALITY 8k (accuracy)", "quality", 8192, lambda r, c: avg(r.get("quality", []), c, "accuracy", bucket="8k", part="all")),
    ("Long QA 16k (contains)", "long", 16384, lambda r, c: avg(r.get("qa_long", []), c, "contains", bucket="16k")),
    ("Long QA 32k (contains)", "long", 32768, lambda r, c: avg(r.get("qa_long", []), c, "contains", bucket="32k")),
]


class Res:
    def __init__(self, root):
        self.root, self.c = root, {}

    def get(self, prefix, step):
        d = {"quality": f"quality_{prefix.split('_')[-1]}"}.get(step, f"{prefix}_{step}") if prefix != "mid_4B" else f"mid_4B_{step}"
        if d not in self.c:
            try:
                self.c[d] = json.load(open(os.path.join(self.root, d, "results.json")))
            except (OSError, json.JSONDecodeError):
                self.c[d] = None
        return self.c[d]

    def ttft(self, prefix, cfg, L):
        r = self.get(prefix, "speed") or {}
        row = next((x for x in r.get("speed_allgraph", {}).get("rows", []) if x["prompt_len"] == L), None)
        return None if row is None or row.get(cfg) is None else row[cfg] * 1e3


def points(R, pair, step, L, fn):
    """[(family, label, x ms, y)] for one panel."""
    prefix = PAIRS[pair][0]
    pts = []
    for lab, pre, cfg in ALONE:
        res = R.get(pre, step)
        x, y = R.ttft(pre, cfg, L), (fn(res, cfg) if res else None)
        if x is not None and y is not None:
            pts.append(("alone", lab, x, y))
    res = R.get(prefix, step)
    for fam, lst in (("ours", OURS), ("spec", SPEC), ("drop", DROP)):
        for lab, cfg in lst:
            x, y = R.ttft(prefix, cfg, L), (fn(res, cfg) if res else None)
            if x is not None and y is not None:
                pts.append((fam, lab, x, y))
    return pts


def frontier(pts):
    """Points not dominated (another point at most as slow and at least as good, one strictly)."""
    return [p for p in pts if not any(q[2] <= p[2] and q[3] >= p[3] and (q[2] < p[2] or q[3] > p[3]) for q in pts)]


OFFSETS = [(5, 4), (5, -11), (-6, 6), (-6, -12), (8, -3), (-10, -3), (0, 10), (0, -16)]


def place_labels(fig, labels):
    """Direct labels with a greedy repel: each label takes the first offset whose box overlaps no label placed before
    it and no marker (bounding boxes in display space). Frontier labels (bold) are placed first."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    taken = []
    marks = {}
    for ax, p, _ in labels:  # marker boxes, so labels do not cover points
        x, y = ax.transData.transform((p[2], p[3]))
        marks.setdefault(id(ax), []).append((x - 4, y - 4, x + 4, y + 4))
    hit = lambda b, c: not (b[2] < c[0] or c[2] < b[0] or b[3] < c[1] or c[3] < b[1])
    for ax, p, on in sorted(labels, key=lambda t: not t[2]):
        best = None
        for dx, dy in OFFSETS:
            ha = "right" if dx < 0 else "center" if dx == 0 else "left"
            t = ax.annotate(p[1], (p[2], p[3]), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, ha=ha,
                            color=INK if on else INK2, fontweight="bold" if on else "normal", zorder=4)
            e = t.get_window_extent(rend)
            b = (e.x0, e.y0, e.x1, e.y1)
            if not any(hit(b, c) for c in taken + marks.get(id(ax), [])) and ax.bbox.contains(e.x0, e.y0) \
                    and ax.bbox.contains(e.x1, e.y1):
                best = (t, b)
                break
            t.remove()
        if best is None:  # nothing free: keep the first offset
            t = ax.annotate(p[1], (p[2], p[3]), textcoords="offset points", xytext=OFFSETS[0], fontsize=7.5,
                            color=INK if on else INK2, fontweight="bold" if on else "normal", zorder=4)
            e = t.get_window_extent(rend)
            best = (t, (e.x0, e.y0, e.x1, e.y1))
        taken.append(best[1])


STYLE = {"ours": (BLUE, "o", "Ours: projected cache, recompute 0/10/20%"),
         "spec": (ORANGE, "s", "SpecPrefill: keep 10/20/30%"),
         "alone": (AQUA, "D", "Model alone: 0.6B / 1.7B / 4B / 8B"),
         "drop": (GREY, "x", "Drop the rest (ablation, 20%)")}


def plot(R, pair, out_dir):
    small = PAIRS[pair][1]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.6))
    summary, labels = [], []
    for ax, (title, step, L, fn) in zip(axes.flat, PANELS):
        pts = points(R, pair, step, L, fn)
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8)
        if not pts:
            ax.set_title(title + " (no data)", fontsize=10, color=INK)
            continue
        front = frontier(pts)
        fx = sorted(front, key=lambda p: p[2])
        ax.step([p[2] for p in fx], [p[3] for p in fx], where="post", color=AXIS, linewidth=1.2, zorder=1)
        for fam in ("drop", "alone", "spec", "ours"):
            q = sorted([p for p in pts if p[0] == fam], key=lambda p: p[2])
            if not q:
                continue
            col, mk, _ = STYLE[fam]
            if fam in ("ours", "spec", "alone"):
                ax.plot([p[2] for p in q], [p[3] for p in q], color=col, linewidth=2, zorder=2, alpha=0.9)
            ax.scatter([p[2] for p in q], [p[3] for p in q], color=col, marker=mk, s=46, zorder=3,
                       edgecolors="#fcfcfb" if mk != "x" else None, linewidths=1.5 if mk != "x" else 2)
            for p in q:
                labels.append((ax, p, p in front))
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel(f"TTFT at {L // 1024}k (ms, log)", fontsize=8, color=INK2)
        summary.append((title, L, sorted(pts, key=lambda p: p[2]), front))
    handles = [plt.Line2D([], [], color=STYLE[f][0], marker=STYLE[f][1], linestyle="-" if f != "drop" else "",
                          markersize=7, linewidth=2, label=STYLE[f][2]) for f in ("ours", "spec", "alone", "drop")]
    handles.append(plt.Line2D([], [], color=AXIS, linewidth=1.2, label="Pareto frontier (bold labels = on it)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Speed vs quality, Qwen3-{small} as the small model, Qwen3-8B target (L40S, all methods CUDA-graphed)",
                 y=1.035, fontsize=12, color=INK)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    place_labels(fig, labels)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"pareto_{pair}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULTS_DIR)
    ap.add_argument("--pairs", default="06,17")
    a = ap.parse_args()
    R = Res(a.results)
    fam_zh = {"ours": "我们", "spec": "SpecPrefill", "alone": "单独", "drop": "丢弃其余"}
    md = ["# 速度–质量 Pareto（目标 Qwen3-8B，L40S）", "",
          f"生成：{time.strftime('%F %T')}，`scripts/plot_pareto.py`；图在 `results/figures/pareto_<pair>.png|pdf`。"
          "每个面板：横轴 = 该长度的首 token 延迟（所有方法 CUDA graph，融合 fold），纵轴 = 质量；自由回答 QA 用“包含答案”。"
          "“我们 x%”= 重算 x% 中间 token（0% = 只用投影缓存）；“SpecPrefill x%”= 保留整段 prompt 的 x%；“单独”= 该模型自己 prefill 和解码；"
          "“丢弃其余”= 和我们 20% 同样的选择，但没选中的直接丢掉（消融）。Pareto 前沿 = 没有别的点同时更快且更好。", ""]
    for pair in [p for p in a.pairs.split(",") if p]:
        summ = plot(R, pair, os.path.join(a.results, "figures"))
        md += [f"## 小模型 Qwen3-{PAIRS[pair][1]}（`{pair}`）", "", "| 面板 | Pareto 前沿上的点（由快到慢：方法 延迟 ms → 质量） | 不在前沿上 |", "|---|---|---|"]
        for title, L, pts, front in summ:
            on = " → ".join(f"{fam_zh[p[0]]} {p[1]} {p[2]:.0f}→{p[3]:.3f}" for p in sorted(front, key=lambda p: p[2]))
            off = "，".join(f"{fam_zh[p[0]]} {p[1]}" for p in pts if p not in front)
            md.append(f"| {title} | {on} | {off} |")
        md.append("")
    open(os.path.join(a.results, "pareto_summary.md"), "w").write("\n".join(md) + "\n")
    print("[plot_pareto]", os.path.join(a.results, "pareto_summary.md"))


if __name__ == "__main__":
    main()
