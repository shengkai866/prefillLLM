#!/usr/bin/env python
"""
Deep-handoff experiment (prefill/deep.py) with the round-2 mix-8k projector. No training.

The small model + projector provide the large model's first 36-k layers for the middle prompt tokens; the large model
then runs its last k layers on ALL prompt tokens, starting from the projector's predicted residual stream.

Configs:  big | hyb | deep:<k> | any eval_recompute.py config (e.g. attn_small:0.1:sent as the selective-recompute reference)
Tasks (stored in results.json as each finishes; finished tasks are skipped on restart):
  sweep    diagnostic over k = 0..36 on QA prompts: KL(8B || deep) and top-1 agreement of the first answer token,
           cosine to the 8B of the handoff residual (input of layer 36-k) and of the final hidden states (middle/tail)
  niah, ruler, qa, speed   same tasks and examples as scripts/eval_recompute.py
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import eval_recompute as R  # noqa: E402
from evaluate_v2 import mean  # noqa: E402
from prefill import config as C  # noqa: E402
from prefill.deep import DeepHybrid  # noqa: E402
from prefill.kv import split_prompt  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402

KS = (4, 8, 12, 18, 24)
DEFAULT_CFGS = ["big", "hyb"] + [f"deep:{k}" for k in KS] + ["attn_small:0.1:sent"]
SPEED_CFGS = DEFAULT_CFGS + ["attn_small:0.05:sent", "random:0.05"]
SWEEP_KS = (0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36)


def is_deep(c):
    return c.startswith("deep:")


def label(c, Ll=36):
    if is_deep(c):
        k = int(c.split(":")[1])
        return f"{C.LARGE_NAME} 跑最后 {k} 层（投影前 {Ll - k} 层）"
    return R.label(c)


class DeepEngine(R.Engine):
    def __init__(self, tok, small, large, proj_path, dev, n_sink, n_tail):
        super().__init__(tok, small, large, proj_path, dev, n_sink, n_tail)
        p, _ = Projector.load(proj_path, dev)
        self.dh = DeepHybrid(self.S, self.L, self.rh.fold, p.eval(), n_sink=n_sink, n_tail=n_tail)

    @torch.no_grad()
    def prefill(self, cfg, ids, answers=()):
        if is_deep(cfg):
            k = int(cfg.split(":")[1])
            return (*self.dh.prefill_deep(ids, k), k / self.L.n_layers)
        return super().prefill(cfg, ids, answers)


def cos_rows(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


@torch.no_grad()
def task_sweep(E, n_per_cell, ks):
    pool = PromptPool("qa_eval")
    cells = collections.defaultdict(list)
    for i, m in enumerate(pool.meta):
        cells[(m["source"], m["bucket"])].append(i)
    L, dh = E.L, E.dh
    acc = collections.defaultdict(lambda: collections.defaultdict(list))  # (bucket, k) -> metric -> values
    for (src, b), idx in sorted(cells.items()):
        for i in idx[:n_per_cell]:
            ids = torch.as_tensor(pool.ids(i), device=E.dev)
            T = ids.shape[0]
            s, m, _ = split_prompt(T, dh.n_sink, dh.n_tail)
            lg_big, hs_true = L.prefill(ids, collect=True)  # hs_true [Ll+1, T, d]: layer inputs + normed final
            lp_big = F.log_softmax(lg_big[0].float(), -1)
            for k in ks:
                _, _, lg, h_mid, hn = dh.prefill_deep(ids, k, return_h=True)
                lp = F.log_softmax(lg[0].float(), -1)
                r = acc[(b, k)]
                r["kl"].append((lp_big.exp() * (lp_big - lp)).sum().item())
                r["top1"].append(float(lg_big[0].argmax() == lg[0].argmax()))
                if h_mid is not None:
                    r["cos_handoff"].append(cos_rows(h_mid, hs_true[L.n_layers - k, s:s + m]))
                if hn is not None:
                    r["cos_final_mid"].append(cos_rows(hn[s:s + m], hs_true[-1, s:s + m]))
                    r["cos_final_tail"].append(cos_rows(hn[s + m:], hs_true[-1, s + m:]))
            del hs_true
        print(f"[sweep] {src} {b} done", flush=True)
    rows = []
    for (b, k), r in sorted(acc.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append({"bucket": b, "k": k, "n": len(r["kl"]), **{mname: mean(v) for mname, v in r.items()}})
    for b in sorted({r["bucket"] for r in rows}):
        print(f"[sweep] {b}: " + " ".join(f"k={r['k']}:KL={r['kl']:.3f}" for r in rows if r["bucket"] == b), flush=True)
    return rows


# ----------------------------------------------------------------------------------------------- summary
def speedup(res, c, T):
    for r in res.get("speed", []):
        if r["prompt_len"] == T and r.get(c) and r.get("big"):
            return r["big"] / r[c]
    return None


def write_summary(res, path):
    cfgs = res["meta"]["configs"]
    Ll = res["meta"].get("n_layers", 36)
    L = ["# 深层交接实验：小模型投影前面的层，大模型自己跑最后 k 层", "",
         f"GPU：{res['meta']['gpu']}；生成：{res['meta']['time']}；projector：混合数据 8k（第 {res['meta'].get('proj_step')} 步，未重新训练）。"
         f"小模型：{C.SMALL_NAME}，大模型：{C.LARGE_NAME}（{Ll} 层）。开头 8 个和最后 64 个 token 始终由大模型完整计算。", "",
         f"“{C.LARGE_NAME} 跑最后 k 层”：中间 token 在前 {Ll}-k 层用 projector 预测的 KV；在第 {Ll}-k 层把 projector 预测的残差流交给 {C.LARGE_NAME}，"
         f"由 {C.LARGE_NAME} 对全部 prompt token 跑完最后 k 层（深层 KV 是大模型自己算的）。k=0 就是原来的混合方案，k={Ll} 就是 {C.LARGE_NAME} 单独。", ""]
    for fig, cap in (("pareto.png", "质量与速度"), ("sweep.png", "k 扫描")):
        if os.path.exists(os.path.join(os.path.dirname(path), "figures", fig)):
            L += [f"![{cap}](figures/{fig})", ""]
    sp_lens = [r["prompt_len"] for r in res.get("speed", [])]
    L += ["## 总表", "", "| 配置 | RULER 4k | RULER 8k | 问答 F1 4k | 问答 F1 8k | 大海捞针 | " + " | ".join(f"加速 {T // 1024}k" for T in sp_lens) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(sp_lens)]
    for c in cfgs:
        L.append(f"| {label(c, Ll)} | {R.fmt(R.ruler_mean(res, c, 4096))} | {R.fmt(R.ruler_mean(res, c, 8192))} | {R.fmt(R.qa_mean(res, c, '4k'))} | "
                 f"{R.fmt(R.qa_mean(res, c, '8k'))} | {R.fmt(R.niah_mean(res, c))} | " + " | ".join(R.fmt(speedup(res, c, T), '{:.2f}×') for T in sp_lens) + " |")
    L += ["", f"加速比是相对 {C.LARGE_NAME} 单独的首 token 延迟（{res['meta']['gpu']}，eager，不含 CUDA graph）。"]
    for Ln in (4096, 8192):
        if not any(r["length"] == Ln for r in res.get("ruler", [])):
            continue
        L += ["", f"## RULER {Ln}，按任务类型", "", "| 配置 | " + " | ".join(R.RULER_GROUPS) + " | 平均 |", "|---|" + "---|" * (len(R.RULER_GROUPS) + 1)]
        for c in cfgs:
            L.append(f"| {label(c, Ll)} | " + " | ".join(R.fmt(R.ruler_mean(res, c, Ln, ts), "{:.2f}") for ts in R.RULER_GROUPS.values())
                     + f" | {R.fmt(R.ruler_mean(res, c, Ln))} |")
        tasks = sorted({r["task"] for r in res["ruler"] if r["length"] == Ln and r["task"] != "_frac"})
        L += ["", f"<details><summary>RULER {Ln} 全部子任务</summary>", "", "| 配置 | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
        for c in cfgs:
            L.append(f"| {label(c, Ll)} | " + " | ".join(R.fmt(R.ruler_mean(res, c, Ln, [t]), "{:.2f}") for t in tasks) + " |")
        L += ["", "</details>"]
    if "qa" in res:
        L += ["", "## 问答 F1", "", "| 配置 | " + " | ".join(f"{s} {b}" for b in ("4k", "8k") for s in ("squad", "hotpot", "trivia")) + " |", "|---|" + "---|" * 6]
        for c in cfgs:
            v = {(r["source"], r["bucket"]): r["f1"] for r in res["qa"] if r["config"] == c}
            if v:
                L.append(f"| {label(c, Ll)} | " + " | ".join(R.fmt(v.get((s, b))) for b in ("4k", "8k") for s in ("squad", "hotpot", "trivia")) + " |")
    if "speed" in res:
        sc = [c for c in res["speed"][0] if c != "prompt_len"]
        L += ["", f"## 首 token 延迟（毫秒，{res['meta']['gpu']}，eager）", "", "| 配置 | " + " | ".join(str(r["prompt_len"]) for r in res["speed"]) + " |",
              "|---|" + "---|" * len(res["speed"])]
        for c in sc:
            L.append(f"| {label(c, Ll)} | " + " | ".join("显存不足" if r.get(c) is None else f"{r[c]*1e3:.1f}" + (f"（{r['big']/r[c]:.2f} 倍）" if r.get("big") else "")
                                                        for r in res["speed"]) + " |")
    if "sweep" in res:
        L += ["", "## k 扫描（诊断，问答 prompt）", "",
              f"KL / top-1：prompt 最后一个位置（第一个答案 token）的分布，相对 {C.LARGE_NAME} 单独。"
              f"交接处余弦：projector 预测的第 {Ll}-k 层输入残差流 vs {C.LARGE_NAME} 真实值（中间 token 平均）。"
              f"最终余弦：最后一层（final norm 之后）隐状态 vs 真实值，分中间 token 和尾部 64 个 token。", ""]
        for b in sorted({r["bucket"] for r in res["sweep"]}):
            L += [f"**{b}**", "", "| k | 理论大模型开销 | KL | top-1 一致 | 交接处余弦 | 最终余弦（中间） | 最终余弦（尾部） |", "|---|---|---|---|---|---|---|"]
            for r in res["sweep"]:
                if r["bucket"] == b:
                    L.append(f"| {r['k']} | {r['k'] / Ll:.0%} | {r['kl']:.3f} | {r['top1']:.2f} | {R.fmt(r.get('cos_handoff'))} | "
                             f"{R.fmt(r.get('cos_final_mid'))} | {R.fmt(r.get('cos_final_tail'))} |")
            L.append("")
    open(path, "w").write("\n".join(L) + "\n")


def plot(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "font.family": "DejaVu Sans",
        "legend.frameon": False})
    PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    cfgs = res["meta"]["configs"]
    lens = [r["prompt_len"] for r in res.get("speed", [])]
    # 1) quality vs speed: deep:k curve vs selective-recompute points, anchors big / hyb
    if lens and "ruler" in res:
        T = 8192 if 8192 in lens else lens[0]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        panels = [("RULER 4k (mean of 13 tasks)", lambda c: R.ruler_mean(res, c, 4096)),
                  ("RULER 8k (mean of 13 tasks)", lambda c: R.ruler_mean(res, c, 8192)),
                  ("Document QA F1, ~7.5k tokens", lambda c: R.qa_mean(res, c, "8k"))]
        for ax, (title, val) in zip(axes, panels):
            deep = sorted((speedup(res, c, T), val(c), int(c.split(":")[1])) for c in cfgs if is_deep(c))
            deep = [d for d in deep if d[0] is not None and d[1] is not None]
            hyb = (speedup(res, "hyb", T), val("hyb"))
            if deep and None not in hyb:
                pts = deep + [(hyb[0], hyb[1], 0)]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PAL[0], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
                        label=f"{C.LARGE_NAME} runs last k layers", zorder=3)
                for x, y, k in pts:
                    ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color=TEXT2)
            others = [c for c in cfgs if not is_deep(c) and c not in ("big", "hyb")]
            for c, col in zip(others, PAL[1:]):
                x, y = speedup(res, c, T), val(c)
                if x is not None and y is not None:
                    ax.plot([x], [y], color=col, marker="s", ms=8, ls="none", mec=SURFACE, mew=1.5, label=f"recompute {c}", zorder=4)
            yb = val("big")
            if yb is not None:
                ax.axhline(yb, color=TEXT2, lw=1.2, ls="--", zorder=2)
                ax.text(ax.get_xlim()[1], yb, f" {C.LARGE_NAME} alone", color=TEXT2, fontsize=8, va="center")
            ax.set_title(title, loc="left")
            ax.set_xlabel(f"TTFT speedup vs {C.LARGE_NAME} at {T} tokens")
        axes[0].set_ylabel("score")
        h, lab = axes[0].get_legend_handles_labels()
        if h:
            fig.legend(h, lab, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"Quality vs speed: deep handoff vs selective recompute ({C.SMALL_NAME} -> {C.LARGE_NAME}, no retraining)", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "pareto.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 2) sweep diagnostics
    if "sweep" in res:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        buckets = sorted({r["bucket"] for r in res["sweep"]})
        for b, col in zip(buckets, PAL):
            rows = [r for r in res["sweep"] if r["bucket"] == b]
            ks = [r["k"] for r in rows]
            axes[0].plot(ks, [r["kl"] for r in rows], color=col, lw=2, marker="o", ms=5, mec=SURFACE, label=f"QA prompts {b}")
            axes[1].plot(ks, [r["top1"] for r in rows], color=col, lw=2, marker="o", ms=5, mec=SURFACE, label=f"QA prompts {b}")
            sel = [r for r in rows if r.get("cos_handoff") is not None]
            axes[2].plot([r["k"] for r in sel], [r["cos_handoff"] for r in sel], color=col, lw=1.5, ls=":", marker="o", ms=4,
                         label=f"handoff residual (layer 36-k), {b}")
            sel = [r for r in rows if r.get("cos_final_mid") is not None]
            axes[2].plot([r["k"] for r in sel], [r["cos_final_mid"] for r in sel], color=col, lw=2, marker="o", ms=5, mec=SURFACE,
                         label=f"final hidden, middle tokens, {b}")
        axes[0].set_yscale("log")
        for ax, t in zip(axes, ("KL(8B || deep) at the first answer token", "top-1 agreement with 8B", "cosine to the 8B's own states")):
            ax.set_title(t, loc="left")
            ax.set_xlabel(f"k = number of final {C.LARGE_NAME} layers run on all tokens")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "sweep.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
    ap.add_argument("--configs", default=",".join(DEFAULT_CFGS))
    ap.add_argument("--speed_configs", default=",".join(SPEED_CFGS))
    ap.add_argument("--tasks", default="sweep,niah,ruler,qa,speed")
    ap.add_argument("--n_sweep", type=int, default=10)
    ap.add_argument("--n_ruler", type=int, default=15)
    ap.add_argument("--ruler_lengths", default="4096,8192")
    ap.add_argument("--n_qa", type=int, default=50)
    ap.add_argument("--speed_lengths", default="4096,8192,16384,32768")
    ap.add_argument("--speed_reps", type=int, default=5)
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "deep_handoff"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plot_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res_path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if a.plot_only:
        plot(res, a.out_dir)
        write_summary(res, os.path.join(a.out_dir, "summary.md"))
        return
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    E = DeepEngine(tok, small, large, a.projector, dev, a.n_sink, a.n_tail)
    cfgs = [c for c in a.configs.split(",") if c]
    res["meta"] = {"gpu": torch.cuda.get_device_name(0), "configs": cfgs, "args": vars(a), "proj_step": E.proj_step,
                   "n_layers": E.L.n_layers, "time": time.strftime("%F %T")}
    log = R.GenLog(os.path.join(a.out_dir, "generations.jsonl"))
    for task in a.tasks.split(","):
        if task in res and not a.force:
            print(f"[task {task}] already done, skipping")
            continue
        t0 = time.time()
        if task == "sweep":
            res["sweep"] = task_sweep(E, a.n_sweep, SWEEP_KS)
        elif task == "ruler":
            res["ruler"] = R.task_ruler(E, cfgs, a.n_ruler, [int(x) for x in a.ruler_lengths.split(",")], log)
        elif task == "qa":
            res["qa"] = R.task_qa(E, cfgs, a.n_qa, log)
        elif task == "niah":
            res["niah"] = R.task_niah(E, cfgs, log)
        elif task == "speed":
            res["speed"] = R.task_speed(E, [c for c in a.speed_configs.split(",") if c],
                                        [int(x) for x in a.speed_lengths.split(",")], a.speed_reps)
        print(f"[task {task}] {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
        try:  # a plotting problem must never stop the evaluation
            plot(res, a.out_dir)
        except Exception as e:
            print(f"[plot] skipped: {type(e).__name__}: {e}", flush=True)
        write_summary(res, os.path.join(a.out_dir, "summary.md"))
    print("[done]", a.out_dir)


if __name__ == "__main__":
    main()
