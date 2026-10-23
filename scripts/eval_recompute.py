#!/usr/bin/env python
"""
Selective-recompute experiment (prefill/recompute.py) with the round-2 mix-8k projector. No training.

Configs:  big (8B alone) | small (small model alone) | hyb (hybrid, nothing recomputed) | oracle
          <scorer>:<fraction of middle tokens>[:<unit>]        recompute, the other middle tokens get projected KV
          drop-<scorer>:<fraction of middle tokens>[:<unit>]   same selection, the other middle tokens are DROPPED
          specprefill:<fraction of the whole prompt>[:lahN]    SpecPrefill baseline (official defaults, look-ahead 8)
          scorer: attn_small | attn_big | spec (SpecPrefill importance) | random | recent
          unit: omitted = single tokens (max-pooled scores; spec: 32-token chunks) | bN = fixed blocks of N tokens |
                sent = sentence/line segments
Tasks (stored in results.json as each finishes; finished tasks are skipped on restart):
  ruler    RULER 4096 / 8192, 13 tasks, same examples as round 2 (same seeds)
  qa       SQuAD v2 / HotpotQA / TriviaQA at ~3.5k / ~7.5k tokens (first n of the round-2 pool)
  niah     round-1 passcode needle at 2048 / 3584 / 7680
  speed    TTFT (eager fast path, H100) vs prompt length for big, hyb and the recompute configs
  gsm8k    GSM8K 8-shot as chat turns (same prompts as scripts/evaluate_v2.py), any config incl. recompute
  code     next-line completion on held-out repositories (same pool as scripts/evaluate_v2.py)
Every generation is written to generations.jsonl for error analysis.
"""
import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from evaluate_v2 import mean, qa_scores, task_code, task_gsm8k  # noqa: E402
from prefill import config as C  # noqa: E402
from prefill.fast import FastBig, FastQwen3, FoldedProjectorKV, kv_to_dynamic_cache, time_call  # noqa: E402
from prefill.kv import greedy_decode, split_prompt, true_prefill  # noqa: E402
from prefill.mixdata import PromptPool, Tok  # noqa: E402
from prefill.models import eos_ids, load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402
from prefill.recompute import RecomputeHybrid  # noqa: E402

FRACS = (0.02, 0.05, 0.1, 0.2)
DEFAULT_CFGS = ["big", "hyb"] + [f"{s}:{f}" for s in ("attn_big", "attn_small") for f in FRACS] + \
               ["random:0.1", "random:0.2", "recent:0.1", "recent:0.2", "oracle"]
SPEED_CFGS = ["big", "hyb"] + [f"{s}:{f}" for s in ("attn_big", "attn_small") for f in (0.05, 0.1, 0.2)] + ["random:0.1"]
SCORER_ZH = {"attn_big": f"{C.LARGE_NAME} 注意力", "attn_small": f"{C.SMALL_NAME} 注意力", "random": "随机", "recent": "最近（加长尾部）",
             "oracle": "理想（答案位置）", "spec": f"SpecPrefill 打分（{C.SMALL_NAME}）"}


def parse(cfg):
    """-> (scorer, fraction, unit) with unit None | int block size | 'sent'"""
    if cfg in ("big", "small", "hyb", "oracle"):
        return cfg, 0.0, None
    s, f, *u = cfg.split(":")
    u = u[0] if u else None
    return s, float(f), (None if u is None else u if u == "sent" or u.startswith("lah") else int(u[1:]))


def unit_zh(u):
    return "逐 token" if u is None else "按句子" if u == "sent" else f"look-ahead {u[3:]}" if str(u).startswith("lah") else f"{u} token 一块"


def label(cfg):
    s, f, u = parse(cfg)
    if s == "big":
        return f"{C.LARGE_NAME} 单独"
    if s == "small":
        return f"{C.SMALL_NAME} 单独"
    if s == "hyb":
        return "混合，不重算"
    if s == "oracle":
        return SCORER_ZH["oracle"]
    if s == "specprefill":
        return f"SpecPrefill 保留整段 {f:.0%}" + (f"，{unit_zh(u)}" if u is not None else "")
    if s.startswith("drop-"):
        return f"丢弃其余（不投影）：{SCORER_ZH[s[5:]]} {f:.0%}" + (f"，{unit_zh(u)}" if u is not None else "")
    return f"{SCORER_ZH[s]} {f:.0%}" + (f"，{unit_zh(u)}" if u is not None else "")


class Engine:
    def __init__(self, tok, small, large, proj_path, dev, n_sink, n_tail):
        self.tok, self.small, self.large, self.dev = tok, small, large, dev
        self.T = Tok(tok)
        self.S, self.L = FastQwen3(small), FastQwen3(large)
        self.fb = FastBig(self.L)
        p, ck = Projector.load(proj_path, dev)
        p.eval()
        self.proj_step = ck.get("step")
        # sentence/line boundary tokens for 'sent' units
        boundary = torch.tensor([any(ch in tok.decode([i]) for ch in ".!?\n") for i in range(len(tok))], device=dev)
        self.eos = eos_ids(tok)
        self.rh = RecomputeHybrid(self.S, self.L, FoldedProjectorKV(p, self.L), n_sink=n_sink, n_tail=n_tail, boundary=boundary,
                                  eos=self.eos)

    def needles(self, answers):
        out = []
        for a in answers:
            a = a.strip()
            for v in {a, " " + a, a.lower(), " " + a.lower()}:
                out.append(self.tok.encode(v, add_special_tokens=False))
        return out

    @torch.no_grad()
    def prefill(self, cfg, ids, answers=()):
        """-> (kbuf, vbuf, logits, fraction of middle tokens recomputed; specprefill: fraction of the prompt kept).
        drop-* and specprefill return a compact cache (shorter than the prompt)."""
        s, f, u = parse(cfg)
        if s == "big":
            return (*self.fb.prefill(ids), 1.0)
        if s == "small":  # small model alone: its own prefill (lean path); decoding is handled in generate()
            return None, None, self.S.prefill(ids)[0], 0.0
        if s == "hyb":
            return (*self.rh.prefill(ids), 0.0)
        if s == "specprefill":
            kb, vb, lg, n = self.rh.prefill_specprefill(ids, f, n_lah=int(u[3:]) if u else None)
            return kb, vb, lg, n / ids.shape[0]
        kb, vb, lg, n = self.rh.prefill_sel(ids, s.removeprefix("drop-"), f, needles=self.needles(answers) if s == "oracle" else None,
                                            seg=u, drop=s.startswith("drop-"))
        _, m, _ = split_prompt(ids.shape[0], self.rh.n_sink, self.rh.n_tail)
        return kb, vb, lg, n / max(m, 1)

    @torch.no_grad()
    def generate(self, cfg, ids, max_new, answers=()):
        ids = torch.as_tensor(ids, device=self.dev)
        if cfg == "small":
            cache, lg = true_prefill(self.small, ids[None])
            out = greedy_decode(self.small, cache, lg, max_new, self.eos)
            return self.tok.decode(out, skip_special_tokens=True), 0.0
        kb, vb, lg, frac = self.prefill(cfg, ids, answers)
        T = ids.shape[0]
        out = greedy_decode(self.large, kv_to_dynamic_cache(kb, vb), lg, max_new, self.eos, pos0=T if kb.shape[2] != T else None)
        return self.tok.decode(out, skip_special_tokens=True), frac

    def chat_ids(self, messages, assistant_prefix=""):
        ids, _ = self.T.chat(messages)
        return ids + (self.T.enc(assistant_prefix) if assistant_prefix else [])


class V2Adapter:
    """Lets scripts/evaluate_v2.py task functions (generate -> str) run any config of this script, incl. recompute."""

    def __init__(self, E):
        self.E = E

    def generate(self, cfg, ids, max_new):
        return self.E.generate(cfg, ids, max_new)[0]

    def chat_ids(self, messages, assistant_prefix=""):
        return self.E.chat_ids(messages, assistant_prefix)


class GenLog:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, **kw):
        self.f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.f.flush()


# ----------------------------------------------------------------------------------------------- tasks
def task_ruler(E, cfgs, n_per_task, lengths, log):
    from datasets import load_dataset
    rows = []
    for L in lengths:
        ds = load_dataset("simonjegou/ruler", str(L), split="test")
        by = collections.defaultdict(list)
        for i, t in enumerate(ds["task"]):
            by[t].append(i)
        rng = random.Random(L)  # same examples as scripts/evaluate_v2.py
        items = []
        for t, idx in sorted(by.items()):
            for i in rng.sample(idx, min(n_per_task, len(idx))):
                ex = ds[i]
                items.append((t, i, E.chat_ids([{"role": "user", "content": ex["context"] + ex["question"]}], ex["answer_prefix"]),
                              ex["answer"], min(128, ex["max_new_tokens"])))
        for c in cfgs:
            t0 = time.time()
            per, fr = collections.defaultdict(list), []
            for t, i, ids, ans, mx in items:
                out, frac = E.generate(c, ids, mx, ans)
                hits = [a.lower() in out.lower() for a in ans]
                sc = float(any(hits)) if t.startswith("qa") else sum(hits) / len(hits)
                per[t].append(sc)
                fr.append(frac)
                log(task="ruler", config=c, length=L, sub=t, idx=i, prompt_len=len(ids), answers=ans, output=out, score=sc, frac=frac)
            for t, v in sorted(per.items()):
                rows.append({"config": c, "length": L, "task": t, "n": len(v), "score": mean(v)})
            rows.append({"config": c, "length": L, "task": "_frac", "n": len(fr), "score": mean(fr)})
            print(f"[ruler] {c:>14} {L}: mean={mean([mean(v) for v in per.values()]):.3f} frac={mean(fr):.3f} "
                  + " ".join(f"{t}={mean(v):.2f}" for t, v in sorted(per.items())) + f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
    return rows


def task_qa(E, cfgs, n, log):
    pool = PromptPool("qa_eval")
    cells = collections.defaultdict(list)
    for i, m in enumerate(pool.meta):
        cells[(m["source"], m["bucket"])].append(i)
    rows = []
    for c in cfgs:
        for (src, b), idx in sorted(cells.items()):
            sc, fr = [], []
            for i in idx[:n]:
                ans = pool.meta[i]["answers"]
                out, frac = E.generate(c, pool.ids(i), 32, ans)
                r = qa_scores(out, ans)
                sc.append(r)
                fr.append(frac)
                log(task="qa", config=c, source=src, bucket=b, idx=i, question=pool.meta[i]["question"], answers=ans, output=out, frac=frac, **r)
            row = {"config": c, "source": src, "bucket": b, "n": len(sc), "frac": mean(fr),
                   **{k: mean([s[k] for s in sc]) for k in ("f1", "em", "contains")}}
            rows.append(row)
            print(f"[qa] {c:>14} {src:>6} {b}: F1={row['f1']:.3f} EM={row['em']:.3f} contains={row['contains']:.3f} frac={row['frac']:.3f}", flush=True)
    return rows


def task_niah(E, cfgs, log):
    pool = PromptPool("niah_eval")
    rows = []
    for c in cfgs:
        per, fr = collections.defaultdict(list), collections.defaultdict(list)
        for i, m in enumerate(pool.meta):
            out, frac = E.generate(c, pool.ids(i), 16, m["answers"])
            ok = float(m["answers"][0] in out)
            per[m["bucket"]].append(ok)
            fr[m["bucket"]].append(frac)
            log(task="niah", config=c, length=int(m["bucket"]), idx=i, answers=m["answers"], output=out, score=ok, frac=frac)
        for b in sorted(per, key=int):
            rows.append({"config": c, "length": int(b), "n": len(per[b]), "accuracy": mean(per[b]), "frac": mean(fr[b])})
        print(f"[niah] {c:>14}: " + " ".join(f"{b}={mean(per[b]):.2f}" for b in sorted(per, key=int)), flush=True)
    return rows


def task_speed(E, cfgs, lengths, reps):
    rows = []
    g = torch.Generator().manual_seed(0)
    for T in lengths:
        ids = torch.randint(1000, 100000, (T,), generator=g).to(E.dev)
        row = {"prompt_len": T}
        for c in cfgs:
            try:
                row[c] = time_call(lambda: E.prefill(c, ids), repeats=reps, warmup=2)
            except torch.OutOfMemoryError:  # e.g. 32B at long prompts on one GPU: record and go on
                row[c] = None
            torch.cuda.empty_cache()
        rows.append(row)
        print(f"[speed] {T}: " + " ".join(f"{c}=" + ("OOM" if row[c] is None else f"{row[c]*1e3:.1f}ms") for c in cfgs), flush=True)
    return rows


# ----------------------------------------------------------------------------------------------- summary
RULER_GROUPS = {"检索": ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
                       "niah_multikey_3", "niah_multiquery", "niah_multivalue"],
                "汇总": ["cwe", "fwe"], "变量追踪": ["vt"], "问答": ["qa_1", "qa_2"]}


def ruler_mean(res, c, L, tasks=None):
    v = [r["score"] for r in res.get("ruler", []) if r["config"] == c and r["length"] == L and r["task"] != "_frac"
         and (tasks is None or r["task"] in tasks)]
    return mean(v) if v else None


def ruler_frac(res, c, L):
    v = [r["score"] for r in res.get("ruler", []) if r["config"] == c and r["length"] == L and r["task"] == "_frac"]
    return v[0] if v else None


def qa_mean(res, c, b):
    v = [r["f1"] for r in res.get("qa", []) if r["config"] == c and r["bucket"] == b]
    return mean(v) if v else None


def niah_mean(res, c):
    v = [r["accuracy"] for r in res.get("niah", []) if r["config"] == c]
    return mean(v) if v else None


def fmt(x, f="{:.3f}"):
    return "-" if x is None else f.format(x)


def write_summary(res, path):
    cfgs = res["meta"]["configs"]
    L = ["# 选择性重算实验", "",
         f"GPU：{res['meta']['gpu']}；生成：{res['meta']['time']}；projector：混合数据 8k（第 {res['meta'].get('proj_step')} 步）。"
         f"小模型：{C.SMALL_NAME}，大模型：{C.LARGE_NAME}。开头 8 个和最后 64 个 token 始终由大模型计算；“重算比例”是中间部分由大模型重新计算的 token 占比。"
         "“丢弃其余”配置的选择方式与对应的重算配置完全相同，只是没选中的中间 token 被丢掉而不是用投影 KV 补上；"
         "SpecPrefill 是原论文方法（小模型打分，按 32 token 一块保留整段 prompt 的一部分，不强制保留开头和结尾，不用 projector），它的比例是整段 prompt 保留的比例。", ""]
    if os.path.exists(os.path.join(os.path.dirname(path), "figures", "quality_vs_frac.png")):
        L += ["![质量与重算比例](figures/quality_vs_frac.png)", ""]
    if os.path.exists(os.path.join(os.path.dirname(path), "figures", "pareto.png")):
        L += ["![质量与首 token 加速比](figures/pareto.png)", ""]
    L += ["## 总表", "", "| 配置 | 实际重算比例（RULER 8k） | RULER 4k | RULER 8k | 问答 F1 4k | 问答 F1 8k | 大海捞针 |", "|---|---|---|---|---|---|---|"]
    for c in cfgs:
        fr = "全部" if c == "big" else fmt(ruler_frac(res, c, 8192), '{:.1%}')
        L.append(f"| {label(c)} | {fr} | {fmt(ruler_mean(res, c, 4096))} | {fmt(ruler_mean(res, c, 8192))} | "
                 f"{fmt(qa_mean(res, c, '4k'))} | {fmt(qa_mean(res, c, '8k'))} | {fmt(niah_mean(res, c))} |")
    for Ln in (4096, 8192):
        if not any(r["length"] == Ln for r in res.get("ruler", [])):
            continue
        L += ["", f"## RULER {Ln}，按任务类型", "", "| 配置 | " + " | ".join(RULER_GROUPS) + " | 平均 |", "|---|" + "---|" * (len(RULER_GROUPS) + 1)]
        for c in cfgs:
            L.append(f"| {label(c)} | " + " | ".join(fmt(ruler_mean(res, c, Ln, ts), "{:.2f}") for ts in RULER_GROUPS.values())
                     + f" | {fmt(ruler_mean(res, c, Ln))} |")
        tasks = sorted({r["task"] for r in res["ruler"] if r["length"] == Ln and r["task"] != "_frac"})
        L += ["", f"<details><summary>RULER {Ln} 全部子任务</summary>", "", "| 配置 | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
        for c in cfgs:
            L.append(f"| {label(c)} | " + " | ".join(fmt(ruler_mean(res, c, Ln, [t]), "{:.2f}") for t in tasks) + " |")
        L += ["", "</details>"]
    if "qa" in res:
        L += ["", "## 问答 F1", "", "| 配置 | " + " | ".join(f"{s} {b}" for b in ("4k", "8k") for s in ("squad", "hotpot", "trivia")) + " |", "|---|" + "---|" * 6]
        for c in cfgs:
            v = {(r["source"], r["bucket"]): r["f1"] for r in res["qa"] if r["config"] == c}
            if v:
                L.append(f"| {label(c)} | " + " | ".join(fmt(v.get((s, b))) for b in ("4k", "8k") for s in ("squad", "hotpot", "trivia")) + " |")
    if "gsm8k" in res:
        L += ["", "## GSM8K（8-shot，对话轮次形式）", "", "| 配置 | 准确率 | 遵守 #### 格式 |", "|---|---|---|"]
        L += [f"| {label(r['config'])} | {r['accuracy']:.3f} | {r['format_follow']:.3f} |" for r in res["gsm8k"]]
    if "code" in res:
        L += ["", "## 代码下一行补全", "", "| 配置 | 4k 精确匹配 | 4k 相似度 | 8k 精确匹配 | 8k 相似度 |", "|---|---|---|---|---|"]
        for c in cfgs:
            v = {r["bucket"]: r for r in res["code"] if r["config"] == c}
            if v:
                L.append(f"| {label(c)} | " + " | ".join(f"{v[b]['exact_match']:.3f} | {v[b]['similarity']:.3f}" if b in v else "- | -" for b in ("4k", "8k")) + " |")
    if "speed" in res:
        sc = [c for c in res["speed"][0] if c != "prompt_len"]
        L += ["", "## 首 token 延迟（毫秒，H100，eager，不含 CUDA graph）", "", "| 配置 | " + " | ".join(str(r["prompt_len"]) for r in res["speed"]) + " |",
              "|---|" + "---|" * len(res["speed"])]
        for c in sc:
            L.append(f"| {label(c)} | " + " | ".join("显存不足" if r.get(c) is None else f"{r[c]*1e3:.1f}" + (f"（{r['big']/r[c]:.2f} 倍）" if r.get("big") else "")
                                                   for r in res["speed"]) + " |")
        L += ["", f"括号内是相对 {C.LARGE_NAME} 单独的加速比。"]
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
    # one series per (scorer, selection unit); categorical slots of the validated palette in fixed order
    PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    NAME = {"attn_big": f"{C.LARGE_NAME} attention", "attn_small": f"{C.SMALL_NAME} attention", "random": "random", "recent": "recent (longer tail)",
            "spec": "SpecPrefill scores"}
    NAME.update({f"drop-{k}": f"{v}, rest DROPPED" for k, v in list(NAME.items())})
    NAME["specprefill"] = "SpecPrefill (x = % of whole prompt kept)"
    unit_en = lambda u: "" if u is None else ", sentences" if u == "sent" else f", look-ahead {u[3:]}" if str(u).startswith("lah") else f", blocks of {u}"
    cfgs = res["meta"]["configs"]
    series = []
    for c in cfgs:
        sc, f, u = parse(c)
        if sc in NAME and (sc, u) not in series:
            series.append((sc, u))
    series = series[:len(PAL)]
    both = lambda c, ts: (lambda v: mean(v) if v else None)([x for x in (ruler_mean(res, c, L, ts) for L in (4096, 8192)) if x is not None])
    panels = [("RULER 4k (mean of 13 tasks)", lambda c: ruler_mean(res, c, 4096)),
              ("RULER 8k (mean of 13 tasks)", lambda c: ruler_mean(res, c, 8192)),
              ("RULER multikey 2+3 (4k & 8k)", lambda c: both(c, ["niah_multikey_2", "niah_multikey_3"])),
              ("Document QA F1, ~3.5k tokens", lambda c: qa_mean(res, c, "4k")),
              ("Document QA F1, ~7.5k tokens", lambda c: qa_mean(res, c, "8k")),
              ("RULER word counting cwe+fwe (4k & 8k)", lambda c: both(c, ["cwe", "fwe"]))]
    fracs = sorted({parse(c)[1] for c in cfgs if parse(c)[0] in NAME} | {0.0})
    xmax = max(fracs) * 100
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax, (title, val) in zip(axes.flat, panels):
        base = val("hyb")
        for (sc, u), col in zip(series, PAL):
            pts = [(0.0, base)] + sorted((parse(c)[1], val(c)) for c in cfgs if parse(c)[0] == sc and parse(c)[2] == u)
            pts = [(x, y) for x, y in pts if y is not None]
            if len(pts) > 1:
                ax.plot([x * 100 for x, _ in pts], [y for _, y in pts], color=col, lw=2, marker="o", ms=6,
                        mec=SURFACE, mew=1.5, label=NAME[sc] + unit_en(u), zorder=3)
        for ref, txt, ls in (("big", f"{C.LARGE_NAME} alone", "--"), ("oracle", "oracle spans", ":")):
            y = val(ref)
            if y is not None:
                ax.axhline(y, color=TEXT2, lw=1.2, ls=ls, zorder=2)
                ax.text(xmax * 1.03, y, txt, color=TEXT2, fontsize=8, va="center", bbox=dict(fc=SURFACE, ec="none", pad=1))
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"middle tokens recomputed by the {C.LARGE_NAME} (%)")
        ax.set_xticks([x * 100 for x in fracs])
        ax.set_xlim(-0.5, xmax * 1.25)
        ax.set_ylim(0, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("score")
    h, lab = max((ax.get_legend_handles_labels() for ax in axes.flat), key=lambda x: len(x[0]))
    if h:
        fig.legend(h, lab, loc="lower center", ncol=min(4, len(h)), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Quality vs. fraction of middle tokens recomputed ({C.SMALL_NAME} prefill -> {C.LARGE_NAME}, projector: mixed data, 8k)", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    fig.savefig(os.path.join(out_dir, "figures", "quality_vs_frac.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # quality vs measured TTFT speedup over the large model at the same prompt length (needs the speed task)
    sp = {r["prompt_len"]: r for r in res.get("speed", [])}
    pan = [(f"RULER {Ln // 1024}k", Ln, (lambda Ln: lambda c: ruler_mean(res, c, Ln))(Ln)) for Ln in (4096, 8192)] + \
          [(f"Document QA F1, ~{b}", Ln, (lambda b: lambda c: qa_mean(res, c, b))(b)) for b, Ln in (("4k", 4096), ("8k", 8192))]
    pan = [p for p in pan if p[1] in sp and sp[p[1]].get("big")]
    if not pan:
        return
    fig, axes = plt.subplots(1, len(pan), figsize=(4.6 * len(pan), 4.4), squeeze=False)
    for ax, (title, Ln, val) in zip(axes[0], pan):
        row = sp[Ln]
        spd = lambda c: row["big"] / row[c] if row.get(c) else None
        for (sc, u), col in zip(series, PAL):
            pts = sorted((spd(c), val(c), parse(c)[1]) for c in cfgs if parse(c)[0] == sc and parse(c)[2] == u
                         and spd(c) is not None and val(c) is not None)
            if pts:
                ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], color=col, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
                        label=NAME[sc] + unit_en(u), zorder=3)
                for x, y, f in pts:
                    ax.annotate(f"{f:.0%}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=col)
        for ref, mk in (("big", "*"), ("small", "s"), ("hyb", "D")):
            x, y = spd(ref) if ref != "big" else 1.0, val(ref)
            if x is not None and y is not None:
                ax.plot([x], [y], marker=mk, ms=9, color=TEXT2, ls="none", zorder=4)
                ax.annotate({"big": f"{C.LARGE_NAME}", "small": f"{C.SMALL_NAME}", "hyb": "hybrid, 0%"}[ref], (x, y),
                            xytext=(5, -10), textcoords="offset points", fontsize=7.5, color=TEXT2)
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"TTFT speedup over {C.LARGE_NAME} at {Ln} tokens (x)")
        ax.set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("score")
    h, lab = max((ax.get_legend_handles_labels() for ax in axes[0]), key=lambda x: len(x[0]))
    if h:
        fig.legend(h, lab, loc="lower center", ncol=min(3, len(h)), bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Quality vs. TTFT speedup ({C.SMALL_NAME} -> {C.LARGE_NAME}; labels = kept / recomputed fraction)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "pareto.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
    ap.add_argument("--configs", default=",".join(DEFAULT_CFGS))
    ap.add_argument("--speed_configs", default=",".join(SPEED_CFGS))
    ap.add_argument("--tasks", default="niah,ruler,qa,speed")
    ap.add_argument("--n_ruler", type=int, default=15)
    ap.add_argument("--ruler_lengths", default="4096,8192")
    ap.add_argument("--n_qa", type=int, default=50)
    ap.add_argument("--n_gsm8k", type=int, default=150)
    ap.add_argument("--n_code", type=int, default=100)
    ap.add_argument("--speed_lengths", default="4096,8192,16384,32768")
    ap.add_argument("--speed_reps", type=int, default=5)
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "recompute"))
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
    E = Engine(tok, small, large, a.projector, dev, a.n_sink, a.n_tail)
    cfgs = [c for c in a.configs.split(",") if c]
    res["meta"] = {"gpu": torch.cuda.get_device_name(0), "configs": cfgs, "args": vars(a), "proj_step": E.proj_step,
                   "time": time.strftime("%F %T")}
    log = GenLog(os.path.join(a.out_dir, "generations.jsonl"))
    for task in a.tasks.split(","):
        if task in res and not a.force:
            print(f"[task {task}] already done, skipping")
            continue
        t0 = time.time()
        if task == "ruler":
            res["ruler"] = task_ruler(E, cfgs, a.n_ruler, [int(x) for x in a.ruler_lengths.split(",")], log)
        elif task == "qa":
            res["qa"] = task_qa(E, cfgs, a.n_qa, log)
        elif task == "niah":
            res["niah"] = task_niah(E, cfgs, log)
        elif task == "gsm8k":
            res["gsm8k"] = task_gsm8k(V2Adapter(E), cfgs, a.n_gsm8k)
        elif task == "code":
            res["code"] = task_code(V2Adapter(E), cfgs, a.n_code)
        elif task == "speed":
            res["speed"] = task_speed(E, [c for c in a.speed_configs.split(",") if c],
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
