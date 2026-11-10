#!/usr/bin/env python
"""
Round-2 evaluation: 8B alone vs 0.6B alone vs hybrid prefill with several projectors, on held-out data of the
same kinds as the round-2 training mixture plus standard benchmarks. Uses the fast prefill path (prefill/fast.py).

Tasks (each stored in results.json as soon as it finishes; finished tasks are skipped on restart):
  kl       KL(8B || config) and top-1 agreement on the 8B's own held-out responses, per data type and length bucket
  qa       SQuAD v2 / HotpotQA / TriviaQA validation questions at ~3.5k and ~7.5k tokens: F1, EM, answer-contained
  ruler    RULER (simonjegou/ruler) 4096 and 8192, all 13 tasks
  niah     round-1 passcode needle test at 2048 / 3584 / 7680
  gsm8k    GSM8K test, 8-shot as chat turns
  code     next-line exact match on held-out repositories at ~3.5k and ~7.5k tokens
  curve    retrieval accuracy and KL vs training step for the mix-4k and control runs (intermediate checkpoints)
"""
import argparse
import collections
import difflib
import glob
import json
import os
import random
import re
import statistics
import string
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import TokenStore  # noqa: E402
from prefill.fast import FastBig, FastHybrid, FastQwen3, FoldedProjectorKV, kv_to_dynamic_cache  # noqa: E402
from prefill.kv import greedy_decode, kl_teacher_student, teacher_forward, true_prefill  # noqa: E402
from prefill.xteacher import NativeTeacher  # noqa: E402
from prefill.mixdata import PromptPool, Tok  # noqa: E402
from prefill.models import eos_ids, load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402


# ----------------------------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, tok, small, large, dev, n_sink, n_tail):
        self.tok, self.small, self.large, self.dev = tok, small, large, dev
        self.T = Tok(tok)
        self.S, self.L = FastQwen3(small), FastQwen3(large)
        self.fb = FastBig(self.L)
        self.hyb = {}
        self.n_sink, self.n_tail = n_sink, n_tail
        self.eos = eos_ids(tok)

    def add_projector(self, tag, path):
        p, ck = Projector.load(path, self.dev)
        p.eval()
        self.hyb[tag] = (p, FastHybrid(self.S, self.L, FoldedProjectorKV(p, self.L), self.n_sink, self.n_tail))
        return ck.get("step")

    def drop(self, tag):
        self.hyb.pop(tag, None)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(self, cfg, ids, max_new):
        ids = torch.as_tensor(ids, device=self.dev)
        if cfg == "small":
            cache, lg = true_prefill(self.small, ids[None])
            out = greedy_decode(self.small, cache, lg, max_new, self.eos)
        else:
            kb, vb, lg = self.fb.prefill(ids) if cfg == "big" else self.hyb[cfg][1].prefill(ids)
            out = greedy_decode(self.large, kv_to_dynamic_cache(kb, vb), lg, max_new, self.eos)
        return self.tok.decode(out, skip_special_tokens=True)

    @torch.no_grad()
    def cont_logits(self, cfg, ids, P):
        """[m, V] logits predicting ids[P:] given the prompt ids[:P]."""
        if cfg == "big":
            return teacher_forward(self.large, ids[None], P)[0]
        if cfg == "small":
            return teacher_forward(self.small, ids[None], P)[0]
        kb, vb, lg0 = self.hyb[cfg][1].prefill(ids[:P])
        m = ids.shape[0] - P
        if m == 1:
            return lg0
        out = self.large(input_ids=ids[None, P:-1], past_key_values=kv_to_dynamic_cache(kb, vb), use_cache=True, logits_to_keep=m - 1)
        return torch.cat([lg0, out.logits[0]], 0)

    def chat_ids(self, messages, assistant_prefix=""):
        ids, _ = self.T.chat(messages)
        return ids + (self.T.enc(assistant_prefix) if assistant_prefix else [])


# ----------------------------------------------------------------------------------------------- metrics
def norm_ans(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = norm_ans(pred).split(), norm_ans(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def qa_scores(pred, answers):
    pred1 = pred.strip().split("\n")[0]
    return {"f1": max(f1(pred1, a) for a in answers), "em": float(max(norm_ans(pred1) == norm_ans(a) for a in answers)),
            "contains": float(any(norm_ans(a) in norm_ans(pred) for a in answers if norm_ans(a)))}


def gsm_extract(text):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    s = m.group(1) if m else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
    try:
        return float(s.replace(",", "")) if s else None
    except ValueError:
        return None


# KL is measured against the teacher's own full prefill: "big" (default) or "small" for reverse pairs (config.TEACHER)
TEACHER_CFG = "small" if C.TEACHER == "small" else "big"


def teacher_logits(E, st, pl, native, i):
    """Teacher logits [m, V] for sd_mixtest sequence i + mask of the positions that count ([m] bool, or None).
    Cross-family reverse pair (native is not None): the teacher reads its own original sequence (prefill/xteacher.py)."""
    if native is not None:
        t, msk = native.logits(E.small, i, E.dev)
        return t[0], msk[0]
    return E.cont_logits(TEACHER_CFG, st.get(i).to(E.dev), int(pl[i])), None


def mmean(x, msk):
    return (x[msk] if msk is not None else x).float().mean().item()


def mean(x):
    return statistics.mean(x) if x else float("nan")


# ----------------------------------------------------------------------------------------------- tasks
def task_kl(E, cfgs, n_per_cell):
    st = TokenStore("sd_mixtest", "all")
    pl = np.load(os.path.join(C.DATA_DIR, "sd_mixtest", "prompt_len.npy"))
    meta = [json.loads(l) for l in open(os.path.join(C.DATA_DIR, "sd_mixtest", "meta.jsonl"))]
    acc = collections.defaultdict(lambda: {"kl": [], "agree": []})
    cnt = collections.Counter()
    native = NativeTeacher.load("sd_mixtest") if C.TEACHER == "small" else None
    for i, m in enumerate(meta):
        cell = (m["type"], m["bucket"])
        if cnt[cell] >= n_per_cell:
            continue
        cnt[cell] += 1
        ids = st.get(i).to(E.dev)
        P = int(pl[i])
        t, msk = teacher_logits(E, st, pl, native, i)
        for c in cfgs:
            if c == TEACHER_CFG:
                continue
            s = E.cont_logits(c, ids, P)
            acc[(c,) + cell]["kl"].append(mmean(kl_teacher_student(t[None], s[None])[0], msk))
            acc[(c,) + cell]["agree"].append(mmean(t.argmax(-1) == s.argmax(-1).to(t.device), msk))
    rows = [{"config": c, "type": t, "bucket": b, "n": len(v["kl"]), "kl": mean(v["kl"]), "top1_agree": mean(v["agree"])}
            for (c, t, b), v in sorted(acc.items())]
    for r in rows:
        print(f"[kl] {r['config']:>8} {r['type']:>9} {r['bucket']}: KL={r['kl']:.4f} agree={r['top1_agree']:.3f}", flush=True)
    return rows


def task_qa(E, cfgs, n):
    pool = PromptPool("qa_eval")
    rows = []
    cells = collections.defaultdict(list)
    for i, m in enumerate(pool.meta):
        cells[(m["source"], m["bucket"])].append(i)
    for c in cfgs:
        for (src, b), idx in sorted(cells.items()):
            sc = [qa_scores(E.generate(c, pool.ids(i), 32), pool.meta[i]["answers"]) for i in idx[:n]]
            r = {"config": c, "source": src, "bucket": b, "n": len(sc), **{k: mean([s[k] for s in sc]) for k in ("f1", "em", "contains")}}
            rows.append(r)
            print(f"[qa] {c:>8} {src:>6} {b}: F1={r['f1']:.3f} EM={r['em']:.3f} contains={r['contains']:.3f}", flush=True)
    return rows


def task_ruler(E, cfgs, n_per_task, lengths):
    from datasets import load_dataset
    rows = []
    for L in lengths:
        ds = load_dataset("simonjegou/ruler", str(L), split="test")
        by = collections.defaultdict(list)
        for i, t in enumerate(ds["task"]):
            by[t].append(i)
        rng = random.Random(L)
        items = []
        for t, idx in sorted(by.items()):
            for i in rng.sample(idx, min(n_per_task, len(idx))):
                ex = ds[i]
                items.append((t, E.chat_ids([{"role": "user", "content": ex["context"] + ex["question"]}], ex["answer_prefix"]),
                              ex["answer"], min(128, ex["max_new_tokens"])))
        for c in cfgs:
            per = collections.defaultdict(list)
            for t, ids, ans, mx in items:
                out = E.generate(c, ids, mx).lower()
                hits = [a.lower() in out for a in ans]
                per[t].append(float(any(hits)) if t.startswith("qa") else sum(hits) / len(hits))
            for t, v in sorted(per.items()):
                rows.append({"config": c, "length": L, "task": t, "n": len(v), "score": mean(v)})
            print(f"[ruler] {c:>8} {L}: mean={mean([mean(v) for v in per.values()]):.3f} " +
                  " ".join(f"{t}={mean(v):.2f}" for t, v in sorted(per.items())), flush=True)
    return rows


def task_niah(E, cfgs):
    pool = PromptPool("niah_eval")
    rows = []
    for c in cfgs:
        per = collections.defaultdict(list)
        for i, m in enumerate(pool.meta):
            per[m["bucket"]].append(float(m["answers"][0] in E.generate(c, pool.ids(i), 16)))
        for b, v in sorted(per.items(), key=lambda x: int(x[0])):
            rows.append({"config": c, "length": int(b), "n": len(v), "accuracy": mean(v)})
        print(f"[niah] {c:>8}: " + " ".join(f"{b}={mean(v):.2f}" for b, v in sorted(per.items(), key=lambda x: int(x[0]))), flush=True)
    return rows


def task_gsm8k(E, cfgs, n, shots=8):
    from datasets import load_dataset
    tr = load_dataset("openai/gsm8k", "main", split="train")
    te = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(0)
    fs = [tr[i] for i in rng.sample(range(len(tr)), shots)]
    msgs = []
    for ex in fs:
        msgs += [{"role": "user", "content": ex["question"]}, {"role": "assistant", "content": ex["answer"]}]
    items = [(E.chat_ids(msgs + [{"role": "user", "content": te[i]["question"]}]), float(te[i]["answer"].split("####")[-1].strip().replace(",", "")))
             for i in range(n)]
    rows = []
    for c in cfgs:
        ok, fmt = 0, 0
        for ids, gold in items:
            out = E.generate(c, ids, 320)
            p = gsm_extract(out)
            ok += p is not None and abs(p - gold) < 1e-4
            fmt += "####" in out
        rows.append({"config": c, "n": n, "accuracy": ok / n, "format_follow": fmt / n, "prompt_tokens": mean([len(i) for i, _ in items])})
        print(f"[gsm8k] {c:>8}: acc={ok/n:.3f} format(####)={fmt/n:.3f}", flush=True)
    return rows


def task_code(E, cfgs, n):
    pool = PromptPool("code_eval")
    rows = []
    for c in cfgs:
        per = collections.defaultdict(lambda: {"em": [], "sim": []})
        for i, m in enumerate(pool.meta):
            if len(per[m["bucket"]]["em"]) >= n:
                continue
            line = E.generate(c, pool.ids(i), 48).split("\n")[0].strip()
            per[m["bucket"]]["em"].append(float(line == m["target"]))
            per[m["bucket"]]["sim"].append(difflib.SequenceMatcher(None, line, m["target"]).ratio())
        for b, v in sorted(per.items()):
            rows.append({"config": c, "bucket": b, "n": len(v["em"]), "exact_match": mean(v["em"]), "similarity": mean(v["sim"])})
            print(f"[code] {c:>8} {b}: EM={mean(v['em']):.3f} sim={mean(v['sim']):.3f}", flush=True)
    return rows


def task_curve(E, runs, n_kl, n_ret):
    """For each run directory: every projector_step*.pt (+ final) -> mean KL (4k bucket) and retrieval accuracy."""
    from datasets import load_dataset
    st = TokenStore("sd_mixtest", "all")
    pl = np.load(os.path.join(C.DATA_DIR, "sd_mixtest", "prompt_len.npy"))
    meta = [json.loads(l) for l in open(os.path.join(C.DATA_DIR, "sd_mixtest", "meta.jsonl"))]
    kl_idx, cnt = [], collections.Counter()
    for i, m in enumerate(meta):
        if m["bucket"] == "4k" and cnt[m["type"]] < n_kl:
            kl_idx.append(i)
            cnt[m["type"]] += 1
    native = NativeTeacher.load("sd_mixtest") if C.TEACHER == "small" else None
    teacher = {}
    for i in kl_idx:
        t, msk = teacher_logits(E, st, pl, native, i)
        teacher[i] = (t.cpu(), msk)
    ds = load_dataset("simonjegou/ruler", "4096", split="test")
    ret = []
    for t in ("niah_single_2", "niah_multikey_1", "niah_multivalue", "niah_single_3"):
        idx = [i for i, x in enumerate(ds["task"]) if x == t][:n_ret]
        ret += [(E.chat_ids([{"role": "user", "content": ds[i]["context"] + ds[i]["question"]}], ds[i]["answer_prefix"]), ds[i]["answer"]) for i in idx]
    niah = PromptPool("niah_eval")
    ret += [(niah.ids(i), m["answers"]) for i, m in enumerate(niah.meta) if m["bucket"] == "3584"][:n_ret]
    rows = []
    for run, d in runs.items():
        ckpts = sorted(glob.glob(os.path.join(d, "projector_step*.pt")), key=lambda p: int(re.findall(r"step(\d+)", p)[0]))
        for path in ckpts:
            step = E.add_projector("_curve", path)
            kls = []
            for i in kl_idx:
                ids = st.get(i).to(E.dev)
                s = E.cont_logits("_curve", ids, int(pl[i]))
                kls.append(mmean(kl_teacher_student(teacher[i][0].to(E.dev)[None], s[None])[0], teacher[i][1]))
            acc = []
            for ids, ans in ret:
                out = E.generate("_curve", ids, 64).lower()
                acc.append(sum(a.lower() in out for a in ans) / len(ans))
            rows.append({"run": run, "step": step, "kl_4k": mean(kls), "retrieval_acc": mean(acc)})
            print(f"[curve] {run} step {step}: KL={mean(kls):.4f} retrieval={mean(acc):.3f}", flush=True)
            E.drop("_curve")
    return rows


# ----------------------------------------------------------------------------------------------- summary
LABEL = {"big": f"{C.LARGE_NAME} 单独", "small": f"{C.SMALL_NAME} 单独", "r1": "混合-第一轮", "ctrl4k": "混合-对照(网页,4k)", "mix4k": "混合-混合数据(4k)", "mix8k": "混合-混合数据(8k)", "s17mix8k": "混合-1.7B 小模型，混合数据(8k)", "s17to32": "混合-1.7B→32B，混合数据(8k)", "cont": "混合，再训练 12000 步（共 24000 步）", "more32": "混合-1.7B→32B，再训练 24000 步（共 48000 步）", "normA": "混合-1.7B，输入归一化", "normB": "混合-1.7B，输入归一化 + 注意力匹配损失", "s17to32n": "混合-1.7B→32B，输入归一化"}


def write_summary(res, path):
    cf = res["meta"]["configs"]
    lab = lambda c: LABEL.get(c, c)
    L = ["# 第二轮实验结果", "", f"GPU: {res['meta']['gpu']}；生成：{res['meta']['time']}。所有测试数据都没有参与训练。", "",
         "配置：" + "，".join(f"`{c}` = {lab(c)}" for c in cf), ""]
    if "figures" in res:
        L += ["## 图", ""] + [f"![{t}](figures/{f})" for f, t in res["figures"]] + [""]
    if "kl" in res:
        types = sorted({r["type"] for r in res["kl"]})
        for b in ("4k", "8k"):
            L += [f"## 与 {C.TEACHER_NAME} 的 KL（{b} prompt，越低越好）", "", "| 配置 | " + " | ".join(types) + " | 平均 |", "|---|" + "---|" * (len(types) + 1)]
            for c in cf:
                v = {r["type"]: r["kl"] for r in res["kl"] if r["config"] == c and r["bucket"] == b}
                if v:
                    L.append(f"| {lab(c)} | " + " | ".join(f"{v.get(t, float('nan')):.3f}" for t in types) + f" | {mean(list(v.values())):.3f} |")
            L.append("")
    if "qa" in res:
        L += ["## 问答 F1", "", "| 配置 | " + " | ".join(f"{s} {b}" for b in ("4k", "8k") for s in ("squad", "hotpot", "trivia")) + " |", "|---|" + "---|" * 6]
        for c in cf:
            v = {(r["source"], r["bucket"]): r["f1"] for r in res["qa"] if r["config"] == c}
            if v:
                L.append(f"| {lab(c)} | " + " | ".join(f"{v.get((s, b), float('nan')):.3f}" for b in ("4k", "8k") for s in ("squad", "hotpot", "trivia")) + " |")
        L.append("")
    if "ruler" in res:
        tasks = sorted({r["task"] for r in res["ruler"]})
        for Ln in sorted({r["length"] for r in res["ruler"]}):
            L += [f"## RULER {Ln}", "", "| 配置 | " + " | ".join(tasks) + " | 平均 |", "|---|" + "---|" * (len(tasks) + 1)]
            for c in cf:
                v = {r["task"]: r["score"] for r in res["ruler"] if r["config"] == c and r["length"] == Ln}
                if v:
                    L.append(f"| {lab(c)} | " + " | ".join(f"{v.get(t, float('nan')):.2f}" for t in tasks) + f" | {mean(list(v.values())):.3f} |")
            L.append("")
    if "niah" in res:
        lens = sorted({r["length"] for r in res["niah"]})
        L += ["## 大海捞针（第一轮模板）", "", "| 配置 | " + " | ".join(map(str, lens)) + " |", "|---|" + "---|" * len(lens)]
        for c in cf:
            v = {r["length"]: r["accuracy"] for r in res["niah"] if r["config"] == c}
            if v:
                L.append(f"| {lab(c)} | " + " | ".join(f"{v[x]:.2f}" for x in lens) + " |")
        L.append("")
    if "gsm8k" in res:
        L += ["## GSM8K（8-shot，对话轮次形式）", "", "| 配置 | 准确率 | 遵守 #### 格式 |", "|---|---|---|"]
        L += [f"| {lab(r['config'])} | {r['accuracy']:.3f} | {r['format_follow']:.3f} |" for r in res["gsm8k"]] + [""]
    if "code" in res:
        L += ["## 代码下一行补全", "", "| 配置 | 4k 精确匹配 | 4k 相似度 | 8k 精确匹配 | 8k 相似度 |", "|---|---|---|---|---|"]
        for c in cf:
            v = {r["bucket"]: r for r in res["code"] if r["config"] == c}
            if v:
                L.append(f"| {lab(c)} | " + " | ".join(f"{v[b]['exact_match']:.3f} | {v[b]['similarity']:.3f}" if b in v else "- | -" for b in ("4k", "8k")) + " |")
        L.append("")
    if "curve" in res:
        L += ["## 训练步数的影响（4k）", "", "| 训练 | 步数 | KL | 检索准确率 |", "|---|---|---|---|"]
        L += [f"| {lab(r['run'])} | {r['step']} | {r['kl_4k']:.4f} | {r['retrieval_acc']:.3f} |" for r in res["curve"]] + [""]
    open(path, "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ck = C.CKPT_DIR
    ap.add_argument("--projectors", default=f"r1={ck}/stage2/projector.pt,ctrl4k={ck}/r2_ctrl4k/projector.pt,"
                                            f"mix4k={ck}/r2_mix4k/projector.pt,mix8k={ck}/r2_mix8k/projector.pt")
    ap.add_argument("--curve_runs", default=f"mix4k={ck}/r2_mix4k,ctrl4k={ck}/r2_ctrl4k")
    ap.add_argument("--tasks", default="kl,qa,ruler,niah,code,gsm8k,curve")
    ap.add_argument("--n_kl", type=int, default=40)
    ap.add_argument("--n_qa", type=int, default=100)
    ap.add_argument("--n_ruler", type=int, default=15)
    ap.add_argument("--ruler_lengths", default="4096,8192")
    ap.add_argument("--n_gsm8k", type=int, default=150)
    ap.add_argument("--n_code", type=int, default=100)
    ap.add_argument("--n_curve_kl", type=int, default=6)
    ap.add_argument("--n_curve_ret", type=int, default=12)
    ap.add_argument("--baselines", default="big,small", help="large/small alone configs to evaluate (empty = skip)")
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "round2"))
    ap.add_argument("--force", action="store_true", help="recompute tasks already in results.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)
    tok, small, large = load_pair(dev)
    E = Engine(tok, small, large, dev, a.n_sink, a.n_tail)
    cfgs = [c for c in a.baselines.split(",") if c]
    for item in [x for x in a.projectors.split(",") if x]:
        tag, path = item.split("=")
        if os.path.exists(path):
            print(f"[projector] {tag}: {path} step {E.add_projector(tag, path)}", flush=True)
            cfgs.append(tag)
        else:
            print(f"[projector] {tag}: {path} missing, skipped", flush=True)
    res_path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["meta"] = {"gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "configs": cfgs,
                   "args": vars(a), "time": time.strftime("%F %T")}
    runs = {k: v for k, v in (x.split("=") for x in a.curve_runs.split(",") if x) if os.path.isdir(v)}
    for task in a.tasks.split(","):
        if task in res and not a.force:
            print(f"[task {task}] already done, skipping")
            continue
        t0 = time.time()
        if task == "kl":
            res["kl"] = task_kl(E, cfgs, a.n_kl)
        elif task == "qa":
            res["qa"] = task_qa(E, cfgs, a.n_qa)
        elif task == "ruler":
            res["ruler"] = task_ruler(E, cfgs, a.n_ruler, [int(x) for x in a.ruler_lengths.split(",")])
        elif task == "niah":
            res["niah"] = task_niah(E, cfgs)
        elif task == "gsm8k":
            res["gsm8k"] = task_gsm8k(E, cfgs, a.n_gsm8k)
        elif task == "code":
            res["code"] = task_code(E, cfgs, a.n_code)
        elif task == "curve":
            res["curve"] = task_curve(E, runs, a.n_curve_kl, a.n_curve_ret)
        print(f"[task {task}] {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
        write_summary(res, os.path.join(a.out_dir, "summary.md"))
    print("[done]", a.out_dir)


if __name__ == "__main__":
    main()
