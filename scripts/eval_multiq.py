#!/usr/bin/env python
"""
Shared document, several questions (prefill/multiq.py): quality and per-question latency of a reusable cache.

Tasks (results.json; finished tasks are skipped on restart):
  qa     multiq_eval pool (scripts/prepare_multiq.py): the document set is prefilled once per config, the questions are
         asked in order from that cache; F1 / EM / contains per bucket, and by question index (1st vs later) because
         spec-q1's cache is selected for the first question only
  speed  document prefill time (once) and time to the first token of a question (extension + selection/recompute,
         no decoding), for the first question and for later questions, at several document lengths; random tokens,
         40-token question
Configs: see prefill/multiq.py. Generations go to generations.jsonl.
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

from eval_recompute import Engine, GenLog, fmt  # noqa: E402
from evaluate_v2 import mean, qa_scores  # noqa: E402
from prefill import config as C  # noqa: E402
from prefill.fast import time_call  # noqa: E402
from prefill.mcq import choose_letter, letter_token_ids  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.multiq import MultiQ, parse_cfg, per_question_frac  # noqa: E402

DEFAULT_CFGS = ["big", "small", "hyb", "ours:0.1:sent", "ours-acc:0.1:sent", "ours-each:0.1:sent",
                "spec-agn:0.1", "spec-agn:0.2", "spec-agn:0.3", "spec-q1:0.1", "spec-q1:0.2", "spec-each:0.1"]
SPEED_CFGS = ["big", "hyb", "ours:0.1:sent", "ours-each:0.1:sent", "spec-agn:0.1", "spec-agn:0.2", "spec-q1:0.1", "spec-each:0.1"]


def label(cfg):
    s, f, u = parse_cfg(cfg)
    g = per_question_frac(cfg)
    unit = "" if u is None else "，按句子" if u == "sent" else f"，{u} token 一块"
    return {"big": f"{C.LARGE_NAME} 单独（缓存复用）", "small": f"{C.SMALL_NAME} 单独（缓存复用）", "hyb": "投影缓存，不重算",
            "ours": f"投影缓存 + 每问重算 {f:.0%}{unit}（不保留）", "ours-acc": f"投影缓存 + 每问重算 {f:.0%}{unit}（累积保留）",
            "pre": f"投影缓存 + 文档阶段重算 {f:.0%}{unit}" + (f" + 每问补 {g:.0%}（累积）" if g else "（每问不重算）"),
            "ours-each": f"我们，每问重新 prefill，重算 {f:.0%}{unit}", "spec-agn": f"SpecPrefill 不看问题，保留 {f:.0%}",
            "spec-q1": f"SpecPrefill 按第 1 问选择，保留 {f:.0%}", "spec-each": f"SpecPrefill 每问重新 prefill，保留 {f:.0%}"}[s]


QBINS = ((1, 1), (2, 5), (6, 10), (11, 20))  # question index ranges (1-based) reported for long question sequences


def task_qa(mq, tok, cfgs, dev, log, max_new=32, pool="multiq_eval", letter_ids=None):
    """Questions with "answers" are generated and scored by F1 / EM / contains; multiple-choice questions ("gold",
    QuALITY) are scored from the first-token logits (prefill/mcq.py) and the accuracy fills all three keys."""
    pool = PromptPool(pool)
    long_seq = max(len(m["questions"]) for m in pool.meta) > 5
    rows = []
    for c in cfgs:
        t0 = time.time()
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for i, m in enumerate(pool.meta):
            st = mq.doc(c, torch.as_tensor(pool.ids(i), device=dev))
            for j, q in enumerate(m["questions"]):
                q_ids = torch.as_tensor(q["ids"], device=dev)
                if "gold" in q:
                    _, _, lg, frac, _ = mq.ask_prefill(st, q_ids)
                    pred = choose_letter(lg, letter_ids)
                    text = "ABCD"[pred]
                    r = {k: float(pred == q["gold"]) for k in ("f1", "em", "contains")}
                else:
                    out, frac = mq.ask(st, q_ids, max_new)
                    text = tok.decode(out, skip_special_tokens=True)
                    r = qa_scores(text, q["answers"])
                for key in ("f1", "em", "contains"):
                    acc[(m["bucket"], "all")][key].append(r[key])
                    acc[(m["bucket"], "q1" if j == 0 else "later")][key].append(r[key])
                    if q.get("difficult"):
                        acc[(m["bucket"], "hard")][key].append(r[key])
                    if long_seq:
                        lo, hi = next(bn for bn in QBINS if bn[0] <= j + 1 <= bn[1])
                        acc[(m["bucket"], f"q{lo}-{hi}")][key].append(r[key])
                acc[(m["bucket"], "all")]["frac"].append(frac)
                log(task="qa", config=c, bucket=m["bucket"], doc=i, q=j, question=q["question"], answers=q.get("answers", q.get("gold")),
                    output=text, frac=frac, **r)
            del st
        for (b, part), v in sorted(acc.items()):
            rows.append({"config": c, "bucket": b, "part": part, "n": len(v["f1"]), **{k: mean(v[k]) for k in ("f1", "em", "contains")},
                         "frac": mean(v["frac"]) if v["frac"] else None})
        msg = " ".join(f"{b}:{mean(acc[(b, 'all')]['f1']):.3f}(q1 {mean(acc[(b, 'q1')]['f1']):.3f}/later {mean(acc[(b, 'later')]['f1']):.3f})"
                       for b in sorted({b for b, _ in acc}))
        print(f"[qa] {c:>18}: F1 {msg}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        torch.cuda.empty_cache()
    return rows


def task_speed(mq, cfgs, dev, lengths, reps, n_qtok=40, n_seq=0):
    """n_seq > 0: also time a sequence of n_seq different questions on one document cache (cumulative configs get
    cheaper / change as their cache fills), twice, element-wise mean -> row[c]["seq"] [n_seq] seconds."""
    rows = []
    g = torch.Generator().manual_seed(0)
    for T in lengths:
        ids = torch.randint(1000, 100000, (T,), generator=g).to(dev)
        q = torch.randint(1000, 100000, (n_qtok,), generator=g).to(dev)
        qseq = [torch.randint(1000, 100000, (n_qtok,), generator=g).to(dev) for _ in range(n_seq)]
        row = {"prompt_len": T}
        for c in cfgs:
            try:
                t_doc = time_call(lambda: mq.doc(c, ids), repeats=reps, warmup=1)
                st = mq.doc(c, ids)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mq.ask_prefill(st, q)
                torch.cuda.synchronize()
                t_first = time.perf_counter() - t0
                t_later = time_call(lambda: mq.ask_prefill(st, q), repeats=reps, warmup=1)
                row[c] = {"doc": t_doc, "first": t_first, "later": t_later}
                del st
                if n_seq:
                    runs = []
                    for _ in range(2):
                        st, ts = mq.doc(c, ids), []
                        for qq in qseq:
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            mq.ask_prefill(st, qq)
                            torch.cuda.synchronize()
                            ts.append(time.perf_counter() - t0)
                        runs.append(ts)
                        del st
                    row[c]["seq"] = [(x + y) / 2 for x, y in zip(*runs)]
            except torch.OutOfMemoryError:
                row[c] = None
            torch.cuda.empty_cache()
        rows.append(row)
        print(f"[speed] {T}: " + " ".join(f"{c}=" + ("OOM" if row[c] is None else f"{row[c]['doc']*1e3:.0f}+{row[c]['first']*1e3:.0f}/{row[c]['later']*1e3:.0f}ms")
                                        for c in cfgs), flush=True)
    return rows


def write_summary(res, path):
    cfgs = res["meta"]["configs"]
    L = ["# 共享文档、多个问题", "",
         f"GPU：{res['meta']['gpu']}；生成：{res['meta']['time']}；小模型 {C.SMALL_NAME}，大模型 {C.LARGE_NAME}，projector 第 {res['meta'].get('proj_step')} 步。",
         "每个文档集（SQuAD 验证集 3 段有答案的段落 + 干扰段落，约 3.5k / 7.5k token）只 prefill 一次，然后依次回答最多 5 个问题；"
         "答案分布在不同段落里。“每问重新 prefill”的配置不复用任何缓存，是单问题基准里的用法。", ""]
    if "qa" in res:
        get = lambda c, b, p, k: next((r[k] for r in res["qa"] if r["config"] == c and r["bucket"] == b and r["part"] == p), None)
        L += ["## 问答 F1", "", "| 配置 | 大模型计算比例 | 4k 全部 | 4k 第 1 问 | 4k 后续问 | 8k 全部 | 8k 第 1 问 | 8k 后续问 | 4k 包含答案 | 8k 包含答案 |", "|---|" + "---|" * 9]
        for c in cfgs:
            if get(c, "4k", "all", "f1") is None and get(c, "8k", "all", "f1") is None:
                continue
            fr = get(c, "8k", "all", "frac")
            L.append(f"| {label(c)} | {fmt(fr, '{:.1%}')} | " + " | ".join(fmt(get(c, b, p, "f1")) for b in ("4k", "8k") for p in ("all", "q1", "later"))
                     + f" | {fmt(get(c, '4k', 'all', 'contains'))} | {fmt(get(c, '8k', 'all', 'contains'))} |")
        L += ["", "“大模型计算比例”：我们 = 每个问题重算的中间 token 比例；SpecPrefill = 保留的文档 token 比例（8k）。"]
        if any(r["part"] == "hard" for r in res["qa"]):
            L += ["", "池是 QuALITY（4 选 1，按首 token 的字母 logits 判分，上表的 F1 / 包含答案都是准确率）。hard 题（需要通读全文）：", "",
                  "| 配置 | 4k 全部 | 4k hard | 8k 全部 | 8k hard |", "|---|---|---|---|---|"]
            for c in cfgs:
                if get(c, "4k", "all", "f1") is not None or get(c, "8k", "all", "f1") is not None:
                    L.append(f"| {label(c)} | " + " | ".join(fmt(get(c, b, p, "f1")) for b in ("4k", "8k") for p in ("all", "hard")) + " |")
        bins = [f"q{lo}-{hi}" for lo, hi in QBINS if any(r["part"] == f"q{lo}-{hi}" for r in res["qa"])]
        if bins:
            L += ["", "### 按问题序号的 F1（第几个问题；缓存累积的配置越往后缓存越完整）", "",
                  "| 配置 | " + " | ".join(f"{b} {x}" for b in ("4k", "8k") for x in bins) + " |", "|---|" + "---|" * (2 * len(bins))]
            for c in cfgs:
                if get(c, "4k", "all", "f1") is not None:
                    L.append(f"| {label(c)} | " + " | ".join(fmt(get(c, b, x, "f1")) for b in ("4k", "8k") for x in bins) + " |")
    if "speed" in res:
        sc = [c for c in res["speed"][0] if c != "prompt_len"]
        L += ["", f"## 延迟（毫秒，{res['meta']['gpu'].replace('NVIDIA ', '')}，eager）", "",
              "每格：文档 prefill（一次）+ 第 1 问首 token / 后续问首 token。“每问重新 prefill”的配置文档 prefill 为 0，每问都是完整开销。", "",
              "| 配置 | " + " | ".join(str(r["prompt_len"]) for r in res["speed"]) + " |", "|---|" + "---|" * len(res["speed"])]
        for c in sc:
            L.append(f"| {label(c)} | " + " | ".join("显存不足" if r.get(c) is None else f"{r[c]['doc']*1e3:.0f} + {r[c]['first']*1e3:.0f} / {r[c]['later']*1e3:.0f}"
                                                   for r in res["speed"]) + " |")
        L += ["", "| 配置 | " + " | ".join(f"{r['prompt_len']}：5 问总时间（倍）" for r in res["speed"]) + " |", "|---|" + "---|" * len(res["speed"])]
        tot = lambda v: v["doc"] + v["first"] + 4 * v["later"]
        for c in sc:
            L.append(f"| {label(c)} | " + " | ".join("-" if r.get(c) is None else f"{tot(r[c])*1e3:.0f}（{tot(r['big'])/tot(r[c]):.2f}）" for r in res["speed"]) + " |")
        L += ["", f"括号内：相对 {C.LARGE_NAME} 单独（缓存复用）的加速比，按“文档 prefill + 5 个问题的首 token”总时间算。"]
        if all("seq" in r[c] for r in res["speed"] for c in sc if r.get(c)):
            ns = [n for n in (1, 5, 10, 15, 20) if n <= len(res["speed"][0]["big"]["seq"])]
            L += ["", f"### 连续 N 个不同问题的总时间（文档 prefill + N 个问题的首 token，毫秒；括号内相对 {C.LARGE_NAME} 复用缓存的加速比）", ""]
            for r in res["speed"]:
                tot = lambda v, n: v["doc"] + sum(v["seq"][:n])
                L += [f"**{r['prompt_len']} token**", "", "| 配置 | " + " | ".join(f"{n} 问" for n in ns) + " |", "|---|" + "---|" * len(ns)]
                for c in sc:
                    if r.get(c):
                        L.append(f"| {label(c)} | " + " | ".join(f"{tot(r[c], n)*1e3:.0f}（{tot(r['big'], n)/tot(r[c], n):.2f}）" for n in ns) + " |")
                L.append("")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
    ap.add_argument("--configs", default=",".join(DEFAULT_CFGS))
    ap.add_argument("--speed_configs", default=",".join(SPEED_CFGS))
    ap.add_argument("--tasks", default="qa,speed")
    ap.add_argument("--speed_lengths", default="4096,8192,16384")
    ap.add_argument("--speed_reps", type=int, default=5)
    ap.add_argument("--speed_nq", type=int, default=0, help="also time a sequence of this many questions per document")
    ap.add_argument("--pool", default="multiq_eval")
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--q_max", type=int, default=128)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "multiq"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--redo", default="", help="qa: configs whose rows are discarded and re-run (others are kept)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res_path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    E = Engine(tok, small, large, a.projector, dev, a.n_sink, a.n_tail)
    mq = MultiQ(E.rh, large, small, E.eos, q_max=a.q_max)
    cfgs = [c for c in a.configs.split(",") if c]
    res["meta"] = {"gpu": torch.cuda.get_device_name(0), "configs": cfgs, "args": vars(a), "proj_step": E.proj_step, "time": time.strftime("%F %T")}
    log = GenLog(os.path.join(a.out_dir, "generations.jsonl"))
    redo = {c for c in a.redo.split(",") if c}
    for task in a.tasks.split(","):
        if task == "qa":  # per config: keep finished configs, run the missing / redone ones (--force: redo all)
            keep = [] if a.force else [r for r in res.get("qa", []) if r["config"] not in redo]
            todo = [c for c in cfgs if c not in {r["config"] for r in keep}]
            if not todo:
                print("[task qa] already done, skipping")
                continue
        elif task in res and not a.force:
            print(f"[task {task}] already done, skipping")
            continue
        t0 = time.time()
        if task == "qa":
            res["qa"] = keep + task_qa(mq, tok, todo, dev, log, pool=a.pool, letter_ids=letter_token_ids(tok))
            res["qa"].sort(key=lambda r: cfgs.index(r["config"]) if r["config"] in cfgs else 99)
        elif task == "speed":
            res["speed"] = task_speed(mq, [c for c in a.speed_configs.split(",") if c], dev, [int(x) for x in a.speed_lengths.split(",")], a.speed_reps,
                                      n_seq=a.speed_nq)
        print(f"[task {task}] {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
        write_summary(res, os.path.join(a.out_dir, "summary.md"))
    print("[done]", a.out_dir)


if __name__ == "__main__":
    main()
