#!/usr/bin/env python
"""
Long-context benchmark suite (pools of scripts/prepare_bench.py) for any config of scripts/eval_recompute.py:
large model alone (big), ours (attn_small:0.1:sent), SpecPrefill (specprefill:0.1), ...

  lastingbench, musique, loogle, loogle_long, qasper   generation (greedy, first line), F1 against the references
                                                       (loogle cloze: mean per-mask F1 of the json answer)
  clipper, longhealth, contractnli                     letter from first-token logits (prefill/mcq.py), accuracy
The question block of every prompt (everything after the context, meta q_len) is kept in the large model's tail
(n_tail raised per prompt), so our configs never project or drop a question or an option; SpecPrefill is unchanged.
Qwen3 models get static YaRN (prefill/models.yarn_config) when --max_len exceeds their 40k window (--yarn auto);
pass --yarn 0 for jobs whose prompts fit. Out-of-memory on one prompt: the context is halved (middle truncation)
and the item is marked trunc_oom in generations.jsonl. Results: results.json (per task, config and subset), summary.md.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from eval_recompute import Engine, GenLog, fmt, label  # noqa: E402
from evaluate_v2 import f1, mean, qa_scores  # noqa: E402
from prefill import config as C  # noqa: E402
from prefill.mcq import choose_letter, letter_token_ids  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prepare_bench import DEFAULT_N, TASKS, pool_name  # noqa: E402

MAX_GEN = {"f1": 48, "f1_cloze": 96}
NAME_ZH = {"lastingbench": "LastingBench（多跳，F1）", "musique": "MuSiQue（F1）", "loogle": "LooGLE（F1）",
           "loogle_long": "LooGLE-long ≥48k（F1）", "qasper": "Qasper（F1）", "clipper": "CLIPPER（acc）",
           "longhealth": "LongHealth（acc）", "contractnli": "ContractNLI（acc）"}
SUBSET_ZH = {"qa": "问答", "cloze": "完形", "span": "片段", "yesno": "是/否", "single": "单章", "multiple": "跨章",
             "entailment": "蕴含", "contradiction": "矛盾", "neutral": "未提及"}


def parse_cloze(out):
    """'{"<mask-0>": "James", "<mask-1>": "Washington DC"}' (or a loose variant) -> {0: 'James', 1: 'Washington DC'}."""
    pred = {}
    try:
        m = re.search(r"\{.*\}", out, re.S)
        for k, v in json.loads(m.group(0)).items():
            mk = re.search(r"(\d+)", str(k))
            if mk:
                pred[int(mk.group(1))] = str(v)
    except Exception:
        pass
    if not pred:
        for k, v in re.findall(r"<?mask-(\d+)>?\"?\s*[:=]\s*\"?([^\"\n,}]+)", out):
            pred.setdefault(int(k), v.strip())
    return pred


def score_item(metric, out, answers):
    if metric == "f1":
        return qa_scores(out, answers)["f1"]
    if metric == "f1_cloze":
        pred = parse_cloze(out)
        return mean([f1(pred.get(k, ""), g) for k, g in enumerate(answers)])
    raise ValueError(metric)


def halve_context(ids, pre_len, q_len):
    """Middle-truncate the context of ids [T] (numpy) to half its length (OOM fallback)."""
    ctx = ids[pre_len:len(ids) - q_len]
    h = len(ctx) // 4
    return list(ids[:pre_len]) + list(ctx[:h]) + list(ctx[len(ctx) - h:]) + list(ids[len(ids) - q_len:])


def run_task(E, task, pool, cfgs, n, log):
    lid = {k: letter_token_ids(E.tok, k) for k in (2, 3, 4, 5)}
    idx = list(range(len(pool.meta)))[:n] if n else list(range(len(pool.meta)))
    n_tail0 = E.rh.n_tail
    rows = []
    for c in cfgs:
        t0 = time.time()
        sc, fr, secs = collections.defaultdict(list), collections.defaultdict(list), []
        n_oom = 0
        for i in idx:
            m = pool.meta[i]
            ids = pool.ids(i)
            E.rh.n_tail = max(n_tail0, m["q_len"])
            oom = 0
            try:
                while True:
                    try:
                        torch.cuda.synchronize()
                        t1 = time.time()
                        if m["metric"] == "acc":
                            _, _, lg, frac = E.prefill(c, torch.as_tensor(ids, device=E.dev))
                            pred = choose_letter(lg, lid[m["n_opt"]])
                            out, s = "ABCDE"[pred], float(pred == m["gold"])
                        else:
                            out, frac = E.generate(c, ids, MAX_GEN[m["metric"]])
                            s = score_item(m["metric"], out, m["answers"])
                        torch.cuda.synchronize()
                        secs.append(time.time() - t1)
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        oom += 1
                        if oom > 3:
                            raise
                        ids = halve_context(ids, m["pre_len"], m["q_len"])
                        print(f"[{task}] OOM on item {m['id']} ({c}); context halved -> {len(ids)} tokens", flush=True)
            finally:
                E.rh.n_tail = n_tail0
            n_oom += oom > 0
            for k in ["all"] + ([m["subset"]] if m["subset"] else []):
                sc[k].append(s)
                fr[k].append(frac)
            log(task=task, config=c, id=m["id"], subset=m["subset"], prompt_len=len(ids), output=out,
                answers=m.get("answers"), gold=m.get("gold"), score=s, frac=frac, trunc_oom=oom > 0)
        for k in sorted(sc, key=lambda k: (k != "all", k)):
            rows.append({"config": c, "task": task, "subset": k, "n": len(sc[k]), "score": mean(sc[k]), "frac": mean(fr[k]),
                         "sec_per_item": mean(secs) if k == "all" else None, "n_oom": n_oom if k == "all" else None})
        print(f"[{task}] {c:>22}: " + " ".join(f"{k}={mean(v):.3f}" for k, v in sc.items())
              + f"  ({mean(secs):.1f} s/item, {n_oom} OOM-truncated, {(time.time()-t0)/60:.1f} min)", flush=True)
        torch.cuda.empty_cache()
    return rows


def write_summary(res, path):
    meta = res["meta"]
    a = meta["args"]
    cfgs = meta["configs"]
    L = ["# 长文本基准：LastingBench / MuSiQue / LooGLE / Qasper / CLIPPER / LongHealth / ContractNLI", "",
         f"GPU：{meta['gpu']}；生成：{meta['time']}；小模型 {C.SMALL_NAME}，大模型 {C.LARGE_NAME}，projector 第 {meta.get('proj_step')} 步；"
         f"prompt 上限 {a['max_len']} token（超出的上下文从中间截断）" + (f"，Qwen3 用 YaRN ×{meta['yarn']}" if meta.get("yarn") else "") + "。",
         "F1 任务贪心生成后取第一行与参考答案算 token F1（LooGLE 完形：每个 mask 的 F1 取平均）；acc 任务按首 token 的字母 logits 判分。"
         "问题/选项始终在大模型自己计算的尾部里。“重算比例”对我们的方法是中间 token 被大模型重算的比例，对 SpecPrefill 是整段 prompt 保留的比例。", ""]
    tasks = [t for t in TASKS if t in res]
    if not tasks:
        return
    L += ["## 总表（每个任务的整体分数）", "", "| 配置 | " + " | ".join(NAME_ZH[t] for t in tasks) + " |", "|---|" + "---|" * len(tasks)]
    for c in cfgs:
        v = []
        for t in tasks:
            r = next((r for r in res[t] if r["config"] == c and r["subset"] == "all"), None)
            v.append(fmt(r["score"]) if r else "-")
        L.append(f"| {label(c)} | " + " | ".join(v) + " |")
    info = []
    for t in tasks:
        r = next((r for r in res[t] if r["subset"] == "all"), None)
        if r:
            info.append(f"{NAME_ZH[t].split('（')[0]} {r['n']} 题（{res['meta']['prompt_tokens'].get(t, 0):.0f} token）")
    L += ["", "题数与平均 prompt 长度：" + "；".join(info), ""]
    for t in tasks:
        subs = [s for s in dict.fromkeys(r["subset"] for r in res[t]) if s != "all"]
        L += [f"## {NAME_ZH[t]}", "", "| 配置 | 重算比例 | 每题秒数 | 整体 | " + " | ".join(SUBSET_ZH.get(s, s) for s in subs) + " |",
              "|---|---|---|---|" + "---|" * len(subs)]
        for c in cfgs:
            rs = {r["subset"]: r for r in res[t] if r["config"] == c}
            if "all" not in rs:
                continue
            r = rs["all"]
            L.append(f"| {label(c)} | {r['frac']:.1%} | {r['sec_per_item']:.1f} | {fmt(r['score'])} | "
                     + " | ".join(fmt(rs[s]["score"]) if s in rs else "-" for s in subs) + " |")
        ns = {r["subset"]: r["n"] for r in res[t] if r["config"] == cfgs[0]}
        L += ["", "题数：" + "，".join(f"{SUBSET_ZH.get(s, s)} {k}" for s, k in ns.items()), ""]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
    ap.add_argument("--configs", default="big,attn_small:0.1:sent,specprefill:0.1,specprefill:0.2")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max_len", type=int, default=65536, help="pool of scripts/prepare_bench.py to use (same --max_len)")
    ap.add_argument("--suffix", default="", help="pool name suffix (e.g. _smoke)")
    ap.add_argument("--n", type=int, default=0, help="items per task (0 = whole pool)")
    ap.add_argument("--yarn", default="auto", help="YaRN factor for Qwen3 models: auto (from --max_len), 0 (off) or a number")
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "bench"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res_path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    tasks = [t for t in a.tasks.split(",") if t]
    pools = {t: PromptPool(pool_name(t, a.max_len, a.suffix)) for t in tasks}
    longest = max(max(len(p.ids(i)) for i in range(len(p))) for p in pools.values())
    yarn = (math.ceil((longest + 1024) / 32768) if longest + 1024 > 40960 else 0) if a.yarn == "auto" else float(a.yarn)
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev, yarn=yarn or None)
    E = Engine(tok, small, large, a.projector, dev, a.n_sink, a.n_tail)
    cfgs = [c for c in a.configs.split(",") if c]
    res["meta"] = {"gpu": torch.cuda.get_device_name(0), "configs": cfgs, "args": vars(a), "proj_step": E.proj_step,
                   "time": time.strftime("%F %T"), "yarn": yarn, "longest_prompt": longest,
                   "prompt_tokens": {**res.get("meta", {}).get("prompt_tokens", {}),
                                     **{t: mean([len(p.ids(i)) for i in range(len(p))][:a.n or None]) for t, p in pools.items()}}}
    log = GenLog(os.path.join(a.out_dir, "generations.jsonl"))
    for task in tasks:
        if task in res and not a.force:
            print(f"[task {task}] already done, skipping")
            continue
        t0 = time.time()
        res[task] = run_task(E, task, pools[task], cfgs, a.n, log)
        print(f"[task {task}] {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
        write_summary(res, os.path.join(a.out_dir, "summary.md"))
    print("[done]", a.out_dir)


if __name__ == "__main__":
    main()
