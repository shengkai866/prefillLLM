#!/usr/bin/env python
"""
Tasks that need the WHOLE context (not one retrievable span), where dropping unselected tokens should hurt more than
filling them with projected KV: LongBench (THUDM/LongBench, data.zip under DATA_DIR/longbench) summarisation and code
completion, plus many-shot GSM8K. Any config of scripts/eval_recompute.py (incl. recompute, drop-*, SpecPrefill).

  gov_report   one-page summary of a government report           ROUGE-L F (LongBench metric)
  qmsum        query-based meeting summary                       ROUGE-L F
  lcc          next line of a long code file                      edit similarity (= fuzz.ratio, Levenshtein based)
  repobench-p  next line with cross-file context                  edit similarity
  gsm32        GSM8K with 32 chat-turn shots (~8k tokens)         accuracy, same prompt builder as evaluate_v2.task_gsm8k
LongBench prompts and max generation lengths follow the official config; prompts longer than --max_len tokens are
truncated in the middle (official rule); summarisation prompts go through the chat template, code prompts are raw
(official build_chat rule); code predictions are cut at the first line. Items with fewer than --min_words words are
skipped so every prompt is long. Generations go to generations.jsonl.
"""
import argparse
import collections
import json
import os
import re
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from eval_recompute import Engine, GenLog, V2Adapter, fmt, label  # noqa: E402
from evaluate_v2 import mean, task_gsm8k  # noqa: E402
from prefill import config as C  # noqa: E402
from prefill.models import load_pair  # noqa: E402

PROMPT = {
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
MAX_GEN = {"gov_report": 512, "qmsum": 512, "lcc": 64, "repobench-p": 64}
CHAT = {"gov_report", "qmsum"}
NAME_ZH = {"gov_report": "政府报告摘要（ROUGE-L）", "qmsum": "会议摘要 QMSum（ROUGE-L）", "lcc": "代码补全 LCC（编辑相似度）",
           "repobench-p": "跨文件代码补全 RepoBench-P（编辑相似度）", "gsm32": "GSM8K 32-shot 准确率"}


def load_task(name):
    z = zipfile.ZipFile(os.path.join(C.DATA_DIR, "longbench", "data.zip"))
    return [json.loads(l) for l in z.open(f"data/{name}.jsonl").read().decode("utf-8").splitlines()]


def rouge_l(pred, ref):
    """ROUGE-L F1 on lower-cased alphanumeric tokens (LCS), as the rouge package computes it."""
    p, r = re.findall(r"[a-z0-9]+", pred.lower()), re.findall(r"[a-z0-9]+", ref.lower())
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    pr, rc = lcs / len(p), lcs / len(r)
    return 2 * pr * rc / (pr + rc)


def edit_sim(a, b):
    """fuzz.ratio: 100 * (1 - Levenshtein distance with substitution cost 2 / (len a + len b))."""
    if not a and not b:
        return 100.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 2)))
        prev = cur
    return 100.0 * (1 - prev[-1] / (len(a) + len(b)))


def truncate_middle(E, text, max_len):
    ids = E.T.enc(text)
    if len(ids) <= max_len:
        return text
    h = max_len // 2
    return E.tok.decode(ids[:h], skip_special_tokens=False) + E.tok.decode(ids[-h:], skip_special_tokens=False)


def task_longbench(E, name, cfgs, n, max_len, min_words, max_gen, log):
    rows_in = [x for x in load_task(name) if x["length"] >= min_words][:n]
    items = []
    for x in rows_in:
        text = truncate_middle(E, PROMPT[name].format(context=x["context"], input=x["input"]), max_len)
        ids = E.chat_ids([{"role": "user", "content": text}]) if name in CHAT else E.T.enc(text)
        items.append((x["_id"], ids, x["answers"]))
    rows = []
    for c in cfgs:
        t0 = time.time()
        sc, fr = [], []
        for _id, ids, answers in items:
            out, frac = E.generate(c, ids, max_gen or MAX_GEN[name])
            if name in CHAT:
                s = max(rouge_l(out, a) for a in answers)
            else:
                line = out.lstrip("\n").split("\n")[0]
                s = max(edit_sim(line, a) for a in answers) / 100
            sc.append(s)
            fr.append(frac)
            log(task=name, config=c, id=_id, prompt_len=len(ids), output=out, score=s, frac=frac)
        rows.append({"config": c, "task": name, "n": len(sc), "score": mean(sc), "frac": mean(fr), "prompt_tokens": mean([len(i) for _, i, _ in items])})
        print(f"[{name}] {c:>22}: {mean(sc):.3f} (frac {mean(fr):.2f}, {len(sc)} items, {(time.time()-t0)/60:.1f} min)", flush=True)
    return rows


def write_summary(res, path):
    cfgs = res["meta"]["configs"]
    a = res["meta"]["args"]
    L = ["# 需要全文信息的任务：LongBench 摘要与代码补全、多示例 GSM8K", "",
         f"GPU：{res['meta']['gpu']}；生成：{res['meta']['time']}；小模型 {C.SMALL_NAME}，大模型 {C.LARGE_NAME}，projector 第 {res['meta'].get('proj_step')} 步。",
         f"LongBench prompt 超过 {a['max_len']} token 时按官方规则从中间截断；摘要任务走对话模板，代码任务是原始文本；"
         f"只取原文不少于 {a['min_words']}（摘要）/ {a['min_words_code']}（代码）个词的样本。"
         f"摘要最多生成 {a['max_gen'] or 512} 个 token。GSM8K 用 32 个示例（约 8k token）。", ""]
    tasks = [t for t in ("gov_report", "qmsum", "lcc", "repobench-p", "gsm32") if t in res]
    if not tasks:
        return
    L += ["| 配置 | " + " | ".join(NAME_ZH[t] for t in tasks) + " |", "|---|" + "---|" * len(tasks)]
    for c in cfgs:
        v = []
        for t in tasks:
            r = next((r for r in res[t] if r["config"] == c), None)
            v.append(fmt(r["accuracy"] if t == "gsm32" else r["score"]) if r else "-")
        L.append(f"| {label(c)} | " + " | ".join(v) + " |")
    n = {t: (res[t][0]["n"], res[t][0].get("prompt_tokens")) for t in tasks if res[t]}
    L += ["", "样本数与平均 prompt 长度：" + "；".join(f"{NAME_ZH[t].split('（')[0]} {k}（{fmt(p, '{:.0f}')} token）" for t, (k, p) in n.items())]
    if "gsm32" in res:
        L += ["", "GSM8K 32-shot：", "", "| 配置 | 准确率 | 遵守 #### 格式 |", "|---|---|---|"]
        L += [f"| {label(r['config'])} | {r['accuracy']:.3f} | {r['format_follow']:.3f} |" for r in res["gsm32"]]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
    ap.add_argument("--configs", default="big,small,hyb,attn_small:0.1:sent,drop-attn_small:0.1:sent,specprefill:0.1,specprefill:0.3")
    ap.add_argument("--tasks", default="gov_report,qmsum,lcc,repobench-p,gsm32")
    ap.add_argument("--n_sum", type=int, default=40)
    ap.add_argument("--n_code", type=int, default=60)
    ap.add_argument("--n_gsm", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=7680)
    ap.add_argument("--min_words", type=int, default=3000, help="summarisation items shorter than this are skipped")
    ap.add_argument("--min_words_code", type=int, default=1500, help="same for lcc / repobench-p (lcc files are shorter)")
    ap.add_argument("--max_gen", type=int, default=400, help="cap for the summarisation tasks (official 512); 0 = official")
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "longbench"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res_path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    E = Engine(tok, small, large, a.projector, dev, a.n_sink, a.n_tail)
    cfgs = [c for c in a.configs.split(",") if c]
    res["meta"] = {"gpu": torch.cuda.get_device_name(0), "configs": cfgs, "args": vars(a), "proj_step": E.proj_step, "time": time.strftime("%F %T")}
    log = GenLog(os.path.join(a.out_dir, "generations.jsonl"))
    for task in a.tasks.split(","):
        if task in res and not a.force:
            print(f"[task {task}] already done, skipping")
            continue
        t0 = time.time()
        if task == "gsm32":
            res[task] = task_gsm8k(V2Adapter(E), cfgs, a.n_gsm, shots=32)
        else:
            res[task] = task_longbench(E, task, cfgs, a.n_sum if task in CHAT else a.n_code, a.max_len,
                                       a.min_words if task in CHAT else a.min_words_code, a.max_gen if task in CHAT else 0, log)
        print(f"[task {task}] {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
        write_summary(res, os.path.join(a.out_dir, "summary.md"))
    print("[done]", a.out_dir)


if __name__ == "__main__":
    main()
