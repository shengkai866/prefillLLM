#!/usr/bin/env python
"""
Long-context benchmark pools (CPU) for scripts/eval_bench.py: one PromptPool per task in the current data tree
(large model's tokenizer), named bench_<task>_<max_len/1024>k[<suffix>]. Every prompt is a chat prompt (thinking off):
  head + instruction + CONTEXT + instruction/question/options + end of turn + assistant prompt.
meta: task, id, subset, metric (f1 | f1_cloze | acc), answers (list of reference strings) or gold (option index) with
n_opt, q_len = tokens after the context (kept in the large model's tail by the evaluation), pre_len = tokens before it,
n_ctx = context tokens before truncation, truncated. Contexts longer than the budget are truncated in the middle
(LongBench rule, on tokens), so a prompt never exceeds --max_len tokens.

Raw data (downloaded once into $PREFILL_ROOT/data/bench/raw, shared by all tokenizers):
  lastingbench  LastingBench (github Seriousss/LastingBench) hotpotqa_antifact.jsonl: counterfactually rewritten
                HotpotQA, LongBench format, 2-hop span answers                                       F1
  musique       LongBench musique.jsonl (THUDM/LongBench data.zip), 2-4 hop                            F1
  loogle        LooGLE (bigai-nlco/LooGLE) shortdep_qa (<= 2 questions per document) + shortdep_cloze
                (1 per document), half each                                                             F1 (cloze: mean
                                                                                                        per-mask F1)
  loogle_long   the same two files, documents of >= --long_min tokens only                             F1
  qasper        allenai/qasper test split, questions whose first annotator gave an extractive span or a
                yes/no answer (unanswerable and free-form questions are skipped); context = title +
                abstract + full text                                                                    F1
  clipper       chtmp223/CLIPPER test split: whole book + one claim, alternately the true and the false
                claim of consecutive rows (balanced)                                                    acc (A/B)
  longhealth    LongHealth (github kbressem/LongHealth benchmark_v5.json): fold k = patients 4k+1..4k+4,
                all their documents + a 5-option question                                              acc (A-E)
  contractnli   ContractNLI (kiddothe2b/contract-nli, contract_nli_long: whole contract) test subset,
                300 (contract, hypothesis) pairs sampled with seed 0                                   acc (A/B/C)
QuALITY is covered by scripts/prepare_quality.py / eval_recompute.py --tasks quality (results/quality_<pair>).
Shakespeare-CF has no public source we could find and is not included.
"""
import argparse
import ast
import json
import os
import random
import re
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.mixdata import POOL_DIR, PromptPool, Tok  # noqa: E402

RAW = os.path.join(C.SCRATCH_ROOT, "data", "bench", "raw")
TASKS = ("lastingbench", "musique", "loogle", "loogle_long", "qasper", "clipper", "longhealth", "contractnli")
DEFAULT_N = {"lastingbench": 150, "musique": 120, "loogle": 300, "loogle_long": 300, "qasper": 304, "clipper": 504,
             "longhealth": 80, "contractnli": 300}

P_MULTIHOP = ("Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
              "The following are given passages.\n",
              "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words."
              "\n\nQuestion: {q}\nAnswer:")
P_QASPER = ("You are given a scientific article and a question. Answer the question as concisely as you can, using a single "
            "phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "
            "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide "
            "any explanation.\n\nArticle: ",
            "\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if "
            "possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If "
            "the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation."
            "\n\nQuestion: {q}\n\nAnswer:")
P_LOOGLE_QA = ("Please answer the question based on the long texts below.\n", "\nQuestion: {q}\nAnswer:")
P_LOOGLE_CLOZE = ("Please fill in the clozes based on the given long texts below. Each of the placeholder '<mask-n>' in the "
                  "question could be an entity of Person, Location or Organization. The same masks represent the same entity. "
                  "Output a json format answer, for example: {\"<mask-0>\": \"James\", \"<mask-1>\": \"Washington DC\"}\n",
                  "\nQuestion: {q} What are the masked entities?\nAnswer:")
P_CLIPPER = ("You are provided with a context and a statement. Your task is to carefully read the context and then determine "
             "whether the statement is true or false.\n\nAnswer TRUE if the statement is true in its entirety based on the "
             "context provided. Answer FALSE if any part of the statement is false based on the context provided.\n\n<context>",
             "</context>\n\n<statement>{q}</statement>\n\n<question>Based on the context provided, is the above statement TRUE "
             "or FALSE?</question>\n\nReply with the letter of the correct option only.\n(A) TRUE\n(B) FALSE\nAnswer:")
P_LONGHEALTH = ("You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in "
                "interpreting and extracting key information from medical documents. Below are the clinical documents of one "
                "patient.\n\n",
                "\n\nAnswer the multiple-choice question about the patient using the documents above. Reply with the letter of "
                "the correct option (A, B, C, D or E) only.\nQuestion: {q}\nAnswer:")
P_CONTRACT = ("You are given a non-disclosure agreement (NDA) and a hypothesis about it. Decide whether the contract entails "
              "the hypothesis, contradicts it, or does not mention it.\n\nContract:\n",
              "\n\nHypothesis: {q}\nDoes the contract above entail, contradict, or not mention the hypothesis? Reply with the "
              "letter of the correct option only.\n(A) Entailment: the contract entails the hypothesis\n(B) Contradiction: the "
              "contract contradicts the hypothesis\n(C) Not mentioned: the contract does not mention it\nAnswer:")
NLI_LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}


def jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def options_block(opts):
    return "\n".join(f"({c}) {o}" for c, o in zip("ABCDE", opts))


class Builder:
    def __init__(self, tok, max_len):
        self.T = Tok(tok)
        self.tok = tok
        self.max_len = max_len
        self.head, self.tail = self.T.chat([{"role": "user", "content": "XXXSPLITXXX"}])[1].split("XXXSPLITXXX")
        self.cache = {}

    def ctx_ids(self, key, text):
        """Context token ids, cached by key (LooGLE / LongHealth / CLIPPER reuse one document for several items)."""
        if key not in self.cache:
            self.cache[key] = self.T.enc(text)
        return self.cache[key]

    def item(self, task, _id, pre, ctx_key, ctx_text, post, metric, subset="", **meta):
        pre_ids = self.T.enc(self.head + pre)
        q_ids = self.T.enc(post + self.tail)
        ctx = self.ctx_ids(ctx_key, ctx_text)
        n_ctx = len(ctx)
        budget = self.max_len - len(pre_ids) - len(q_ids)
        truncated = n_ctx > budget
        if truncated:
            h = budget // 2
            ctx = list(ctx[:h]) + list(ctx[n_ctx - (budget - h):])
        return {"ids": pre_ids + list(ctx) + q_ids, "task": task, "id": str(_id), "subset": subset, "metric": metric, "fmt": "chat",
                "q_len": len(q_ids), "pre_len": len(pre_ids), "n_ctx": n_ctx, "truncated": truncated, **meta}


# ------------------------------------------------------------------------------------------------ tasks
def lastingbench(B, n):
    rows = jsonl(os.path.join(RAW, "hotpotqa_antifact.jsonl"))[:n]
    return [B.item("lastingbench", r["_id"], P_MULTIHOP[0], r["_id"], r["context"], P_MULTIHOP[1].format(q=r["input"]), "f1",
                   answers=ast.literal_eval(r["answers"]) if isinstance(r["answers"], str) else r["answers"], question=r["input"])
            for r in rows]


def musique(B, n):
    z = zipfile.ZipFile(os.path.join(C.SCRATCH_ROOT, "data", "longbench", "data.zip"))
    rows = [json.loads(l) for l in z.open("data/musique.jsonl").read().decode("utf-8").splitlines()][:n]
    return [B.item("musique", r["_id"], P_MULTIHOP[0], r["_id"], r["context"], P_MULTIHOP[1].format(q=r["input"]), "f1",
                   answers=r["answers"], question=r["input"]) for r in rows]


def loogle_rows(B, long_min=0):
    """[qa docs, cloze docs], each doc_id -> {"ctx": document text, "rows": question rows without the context} in file
    order. Every row of the raw files repeats the whole document, so the context is kept once per document (the login
    node allows 4 GB). long_min > 0 keeps documents of >= long_min tokens."""
    out = []
    for f in ("loogle_shortdep_qa.jsonl", "loogle_shortdep_cloze.jsonl"):
        docs = {}
        for l in open(os.path.join(RAW, f), encoding="utf-8"):
            r = json.loads(l)
            d = docs.setdefault(r["doc_id"], {"ctx": r.pop("context"), "rows": []})
            r.pop("context", None)
            d["rows"].append(r)
        if long_min:
            docs = {k: d for k, d in docs.items() if len(B.ctx_ids(k, d["ctx"])) >= long_min}
        out.append(docs)
    return out


def loogle_item(B, r, ctx, task):
    cloze = r["task"] == "shortdep_cloze"
    P = P_LOOGLE_CLOZE if cloze else P_LOOGLE_QA
    ans = r["answer"]
    if cloze:
        ans = ast.literal_eval(ans) if isinstance(ans, str) and ans.startswith("[") else [ans]
        answers = [str(a) for a in ans]  # one reference per mask, in mask order
    else:
        answers = [ans]
    return B.item(task, r["id"], P[0], r["doc_id"], ctx, P[1].format(q=r["question"]), "f1_cloze" if cloze else "f1",
                  subset="cloze" if cloze else "qa", answers=answers, question=r["question"][:300], doc_id=r["doc_id"])


def round_robin(docs, per_doc, n):
    """Up to per_doc questions per document, documents in order, until n items -> [(row, context)]."""
    out, k = [], 0
    while len(out) < n and k < per_doc:
        for d in docs.values():
            if k < len(d["rows"]) and len(out) < n:
                out.append((d["rows"][k], d["ctx"]))
        k += 1
    return out


def loogle(B, n):
    qa, cloze = loogle_rows(B)
    return [loogle_item(B, r, ctx, "loogle") for r, ctx in round_robin(qa, 2, n - n // 2) + round_robin(cloze, 1, n // 2)]


def loogle_long(B, n, long_min):
    qa, cloze = loogle_rows(B, long_min)
    print(f"[loogle_long] documents >= {long_min} tokens: {len(qa)} qa docs, {len(cloze)} cloze docs", flush=True)
    rows = round_robin(qa, 50, n) + round_robin(cloze, 50, n)
    return [loogle_item(B, r, ctx, "loogle_long") for r, ctx in rows[:n]]


def qasper(B, n):
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(RAW, "qasper_test.parquet")).to_pylist()
    items = []
    for paper in t:
        ft = paper["full_text"]
        body = "".join(f"\n\n{name}\n" + "\n".join(ps) for name, ps in zip(ft["section_name"], ft["paragraphs"]))
        ctx = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}{body}"
        for q, qid, ans in zip(paper["qas"]["question"], paper["qas"]["question_id"], paper["qas"]["answers"]):
            first = ans["answer"][0]
            if first["unanswerable"] or (first["yes_no"] is None and not first["extractive_spans"]):
                continue
            refs = []
            for a in ans["answer"]:  # every annotator with a span / yes-no answer
                if a["unanswerable"]:
                    continue
                if a["yes_no"] is not None:
                    refs.append("yes" if a["yes_no"] else "no")
                elif a["extractive_spans"]:
                    refs.append(", ".join(a["extractive_spans"]))
            items.append(B.item("qasper", qid, P_QASPER[0], paper["id"], ctx, P_QASPER[1].format(q=q), "f1",
                                subset="yesno" if first["yes_no"] is not None else "span", answers=refs, question=q))
            if len(items) >= n:
                return items
    return items


def clipper(B, n):
    """First n rows of the test split by id, streamed (one row holds a whole book, so the files are never fully loaded)."""
    import pyarrow.parquet as pq
    files = [os.path.join(RAW, f"clipper_test{i}.parquet") for i in (0, 1)]
    ids = sorted(int(x) for f in files for x in pq.read_table(f, columns=["id"]).column("id").to_pylist())
    keep = set(ids[:n])
    rows = []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=16, columns=["id", "book_name", "book_title", "claim_type",
                                                                           "true_claim", "false_claim", "messages"]):
            rows += [r for r in batch.to_pylist() if int(r["id"]) in keep]
            if len(rows) >= n:
                break
        if len(rows) >= n:
            break
    rows.sort(key=lambda r: int(r["id"]))
    items = []
    for i, r in enumerate(rows[:n]):
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        m = re.search(r"<context>(.*)</context>", user, re.S)
        ctx = m.group(1) if m else user
        truth = i % 2 == 0  # alternate: even rows ask about the true claim, odd rows about the false one
        claim = r["true_claim"] if truth else r["false_claim"]
        items.append(B.item("clipper", r["id"], P_CLIPPER[0], r["book_name"], ctx, P_CLIPPER[1].format(q=claim), "acc",
                            subset=r["claim_type"], gold=0 if truth else 1, n_opt=2, label="true" if truth else "false",
                            question=claim, book=r["book_title"]))
        B.cache.pop(r["book_name"], None)  # books are not shared between rows; keep memory flat
    return items


def longhealth(B, n, fold):
    d = json.load(open(os.path.join(RAW, "longhealth_v5.json")))
    pats = sorted(d)[fold * 4:(fold + 1) * 4]
    items = []
    for p in pats:
        pat = d[p]
        ctx = "\n\n".join(f"Document {k}:\n{v.strip()}" for k, v in sorted(pat["texts"].items(), key=lambda kv: int(kv[0].split("_")[1])))
        for q in pat["questions"]:
            opts = [q[f"answer_{c}"] for c in "abcde"]
            gold = opts.index(q["correct"])
            post = P_LONGHEALTH[1].format(q=q["question"] + "\n" + options_block(opts))
            items.append(B.item("longhealth", f"{p}_q{q['No']}", P_LONGHEALTH[0], p, ctx, post, "acc", subset=p, gold=gold, n_opt=5,
                                question=q["question"]))
    return items[:n]


def contractnli(B, n):
    z = zipfile.ZipFile(os.path.join(RAW, "contract_nli_long.zip"))
    rows = [json.loads(l) for l in z.open("contract_nli_long.jsonl").read().decode("utf-8").splitlines()]
    pairs = []
    for ci, r in enumerate(rows):
        if r["subset"] != "test":
            continue
        hl = r["hypothesises/labels"]
        hl = ast.literal_eval(hl) if isinstance(hl, str) else hl
        for hi, h in enumerate(hl):
            pairs.append((ci, hi, r["premise"], h["hypothesis"], h["label"]))
    random.Random(0).shuffle(pairs)
    pairs = sorted(pairs[:n])
    return [B.item("contractnli", f"c{ci}_h{hi}", P_CONTRACT[0], f"c{ci}", prem, P_CONTRACT[1].format(q=hyp), "acc",
                   subset=lab, gold=NLI_LABELS[lab], n_opt=3, question=hyp) for ci, hi, prem, hyp, lab in pairs]


def pool_name(task, max_len, suffix=""):
    return f"bench_{task}_{max_len // 1024}k{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max_len", type=int, default=65536, help="prompt token budget; longer contexts are truncated in the middle")
    ap.add_argument("--long_min", type=int, default=49152, help="loogle_long: minimum document length in tokens")
    ap.add_argument("--fold", type=int, default=0, help="longhealth fold (4 patients each)")
    ap.add_argument("--n", type=int, default=0, help="items per task (0 = the per-task defaults)")
    ap.add_argument("--suffix", default="", help="pool name suffix (e.g. _smoke)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(C.LARGE_DIR)
    B = Builder(tok, a.max_len)
    # several jobs of one tokenizer tree may start together: build under a lock, then the others find the pools
    import fcntl
    os.makedirs(POOL_DIR, exist_ok=True)
    lock = open(os.path.join(POOL_DIR, "bench.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    for task in a.tasks.split(","):
        name = pool_name(task, a.max_len, a.suffix)
        if PromptPool.exists(name) and not a.force:
            print(f"[pool] {name} exists, skip")
            continue
        t0 = time.time()
        n = a.n or DEFAULT_N[task]
        if task == "loogle_long":
            items = loogle_long(B, n, a.long_min)
        elif task == "longhealth":
            items = longhealth(B, n, a.fold)
        else:
            items = globals()[task](B, n)
        ls = [len(x["ids"]) for x in items]
        print(f"[{task}] {len(items)} items, prompt {min(ls)}-{max(ls)} tokens (mean {sum(ls)/len(ls):.0f}), "
              f"{sum(x['truncated'] for x in items)} truncated, question block <= {max(x['q_len'] for x in items)} tokens, "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        PromptPool.save(name, items)


if __name__ == "__main__":
    main()
