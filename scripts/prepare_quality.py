#!/usr/bin/env python
"""
QuALITY (Pang et al. 2022, v1.0.1 dev split, html-stripped) as two evaluation pools of the current data tree (CPU):
  quality_eval    one prompt per question: chat prompt = instruction + article + question + 4 options; the model answers
                  with a letter (scored from first-token logits, prefill/mcq.py). meta: gold (0-3), difficult, bucket
                  ("4k": article < 4096 tokens, else "8k"), q_len = tokens of the question block + end of turn + assistant
                  prompt, so that the evaluation can keep the whole question in the large model's tail.
  quality_multiq  one document per article (shared prefix = instruction + article), all its questions as meta
                  ["questions"][i]["ids"] (the question block + end of turn + assistant prompt), like multiq_eval.
Raw data: $PREFILL_ROOT/data/quality/QuALITY.v1.0.1.htmlstripped.dev (the test split has no public labels).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.mixdata import PromptPool, Tok  # noqa: E402

INSTR = ("Read the story and answer the multiple-choice question about it. Reply with the letter of the correct option "
         "(A, B, C or D) only.")


def question_block(q):
    opts = "\n".join(f"({c}) {o}" for c, o in zip("ABCD", q["options"]))
    return f"Question: {q['question']}\n{opts}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n_articles", type=int, default=0, help="0 = all")
    ap.add_argument("--suffix", default="", help="pool name suffix (e.g. _smoke)")
    a = ap.parse_args()
    names = ("quality_eval" + a.suffix, "quality_multiq" + a.suffix)
    if all(PromptPool.exists(n) for n in names):
        print(f"[pool] {names} exist, skip")
        return
    t0 = time.time()
    path = os.path.join(C.DATA_DIR, "quality", f"QuALITY.v1.0.1.htmlstripped.{a.split}")
    arts = [json.loads(l) for l in open(path)]
    if a.n_articles:
        arts = arts[:a.n_articles]
    tok = AutoTokenizer.from_pretrained(C.LARGE_DIR)
    T = Tok(tok)
    head, tail = T.chat([{"role": "user", "content": "XXXSPLITXXX"}])[1].split("XXXSPLITXXX")
    single, multi = [], []
    for ai, art in enumerate(arts):
        prefix_text = head + f"{INSTR}\n\n{art['article'].strip()}\n\n"
        prefix = T.enc(prefix_text)
        n_art = T.n(art["article"])
        bucket = "4k" if n_art < 4096 else "8k"
        qs = []
        for qi, q in enumerate(art["questions"]):
            gold = int(q["gold_label"]) - 1
            qids = T.enc(question_block(q) + tail)
            base = dict(article=ai, article_id=art["article_id"], q=qi, question_id=q["question_unique_id"], gold=gold,
                        difficult=int(q.get("difficult", 0)), source=art.get("source", ""), bucket=bucket)
            single.append({"ids": prefix + qids, "type": "quality", "fmt": "chat", "q_len": len(qids),
                           "question": q["question"], "options": q["options"], **base})
            qs.append({"ids": qids, "question": q["question"], "options": q["options"], "gold": gold, "difficult": base["difficult"],
                       "question_id": q["question_unique_id"]})
        multi.append({"ids": prefix, "type": "quality_multiq", "fmt": "chat", "article": ai, "article_id": art["article_id"],
                      "source": art.get("source", ""), "bucket": bucket, "n_article_tokens": n_art, "questions": qs})
    for b in ("4k", "8k"):
        ls = [len(x["ids"]) for x in single if x["bucket"] == b]
        if not ls:  # e.g. a 2-article smoke pool
            continue
        print(f"[quality] {b}: {sum(1 for x in multi if x['bucket'] == b)} articles, {len(ls)} questions, prompt {min(ls)}-{max(ls)} tokens, "
              f"{sum(x['difficult'] for x in single if x['bucket'] == b)} hard; question block <= {max(x['q_len'] for x in single if x['bucket'] == b)} tokens", flush=True)
    PromptPool.save(names[0], single)
    PromptPool.save(names[1], multi)
    print(f"[done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
