#!/usr/bin/env python
"""
QA error types from an eval_recompute run's generations.jsonl (no GPU; decodes the qa_eval prompts with the 8B tokenizer).
An answer is wrong if its first line neither contains a gold answer nor reaches F1 0.5. Wrong answers are split into
  refusal       says the document does not give the answer
  elsewhere     everything it says appears in the document (substring, or every content word occurs there):
                picked the wrong span / distractor
  not_in_doc    contains words the document does not have: fabricated
Usage: python scripts/analyze_qa_errors.py results/spec_06 [results/spec_17 ...]   -> <dir>/qa_errors.json + table on stdout
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_v2 import f1, norm_ans  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

REFUSAL = re.compile(r"not (mentioned|provided|stated|specified|given|available|clear|found)|no information|cannot be determined"
                     r"|can't be determined|does not (say|mention|specify|provide|state)|doesn't (say|mention|specify)"
                     r"|unanswerable|i don't know|not enough information|unknown")
STOP = set("a an the of in on at to for by with and or is was are were be been it its this that as from which who what "
           "when where how his her their he she they".split())


def classify(output, answers, doc_norm, doc_words):
    p1 = output.strip().split("\n")[0]
    if any(norm_ans(a) and norm_ans(a) in norm_ans(output) for a in answers) or max(f1(p1, a) for a in answers) >= 0.5:
        return "correct"
    if REFUSAL.search(p1.lower()):
        return "refusal"
    n = norm_ans(p1)
    content = [w for w in n.split() if w not in STOP]
    if n and (n in doc_norm or all(w in doc_words for w in content)):
        return "elsewhere"
    return "not_in_doc"


def main():
    tok = AutoTokenizer.from_pretrained(os.path.join(os.environ["PREFILL_ROOT"], "models", "Qwen3-8B"))
    pool, docs = PromptPool("qa_eval"), {}
    for d in sys.argv[1:]:
        counts = collections.defaultdict(collections.Counter)
        for line in open(os.path.join(d, "generations.jsonl")):
            g = json.loads(line)
            if g["task"] != "qa":
                continue
            if g["idx"] not in docs:
                n = norm_ans(tok.decode(pool.ids(g["idx"]), skip_special_tokens=True))
                docs[g["idx"]] = (n, set(n.split()))
            counts[g["config"]][classify(g["output"], g["answers"], *docs[g["idx"]])] += 1
        res = {c: {k: v[k] / sum(v.values()) for k in ("correct", "elsewhere", "not_in_doc", "refusal")} | {"n": sum(v.values())}
               for c, v in counts.items()}
        json.dump(res, open(os.path.join(d, "qa_errors.json"), "w"), indent=1)
        print(f"\n{d}\n{'config':28s} correct elsewhere not_in_doc refusal   n")
        for c, r in res.items():
            print(f"{c:28s} {r['correct']:7.1%} {r['elsewhere']:9.1%} {r['not_in_doc']:10.1%} {r['refusal']:7.1%} {r['n']:4d}")


if __name__ == "__main__":
    main()
