#!/usr/bin/env python
"""
Shared-document multi-question evaluation pool (CPU): `multiq_eval`.

Each prompt is a document set built like qa_eval's SQuAD prompts (scripts/prepare_mix.py): 3 gold SQuAD v2 validation
paragraphs that each have >= 2 answerable questions, padded with distractor paragraphs to ~3.5k / ~7.5k tokens and
shuffled. Up to 5 questions per document set (<= 2 per gold paragraph), so the answers sit in different places of the
context and no single question-specific selection covers all of them.
Stored ids = the shared prefix (chat template start + instruction + documents + "\n\n"); every question's own ids
("Question: ..." + end of the user turn + assistant prompt) are in meta["questions"][i]["ids"]. prefix + question ids
together reproduce the qa_eval prompt layout.
"""
import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_mix import EVAL_TARGET, load_sources  # noqa: E402  (sets HF_HOME)
from transformers import AutoTokenizer  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.mixdata import QA_INSTR, Builders, PromptPool, Tok, fill_docs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=30, help="document sets per length bucket")
    ap.add_argument("--n_gold", type=int, default=3)
    ap.add_argument("--n_q", type=int, default=5)
    ap.add_argument("--name", default="multiq_eval")
    a = ap.parse_args()
    if PromptPool.exists(a.name):
        print(f"[pool] {a.name} exists, skip")
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(C.LARGE_DIR)
    T = Tok(tok)
    B = Builders(tok, "test", seed=400)
    load_sources(tok, None, B, 1.0, parts=("qa",))
    by_ctx = collections.defaultdict(list)
    for ex in B.squad:
        by_ctx[ex["context"]].append(ex)
    multi = [c for c, v in by_ctx.items() if len(v) >= 2]
    print(f"[src] {len(by_ctx)} SQuAD contexts, {len(multi)} with >= 2 answerable questions ({(time.time()-t0)/60:.1f} min)", flush=True)
    head, tail = T.chat([{"role": "user", "content": "XXXSPLITXXX"}])[1].split("XXXSPLITXXX")
    rng = B.rng
    out = []
    for b, L in EVAL_TARGET.items():
        for _ in range(a.n_docs):
            for _try in range(50):
                golds = rng.sample(multi, a.n_gold)
                qs = []
                for c in golds:
                    qs += [(c, ex) for ex in rng.sample(by_ctx[c], 2)]
                rng.shuffle(qs)
                qs = qs[:a.n_q]
                qmax = max(T.n(f"Question: {ex['question']}" + tail) for _, ex in qs)
                dis = [(c, n) for c, n in B.squad_ctx if c not in golds]
                text = fill_docs(T, golds, dis, L - 60 - qmax, rng)
                if text is None:
                    continue
                prefix = T.enc(head + f"{QA_INSTR}\n\n{text}\n\n")
                if len(prefix) + qmax > L:
                    continue
                out.append({"ids": prefix, "type": "multiq", "source": "squad", "fmt": "chat", "bucket": b, "n_gold": a.n_gold,
                            "questions": [{"question": ex["question"], "answers": ex["answers"], "gold": golds.index(c),
                                           "ids": T.enc(f"Question: {ex['question']}" + tail)} for c, ex in qs]})
                break
        lens = [len(x["ids"]) for x in out if x["bucket"] == b]
        print(f"[{a.name}] {b}: {len(lens)} document sets, {min(lens)}-{max(lens)} prefix tokens, "
              f"{sum(len(x['questions']) for x in out if x['bucket'] == b)} questions", flush=True)
    PromptPool.save(a.name, out)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
