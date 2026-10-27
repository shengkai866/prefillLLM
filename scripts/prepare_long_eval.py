#!/usr/bin/env python
"""
Long-context evaluation pools (CPU), built exactly like qa_eval / niah_eval in scripts/prepare_mix.py but longer:
  qa_eval_long    SQuAD v2 / HotpotQA / TriviaQA validation questions padded with distractor documents to ~15.9k and
                  ~32.3k tokens (buckets "16k", "32k"; 512 tokens below the nominal length, like 3584 / 7680 for 4k / 8k)
  niah_eval_long  round-1 passcode needle at 15872 / 32256 tokens
Same held-out test material and streams as prepare_mix.py (fresh builder / needle seeds). Idempotent.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_mix import load_sources  # noqa: E402  (sets HF_HOME)
from transformers import AutoTokenizer  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.mixdata import Builders, PromptPool, Tok, passcode_niah  # noqa: E402

LONG_TARGET = {"16k": 15872, "32k": 32256}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qa", type=int, default=60)
    ap.add_argument("--n_niah", type=int, default=30)
    a = ap.parse_args()
    if PromptPool.exists("qa_eval_long") and PromptPool.exists("niah_eval_long"):
        print("[pool] long pools exist, skip")
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(C.LARGE_DIR)
    B_te = Builders(tok, "test", seed=300)
    load_sources(tok, None, B_te, 1.0, parts=("web", "qa"))
    print(f"[src] loaded in {(time.time()-t0)/60:.1f} min", flush=True)

    if not PromptPool.exists("qa_eval_long"):
        out = []
        for b, L in LONG_TARGET.items():
            for kind in ("squad", "hotpot", "trivia"):
                got = 0
                for _ in range(a.n_qa * 5):
                    x = B_te.qa(L, kind)
                    if x is not None:
                        out.append(dict(x, bucket=b))
                        got += 1
                    if got >= a.n_qa:
                        break
                lens = [len(x["ids"]) for x in out if x["bucket"] == b and x["source"] == kind]
                print(f"[qa_eval_long] {b} {kind}: {got} prompts, {min(lens)}-{max(lens)} tokens", flush=True)
        PromptPool.save("qa_eval_long", out)
    if not PromptPool.exists("niah_eval_long"):
        rng, T = random.Random(78), Tok(tok)
        PromptPool.save("niah_eval_long", [dict(passcode_niah(T, B_te.web_texts, L, rng), bucket=str(L))
                                           for L in LONG_TARGET.values() for _ in range(a.n_niah)])
    print(f"[done] long pools ready in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
