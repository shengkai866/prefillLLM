#!/usr/bin/env python
"""
Round-2 data preparation (CPU). Loads web / QA / code / chat / math sources with disjoint train and test material,
then builds prompt pools under DATA_DIR/pools/:

  mix4k_train     mixed prompts, 256-3968 tokens           (stage-2 data for the mix-4k and mix-8k runs)
  mixlong_train   mixed prompts, 4096-8064 tokens          (extra stage-2 data for the mix-8k run)
  web4k_train     web-only prompts, 256-3968 tokens        (control run)
  mixtest         held-out, per type x {4k: 2048-3968, 8k: 6144-8064}   (KL vs 8B, and training validation)
  qa_eval         SQuAD v2 / HotpotQA / TriviaQA validation questions at ~3.5k and ~7.5k tokens (EM / F1)
  code_eval       next-line completion on held-out repos at ~3.5k and ~7.5k tokens (exact match)
  niah_eval       round-1 passcode needle test at 2048 / 3584 / 7680 tokens
Idempotent: pools that already exist are skipped. Use --scale to shrink everything for a smoke test.
"""
import argparse
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

os.environ.setdefault("HF_HOME", C.HF_CACHE_DIR)
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from prefill.data import TokenStore, save_store  # noqa: E402
from prefill.mixdata import Builders, PromptPool, Tok, build_pool, chunk_text, passcode_niah, stable_hash  # noqa: E402

BUCKETS = {"4k": (2048, 3968), "8k": (6144, 8064)}
EVAL_TARGET = {"4k": 3584, "8k": 7680}


def stream(repo, cfg, split, n, seed=0, buffer=5000):
    ds = load_dataset(repo, cfg, split=split, streaming=True).shuffle(seed=seed, buffer_size=buffer)
    for i, ex in enumerate(ds):
        if i >= n:
            break
        yield ex


def long_web(tok, n_per_source, n_test, min_tok=4224, max_tok=8448):
    """Long web documents (>= min_tok tokens) for 4k-8k prompts; stored as TokenStores 'web_long'."""
    if os.path.exists(os.path.join(C.DATA_DIR, "web_long", "test_offsets.npy")):
        return
    tr, te = [], []
    for name in ("fineweb_edu", "slimpajama"):
        repo, cfg, field = C.DATASETS[name]
        got, scanned, t0 = [], 0, time.time()
        for ex in stream(repo, cfg, "train", 10 ** 7, seed=11):
            scanned += 1
            if len(ex[field]) < min_tok * 3:
                continue
            ids = tok(ex[field], add_special_tokens=False)["input_ids"]
            if len(ids) >= min_tok:
                got.append(ids[:max_tok])
            if len(got) >= n_per_source + n_test:
                break
        print(f"[web_long] {name}: {len(got)} long docs from {scanned} scanned ({time.time()-t0:.0f}s)", flush=True)
        te += got[:n_test]
        tr += got[n_test:]
    save_store("web_long", "train", tr)
    save_store("web_long", "test", te)


def load_sources(tok, B_tr: Builders, B_te: Builders, scale, parts=("web", "qa", "code", "chat", "math")):
    """B_tr may be None (test material only); parts selects which sources to load (same streams and seeds either way)."""
    T = Tok(tok)
    n = lambda x: max(20, int(x * scale))
    # web: token docs for raw prompts, decoded texts for haystacks, a word list for synthetic names
    for B, split in ((B_tr, "train"), (B_te, "test")) if "web" in parts else ():
        if B is None:
            continue
        for name in ("fineweb_edu", "slimpajama", "web_long"):
            s = TokenStore(name, split)
            k = len(s) if split == "test" else min(len(s), n(12000))
            B.web_docs += [s.get(i).tolist() for i in range(k)]
        s = TokenStore("fineweb_edu", split)
        B.web_texts = [tok.decode(s.get(i, 1500)) for i in range(min(len(s), n(3000)))]
        words = set()
        for t in B.web_texts[:300]:
            words.update(w.lower() for w in re.findall(r"\b[a-zA-Z]{4,9}\b", t))
        B.words = sorted(words)
        print(f"[src] web {split}: {len(B.web_docs)} docs, {len(B.web_texts)} haystack texts, {len(B.words)} words", flush=True)

    # SQuAD v2 (answerable only)
    for B, split, k in ((B_tr, "train", n(30000)), (B_te, "validation", n(6000))) if "qa" in parts else ():
        if B is None:
            continue
        rows = [ex for ex in stream("rajpurkar/squad_v2", None, split, k, seed=1)]
        ctx = {}
        for ex in rows:
            ctx[ex["context"]] = 1
            if ex["answers"]["text"]:
                B.squad.append({"context": ex["context"], "question": ex["question"], "answers": list(dict.fromkeys(ex["answers"]["text"]))})
        B.squad_ctx = [(c, T.n(c)) for c in ctx]
        print(f"[src] squad {split}: {len(B.squad)} answerable, {len(B.squad_ctx)} contexts", flush=True)

    # HotpotQA distractor
    for B, split, k in ((B_tr, "train", n(6000)), (B_te, "validation", n(2000))) if "qa" in parts else ():
        if B is None:
            continue
        for ex in stream("hotpotqa/hotpot_qa", "distractor", split, k, seed=2):
            paras = [f"{t}: {''.join(s)}".strip() for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
            B.hotpot.append({"paragraphs": paras, "question": ex["question"], "answer": ex["answer"]})
            B.hotpot_par += [(p, T.n(p)) for p in paras[:3]]
        print(f"[src] hotpot {split}: {len(B.hotpot)} questions, {len(B.hotpot_par)} distractor paragraphs", flush=True)

    # TriviaQA (reading comprehension, Wikipedia evidence)
    for B, split, k in ((B_tr, "train", n(3000)), (B_te, "validation", n(1200))) if "qa" in parts else ():
        if B is None:
            continue
        for ex in stream("mandarjoshi/trivia_qa", "rc.wikipedia", split, k, seed=3, buffer=500):
            ctx = "\n\n".join(ex["entity_pages"]["wiki_context"])
            if not ctx:
                continue
            ans = list(dict.fromkeys([ex["answer"]["value"]] + ex["answer"]["aliases"]))
            if any(a.lower() in ctx.lower() for a in ans):
                B.trivia.append({"context": ctx, "question": ex["question"], "answers": ans})
            for ch in chunk_text(ctx)[:4]:
                B.trivia_ctx.append((ch, T.n(ch)))
        print(f"[src] trivia {split}: {len(B.trivia)} questions, {len(B.trivia_ctx)} distractor chunks", flush=True)

    if not {"code", "chat", "math"} & set(parts):
        return
    # code: held-out repositories by hash
    tr_files, te_files = [], []
    for ex in stream("codeparrot/codeparrot-clean", None, "train", n(120000), seed=4, buffer=20000):
        if ex["autogenerated"] or not (800 <= len(ex["content"]) <= 30000) or ex["alpha_frac"] < 0.3:
            continue
        item = (f"{ex['repo_name']}/{ex['path']}", ex["content"])
        (te_files if stable_hash(ex["repo_name"], 40) == 0 else tr_files).append(item)
        if len(tr_files) >= n(20000) and len(te_files) >= n(600):
            break
    B_tr.code_files, B_te.code_files = tr_files, te_files
    print(f"[src] code: {len(tr_files)} train files, {len(te_files)} held-out files", flush=True)

    # chat
    for B, split, k in ((B_tr, "train_sft", n(12000)), (B_te, "test_sft", n(2000))):
        B.chats = [[{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
                   for ex in stream("HuggingFaceH4/ultrachat_200k", None, split, k, seed=5)
                   if len(ex["messages"]) >= 2 and ex["messages"][0]["role"] == "user"]
        print(f"[src] chat {split}: {len(B.chats)} conversations", flush=True)

    # math (hash split, MetaMathQA has no test split)
    for ex in stream("meta-math/MetaMathQA", None, "train", n(40000), seed=6, buffer=20000):
        (B_te.math if stable_hash(ex["query"], 20) == 0 else B_tr.math).append((ex["query"], ex["response"]))
    print(f"[src] math: {len(B_tr.math)} train, {len(B_te.math)} held-out", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all sizes (smoke tests)")
    ap.add_argument("--n_mix4k", type=int, default=20000)
    ap.add_argument("--n_mixlong", type=int, default=6000)
    ap.add_argument("--n_web4k", type=int, default=20000)
    ap.add_argument("--n_test_per_cell", type=int, default=40)
    ap.add_argument("--n_qa_eval", type=int, default=100)
    ap.add_argument("--n_code_eval", type=int, default=100)
    ap.add_argument("--n_niah_eval", type=int, default=30)
    a = ap.parse_args()
    sc = lambda x: max(4, int(x * a.scale))
    tok = AutoTokenizer.from_pretrained(C.LARGE_DIR)
    t0 = time.time()
    long_web(tok, sc(1500), sc(80))
    B_tr, B_te = Builders(tok, "train", seed=100), Builders(tok, "test", seed=200)
    load_sources(tok, B_tr, B_te, a.scale)
    print(f"[src] all sources loaded in {(time.time()-t0)/60:.1f} min", flush=True)

    def make(name, fn):
        if PromptPool.exists(name):
            print(f"[pool] {name} exists, skip")
            return
        t = time.time()
        PromptPool.save(name, fn())
        print(f"[pool] {name} took {(time.time()-t)/60:.1f} min", flush=True)

    make("mixtest", lambda: [dict(x, bucket=b) for b, (lo, hi) in BUCKETS.items()
                             for typ in ("web", "qa", "code", "chat", "math", "retrieval")
                             for x in build_pool(B_te, sc(a.n_test_per_cell), lo, hi, mix={typ: 1.0}, strict=True)])
    make("mix4k_train", lambda: build_pool(B_tr, sc(a.n_mix4k), 256, 3968))
    make("web4k_train", lambda: build_pool(B_tr, sc(a.n_web4k), 256, 3968, mix={"web": 1.0}))
    make("mixlong_train", lambda: build_pool(B_tr, sc(a.n_mixlong), 4096, 8064))

    def qa_eval():
        out = []
        for b, L in EVAL_TARGET.items():
            for kind in ("squad", "hotpot", "trivia"):
                got = 0
                for _ in range(sc(a.n_qa_eval) * 5):
                    x = B_te.qa(L, kind)
                    if x is not None:
                        out.append(dict(x, bucket=b))
                        got += 1
                    if got >= sc(a.n_qa_eval):
                        break
        return out
    make("qa_eval", qa_eval)

    def code_eval():
        out = []
        for b, L in EVAL_TARGET.items():
            got = 0
            for _ in range(sc(a.n_code_eval) * 10):
                x = B_te.code_nextline(L)
                if x is not None:
                    out.append(dict(x, bucket=b))
                    got += 1
                if got >= sc(a.n_code_eval):
                    break
        return out
    make("code_eval", code_eval)

    rng = random.Random(77)
    T = Tok(tok)
    make("niah_eval", lambda: [dict(passcode_niah(T, B_te.web_texts, L, rng), bucket=str(L))
                               for L in (2048, 3584, 7680) for _ in range(sc(a.n_niah_eval))])
    print(f"[done] pools ready in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
