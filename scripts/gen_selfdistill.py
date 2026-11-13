#!/usr/bin/env python
"""
Self-distillation data: Qwen3-8B continues (raw prompts: exactly --max_new tokens) or answers (chat prompts: until
end of turn, at most --max_new tokens) every prompt of a pool. Writes DATA_DIR/<out>/:
  all_tokens.npy / all_offsets.npy   prompt + response tokens
  prompt_len.npy                     prompt length of every sequence
  meta.jsonl                         pool metadata (+ "pool_idx")
Work is done in shards (resumable after a crash); prompts are length-sorted and batched under a token budget.
Generator: --model (default the large model, config.LARGE_DIR); a model too big for one GPU (Qwen3-30B-A3B on L40S)
is split over the visible GPUs with --maxmem "22GiB,44GiB" (models.load_split). Same tokenizer as the pool required.
Several GPUs: run one process per GPU with --worker k --n_workers N (each does shards k, k+N, ...), then once more
without them to merge (all shards then exist, so it only merges).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import save_store  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import eos_ids, load_frozen, load_split  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


@torch.no_grad()
def generate_batch(model, prompts, raw, a, pad, eos, dev):
    maxL = max(len(p) for p in prompts)
    inp = torch.full((len(prompts), maxL), pad, dtype=torch.long)
    att = torch.zeros_like(inp)
    for i, p in enumerate(prompts):
        inp[i, maxL - len(p):] = torch.as_tensor(p)
        att[i, maxL - len(p):] = 1
    kw = dict(max_new_tokens=a.max_new, pad_token_id=pad)
    if a.greedy:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        kw.update(do_sample=True, temperature=a.temperature, top_p=a.top_p)
    if raw:
        kw["min_new_tokens"] = a.max_new
    out = model.generate(input_ids=inp.to(dev), attention_mask=att.to(dev), **kw)[:, maxL:].cpu().tolist()
    res = []
    for row in out:
        if not raw:
            cut = next((j for j, t in enumerate(row) if t in eos), None)
            row = row[:cut + 1] if cut is not None else row
        res.append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_tokens", type=int, default=98304)
    ap.add_argument("--max_batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--model", default=C.LARGE_DIR, help="generator (default: the large model)")
    ap.add_argument("--maxmem", default="", help="split the generator over GPUs 0..n-1 with these per-GPU caps")
    a = ap.parse_args()
    out_dir = os.path.join(C.DATA_DIR, a.out)
    if os.path.exists(os.path.join(out_dir, "meta.jsonl")):
        print(f"[selfdistill] {a.out} exists, skip")
        return
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = set(eos_ids(tok))
    model = load_split(a.model, a.maxmem) if a.maxmem else load_frozen(a.model, dev)
    print(f"[selfdistill] generator {a.model}", flush=True)
    pool = PromptPool(a.pool)
    lens = np.diff(pool.offsets)
    order = sorted(range(len(pool)), key=lambda i: (pool.meta[i]["fmt"], -lens[i]))
    shards = [order[k:k + a.shard] for k in range(0, len(order), a.shard)]
    print(f"[selfdistill] {a.pool} -> {a.out}: {len(pool)} prompts, {len(shards)} shards, greedy={a.greedy}", flush=True)
    t0, done = time.time(), 0
    for si, idxs in enumerate(shards):
        path = os.path.join(shard_dir, f"{si:04d}.json")
        if si % a.n_workers != a.worker:
            continue
        if os.path.exists(path):
            done += len(idxs)
            continue
        recs = []
        for fmt in ("raw", "chat"):
            group = [i for i in idxs if pool.meta[i]["fmt"] == fmt]
            group.sort(key=lambda i: -lens[i])
            j = 0
            while j < len(group):
                bs = max(1, min(a.max_batch, a.batch_tokens // (int(lens[group[j]]) + a.max_new)))
                batch = group[j:j + bs]
                j += bs
                resp = generate_batch(model, [pool.ids(i).tolist() for i in batch], fmt == "raw", a, pad, eos, dev)
                recs += [{"i": i, "r": r} for i, r in zip(batch, resp)]
        json.dump(recs, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        done += len(idxs)
        el = time.time() - t0
        print(f"[selfdistill] shard {si + 1}/{len(shards)}: {done}/{len(pool)} prompts, {el/60:.1f} min elapsed", flush=True)

    missing = [si for si in range(len(shards)) if not os.path.exists(os.path.join(shard_dir, f"{si:04d}.json"))]
    if missing:
        print(f"[selfdistill] worker {a.worker}/{a.n_workers} done; {len(missing)} shards left to other workers, not merging", flush=True)
        return
    seqs, plens, metas = [], [], []
    for si in range(len(shards)):
        for rec in json.load(open(os.path.join(shard_dir, f"{si:04d}.json"))):
            p = pool.ids(rec["i"]).tolist()
            if not rec["r"]:
                continue
            seqs.append(p + rec["r"])
            plens.append(len(p))
            metas.append(dict(pool.meta[rec["i"]], pool_idx=rec["i"], resp_len=len(rec["r"])))
    save_store(a.out, "all", seqs)
    np.save(os.path.join(out_dir, "prompt_len.npy"), np.array(plens, dtype=np.int64))
    with open(os.path.join(out_dir, "meta.jsonl"), "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    rl = np.array([m["resp_len"] for m in metas])
    print(f"[selfdistill] {a.out}: {len(seqs)} sequences, mean response {rl.mean():.0f} tokens, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
