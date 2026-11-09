#!/usr/bin/env python
"""
Serving costs beyond batch-1 TTFT, for one model setup (small / large from PREFILL_SMALL_DIR / PREFILL_LARGE_DIR):

  decode  end-to-end latency = TTFT (every method CUDA-graphed, prefill/graphs.py, as speed_allgraph) + greedy decoding
          of N-1 more tokens with a CUDA-graphed decode step (prefill/serving.Decoder) on the model that decodes for
          that method: the large model for big / hyb / ours / SpecPrefill (SpecPrefill: on its compact cache), the small
          model for `small`. Cache contents are random for the decode part (its cost does not depend on them).
  memory  per method: weights it needs (models it runs + projector fold tensors), peak extra memory of one eager
          prefill, and the KV cache that decoding keeps (cache rows x bytes per row of the decoding model).
  batch   prefill throughput with B equal-length prompts in one batched call (prefill/serving.py): big / small alone
          (dense_prefill_b) and hyb / ours (OursBatched: one batched small call, eager per-prompt selection, per-prompt
          fold, one batched large call). SpecPrefill is not batched (its look-ahead loop is batch-1 only).
  check   GPU sanity: graphed Decoder vs HF greedy decoding from a real large-model cache; batched vs batch-1 logits.

  python scripts/bench_e2e.py --projector P --out_dir results/e2e_06 --configs big,small,hyb,attn_small:0.1:sent,specprefill:0.1
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from eval_recompute import Engine, graph_kind, parse  # noqa: E402
from prefill import config as C  # noqa: E402
from prefill.fast import FastBig, Graphed, kv_to_dynamic_cache, time_call  # noqa: E402
from prefill.graphs import build as build_graphs  # noqa: E402
from prefill.kv import greedy_decode  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.recompute import extend_at  # noqa: E402
from prefill.serving import Decoder, OursBatched, dense_prefill_b  # noqa: E402


def ints(s):
    return [int(x) for x in s.split(",") if x]


def rand_ids(T, B=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (T,) if B is None else (B, T), generator=g).cuda()


def free():
    gc.collect()
    torch.cuda.empty_cache()


def kv_row_bytes(M):
    return 2 * M.n_layers * M.Hkv * M.D * M.embed.element_size()


def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def fold_bytes(fold):
    return sum(v.numel() * v.element_size() for v in vars(fold).values() if torch.is_tensor(v))


def decode_setup(E, cfg, ids):
    """(decoding model, cache rows, RoPE position of the first fed-back token) for method cfg on prompt ids."""
    T = ids.shape[0]
    if cfg == "small":
        return E.S, T, T
    s, _, _ = parse(cfg)
    if s == "specprefill" or s.startswith("drop-"):
        kb = E.prefill(cfg, ids)[0]
        rows = kb.shape[2]
        del kb
        return E.L, rows, T
    return E.L, T, T


def time_decode(M, rows, pos0, gens):
    """Seconds for N-1 graphed decode steps after a cache of `rows` rows, for every N in gens."""
    nmax = max(gens)
    dec = Decoder(M, rows + nmax)
    dec.reset(rows, pos0, 1000)
    for _ in range(4):
        dec.step()
    dec.reset(rows, pos0, 1000)
    out = {1: 0.0}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for j in range(1, nmax):
        dec.step()
        if j + 1 in gens:
            torch.cuda.synchronize()
            out[j + 1] = time.perf_counter() - t0
    del dec
    free()
    return out


def task_decode(E, cfgs, lengths, gens, reps):
    rows = []
    for T in lengths:
        ids = rand_ids(T)
        for c in cfgs:
            row = {"prompt_len": T, "config": c}
            try:
                kind, f = graph_kind(c)
                run = build_graphs(E.rh, kind, f, ids)
                row["ttft"] = time_call(lambda: run(ids), repeats=reps, warmup=2)
                del run
                free()
                M, n, p0 = decode_setup(E, c, ids)
                row.update(decoder="small" if M is E.S else "large", cache_rows=n)
                row["decode"] = {str(k): v for k, v in time_decode(M, n, p0, gens).items()}
                row["e2e"] = {k: row["ttft"] + v for k, v in row["decode"].items()}
                row["ms_per_token"] = row["decode"][str(max(gens))] / (max(gens) - 1) * 1e3
            except torch.OutOfMemoryError:
                row["oom"] = True
            free()
            rows.append(row)
            print(f"[decode] {T} {c:>22}: " + ("OOM" if row.get("oom") else
                  f"ttft {row['ttft']*1e3:.0f} ms, {row['ms_per_token']:.1f} ms/token ({row['decoder']}, {row['cache_rows']} rows), "
                  + " ".join(f"N={k}:{v*1e3:.0f}" for k, v in row["e2e"].items())), flush=True)
    return rows


def task_memory(E, cfgs, lengths, gens, small_hf, large_hf):
    W = {"small": param_bytes(small_hf), "large": param_bytes(large_hf), "fold": fold_bytes(E.rh.fold)}
    rows = []
    for T in lengths:
        ids = rand_ids(T)
        for c in cfgs:
            s, _, _ = parse(c)
            uses = ["small"] if c == "small" else ["large"] if c == "big" else ["small", "large"] + (["fold"] if s in ("hyb", "attn_small") else [])
            row = {"prompt_len": T, "config": c, "weights": sum(W[u] for u in uses), "uses": uses}
            free()
            try:
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                out = E.prefill(c, ids)
                torch.cuda.synchronize()
                row["prefill_peak"] = torch.cuda.max_memory_allocated() - base
                del out
                M, n, _ = decode_setup(E, c, ids)
                row["kv_rows"] = n
                row["kv_cache"] = {str(N): (n + N) * kv_row_bytes(M) for N in gens}
            except torch.OutOfMemoryError:
                row["oom"] = True
            free()
            rows.append(row)
            g = 2 ** 30
            print(f"[memory] {T} {c:>22}: " + ("OOM" if row.get("oom") else
                  f"weights {row['weights']/g:.2f} GB, prefill peak +{row['prefill_peak']/g:.2f} GB, "
                  f"KV ({n} rows +{max(gens)}) {row['kv_cache'][str(max(gens))]/g:.2f} GB"), flush=True)
    return rows


def batched_runner(E, cfg, T, B):
    """-> run() for a batch of B random prompts of length T (graph-captured), and the prompts."""
    ids = rand_ids(T, B, seed=B)
    s, f, _ = parse(cfg)
    if cfg in ("big", "small"):
        M = E.L if cfg == "big" else E.S
        kb = torch.empty(B, M.n_layers, M.Hkv, T, M.D, dtype=M.embed.dtype, device="cuda")
        vb = torch.empty_like(kb)
        g = Graphed(lambda x: dense_prefill_b(M, x, kb, vb)[0], ids, copy_inputs=False)
        return (lambda: g(ids)), ids
    assert s in ("hyb", "attn_small"), cfg
    ob = OursBatched(E.rh, T, f if s == "attn_small" else 0.0)
    g1 = Graphed(ob.stage1, ids, copy_inputs=False)
    hs, sc = g1(ids)
    pos0 = ob.select(ids, sc)
    kb, vb = ob.new_buffers(B)
    g2 = Graphed(lambda h, x, p: ob.stage2(h, x, p, kb, vb), hs, ids, pos0, copy_inputs=False)

    def run():
        h, sc_ = g1(ids)
        pos = ob.select(ids, sc_)
        assert pos.shape == pos0.shape
        return g2(h, ids, pos)
    return run, ids


def task_batch(E, cfgs, lengths, batches, reps):
    rows = []
    for T in lengths:
        for B in batches:
            row = {"prompt_len": T, "batch": B}
            for c in cfgs:
                try:
                    run, _ = batched_runner(E, c, T, B)
                    row[c] = time_call(run, repeats=reps, warmup=2)
                    del run
                except torch.OutOfMemoryError:
                    row[c] = None
                free()
            rows.append(row)
            print(f"[batch] T={T} B={B}: " + " ".join(f"{c}=" + ("OOM" if row[c] is None else
                  f"{row[c]*1e3:.0f}ms ({B*T/row[c]/1e3:.1f}k tok/s)") for c in cfgs), flush=True)
    return rows


def task_check(E, large_hf):
    out = {}
    T, N = 4096, 16
    ids = rand_ids(T, seed=123)
    kb, vb, lg = FastBig(E.L).prefill(ids)
    ref = greedy_decode(large_hf, kv_to_dynamic_cache(kb, vb), lg, N)
    rows = Decoder.capacity_rows(T + N)
    kbd = torch.zeros(E.L.n_layers, E.L.Hkv, rows, E.L.D, dtype=kb.dtype, device="cuda")
    vbd = torch.zeros_like(kbd)
    kbd[:, :, :T], vbd[:, :, :T] = kb, vb
    dec = Decoder(E.L, T + N, kbd, vbd)
    dec.reset(T, T, int(lg.argmax(-1)))
    got = [int(dec.tok)]
    for _ in range(N - 1):
        dec.step()
        got.append(int(dec.tok))
    out["decoder_tokens_match"] = sum(a == b for a, b in zip(got, ref)) / N
    del kb, vb, kbd, vbd, dec
    free()
    B, T = 2, 4096
    ids = rand_ids(T, B, seed=7)
    lgb = dense_prefill_b(E.L, ids)[0]
    lg1 = torch.cat([FastBig(E.L).prefill(ids[b])[2] for b in range(B)])
    out["dense_batched_rel_diff"] = ((lgb.float() - lg1.float()).abs().max() / lg1.float().abs().max()).item()
    ob = OursBatched(E.rh, T, 0.1)
    hs, sc = ob.stage1(ids)
    pos = ob.select(ids, sc)
    kbo, vbo = ob.new_buffers(B)
    lgo = ob.stage2(hs, ids, pos, kbo, vbo)
    ref = []
    for b in range(B):
        k1, v1 = E.rh.stage_fold(E.rh.stage_small(ids[b, :ob.s + ob.m])[:, ob.s:], ob.s, T)
        ref.append(extend_at(E.L, ids[b][pos[b]], pos[b], k1, v1)[0])
    ref = torch.cat(ref)
    out["ours_batched_rel_diff"] = ((lgo.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    free()
    print("[check]", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
    ap.add_argument("--configs", default="big,small,hyb,attn_small:0.1:sent,specprefill:0.1")
    ap.add_argument("--batch_configs", default="big,small,hyb,attn_small:0.1:sent")
    ap.add_argument("--tasks", default="check,decode,memory,batch")
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--gens", default="1,16,64,256,1024", help="output lengths N (tokens incl. the first)")
    ap.add_argument("--batch_lengths", default="4096,8192,16384")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    E = Engine(tok, small, large, a.projector, dev, a.n_sink, a.n_tail)
    cfgs = [c for c in a.configs.split(",") if c]
    res["meta"] = {"gpu": torch.cuda.get_device_name(0), "small": C.SMALL_NAME, "large": C.LARGE_NAME, "args": vars(a),
                   "time": time.strftime("%F %T"), "fold": os.environ.get("PREFILL_FOLD", "fused")}
    gens = ints(a.gens)
    with torch.no_grad():
        for task in a.tasks.split(","):
            t0 = time.time()
            if task == "check":
                res["check"] = task_check(E, large)
            elif task == "decode":
                res["decode"] = task_decode(E, cfgs, ints(a.lengths), gens, a.reps)
            elif task == "memory":
                res["memory"] = task_memory(E, cfgs, ints(a.lengths), gens, small, large)
            elif task == "batch":
                res["batch"] = task_batch(E, [c for c in a.batch_configs.split(",") if c], ints(a.batch_lengths),
                                          ints(a.batches), a.reps)
            else:
                raise ValueError(task)
            json.dump(res, open(path, "w"), indent=1)
            print(f"[task {task}] {(time.time() - t0) / 60:.1f} min", flush=True)
    print("[done]", a.out_dir)


if __name__ == "__main__":
    main()
