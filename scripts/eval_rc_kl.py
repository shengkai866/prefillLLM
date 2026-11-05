#!/usr/bin/env python
"""
KL(large || ours) on sd_mixtest (the large model's own held-out responses, 6 types x 4k / 8k) for several projectors and
recompute fractions, on the SAME examples (paired comparison). Inference path: RecomputeHybrid.prefill_sel with
attn_small, sentence units (= "ours"), fraction 0 = plain hybrid. Used by the moe-router branch to compare a
recompute-aware projector (rc_*) with its base.
  python scripts/eval_rc_kl.py --projectors base=<path>,rc=<path> --fracs 0,0.05,0.1,0.2 --out_dir results/rc_kl_06
Outputs results.json + summary.md; per-example rows in per_example.jsonl (restart resumes).
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import TokenStore  # noqa: E402
from prefill.fast import FastBig, FastQwen3, build_fold, kv_to_dynamic_cache  # noqa: E402
from prefill.kv import kl_teacher_student  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402
from prefill.recompute import RecomputeHybrid  # noqa: E402


def mean(x):
    return statistics.mean(x) if x else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projectors", required=True, help="tag=path,...")
    ap.add_argument("--fracs", default="0,0.05,0.1,0.2")
    ap.add_argument("--n_per_cell", type=int, default=20)
    ap.add_argument("--buckets", default="4k,8k")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    S, L = FastQwen3(small), FastQwen3(large)
    fb = FastBig(L)
    boundary = torch.tensor([any(ch in tok.decode([i]) for ch in ".!?\n") for i in range(len(tok))], device=dev)
    projs = {}
    for item in a.projectors.split(","):
        tag, path = item.split("=")
        p, ck = Projector.load(path, dev)
        p.eval()
        projs[tag] = RecomputeHybrid(S, L, build_fold(p, L), n_sink=8, n_tail=64, boundary=boundary)
        print(f"[projector] {tag}: {path} step {ck.get('step')}", flush=True)
    fracs = [float(f) for f in a.fracs.split(",")]
    st = TokenStore("sd_mixtest", "all")
    pl = np.load(os.path.join(C.DATA_DIR, "sd_mixtest", "prompt_len.npy"))
    meta = [json.loads(l) for l in open(os.path.join(C.DATA_DIR, "sd_mixtest", "meta.jsonl"))]
    cnt, todo = collections.Counter(), []
    for i, m in enumerate(meta):
        if m["bucket"] in a.buckets.split(",") and cnt[(m["type"], m["bucket"])] < a.n_per_cell:
            cnt[(m["type"], m["bucket"])] += 1
            todo.append(i)
    raw = os.path.join(a.out_dir, "per_example.jsonl")
    done = {}
    if os.path.exists(raw):
        for l in open(raw):
            r = json.loads(l)
            done[r["i"]] = r
    keys = [f"{t}@{f:g}" for t in projs for f in fracs]
    rows = [done[i] for i in todo if i in done and all(k in done[i]["kl"] for k in keys)]
    have = {r["i"] for r in rows}
    t0 = time.time()

    @torch.no_grad()
    def cont(ids, P, kb, vb, lg0):
        mc = ids.shape[0] - P
        if mc == 1:
            return lg0
        out = large(input_ids=ids[None, P:-1], past_key_values=kv_to_dynamic_cache(kb, vb), use_cache=True, logits_to_keep=mc - 1)
        return torch.cat([lg0, out.logits[0]], 0)

    with open(raw, "a") as fo, torch.no_grad():
        for i in todo:
            if i in have:
                continue
            ids = st.get(i).to(dev)
            P = int(pl[i])
            kb, vb, lg = fb.prefill(ids[:P])
            teacher = cont(ids, P, kb, vb, lg)
            del kb, vb
            r = {"i": i, "type": meta[i]["type"], "bucket": meta[i]["bucket"], "P": P, "kl": {}, "agree": {}}
            for tag, rh in projs.items():
                for f in fracs:
                    kb, vb, lg0, n = rh.prefill_sel(ids[:P], "attn_small", f, seg="sent")
                    s = cont(ids, P, kb, vb, lg0)
                    r["kl"][f"{tag}@{f:g}"] = kl_teacher_student(teacher[None], s[None]).mean().item()
                    r["agree"][f"{tag}@{f:g}"] = (teacher.argmax(-1) == s.argmax(-1)).float().mean().item()
                    del kb, vb
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            rows.append(r)
            print(f"[{len(rows)}/{len(todo)}] {r['type']:>9} {r['bucket']} " +
                  " ".join(f"{k}={v:.3f}" for k, v in r["kl"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)

    types = sorted({r["type"] for r in rows})
    res = {"meta": {"args": vars(a), "n": len(rows), "time": time.strftime("%F %T"), "gpu": torch.cuda.get_device_name(0)},
           "kl": {}}
    md = [f"# KL(large || ours) on sd_mixtest, paired ({len(rows)} examples)", "",
          "ours = recompute the given fraction of middle tokens (small-model attention, sentence units); 0 = plain hybrid.", ""]
    for b in a.buckets.split(","):
        rs = [r for r in rows if r["bucket"] == b]
        md += [f"## {b} (n = {len(rs)})", "", "| projector @ frac | KL | top-1 | " + " | ".join(types) + " |",
               "|---|---|---|" + "---|" * len(types)]
        for k in keys:
            v = {"kl": mean([r["kl"][k] for r in rs]), "agree": mean([r["agree"][k] for r in rs]),
                 "kl_by_type": {t: mean([r["kl"][k] for r in rs if r["type"] == t]) for t in types}}
            res["kl"].setdefault(k, {})[b] = v
            md.append(f"| {k} | {v['kl']:.4f} | {v['agree']:.3f} | " + " | ".join(f"{v['kl_by_type'][t]:.3f}" for t in types) + " |")
        md.append("")
    json.dump(res, open(os.path.join(a.out_dir, "results.json"), "w"), indent=1)
    open(os.path.join(a.out_dir, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
