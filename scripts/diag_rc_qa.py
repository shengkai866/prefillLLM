#!/usr/bin/env python
"""
Why does the recompute-aware projector (rc) lower KL on sd_mixtest but not QA F1? Per QA benchmark prompt (qa_eval, the
same first 50 per source x bucket as eval_recompute's qa task):
  * the large model's own greedy answer (32 tokens, as the benchmark decodes)
  * KL(large || base) and KL(large || rc), teacher-forced on that answer, at recompute 0 and 10% (attn_small, sentences)
  * the selected middle tokens (identical for base and rc: they depend only on the small model) and whether they cover
    the gold answer span in the document ("covered" / "missed" / "absent" when no gold string occurs in the prompt)
Per-example rows -> per_example.jsonl; joined offline with the benchmark generations (F1 of base / rc).
  python scripts/diag_rc_qa.py --base <path> --rc <path> --out_dir results/rc_qa_<tag>
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import collections  # noqa: E402

import torch  # noqa: E402

from prefill.fast import FastBig, FastQwen3, build_fold, kv_to_dynamic_cache  # noqa: E402
from prefill.kv import greedy_decode, kl_teacher_student, split_prompt  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import eos_ids, load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402
from prefill.recompute import RecomputeHybrid, choose_units, find_spans, segment_ids, small_prefill_scored  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rc", required=True)
    ap.add_argument("--n", type=int, default=50, help="per source x bucket (eval_recompute --n_qa)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    S, L = FastQwen3(small), FastQwen3(large)
    fb = FastBig(L)
    eos = eos_ids(tok)
    boundary = torch.tensor([any(ch in tok.decode([i]) for ch in ".!?\n") for i in range(len(tok))], device=dev)
    rh = {}
    for tag, path in (("base", a.base), ("rc", a.rc)):
        p, _ = Projector.load(path, dev)
        p.eval()
        rh[tag] = RecomputeHybrid(S, L, build_fold(p, L), n_sink=8, n_tail=64, boundary=boundary)
    pool = PromptPool("qa_eval")
    cells = collections.defaultdict(list)
    for i, m in enumerate(pool.meta):
        cells[(m["source"], m["bucket"])].append(i)
    todo = [i for (src, b), idx in sorted(cells.items()) if b in ("4k", "8k") for i in idx[:a.n]]
    t0 = time.time()
    with open(os.path.join(a.out_dir, "per_example.jsonl"), "w") as fo, torch.no_grad():
        for j, i in enumerate(todo):
            m = pool.meta[i]
            ids = torch.as_tensor(pool.ids(i), device=dev)
            P = ids.shape[0]
            kb, vb, lg = fb.prefill(ids)
            ans_ids = greedy_decode(large, kv_to_dynamic_cache(kb, vb), lg, 32, eos)
            del kb, vb
            full = torch.cat([ids, torch.as_tensor(ans_ids, device=dev)])
            mc = len(ans_ids)

            def cont(kb, vb, lg0):
                if mc == 1:
                    return lg0
                out = large(input_ids=full[None, P:-1], past_key_values=kv_to_dynamic_cache(kb, vb), use_cache=True,
                            logits_to_keep=mc - 1)
                return torch.cat([lg0, out.logits[0]], 0)

            kb, vb, lg = fb.prefill(ids)
            teacher = cont(kb, vb, lg)
            del kb, vb
            r = {"i": i, "source": m["source"], "bucket": m["bucket"], "P": P, "answer_8b": tok.decode(ans_ids, skip_special_tokens=True),
                 "n_ans_tok": mc, "kl": {}, "top1": {}}
            for tag in ("base", "rc"):
                for f in (0.0, 0.1):
                    kb, vb, lg0, _ = rh[tag].prefill_sel(ids, "attn_small", f, seg="sent")
                    s = cont(kb, vb, lg0)
                    del kb, vb
                    k = f"{tag}@{f:g}"
                    r["kl"][k] = kl_teacher_student(teacher[None], s[None]).mean().item()
                    r["kl_first"] = r.get("kl_first", {})
                    r["kl_first"][k] = kl_teacher_student(teacher[None, :1], s[None, :1]).mean().item()
                    r["top1"][k] = (teacher.argmax(-1) == s.argmax(-1)).float().mean().item()
            # selection (same for both projectors) vs gold answer span
            s0, mm, t = split_prompt(P, 8, 64)
            _, score = small_prefill_scored(S, ids, t, s0, s0 + mm)
            sel = choose_units(score, segment_ids(ids[s0:s0 + mm], "sent", boundary, 64), int(round(0.1 * mm)))
            needles = []
            for x in m["answers"]:
                for v in (x, " " + x):
                    needles.append(tok.encode(v, add_special_tokens=False))
            span = find_spans(ids[s0:s0 + mm], needles, pad=0)
            if not bool(span.any()):
                cov = "absent"
            else:
                hit = torch.zeros(mm, dtype=torch.bool, device=dev)
                hit[sel] = True
                cov = "covered" if bool((hit & span).any()) else "missed"
            r["coverage"] = cov
            r["span_frac_selected"] = float((span[sel].sum() / span.sum().clamp_min(1)).item())
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if j % 25 == 0:
                print(f"[{j + 1}/{len(todo)}] {m['source']} {m['bucket']} cov={cov} " +
                      " ".join(f"{k}={v:.3f}" for k, v in r["kl"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
