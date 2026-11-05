#!/usr/bin/env python
"""
Routing diagnostic for a "MoE projector" (branch moe-router): which cells of the (layer x middle-token) grid should the
large model compute itself instead of taking the projected K/V? No training; one existing projector.

For every held-out example (sd_mixtest: the large model's own responses, 6 data types x 4k / 8k) we build
  * the true large-model prefill  (K/V and residual stream of every layer)
  * the plain hybrid               (sink + tail true, middle projected)
and score many cache variants by KL(large || variant) and top-1 agreement on the continuation.

Oracle substitution ("o:*"): the true K/V is pasted into a set of (layer, token) cells, then the large model runs the
tail as usual. Upper bound of routing that cell to the large model (the real recompute of a cell sees projected
neighbours, the pasted one does not).
  o:none / o:all                  plain hybrid / every middle cell true (sanity: KL ~ 0)
  o:band<a>                       layers a..a+3, all tokens                       (11% of the cells)
  o:first<n> / o:last<n>          first / last n layers, all tokens
  o:tok_{rand,attn,err}           10% of the tokens, all layers: random / small-model attention on the tail, sentence
                                  units (the selection of "ours") / largest projection error summed over layers
  o:cell_err                      the 10% of all cells with the largest error (unstructured routing)
  o:cell_err_pl                   per layer, the 10% of tokens with the largest error in that layer
  o:tok_{err,attn}20_{last,first}18   20% of the tokens, only in the last / first 18 layers (10% of the cells)
Real recompute ("r:*"): the large model actually computes the cells.
  r:tok_{rand,attn,err}[20]       whole tokens, all layers (r:tok_attn = "ours", recompute 10% sentences)
  r:sfx_{err,attn}20_a18          20% of the tokens join the large model at layer 18 with the PROJECTOR's predicted
                                  residual as input and run layers 18..35 (same cells as o:tok_*20_last18)
  r:sfx_err20_a18_true            same with the TRUE residual at layer 18 as input (residual error removed)
Error of a cell = |k^ - k|^2/|k|^2 + |v^ - v|^2/|v|^2 (per token, per layer, over all KV heads).
Statistics per layer: mean error, cosine of the predicted vs true residual stream, share of the error in the top
10% cells, rank correlation of token errors across layers, overlap of the err / attn token selections.

Outputs results/<out_dir>/results.json + summary.md (written every --save_every examples; restart resumes).
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
import torch.nn.functional as F  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import TokenStore  # noqa: E402
from prefill.fast import FastHybrid, FastQwen3, build_fold, causal_mask_for, kv_to_dynamic_cache, rms  # noqa: E402
from prefill.kv import kl_teacher_student, split_prompt  # noqa: E402
from prefill.models import load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402
from prefill.recompute import choose_units, extend_at, segment_ids, small_prefill_scored  # noqa: E402


def topk_idx(score, k):
    return torch.topk(score, k).indices.sort().values


def run_suffix(L, ids, kbuf, vbuf, s, m, idx, a, h_in):
    """Sink + tail tokens run every layer; middle tokens s+idx join at layer a with residual h_in [n, d] and run
    layers a..Ll-1. Their K/V overwrite the buffer rows in those layers. Returns last-token logits [1, V]."""
    P, dev = ids.shape[0], ids.device
    ar = lambda x, y: torch.arange(x, y, device=dev)
    base = torch.cat([ar(0, s), ar(s + m, P)])
    allp = torch.cat([base, s + idx])
    order = torch.argsort(allp)
    pos_all = allp[order]
    h = F.embedding(L.ids_in(ids[base]), L.embed)
    pos = base
    mask = causal_mask_for(pos, P, L.embed.dtype, dev, L.rep)[None, None]
    for i, lw in enumerate(L.layers):
        if i == a:
            h = torch.cat([h, h_in.to(h.dtype)])[order]
            pos = pos_all
            mask = causal_mask_for(pos, P, L.embed.dtype, dev, L.rep)[None, None]
        Q = pos.shape[0]
        q, k, v = L._qkv(lw, rms(h, lw["ln1"], L.eps), L.cos[pos], L.sin[pos])
        kbuf[i].index_copy_(1, pos, k)
        vbuf[i].index_copy_(1, pos, v)
        qg = q.reshape(L.Hkv, L.rep * Q, L.D)
        o = F.scaled_dot_product_attention(qg[None], kbuf[i][None], vbuf[i][None], attn_mask=mask, scale=L.scaling)[0]
        h = L._mlp(lw, L._out(lw, h, o.reshape(L.H, Q, L.D)))
    return F.linear(rms(h[-1:], L.norm, L.eps), L.lm_head)


def spearman_rows(x):
    """Mean off-diagonal Spearman correlation between the rows of x [R, n]."""
    r = x.argsort(1).argsort(1).float()
    r = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    c = (r @ r.T) / (x.shape[1] - 1)
    R = c.shape[0]
    return ((c.sum() - c.diag().sum()) / (R * (R - 1))).item()


class Diag:
    def __init__(self, a, dev):
        self.a, self.dev = a, dev
        self.tok, self.small, self.large = load_pair(dev)
        self.S, self.L = FastQwen3(self.small), FastQwen3(self.large)
        self.proj, ck = Projector.load(a.projector, dev)
        self.proj.eval()
        print(f"[projector] {a.projector} step {ck.get('step')}", flush=True)
        self.H = FastHybrid(self.S, self.L, build_fold(self.proj, self.L), a.n_sink, a.n_tail)
        self.boundary = torch.tensor([any(ch in self.tok.decode([i]) for ch in ".!?\n") for i in range(len(self.tok))],
                                     device=dev)
        self.gen = torch.Generator(device="cpu").manual_seed(0)

    @torch.no_grad()
    def cont(self, ids, P, kb, vb, lg0):
        """[m, V] logits predicting ids[P:] from the prompt cache kb/vb and the first-token logits lg0."""
        mc = ids.shape[0] - P
        if mc == 1:
            return lg0
        out = self.large(input_ids=ids[None, P:-1], past_key_values=kv_to_dynamic_cache(kb, vb), use_cache=True,
                         logits_to_keep=mc - 1)
        return torch.cat([lg0, out.logits[0]], 0)

    @torch.no_grad()
    def example(self, ids, P):
        a, L, dev = self.a, self.L, self.dev
        prompt = ids[:P]
        s, m, t = split_prompt(P, a.n_sink, a.n_tail)
        Ll = L.n_layers
        ar = lambda x, y: torch.arange(x, y, device=dev)
        # true large-model prefill
        kT = torch.empty(Ll, L.Hkv, P, L.D, dtype=L.embed.dtype, device=dev)
        vT = torch.empty_like(kT)
        lgT, hT = L.prefill(prompt, kT, vT, collect=True)
        hT_mid = hT[:Ll, s:s + m]  # residual-stream INPUT of every layer, middle tokens
        del hT
        teacher = self.cont(ids, P, kT, vT, lgT)
        # small model (+ its tail attention on the middle = the "ours" scorer) and the projected middle K/V
        hs, att = small_prefill_scored(self.S, prompt, t, s, s + m)
        hs = hs[:, s:s + m]
        kP, vP = self.H.stage_fold(hs, s, P)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h_hat = self.proj(hs[:, None])[:, 0]  # [Ll, m, d_l] predicted residual inputs
        kTm, vTm = kT[:, :, s:s + m], vT[:, :, s:s + m]
        dk = (kP[:, :, s:s + m].float() - kTm.float()).pow(2).sum((1, 3)) / kTm.float().pow(2).sum((1, 3)).clamp_min(1e-6)
        dv = (vP[:, :, s:s + m].float() - vTm.float()).pow(2).sum((1, 3)) / vTm.float().pow(2).sum((1, 3)).clamp_min(1e-6)
        err = dk + dv  # [Ll, m]
        rcos = torch.stack([F.cosine_similarity(h_hat[l].float(), hT_mid[l].float(), dim=-1) for l in range(Ll)])  # [Ll, m]

        k10 = max(1, int(round(0.1 * m)))
        k20 = max(1, int(round(0.2 * m)))
        units = segment_ids(prompt[s:s + m], "sent", self.boundary, 64)
        sel = {
            "rand": torch.randperm(m, generator=self.gen)[:k10].sort().values.to(dev),
            "attn": choose_units(att, units, k10),
            "err": topk_idx(err.sum(0), k10),
            "attn20": choose_units(att, units, k20),
            "err20": topk_idx(err.sum(0), k20),
            "rand20": torch.randperm(m, generator=self.gen)[:k20].sort().values.to(dev),
        }
        nc = Ll * m
        stats = {
            "m": m,
            "err_layer": err.mean(1).tolist(),
            "rcos_layer": rcos.mean(1).tolist(),
            "err_share_top10_cells": (torch.topk(err.flatten(), max(1, nc // 10)).values.sum() / err.sum()).item(),
            "err_share_top10_tokens": (torch.topk(err.sum(0), k10).values.sum() / err.sum()).item(),
            "err_layer_spearman": spearman_rows(err),
            "overlap_err_attn": len(set(sel["err"].tolist()) & set(sel["attn"].tolist())) / k10,
            "overlap_err_rand": len(set(sel["err"].tolist()) & set(sel["rand"].tolist())) / k10,
            "n_attn": int(sel["attn"].numel()),
        }

        def tok_mask(idx, lo=0, hi=Ll):
            M = torch.zeros(Ll, m, dtype=torch.bool, device=dev)
            M[lo:hi, idx] = True
            return M

        masks = {"o:none": torch.zeros(Ll, m, dtype=torch.bool, device=dev),
                 "o:all": torch.ones(Ll, m, dtype=torch.bool, device=dev)}
        for b in range(0, Ll, 4):
            masks[f"o:band{b:02d}"] = tok_mask(ar(0, m), b, min(Ll, b + 4))
        for n in (9, 18):
            masks[f"o:first{n}"] = tok_mask(ar(0, m), 0, n)
            masks[f"o:last{n}"] = tok_mask(ar(0, m), Ll - n, Ll)
        for key in ("rand", "attn", "err"):
            masks[f"o:tok_{key}"] = tok_mask(sel[key])
        M = torch.zeros(nc, dtype=torch.bool, device=dev)
        M[torch.topk(err.flatten(), max(1, nc // 10)).indices] = True
        masks["o:cell_err"] = M.view(Ll, m)
        M = torch.zeros(Ll, m, dtype=torch.bool, device=dev)
        M.scatter_(1, torch.topk(err, k10, dim=1).indices, True)
        masks["o:cell_err_pl"] = M
        half = Ll // 2
        for key in ("err20", "attn20"):
            masks[f"o:tok_{key}_last{half}"] = tok_mask(sel[key], half, Ll)
            masks[f"o:tok_{key}_first{half}"] = tok_mask(sel[key], 0, half)

        out = {}

        def score(name, kb, vb, lg0):
            sl = self.cont(ids, P, kb, vb, lg0)
            out[name] = (kl_teacher_student(teacher[None], sl[None]).mean().item(),
                         (teacher.argmax(-1) == sl.argmax(-1)).float().mean().item())

        for name, Mk in masks.items():
            kb, vb = kP.clone(), vP.clone()
            w = Mk[:, None, :, None]
            kb[:, :, s:s + m] = torch.where(w, kTm, kP[:, :, s:s + m])
            vb[:, :, s:s + m] = torch.where(w, vTm, vP[:, :, s:s + m])
            score(name, kb, vb, self.H.stage_tail(prompt, kb, vb, s, m))
        for key in ("rand", "attn", "err", "attn20", "rand20"):
            kb, vb = kP.clone(), vP.clone()
            pos = torch.cat([ar(0, s), s + sel[key], ar(s + m, P)])
            lg0, _ = extend_at(L, prompt[pos], pos, kb, vb)
            score(f"r:tok_{key}", kb, vb, lg0)
        for key in ("err20", "attn20"):
            for src, suffix in ((h_hat, ""), (hT_mid, "_true")):
                if suffix and key != "err20":
                    continue
                kb, vb = kP.clone(), vP.clone()
                lg0 = run_suffix(L, prompt, kb, vb, s, m, sel[key], half, src[half, sel[key]])
                score(f"r:sfx_{key}_a{half}{suffix}", kb, vb, lg0)
        return out, stats


def mean(x):
    return statistics.mean(x) if x else float("nan")


def summarize(rows, a):
    """rows: list of {type, bucket, kl: {cfg: (kl, agree)}, stats}. Returns (json dict, markdown)."""
    cfgs = list(rows[0]["kl"].keys())
    cells = sorted({(r["type"], r["bucket"]) for r in rows})
    buckets = sorted({r["bucket"] for r in rows})
    agg = {}
    for c in cfgs:
        agg[c] = {}
        for b in buckets + ["all"]:
            rs = [r for r in rows if b in ("all", r["bucket"])]
            agg[c][b] = {"kl": mean([r["kl"][c][0] for r in rs]), "agree": mean([r["kl"][c][1] for r in rs]),
                         "kl_by_type": {t: mean([r["kl"][c][0] for r in rs if r["type"] == t])
                                        for t in sorted({r["type"] for r in rs})}}
    st = {}
    for b in buckets:
        rs = [r["stats"] for r in rows if r["bucket"] == b]
        st[b] = {"n": len(rs),
                 "err_layer": np.mean([x["err_layer"] for x in rs], 0).round(4).tolist(),
                 "rcos_layer": np.mean([x["rcos_layer"] for x in rs], 0).round(4).tolist(),
                 **{k: mean([x[k] for x in rs]) for k in ("err_share_top10_cells", "err_share_top10_tokens",
                                                         "err_layer_spearman", "overlap_err_attn", "overlap_err_rand")}}
    res = {"meta": {"args": vars(a), "n": len(rows), "cells": [list(c) for c in cells], "time": time.strftime("%F %T"),
                    "gpu": torch.cuda.get_device_name(0)},
           "configs": agg, "stats": st}
    types = sorted({r["type"] for r in rows})
    md = [f"# MoE-routing diagnostic ({len(rows)} examples, projector `{os.path.basename(os.path.dirname(a.projector))}`)", "",
          "KL(large || variant) on the continuation; `o:` = true K/V pasted into the cells (oracle), `r:` = real recompute.", ""]
    for b in buckets:
        md += [f"## {b}  (n = {st[b]['n']})", "",
               "| config | KL | top-1 | " + " | ".join(types) + " |", "|---|---|---|" + "---|" * len(types)]
        for c in cfgs:
            v = agg[c][b]
            md.append(f"| {c} | {v['kl']:.4f} | {v['agree']:.3f} | " + " | ".join(f"{v['kl_by_type'].get(t, float('nan')):.3f}" for t in types) + " |")
        x = st[b]
        md += ["", f"- error share of the top 10% cells: {x['err_share_top10_cells']:.3f}; of the top 10% tokens (all layers): {x['err_share_top10_tokens']:.3f}",
               f"- mean Spearman correlation of token errors between layers: {x['err_layer_spearman']:.3f}",
               f"- overlap of the top-10% error tokens with the attn selection: {x['overlap_err_attn']:.3f} (random: {x['overlap_err_rand']:.3f})",
               "", "| layer | " + " | ".join(str(i) for i in range(len(x["err_layer"]))) + " |",
               "|---|" + "---|" * len(x["err_layer"]),
               "| KV rel. error | " + " | ".join(f"{e:.3f}" for e in x["err_layer"]) + " |",
               "| residual cos | " + " | ".join(f"{e:.2f}" for e in x["rcos_layer"]) + " |", ""]
    return res, "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
    ap.add_argument("--n_per_cell", type=int, default=10)
    ap.add_argument("--buckets", default="4k,8k")
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--save_every", type=int, default=6)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "moe_diag"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    raw_path = os.path.join(a.out_dir, "per_example.jsonl")
    done = {}
    if os.path.exists(raw_path):
        for l in open(raw_path):
            r = json.loads(l)
            done[r["i"]] = r
    D = Diag(a, dev)
    st = TokenStore("sd_mixtest", "all")
    pl = np.load(os.path.join(C.DATA_DIR, "sd_mixtest", "prompt_len.npy"))
    meta = [json.loads(l) for l in open(os.path.join(C.DATA_DIR, "sd_mixtest", "meta.jsonl"))]
    buckets = a.buckets.split(",")
    cnt = collections.Counter()
    todo = []
    for i, mt in enumerate(meta):
        cell = (mt["type"], mt["bucket"])
        if mt["bucket"] in buckets and cnt[cell] < a.n_per_cell:
            cnt[cell] += 1
            todo.append(i)
    rows = [done[i] for i in todo if i in done]
    print(f"[diag] {len(todo)} examples, {len(rows)} already done", flush=True)

    def flush():
        res, md = summarize(rows, a)
        json.dump(res, open(os.path.join(a.out_dir, "results.json"), "w"), indent=1)
        open(os.path.join(a.out_dir, "summary.md"), "w").write(md)

    t0 = time.time()
    with open(raw_path, "a") as fo:
        for j, i in enumerate([i for i in todo if i not in done]):
            ids = st.get(i).to(dev).long()
            kl, stats = D.example(ids, int(pl[i]))
            r = {"i": i, "type": meta[i]["type"], "bucket": meta[i]["bucket"], "P": int(pl[i]), "kl": kl, "stats": stats}
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            rows.append(r)
            print(f"[{len(rows)}/{len(todo)}] {r['type']:>9} {r['bucket']} P={r['P']} none={kl['o:none'][0]:.3f} "
                  f"all={kl['o:all'][0]:.4f} ours={kl['r:tok_attn'][0]:.3f} o:tok_err={kl['o:tok_err'][0]:.3f} "
                  f"o:cell_err={kl['o:cell_err'][0]:.3f} r:sfx={kl['r:sfx_err20_a18'][0]:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if len(rows) % a.save_every == 0:
                flush()
            torch.cuda.empty_cache()
    flush()
    print(open(os.path.join(a.out_dir, "summary.md")).read())


if __name__ == "__main__":
    main()
