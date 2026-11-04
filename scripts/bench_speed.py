#!/usr/bin/env python
"""
Speed benchmark of the optimised hybrid prefill (prefill/fast.py) with the already-trained projector, plus
diagnostics for "why does the hybrid forget exact details of the prompt".

Sections (all written to <out_dir>/results.json and summary.md):
  check    correctness on GPU: lean == HF, fast hybrid == reference hybrid (KL vs 8B on held-out text), graph == eager
  speed    TTFT vs prompt length for: HF eager (old code), lean eager, lean + CUDA graph; 8B alone and hybrid,
           plus per-stage breakdown of the graphed hybrid and an analytic FLOP ratio
  niah     needle-in-a-haystack with selected parts of the projected KV replaced by the true 8B KV
  probe    per-layer fidelity (cosine) of projected hidden states / K / V vs. the truth, stage1 vs stage2
"""
import argparse
import gc
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import TokenStore  # noqa: E402
from prefill.fast import (FastBig, FastHybrid, FastQwen3, FoldedProjectorKV, Graphed, build_fold, kv_to_dynamic_cache,  # noqa: E402
                          time_call)
from prefill.kv import (greedy_decode, hidden_to_kv, hybrid_prefill, kl_teacher_student, large_hidden_states,  # noqa: E402
                        small_hidden_states, split_prompt, student_forward, teacher_forward, true_prefill)
from prefill.models import eos_ids, load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402


def thirds(n):
    a, b = round(n / 3), round(2 * n / 3)
    return [range(0, a), range(a, b), range(b, n)]


def gname(r):
    return f"L{r.start}-{r.stop - 1}"


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def kl_last(a, b):
    return kl_teacher_student(a[None].float(), b[None].float()).mean().item()


# ----------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def section_check(ctx, n_seq=16, P=2048, cont=256):
    tok, small, large, proj, S, L, fh, fb, dev = (ctx[k] for k in "tok small large proj S L fh fb dev".split())
    store = TokenStore("fineweb_edu", "test")
    idx = [int(i) for i in store.indices_with_min_len(P + cont)][:n_seq]
    res = {"lean_vs_hf_big_kl": [], "kl_ref_hybrid": [], "kl_fast_hybrid": [], "kl_ref_vs_fast": [], "agree_ref": [], "agree_fast": []}
    for i in idx:
        ids = store.get(i, P + cont).to(dev)
        _, lg_hf = true_prefill(large, ids[None, :P])
        _, _, lg_lean = fb.prefill(ids[:P])
        res["lean_vs_hf_big_kl"].append(kl_last(lg_hf, lg_lean))
        t = teacher_forward(large, ids[None], P)[0]
        r = student_forward(large, small, proj, ids[None], P, fh.n_sink, fh.n_tail)[0]
        kb, vb, lg0 = fh.prefill(ids[:P])
        out = large(input_ids=ids[None, P:], past_key_values=kv_to_dynamic_cache(kb, vb), use_cache=True, logits_to_keep=cont)
        f = torch.cat([lg0, out.logits[0, :-1]], 0)
        res["kl_ref_hybrid"].append(kl_teacher_student(t[None], r[None]).mean().item())
        res["kl_fast_hybrid"].append(kl_teacher_student(t[None], f[None]).mean().item())
        res["kl_ref_vs_fast"].append(kl_teacher_student(r[None], f[None]).mean().item())
        res["agree_ref"].append((t.argmax(-1) == r.argmax(-1)).float().mean().item())
        res["agree_fast"].append((t.argmax(-1) == f.argmax(-1)).float().mean().item())
    out = {k: statistics.mean(v) for k, v in res.items()}
    # graph replay == eager
    out["graph_vs_eager_hybrid_maxdiff"] = float("nan")
    if dev.type == "cuda":
        ids = store.get(idx[0], P).to(dev)
        g = Graphed(lambda x: fh.prefill(x), ids)
        lg_g = g(ids)[2].clone()
        out["graph_vs_eager_hybrid_maxdiff"] = (lg_g.float() - fh.prefill(ids)[2].float()).abs().max().item()
        del g
        free()
    out.update(n=len(idx), prompt_len=P, cont_len=cont)
    print("[check]", json.dumps(out, indent=1), flush=True)
    return out


# ----------------------------------------------------------------------------------------------- speed
def flops(cfg, n_tok, kv_len_avg):
    """Forward FLOPs of n_tok tokens attending to kv_len_avg keys on average (no lm_head)."""
    d, H, Hkv, D, I, Lyr = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size, cfg.num_hidden_layers
    per_layer_params = d * (H + 2 * Hkv) * D + H * D * d + 3 * d * I
    return Lyr * (2 * per_layer_params * n_tok + 4 * H * D * n_tok * kv_len_avg)


def analytic_ratio(ctx, T):
    sc, lc = ctx["small"].config, ctx["large"].config
    fold = ctx["fold"]
    s, m, t = split_prompt(T, ctx["fh"].n_sink, ctx["fh"].n_tail)
    big = flops(lc, T, T / 2)
    Ls, d_s = ctx["proj"].cfg["Ls"], sc.hidden_size
    d_mid, ncol = fold.Wcat.shape[1], fold.Wcat.shape[2]
    hyb = flops(sc, s + m, (s + m) / 2) + m * (2 * Ls * d_s * d_mid + 2 * lc.num_hidden_layers * (Ls * d_mid + d_mid * ncol)) \
        + flops(lc, s + t, T - t / 2)
    return big / hyb


def section_speed(ctx, lengths, reps_graph=10, reps_eager=5, reps_hf=3):
    small, large, proj, fh, fb, dev = (ctx[k] for k in "small large proj fh fb dev".split())
    store = TokenStore("fineweb_edu", "test")
    base = torch.cat([store.get(i) for i in range(len(store))])  # long token stream to cut prompts from
    rows = []
    for T in lengths:
        ids = base[:T].to(dev)
        s, m, t = split_prompt(T, fh.n_sink, fh.n_tail)
        row = {"prompt_len": T, "analytic_flop_ratio": analytic_ratio(ctx, T)}

        def run(name, fn, reps):
            try:
                row[name] = time_call(fn, reps)
            except Exception as e:  # OOM at long lengths, or anything else: record and keep going
                row[name] = None
                row.setdefault("errors", {})[name] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"[speed] T={T} {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            free()

        with torch.no_grad():
            run("big_hf", lambda: true_prefill(large, ids[None]), reps_hf)
            run("hybrid_hf", lambda: hybrid_prefill(large, small, proj, ids[None], fh.n_sink, fh.n_tail), reps_hf)
            run("big_lean", lambda: fb.prefill(ids), reps_eager)
            run("hybrid_lean", lambda: fh.prefill(ids), reps_eager)
            for name, fn in (("big_graph", fb.prefill), ("hybrid_graph", fh.prefill)):
                holder = {}

                def graphed_call(fn=fn, holder=holder):
                    if "g" not in holder:
                        holder["g"] = Graphed(fn, ids)
                    return holder["g"](ids)
                run(name, graphed_call, reps_graph)
                holder.clear()
                free()
            # per-stage breakdown of the graphed hybrid
            try:
                g1 = Graphed(fh.stage_small, ids[:s + m])
                row["stage_small_graph"] = time_call(lambda: g1(ids[:s + m]), reps_graph)
                hs = g1.out[:, s:].clone()
                del g1
                free()
                g2 = Graphed(lambda h: fh.stage_fold(h, s, T), hs)
                row["stage_fold_graph"] = time_call(lambda: g2(hs), reps_graph)
                kb, vb = (x.clone() for x in g2.out)
                del g2, hs
                free()
                g3 = Graphed(lambda x, k, v: fh.stage_tail(x, k, v, s, m), ids, kb, vb)
                row["stage_tail_graph"] = time_call(lambda: g3(ids, g3.static_in[1], g3.static_in[2]), reps_graph)
                del g3, kb, vb
            except Exception as e:
                row.setdefault("errors", {})["breakdown"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"[speed] T={T} breakdown: {type(e).__name__}: {str(e)[:300]}", flush=True)
            free()
        for a, b in (("big_hf", "hybrid_hf"), ("big_lean", "hybrid_lean"), ("big_graph", "hybrid_graph")):
            if row.get(a) and row.get(b):
                row[f"speedup_{a.split('_')[1]}"] = row[a] / row[b]
        rows.append(row)
        print("[speed] " + " ".join(f"{k}={v*1000:.1f}ms" if isinstance(v, float) and k not in ("analytic_flop_ratio",) and not k.startswith("speedup") else f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return rows


# ----------------------------------------------------------------------------------------------- niah diagnostics
def make_niah(tok, n, L, seed):
    store = TokenStore("fineweb_edu", "test")
    rng = random.Random(seed)
    docs = [tok.decode(store.get(i, 1024)) for i in range(min(200, len(store)))]
    cases = []
    for _ in range(n):
        code = "".join(rng.choice("0123456789") for _ in range(6))
        needle_core = f"The secret passcode is {code}."
        depth = rng.uniform(0.05, 0.95)
        words = []
        while len(tok(" ".join(words), add_special_tokens=False)["input_ids"]) < L - 80:
            words.extend(rng.choice(docs).split())
        hay = tok.decode(tok(" ".join(words), add_special_tokens=False)["input_ids"][: L - 80])
        cut = int(len(hay) * depth)
        cut = hay.find(" ", cut) if hay.find(" ", cut) > 0 else cut
        prompt = hay[:cut] + " " + needle_core + " " + hay[cut:] + \
            "\n\nQuestion: What is the secret passcode mentioned in the text above? Reply with only the passcode."
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        cs = text.find(needle_core)
        ce = cs + len(needle_core)
        span = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if a < ce and b > cs]
        cases.append({"ids": enc["input_ids"], "code": code, "span": span, "depth": depth})
    return cases


@torch.no_grad()
def section_niah(ctx, lengths, n):
    tok, large, fh, fb, dev = (ctx[k] for k in "tok large fh fb dev".split())
    Ll = large.config.num_hidden_layers
    g = thirds(Ll)  # early / mid / late layer groups (0-11, 12-23, 24-35 for Qwen3-8B)

    def patcher(kind, kt, vt, span):
        def p(kb, vb, s, m):
            mid = list(range(s, s + m))
            sp = torch.tensor([j for j in span if s <= j < s + m], dtype=torch.long, device=dev)
            if kind == "all_true":
                kb[:, :, s:s + m] = kt[:, :, s:s + m]; vb[:, :, s:s + m] = vt[:, :, s:s + m]
            elif kind.startswith("needle_true"):
                layers = {"needle_true": range(Ll), "needle_true_early": g[0], "needle_true_mid": g[1], "needle_true_late": g[2],
                          "needle_trueK": range(Ll), "needle_trueV": range(Ll)}[kind]
                for l in layers:
                    if kind != "needle_trueV":
                        kb[l][:, sp] = kt[l][:, sp]
                    if kind != "needle_trueK":
                        vb[l][:, sp] = vt[l][:, sp]
            elif kind.startswith("layers_true_"):
                rr = {"layers_true_early": g[0], "layers_true_mid": g[1], "layers_true_late": g[2]}[kind]
                kb[rr.start:rr.stop, :, s:s + m] = kt[rr.start:rr.stop, :, s:s + m]
                vb[rr.start:rr.stop, :, s:s + m] = vt[rr.start:rr.stop, :, s:s + m]
            del mid
        return p

    kinds = ["hybrid", "all_true", "needle_true", "needle_trueK", "needle_trueV", "needle_true_early", "needle_true_mid",
             "needle_true_late", "layers_true_early", "layers_true_mid", "layers_true_late"]
    rows, fidelity, fails = [], [], []
    for Lt in lengths:
        cases = make_niah(tok, n, Lt, seed=7 + Lt)
        hits = {k: 0 for k in kinds}
        cos_acc = {"needle_K": torch.zeros(Ll), "needle_V": torch.zeros(Ll), "other_K": torch.zeros(Ll), "other_V": torch.zeros(Ll)}
        for c in cases:
            ids = torch.tensor(c["ids"], device=dev)
            kt, vt, _ = (x.clone() for x in fb.prefill(ids))
            for k in kinds:
                kb, vb, lg = fh.prefill(ids, patch=None if k == "hybrid" else patcher(k, kt, vt, c["span"]))
                if k == "hybrid":
                    s, m, _ = split_prompt(ids.shape[0], fh.n_sink, fh.n_tail)
                    sp = [j for j in c["span"] if s <= j < s + m]
                    oth = [j for j in range(s, s + m) if j not in set(sp)]
                    for nm, a, b in (("K", kb, kt), ("V", vb, vt)):
                        cs = F.cosine_similarity(a.float(), b.float(), dim=-1).mean(1)  # [L, T]
                        cos_acc[f"needle_{nm}"] += cs[:, sp].mean(1).cpu()
                        cos_acc[f"other_{nm}"] += cs[:, oth].mean(1).cpu()
                out = greedy_decode(large, kv_to_dynamic_cache(kb, vb), lg, 16, ctx["eos"])
                text = tok.decode(out, skip_special_tokens=True)
                ok = c["code"] in text
                hits[k] += ok
                if k == "hybrid" and not ok and len(fails) < 8:
                    fails.append({"len": Lt, "code": c["code"], "output": text, "depth": round(c["depth"], 2)})
        for k in kinds:
            rows.append({"target_len": Lt, "variant": k, "accuracy": hits[k] / len(cases), "n": len(cases)})
            print(f"[niah] L={Lt} {k}: {hits[k]}/{len(cases)}", flush=True)
        fidelity.append({"target_len": Lt, **{k: (v / len(cases)).tolist() for k, v in cos_acc.items()}})
    return {"rows": rows, "kv_cosine": fidelity, "hybrid_failures": fails}


# ----------------------------------------------------------------------------------------------- probe
@torch.no_grad()
def section_probe(ctx, projs, n_seq=8, T=2048, n_sink=8):
    small, large, dev = ctx["small"], ctx["large"], ctx["dev"]
    store = TokenStore("fineweb_edu", "test")
    idx = [int(i) for i in store.indices_with_min_len(T)][:n_seq]
    out = {}
    for tag, p in projs.items():
        acc = {"h": 0, "k": 0, "v": 0}
        for i in idx:
            ids = store.get(i, T).to(dev)[None]
            ht = large_hidden_states(large, ids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                hh = p(small_hidden_states(small, ids))
            pos = torch.arange(T, device=dev)[None]
            kt, vt = hidden_to_kv(large, ht, pos)
            kh, vh = hidden_to_kv(large, hh, pos)
            acc["h"] = acc["h"] + F.cosine_similarity(hh.float(), ht.float(), dim=-1)[:, 0, n_sink:].mean(-1).cpu() / len(idx)
            acc["k"] = acc["k"] + torch.stack([F.cosine_similarity(a.float(), b.float(), dim=-1)[0, :, n_sink:].mean() for a, b in zip(kh, kt)]).cpu() / len(idx)
            acc["v"] = acc["v"] + torch.stack([F.cosine_similarity(a.float(), b.float(), dim=-1)[0, :, n_sink:].mean() for a, b in zip(vh, vt)]).cpu() / len(idx)
            del ht, hh
            free()
        out[tag] = {k: v.tolist() for k, v in acc.items()}
        print(f"[probe] {tag}: h_cos L0={out[tag]['h'][0]:.3f} mean={sum(out[tag]['h'])/len(out[tag]['h']):.3f} | "
              f"K mean={sum(out[tag]['k'])/len(out[tag]['k']):.3f} | V mean={sum(out[tag]['v'])/len(out[tag]['v']):.3f}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------- report
def fmt_ms(x):
    return "-" if x is None else f"{x*1000:.1f}"


def write_summary(res, path):
    L = [f"# Speed v2 benchmark ({res['meta']['gpu']})", ""]
    if "check" in res:
        c = res["check"]
        L += ["## Correctness of the fast path", "",
              f"- lean 8B vs HF 8B, first-token KL: {c['lean_vs_hf_big_kl']:.2e}",
              f"- held-out continuation KL vs 8B ({c['n']} seqs, prompt {c['prompt_len']}): reference hybrid {c['kl_ref_hybrid']:.4f}, "
              f"fast hybrid {c['kl_fast_hybrid']:.4f}; KL(reference || fast) {c['kl_ref_vs_fast']:.2e}",
              f"- top-1 agreement with 8B: reference {c['agree_ref']:.3f}, fast {c['agree_fast']:.3f}",
              f"- CUDA graph vs eager max |logit diff|: {c['graph_vs_eager_hybrid_maxdiff']:.2e}", ""]
    if "speed" in res:
        L += ["## TTFT (ms, median)", "",
              "| prompt | 8B HF | hybrid HF | 8B lean | hybrid lean | 8B graph | hybrid graph | speedup HF | speedup lean | speedup graph | FLOP ratio |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["speed"]:
            sp = lambda k: f"{r[k]:.2f}x" if r.get(k) else "-"
            L.append(f"| {r['prompt_len']} | {fmt_ms(r.get('big_hf'))} | {fmt_ms(r.get('hybrid_hf'))} | {fmt_ms(r.get('big_lean'))} | "
                     f"{fmt_ms(r.get('hybrid_lean'))} | {fmt_ms(r.get('big_graph'))} | {fmt_ms(r.get('hybrid_graph'))} | "
                     f"{sp('speedup_hf')} | {sp('speedup_lean')} | {sp('speedup_graph')} | {r['analytic_flop_ratio']:.2f}x |")
        L += ["", "Graphed hybrid, per stage (ms): small prefill / folded projector+KV / sink+tail 8B call", ""]
        for r in res["speed"]:
            L.append(f"- {r['prompt_len']}: {fmt_ms(r.get('stage_small_graph'))} / {fmt_ms(r.get('stage_fold_graph'))} / {fmt_ms(r.get('stage_tail_graph'))}")
        L.append("")
    if "niah" in res:
        n = res["niah"]
        lens = sorted({r["target_len"] for r in n["rows"]})
        kinds = list(dict.fromkeys(r["variant"] for r in n["rows"]))
        L += ["## Needle-in-a-haystack: which part of the projected KV loses the needle", "",
              "| variant | " + " | ".join(str(x) for x in lens) + " |", "|---|" + "---|" * len(lens)]
        for k in kinds:
            L.append(f"| {k} | " + " | ".join(f"{r['accuracy']:.2f}" for x in lens for r in n["rows"] if r["variant"] == k and r["target_len"] == x) + " |")
        G = thirds(len(n["kv_cosine"][0]["needle_K"]))
        L += ["", f"early / mid / late = layers {', '.join(gname(r) for r in G)}.", "",
              "K/V cosine to the true 8B cache, mean over layer groups (needle tokens vs other middle tokens):", "",
              "| len | part | " + " | ".join(gname(r) for r in G) + " |", "|---|---|---|---|---|"]
        for f in n["kv_cosine"]:
            for part in ("needle_K", "other_K", "needle_V", "other_V"):
                v = f[part]
                L.append(f"| {f['target_len']} | {part} | " + " | ".join(f"{statistics.mean(v[r.start:r.stop]):.3f}" for r in G) + " |")
        L += ["", "Example hybrid failures:", ""] + [f"- len {x['len']}, code {x['code']}, depth {x['depth']}: `{x['output']!r}`" for x in n["hybrid_failures"]]
        L.append("")
    if "probe" in res:
        n_l = len(next(iter(res["probe"].values()))["h"])
        G = thirds(n_l)
        G = [range(0, 1), range(1, G[0].stop)] + G[1:]
        L += ["## Per-layer fidelity on FineWeb-Edu (cosine to truth, positions >= 8)", "",
              "| projector | quantity | " + " | ".join(gname(r) for r in G) + " |", "|---|---|" + "---|" * len(G)]
        for tag, d in res["probe"].items():
            for q in ("h", "k", "v"):
                v = d[q]
                L.append(f"| {tag} | {q} | " + " | ".join(f"{statistics.mean(v[r.start:r.stop]):.3f}" for r in G) + " |")
        L.append("")
    open(path, "w").write("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "stage2", "projector.pt"))
    ap.add_argument("--stage1", default=os.path.join(C.CKPT_DIR, "stage1", "projector.pt"))
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--sections", default="check,speed,niah,probe")
    ap.add_argument("--niah_lengths", default="2048,3584")
    ap.add_argument("--n_niah", type=int, default=30)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "speed_v2"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_check", type=int, default=16)
    ap.add_argument("--check_len", type=int, default=2048)
    ap.add_argument("--n_probe", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)
    tok, small, large = load_pair(dev)
    proj, ck = Projector.load(a.projector, dev)
    proj.eval()
    print(f"[load] projector {a.projector} step {ck.get('step')}", flush=True)
    t0 = time.time()
    S, L = FastQwen3(small), FastQwen3(large)
    fold = build_fold(proj, L)
    print(f"[load] fast path built in {time.time()-t0:.1f}s; folded weights {fold.Wcat.numel()*2/1e9:.2f} GB", flush=True)
    ctx = dict(tok=tok, small=small, large=large, proj=proj, S=S, L=L, fold=fold, fh=FastHybrid(S, L, fold, a.n_sink, a.n_tail),
               fb=FastBig(L), dev=dev, eos=eos_ids(tok))
    res_path = os.path.join(a.out_dir, "results.json")
    res = {"meta": {"gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "args": vars(a), "projector_step": ck.get("step"), "time": time.strftime("%F %T")}}
    for sec in a.sections.split(","):
        t0 = time.time()
        if sec == "check":
            res["check"] = section_check(ctx, a.n_check, a.check_len)
        elif sec == "speed":
            res["speed"] = section_speed(ctx, [int(x) for x in a.lengths.split(",")])
        elif sec == "niah":
            res["niah"] = section_niah(ctx, [int(x) for x in a.niah_lengths.split(",")], a.n_niah)
        elif sec == "probe":
            projs = {"stage2": proj}
            if os.path.exists(a.stage1):
                projs["stage1"] = Projector.load(a.stage1, dev)[0].eval()
            res["probe"] = section_probe(ctx, projs, a.n_probe, a.check_len)
        print(f"[section {sec}] {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
        write_summary(res, os.path.join(a.out_dir, "summary.md"))
    print("[done]", a.out_dir)


if __name__ == "__main__":
    main()
