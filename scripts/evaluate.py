#!/usr/bin/env python
"""
Compare three ways of answering:
  big    : Qwen3-8B does its own prefill and decodes
  small  : Qwen3-0.6B alone
  hybrid : Qwen3-0.6B prefill -> projector -> Qwen3-8B KV cache (sink/tail recomputed by 8B) -> 8B decodes
Tasks: ppl (held-out continuation NLL/KL vs prompt length), oracle (per-layer KV replacement), gsm8k,
       niah (needle in a haystack), speed (TTFT / decode latency vs prompt length).
Writes results.json + summary.md into the output directory.
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import TokenStore  # noqa: E402
from prefill.kv import (cache_from_kv, greedy_decode, hidden_to_kv, hybrid_prefill, kl_teacher_student,  # noqa: E402
                        large_hidden_states, small_hidden_states, split_prompt, student_forward, teacher_forward, true_prefill)
from prefill.models import eos_ids, load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402


class Cfg:
    def __init__(self, name, kind, projector=None, n_sink=8, n_tail=64):
        self.name, self.kind, self.projector, self.n_sink, self.n_tail = name, kind, projector, n_sink, n_tail


class Runner:
    def __init__(self, tok, small, large, device):
        self.tok, self.small, self.large, self.device = tok, small, large, device
        self.eos = eos_ids(tok)

    def model_for(self, cfg):
        return self.small if cfg.kind == "small" else self.large

    def prefill(self, cfg, ids, timing=None):
        t0 = time.perf_counter()
        if cfg.kind == "hybrid":
            cache, logits = hybrid_prefill(self.large, self.small, cfg.projector, ids, cfg.n_sink, cfg.n_tail, timing)
        else:
            cache, logits = true_prefill(self.model_for(cfg), ids)
        torch.cuda.synchronize() if ids.is_cuda else None
        if timing is not None:
            timing["ttft"] = time.perf_counter() - t0
        return cache, logits

    def generate(self, cfg, ids, max_new, timing=None):
        timing = {} if timing is None else timing
        cache, logits = self.prefill(cfg, ids, timing)
        out = greedy_decode(self.model_for(cfg), cache, logits, max_new, self.eos, timing)
        return out, timing

    @torch.no_grad()
    def cont_logits(self, cfg, ids, P):
        if cfg.kind == "hybrid":
            return student_forward(self.large, self.small, cfg.projector, ids, P, cfg.n_sink, cfg.n_tail)
        return teacher_forward(self.model_for(cfg), ids, P)

    def chat_ids(self, user_text):
        text = self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"]], device=self.device)


# --------------------------------------------------------------------------------------------- tasks
def task_ppl(R, cfgs, sources, lengths, n_per, cont):
    rows = []
    big = [c for c in cfgs if c.kind == "big"][0]
    for src in sources:
        store = TokenStore(src, "test")
        for L in lengths:
            idx = [int(i) for i in store.indices_with_min_len(L + cont)][:n_per]
            acc = {c.name: {"nll": [], "kl_vs_big": [], "top1_agree": []} for c in cfgs}
            for i in idx:
                ids = store.get(i, L + cont).unsqueeze(0).to(R.device)
                tgt = ids[:, L:]
                t = R.cont_logits(big, ids, L)
                for c in cfgs:
                    s = t if c is big else R.cont_logits(c, ids, L)
                    acc[c.name]["nll"].append(F.cross_entropy(s.float().flatten(0, 1), tgt.flatten()).item())
                    acc[c.name]["kl_vs_big"].append(kl_teacher_student(t, s).mean().item())
                    acc[c.name]["top1_agree"].append((t.argmax(-1) == s.argmax(-1)).float().mean().item())
            for c in cfgs:
                m = {k: statistics.mean(v) for k, v in acc[c.name].items() if v}
                rows.append({"source": src, "prompt_len": L, "config": c.name, "n": len(idx), **m, "ppl": float(torch.tensor(m["nll"]).exp())})
                print(f"[ppl] {src} L={L} {c.name}: nll={m['nll']:.4f} ppl={rows[-1]['ppl']:.3f} kl={m['kl_vs_big']:.4f} agree={m['top1_agree']:.3f}", flush=True)
    return rows


@torch.no_grad()
def task_oracle(R, projector, n_seqs, P, cont, n_sink):
    """Replace the true KV of one layer at a time (positions >= n_sink) with the projected KV; KL vs. true."""
    store = TokenStore("fineweb_edu", "test")
    P = min(P, int(max(store.length(i) for i in range(len(store)))) - cont)  # adapt to available data
    idx = [int(i) for i in store.indices_with_min_len(P + cont)][:n_seqs]
    Ll = R.large.config.num_hidden_layers
    per_layer = {l: [] for l in range(Ll)}
    all_proj, none_proj = [], []
    for i in idx:
        ids = store.get(i, P + cont).unsqueeze(0).to(R.device)
        prompt, c_ids = ids[:, :P], ids[:, P:]
        t = teacher_forward(R.large, ids, P)
        true_cache, _ = true_prefill(R.large, prompt)
        kt = [true_cache.key_cache[l] for l in range(Ll)]
        vt = [true_cache.value_cache[l] for l in range(Ll)]
        hs = small_hidden_states(R.small, prompt)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=R.device.type == "cuda"):
            h_hat = projector(hs)
        pos = torch.arange(P, device=R.device).unsqueeze(0)
        kp, vp = hidden_to_kv(R.large, h_hat, pos)
        kp = [torch.cat([kt[l][:, :, :n_sink], kp[l][:, :, n_sink:]], 2) for l in range(Ll)]
        vp = [torch.cat([vt[l][:, :, :n_sink], vp[l][:, :, n_sink:]], 2) for l in range(Ll)]

        def run(ks, vs):
            cache = cache_from_kv(ks, vs)
            out = R.large(input_ids=c_ids, past_key_values=cache, use_cache=True, logits_to_keep=cont)
            # logits at position j predict c_ids[j+1]; the last prompt token's logits are not available here,
            # so compare on c_ids[1:]
            return kl_teacher_student(t[:, 1:], out.logits[:, :-1]).mean().item()

        none_proj.append(run(kt, vt))
        all_proj.append(run(kp, vp))
        for l in range(Ll):
            ks, vs = list(kt), list(vt)
            ks[l], vs[l] = kp[l], vp[l]
            per_layer[l].append(run(ks, vs))
    res = {"n": len(idx), "prompt_len": P, "kl_none_projected": statistics.mean(none_proj),
           "kl_all_projected_no_tail": statistics.mean(all_proj),
           "kl_per_layer": {l: statistics.mean(v) for l, v in per_layer.items()}}
    print("[oracle] none=%.4f all=%.4f" % (res["kl_none_projected"], res["kl_all_projected_no_tail"]))
    print("[oracle] per-layer KL:", {l: round(v, 4) for l, v in res["kl_per_layer"].items()})
    return res


def extract_answer(text):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def task_gsm8k(R, cfgs, n, n_shot, max_new):
    from datasets import load_from_disk
    ds = load_from_disk(C.GSM8K_DIR)
    rng = random.Random(0)
    train = ds["train"]
    shots = [train[i] for i in rng.sample(range(len(train)), n_shot)]
    header = "Solve each math problem step by step. Finish with a line of the form '#### <final number>'.\n\n"
    fewshot = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
    test = ds["test"]
    items = [test[i] for i in range(min(n, len(test)))]
    rows = []
    for c in cfgs:
        correct, ttft, total, ntok, plen = 0, [], [], [], []
        samples = []
        for k, ex in enumerate(items):
            ids = R.chat_ids(header + fewshot + f"Question: {ex['question']}\nAnswer:")
            out, tm = R.generate(c, ids, max_new)
            text = R.tok.decode(out, skip_special_tokens=True)
            gold = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
            pred = extract_answer(text)
            ok = pred is not None and abs(pred - gold) < 1e-4
            correct += ok
            ttft.append(tm["ttft"]); total.append(tm["ttft"] + tm["decode"]); ntok.append(tm["n_decoded"]); plen.append(ids.shape[1])
            if k < 3:
                samples.append({"q": ex["question"], "gold": gold, "pred": pred, "output": text[:800]})
            if (k + 1) % 25 == 0:
                print(f"[gsm8k] {c.name}: {k+1}/{len(items)} acc={correct/(k+1):.3f}", flush=True)
        rows.append({"config": c.name, "n": len(items), "accuracy": correct / len(items), "mean_prompt_tokens": statistics.mean(plen),
                     "mean_ttft_s": statistics.mean(ttft), "mean_total_s": statistics.mean(total), "mean_new_tokens": statistics.mean(ntok),
                     "samples": samples})
        print(f"[gsm8k] {c.name}: acc={rows[-1]['accuracy']:.3f} ttft={rows[-1]['mean_ttft_s']:.3f}s total={rows[-1]['mean_total_s']:.2f}s", flush=True)
    return rows


def task_niah(R, cfgs, lengths, n_per):
    store = TokenStore("fineweb_edu", "test")
    rng = random.Random(1)
    docs = [R.tok.decode(store.get(i, 1024)) for i in range(min(200, len(store)))]
    rows = []
    cases = []
    for L in lengths:
        for _ in range(n_per):
            code = "".join(rng.choice("0123456789") for _ in range(6))
            needle = f" The secret passcode is {code}. "
            depth = rng.uniform(0.05, 0.95)
            # build a haystack of roughly L tokens from random docs
            words = []
            while len(R.tok(" ".join(words), add_special_tokens=False)["input_ids"]) < L - 80:
                words.extend(rng.choice(docs).split())
            hay_ids = R.tok(" ".join(words), add_special_tokens=False)["input_ids"][: L - 80]
            hay = R.tok.decode(hay_ids)
            cut = int(len(hay) * depth)
            cut = hay.find(" ", cut) if hay.find(" ", cut) > 0 else cut
            text = hay[:cut] + needle + hay[cut:]
            cases.append((L, text + "\n\nQuestion: What is the secret passcode mentioned in the text above? Reply with only the passcode.", code, depth))
    for c in cfgs:
        for L in lengths:
            hits, ttfts, plens = 0, [], []
            sub = [x for x in cases if x[0] == L]
            for (_, prompt, code, depth) in sub:
                ids = R.chat_ids(prompt)
                out, tm = R.generate(c, ids, 16)
                text = R.tok.decode(out, skip_special_tokens=True)
                hits += code in text
                ttfts.append(tm["ttft"]); plens.append(ids.shape[1])
            rows.append({"config": c.name, "target_len": L, "mean_prompt_tokens": statistics.mean(plens), "n": len(sub),
                         "accuracy": hits / len(sub), "mean_ttft_s": statistics.mean(ttfts)})
            print(f"[niah] {c.name} L={L}: acc={hits/len(sub):.3f} ttft={statistics.mean(ttfts):.3f}s", flush=True)
    return rows


def task_speed(R, cfgs, lengths, repeats, decode_tokens):
    store = TokenStore("fineweb_edu", "test")
    rows = []
    for L in lengths:
        idx = [int(i) for i in store.indices_with_min_len(L)]
        ids = store.get(idx[0], L).unsqueeze(0).to(R.device)
        for c in cfgs:
            ttfts, dec, parts = [], [], {}
            for r in range(repeats + 2):
                tm = {}
                out, tm = R.generate(c, ids, decode_tokens, tm)
                if r >= 2:  # skip warmup
                    ttfts.append(tm["ttft"]); dec.append(tm["decode"] / max(1, tm["n_decoded"]))
                    for k in ("sink", "small", "projector", "kv", "tail"):
                        if k in tm:
                            parts.setdefault(k, []).append(tm[k])
            row = {"config": c.name, "prompt_len": L, "ttft_median_s": statistics.median(ttfts), "ttft_min_s": min(ttfts),
                   "decode_per_token_s": statistics.median(dec)}
            row.update({f"ttft_{k}_s": statistics.median(v) for k, v in parts.items()})
            rows.append(row)
            print(f"[speed] L={L} {c.name}: ttft={row['ttft_median_s']*1000:.1f}ms decode={row['decode_per_token_s']*1000:.1f}ms/tok " +
                  " ".join(f"{k}={v*1000:.0f}ms" for k, v in row.items() if k.startswith("ttft_") and k.endswith("_s") and k not in ("ttft_median_s", "ttft_min_s")), flush=True)
    return rows


# --------------------------------------------------------------------------------------------- report
def write_summary(res, path, gpu):
    L = [f"# prefillLLM evaluation ({gpu})", ""]
    if "speed" in res:
        L += ["## Speed: time-to-first-token (median, ms) vs prompt length", "", "| prompt_len | " + " | ".join(sorted({r['config'] for r in res['speed']})) + " |", "|---|" + "---|" * len({r['config'] for r in res['speed']})]
        names = sorted({r["config"] for r in res["speed"]})
        for Lp in sorted({r["prompt_len"] for r in res["speed"]}):
            d = {r["config"]: r for r in res["speed"] if r["prompt_len"] == Lp}
            L.append(f"| {Lp} | " + " | ".join(f"{d[n]['ttft_median_s']*1000:.1f}" if n in d else "-" for n in names) + " |")
        L += ["", "Decode latency per token (ms): " + ", ".join(f"{r['config']}={r['decode_per_token_s']*1000:.1f}" for r in res["speed"] if r["prompt_len"] == max(x["prompt_len"] for x in res["speed"])), ""]
        hy = [r for r in res["speed"] if "ttft_small_s" in r]
        if hy:
            L += ["Hybrid TTFT breakdown (ms) at the longest prompt:", ""]
            for r in hy:
                if r["prompt_len"] == max(x["prompt_len"] for x in res["speed"]):
                    L.append(f"- {r['config']}: " + ", ".join(f"{k[5:-2]}={r[k]*1000:.1f}" for k in r if k.startswith("ttft_") and k not in ("ttft_median_s", "ttft_min_s")))
            L.append("")
    if "ppl" in res:
        L += ["## Held-out continuation quality (256 tokens after the prompt)", "", "| source | prompt_len | config | NLL | PPL | KL vs big | top-1 agree |", "|---|---|---|---|---|---|---|"]
        for r in res["ppl"]:
            L.append(f"| {r['source']} | {r['prompt_len']} | {r['config']} | {r['nll']:.4f} | {r['ppl']:.3f} | {r['kl_vs_big']:.4f} | {r['top1_agree']:.3f} |")
        L.append("")
    if "gsm8k" in res:
        L += ["## GSM8K (few-shot, greedy, non-thinking mode)", "", "| config | n | accuracy | prompt tokens | TTFT (s) | total (s) | new tokens |", "|---|---|---|---|---|---|---|"]
        for r in res["gsm8k"]:
            L.append(f"| {r['config']} | {r['n']} | {r['accuracy']:.3f} | {r['mean_prompt_tokens']:.0f} | {r['mean_ttft_s']:.3f} | {r['mean_total_s']:.2f} | {r['mean_new_tokens']:.0f} |")
        L.append("")
    if "niah" in res:
        L += ["## Needle in a haystack (exact passcode retrieval)", "", "| config | target_len | prompt tokens | accuracy | TTFT (s) |", "|---|---|---|---|---|"]
        for r in res["niah"]:
            L.append(f"| {r['config']} | {r['target_len']} | {r['mean_prompt_tokens']:.0f} | {r['accuracy']:.3f} | {r['mean_ttft_s']:.3f} |")
        L.append("")
    if "oracle" in res:
        for tag, o in res["oracle"].items():
            L += [f"## Oracle: single-layer KV replacement ({tag}, prompt {o['prompt_len']}, KL vs true, no tail recompute)", "",
                  f"none projected = {o['kl_none_projected']:.4f}; all layers projected = {o['kl_all_projected_no_tail']:.4f}", "",
                  "| layer | KL |", "|---|---|"]
            L += [f"| {l} | {v:.4f} |" for l, v in o["kl_per_layer"].items()]
            L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projectors", default=f"stage1={C.CKPT_DIR}/stage1/projector.pt,stage2={C.CKPT_DIR}/stage2/projector.pt")
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--variants", default="16,128;4,32", help="extra sink,tail settings for ppl/niah (applied to the last projector)")
    ap.add_argument("--tasks", default="speed,ppl,oracle,niah,gsm8k")
    ap.add_argument("--ppl_lengths", default="512,1024,2048,3584")
    ap.add_argument("--n_ppl", type=int, default=40)
    ap.add_argument("--n_oracle", type=int, default=12)
    ap.add_argument("--oracle_len", type=int, default=2048)
    ap.add_argument("--n_gsm8k", type=int, default=150)
    ap.add_argument("--gsm8k_shots", type=int, default=5)
    ap.add_argument("--gsm8k_max_new", type=int, default=400)
    ap.add_argument("--niah_lengths", default="1024,2048,3584")
    ap.add_argument("--n_niah", type=int, default=30)
    ap.add_argument("--speed_lengths", default="512,1024,2048,4096")
    ap.add_argument("--speed_repeats", type=int, default=5)
    ap.add_argument("--speed_decode_tokens", type=int, default=32)
    ap.add_argument("--out_dir", default=os.path.join(C.RESULTS_DIR, "main"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    tok, small, large = load_pair(device)
    R = Runner(tok, small, large, device)

    cfgs = [Cfg("big", "big"), Cfg("small", "small")]
    projs = {}
    for item in [x for x in a.projectors.split(",") if x]:
        tag, path = item.split("=")
        if not os.path.exists(path):
            print(f"[warn] projector {path} missing, skipping {tag}")
            continue
        projs[tag], ck = Projector.load(path, device)
        projs[tag].eval()
        print(f"[projector] {tag}: {path} step={ck.get('step')}")
        cfgs.append(Cfg(f"hybrid_{tag}_s{a.n_sink}_t{a.n_tail}", "hybrid", projs[tag], a.n_sink, a.n_tail))
    variant_cfgs = []
    if projs and a.variants:
        tag = list(projs)[-1]
        for v in a.variants.split(";"):
            s, t = map(int, v.split(","))
            variant_cfgs.append(Cfg(f"hybrid_{tag}_s{s}_t{t}", "hybrid", projs[tag], s, t))

    res_path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["meta"] = {"gpu": gpu, "args": vars(a), "time": time.strftime("%Y-%m-%d %H:%M")}
    tasks = a.tasks.split(",")
    for task in tasks:
        t0 = time.time()
        if task == "speed":
            res["speed"] = task_speed(R, cfgs, [int(x) for x in a.speed_lengths.split(",")], a.speed_repeats, a.speed_decode_tokens)
        elif task == "ppl":
            res["ppl"] = task_ppl(R, cfgs + variant_cfgs, ["fineweb_edu", "slimpajama"], [int(x) for x in a.ppl_lengths.split(",")], a.n_ppl, 256)
        elif task == "oracle":
            res["oracle"] = {tag: task_oracle(R, p, a.n_oracle, a.oracle_len, 128, a.n_sink) for tag, p in projs.items()}
        elif task == "niah":
            res["niah"] = task_niah(R, cfgs + variant_cfgs, [int(x) for x in a.niah_lengths.split(",")], a.n_niah)
        elif task == "gsm8k":
            res["gsm8k"] = task_gsm8k(R, cfgs, a.n_gsm8k, a.gsm8k_shots, a.gsm8k_max_new)
        print(f"[task {task}] done in {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
        write_summary(res, os.path.join(a.out_dir, "summary.md"), gpu)
    print("[done] results in", a.out_dir)


if __name__ == "__main__":
    main()
