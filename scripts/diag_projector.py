#!/usr/bin/env python
"""
Where is the projected KV wrong, and which of its errors matter?  (No training.)

Oracle configs (everything else exactly as `hyb`, i.e. nothing recomputed):
  or:<a>-<b>:<kv>   middle-token K and/or V of large layers a..b (inclusive) replaced by the large model's TRUE K/V
  or:all:<kv>       all layers; or:all:kv is the sanity check (must match the large model alone)
  kv = k | v | kv.  Any config of scripts/eval_recompute.py (big, hyb, attn_small:0.1:sent, ...) also works.

Tasks (results.json; finished tasks are skipped on restart unless --force):
  kl      KL(large || config) on the held-out sd_mixtest answers (same teacher as scripts/evaluate_v2.py), per type
  ruler   RULER 4k / 8k (scripts/eval_recompute.task_ruler, same examples)
  qa      SQuAD / HotpotQA / TriviaQA 4k / 8k (scripts/eval_recompute.task_qa, same examples)
  tokerr  token-level error of the projected middle KV vs the true KV (no generation), on RULER and QA prompts:
          relative error ||K_proj - K_true|| / ||K_true|| (same for V) per layer; uniform mean vs weighted by the
          attention of the last prompt token (true query, true keys); by token type (digit / word / other / answer
          tokens) and by position in the middle; attention mass on the answer tokens with the true vs projected keys
          (true query, so this isolates the key error: can the large model still find the evidence?)
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_recompute import Engine, GenLog, task_qa, task_ruler  # noqa: E402
from evaluate_v2 import mean  # noqa: E402
from prefill import config as C  # noqa: E402
from prefill.data import TokenStore  # noqa: E402
from prefill.fast import kv_to_dynamic_cache  # noqa: E402
from prefill.kv import kl_teacher_student, split_prompt, teacher_forward  # noqa: E402
from prefill.mixdata import PromptPool  # noqa: E402
from prefill.models import load_pair  # noqa: E402

RULER_CAT = {"检索": lambda t: t.startswith("niah"), "汇总": lambda t: t in ("cwe", "fwe"), "变量追踪": lambda t: t == "vt",
             "问答": lambda t: t.startswith("qa")}


def layer_groups(Ll, n=4):
    b = [round(i * Ll / n) for i in range(n + 1)]
    return [(b[i], b[i + 1] - 1) for i in range(n)]


def default_configs(Ll):
    return ["hyb"] + [f"or:{a}-{b}:kv" for a, b in layer_groups(Ll)] + ["or:all:k", "or:all:v"]


def parse_or(cfg, Ll):
    _, rng, kv = cfg.split(":")
    a, b = (0, Ll - 1) if rng == "all" else map(int, rng.split("-"))
    assert kv in ("k", "v", "kv") and 0 <= a <= b < Ll, cfg
    return list(range(a, b + 1)), kv


def label(cfg, Ll):
    if cfg.startswith("or:"):
        layers, kv = parse_or(cfg, Ll)
        what = {"k": "K", "v": "V", "kv": "K+V"}[kv]
        rng = "全部层" if len(layers) == Ll else f"层 {layers[0]}–{layers[-1]}"
        return f"真 {what}：{rng}"
    return {"big": f"{C.LARGE_NAME} 单独", "hyb": "混合，不重算"}.get(cfg, cfg)


class DiagEngine(Engine):
    @torch.no_grad()
    def prefill(self, cfg, ids, answers=()):
        if not cfg.startswith("or:"):
            return super().prefill(cfg, ids, answers)
        layers, kv = parse_or(cfg, self.L.n_layers)
        li = torch.tensor(layers, device=ids.device)
        kt, vt, _ = self.fb.prefill(ids)

        def patch(kb, vb, s, m):
            if "k" in kv:
                kb[li, :, s:s + m] = kt[li, :, s:s + m]
            if "v" in kv:
                vb[li, :, s:s + m] = vt[li, :, s:s + m]

        kb, vb, lg = self.rh.prefill(ids, patch=patch)
        return kb, vb, lg, 0.0

    @torch.no_grad()
    def cont_logits(self, cfg, ids, P):
        """[m, V] logits predicting ids[P:] given the prompt ids[:P] (as scripts/evaluate_v2.Engine.cont_logits)."""
        if cfg == "big":
            return teacher_forward(self.large, ids[None], P)[0]
        kb, vb, lg0, _ = self.prefill(cfg, ids[:P])
        m = ids.shape[0] - P
        if m == 1:
            return lg0
        out = self.large(input_ids=ids[None, P:-1], past_key_values=kv_to_dynamic_cache(kb, vb), use_cache=True,
                         logits_to_keep=m - 1)
        return torch.cat([lg0, out.logits[0]], 0)


# ----------------------------------------------------------------------------------------------- tasks
def task_kl(E, cfgs, n_per_cell):
    st = TokenStore("sd_mixtest", "all")
    pl = np.load(os.path.join(C.DATA_DIR, "sd_mixtest", "prompt_len.npy"))
    meta = [json.loads(l) for l in open(os.path.join(C.DATA_DIR, "sd_mixtest", "meta.jsonl"))]
    acc = collections.defaultdict(lambda: {"kl": [], "agree": []})
    cnt = collections.Counter()
    for i, m in enumerate(meta):
        cell = (m["type"], m["bucket"])
        if cnt[cell] >= n_per_cell:
            continue
        cnt[cell] += 1
        ids = st.get(i).to(E.dev)
        P = int(pl[i])
        t = E.cont_logits("big", ids, P)
        for c in cfgs:
            if c == "big":
                continue
            s = E.cont_logits(c, ids, P)
            acc[(c,) + cell]["kl"].append(kl_teacher_student(t[None], s[None]).mean().item())
            acc[(c,) + cell]["agree"].append((t.argmax(-1) == s.argmax(-1)).float().mean().item())
    rows = [{"config": c, "type": t, "bucket": b, "n": len(v["kl"]), "kl": mean(v["kl"]), "top1_agree": mean(v["agree"])}
            for (c, t, b), v in sorted(acc.items())]
    for c in cfgs:
        v = [r for r in rows if r["config"] == c]
        if v:
            print(f"[kl] {c:>16}: KL={mean([r['kl'] for r in v]):.4f} agree={mean([r['top1_agree'] for r in v]):.3f}", flush=True)
    return rows


def find_spans(ids, needles, lo, hi):
    """bool [len(ids)] marking every occurrence of any needle token sequence inside [lo, hi)."""
    mark = np.zeros(len(ids), bool)
    for nd in needles:
        k = len(nd)
        if k == 0:
            continue
        for j in range(lo, hi - k + 1):
            if ids[j] == nd[0] and ids[j:j + k] == nd:
                mark[j:j + k] = True
    return mark


def tokerr_items(E, n_ruler, n_qa):
    from datasets import load_dataset
    import random
    items = []
    for L in (4096, 8192):
        ds = load_dataset("simonjegou/ruler", str(L), split="test")
        by = collections.defaultdict(list)
        for i, t in enumerate(ds["task"]):
            by[t].append(i)
        rng = random.Random(L)  # same examples as task_ruler (first n of its per-task sample)
        for t, idx in sorted(by.items()):
            pick = rng.sample(idx, min(15, len(idx)))
            if t in ("vt", "cwe", "fwe", "niah_multikey_3", "niah_multivalue", "qa_1", "qa_2"):
                for i in pick[:n_ruler]:
                    ex = ds[i]
                    ids = E.chat_ids([{"role": "user", "content": ex["context"] + ex["question"]}], ex["answer_prefix"])
                    items.append((f"ruler{L // 1024}k_{t}", ids, ex["answer"]))
    pool = PromptPool("qa_eval")
    cells = collections.defaultdict(list)
    for i, m in enumerate(pool.meta):
        cells[(m["source"], m["bucket"])].append(i)
    for (src, b), idx in sorted(cells.items()):
        for i in idx[:n_qa]:
            items.append((f"qa{b}", pool.ids(i), pool.meta[i]["answers"]))
    return items


@torch.no_grad()
def task_tokerr(E, n_ruler, n_qa):
    Lf = E.L
    Ll, Hkv, rep, D = Lf.n_layers, Lf.Hkv, Lf.rep, Lf.D
    tok_txt = {}

    def ttype(t):
        if t not in tok_txt:
            s = E.tok.decode([t])
            tok_txt[t] = "数字" if any(ch.isdigit() for ch in s) else "单词" if any(ch.isalpha() for ch in s) else "其他"
        return tok_txt[t]

    # accumulators: per (source, key) -> sums over tokens (per layer)
    S = collections.defaultdict(lambda: torch.zeros(Ll, dtype=torch.float64))
    Nn = collections.Counter()
    ans_mass = collections.defaultdict(list)  # source -> [(mass_true [Ll], mass_proj [Ll])]
    items = tokerr_items(E, n_ruler, n_qa)
    t0 = time.time()
    for src, ids_l, answers in items:
        ids = torch.as_tensor(ids_l, device=E.dev)
        T = ids.shape[0]
        s, m, _ = split_prompt(T, E.rh.n_sink, E.rh.n_tail)
        if m <= 0:
            continue
        kt = torch.empty(Ll, Hkv, T, D, dtype=Lf.embed.dtype, device=E.dev)
        vt = torch.empty_like(kt)
        lq = []
        Lf.prefill(ids, kt, vt, logits=False, last_q=lq)
        kh, vh, _ = E.rh.prefill(ids)
        sl = slice(s, s + m)
        Kt, Vt, Kh, Vh = (x[:, :, sl].float() for x in (kt, vt, kh, vh))
        ek = ((Kh - Kt).pow(2).sum((1, 3)) / Kt.pow(2).sum((1, 3)).clamp_min(1e-12)).sqrt()  # [Ll, m]
        ev = ((Vh - Vt).pow(2).sum((1, 3)) / Vt.pow(2).sum((1, 3)).clamp_min(1e-12)).sqrt()
        q = torch.stack(lq).float().view(Ll, Hkv, rep, D)  # last prompt token's true queries

        def attn(keys):  # [Ll, T] attention of the last token (mean over heads)
            sc = torch.einsum("lgrd,lgtd->lgrt", q, keys.float()) * Lf.scaling
            return sc.softmax(-1).mean((1, 2))

        a_true = attn(kt)
        kmix = kt.clone()
        kmix[:, :, sl] = kh[:, :, sl]
        a_proj = attn(kmix)
        w = a_true[:, sl].double()
        types = [ttype(t) for t in ids_l[s:s + m]]
        ans = find_spans(list(ids_l), E.needles(answers), s, s + m)[s:s + m]
        pos_bin = (np.arange(m) * 10 // m)
        for key_src in (src, "all"):
            S[(key_src, "k_sum")] += ek.double().sum(1).cpu()
            S[(key_src, "v_sum")] += ev.double().sum(1).cpu()
            Nn[(key_src, "tok")] += m
            S[(key_src, "k_w")] += (ek.double() * w).sum(1).cpu()
            S[(key_src, "v_w")] += (ev.double() * w).sum(1).cpu()
            S[(key_src, "w")] += w.sum(1).cpu()
            S[(key_src, "mass_mid_true")] += a_true[:, sl].double().sum(1).cpu()
            S[(key_src, "mass_mid_proj")] += a_proj[:, sl].double().sum(1).cpu()
            Nn[(key_src, "prompts")] += 1
            groups = collections.defaultdict(list)
            for j, ty in enumerate(types):
                groups["type:" + ty].append(j)
            groups["type:答案"] = list(np.nonzero(ans)[0])
            for bi in range(10):
                groups[f"pos:{bi}"] = list(np.nonzero(pos_bin == bi)[0])
            for g, idx in groups.items():
                if not idx:
                    continue
                ix = torch.as_tensor(idx, device=E.dev)
                S[(key_src, g + ":k")] += ek[:, ix].double().sum(1).cpu()
                S[(key_src, g + ":v")] += ev[:, ix].double().sum(1).cpu()
                Nn[(key_src, g)] += len(idx)
            if ans.any():
                ia = torch.as_tensor(np.nonzero(ans)[0] + s, device=E.dev)
                ans_mass[key_src].append((a_true[:, ia].sum(1).cpu().double(), a_proj[:, ia].sum(1).cpu().double()))
    print(f"[tokerr] {len(items)} prompts, {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = []
    for src in sorted({k[0] for k in Nn}):
        n = Nn[(src, "tok")]
        r = {"source": src, "prompts": Nn[(src, "prompts")], "tokens": n,
             "err_k": (S[(src, "k_sum")] / n).tolist(), "err_v": (S[(src, "v_sum")] / n).tolist(),
             "err_k_attnw": (S[(src, "k_w")] / S[(src, "w")]).tolist(), "err_v_attnw": (S[(src, "v_w")] / S[(src, "w")]).tolist(),
             "mid_mass_true": (S[(src, "mass_mid_true")] / Nn[(src, "prompts")]).tolist(),
             "mid_mass_proj": (S[(src, "mass_mid_proj")] / Nn[(src, "prompts")]).tolist(), "groups": {}}
        for (s2, g), c in Nn.items():
            if s2 == src and (g.startswith("type:") or g.startswith("pos:")):
                r["groups"][g] = {"n": c, "err_k": (S[(src, g + ":k")] / c).tolist(), "err_v": (S[(src, g + ":v")] / c).tolist()}
        if ans_mass[src]:
            r["ans_mass_true"] = torch.stack([x[0] for x in ans_mass[src]]).mean(0).tolist()
            r["ans_mass_proj"] = torch.stack([x[1] for x in ans_mass[src]]).mean(0).tolist()
            r["ans_prompts"] = len(ans_mass[src])
        rows.append(r)
        print(f"[tokerr] {src:>22}: errK={np.mean(r['err_k']):.3f} errV={np.mean(r['err_v']):.3f} "
              f"attn-weighted errK={np.mean(r['err_k_attnw']):.3f} errV={np.mean(r['err_v_attnw']):.3f}"
              + (f" ans mass true={np.mean(r['ans_mass_true']):.4f} proj={np.mean(r['ans_mass_proj']):.4f}" if "ans_mass_true" in r else ""),
              flush=True)
    return rows


# ----------------------------------------------------------------------------------------------- summary
def fmt(x, f="{:.3f}"):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


def write_summary(res, path, Ll):
    cfgs = res["meta"]["configs"]
    groups = layer_groups(Ll)
    L = ["# projector 误差诊断（不训练）", "",
         f"GPU：{res['meta']['gpu']}；生成：{res['meta']['time']}；projector：`{res['meta']['args']['projector']}`（第 {res['meta'].get('proj_step')} 步）。"
         f"小模型 {C.SMALL_NAME}，大模型 {C.LARGE_NAME}（{Ll} 层）。所有配置都不重算；“真 K+V：层 a–b”= 中间 token 在这些层的投影 KV 换成大模型真实的 KV，其余同“混合，不重算”。", ""]
    if res.get("kl"):
        types = sorted({r["type"] for r in res["kl"]})
        kl = lambda c, t=None: mean([r["kl"] for r in res["kl"] if r["config"] == c and (t is None or r["type"] == t)])
        base = kl("hyb") if any(r["config"] == "hyb" for r in res["kl"]) else None
        L += ["## 与大模型的 KL（sd_mixtest 答案，越低越好；“补回”= 相对“混合，不重算”降低的比例）", "",
              "| 配置 | " + " | ".join(types) + " | 平均 | top-1 一致 | 补回 |", "|---|" + "---|" * (len(types) + 3)]
        for c in cfgs:
            if c == "big" or not any(r["config"] == c for r in res["kl"]):
                continue
            ag = mean([r["top1_agree"] for r in res["kl"] if r["config"] == c])
            rec = None if base is None or c == "hyb" else (base - kl(c)) / base
            L.append(f"| {label(c, Ll)} | " + " | ".join(fmt(kl(c, t)) for t in types) + f" | {fmt(kl(c))} | {fmt(ag)} | "
                     + ("-" if rec is None else f"{rec:.0%}") + " |")
        L.append("")
    if res.get("ruler"):
        lens = sorted({r["length"] for r in res["ruler"]})
        sc = lambda c, Lx, f=None: mean([r["score"] for r in res["ruler"] if r["config"] == c and r["length"] == Lx
                                         and r["task"] != "_frac" and (f is None or f(r["task"]))])
        for Lx in lens:
            L += [f"## RULER {Lx}（“补回”= 补回“混合，不重算”与大模型之间差距的比例）", "",
                  "| 配置 | " + " | ".join(RULER_CAT) + " | 平均 | 补回 |", "|---|" + "---|" * (len(RULER_CAT) + 2)]
            hb, bb = sc("hyb", Lx), sc("big", Lx)
            for c in cfgs:
                if not any(r["config"] == c and r["length"] == Lx for r in res["ruler"]):
                    continue
                rec = (sc(c, Lx) - hb) / (bb - hb) if c not in ("hyb", "big") and hb == hb and bb == bb and bb != hb else None
                L.append(f"| {label(c, Ll)} | " + " | ".join(fmt(sc(c, Lx, f)) for f in RULER_CAT.values()) + f" | {fmt(sc(c, Lx))} | "
                         + ("-" if rec is None else f"{rec:.0%}") + " |")
            L.append("")
    if res.get("qa"):
        bks = sorted({r["bucket"] for r in res["qa"]})
        f1 = lambda c, b: mean([r["f1"] for r in res["qa"] if r["config"] == c and r["bucket"] == b])
        L += ["## 问答 F1（squad / hotpot / trivia 平均）", "", "| 配置 | " + " | ".join(bks) + " |", "|---|" + "---|" * len(bks)]
        for c in cfgs:
            if any(r["config"] == c for r in res["qa"]):
                L.append(f"| {label(c, Ll)} | " + " | ".join(fmt(f1(c, b)) for b in bks) + " |")
        L.append("")
    if res.get("tokerr"):
        rows = {r["source"]: r for r in res["tokerr"]}
        gm = lambda v, a, b: float(np.mean(v[a:b + 1]))
        L += ["## 逐 token 误差（相对误差 ‖投影 − 真实‖ / ‖真实‖，中间 token）", "",
              "“加权”= 按最后一个 prompt token（真实 query、真实 key）对该 token 的注意力加权，即“真正被读到的 token”的误差。", "",
              "### 按层组（全部 prompt）", "", "| 层组 | K 误差 | K 误差（加权） | V 误差 | V 误差（加权） | 注意力落在中间的比例：真 key | 投影 key |",
              "|---|---|---|---|---|---|---|"]
        r = rows["all"]
        for a, b in groups:
            L.append(f"| {a}–{b} | {gm(r['err_k'], a, b):.3f} | {gm(r['err_k_attnw'], a, b):.3f} | {gm(r['err_v'], a, b):.3f} | "
                     f"{gm(r['err_v_attnw'], a, b):.3f} | {gm(r['mid_mass_true'], a, b):.3f} | {gm(r['mid_mass_proj'], a, b):.3f} |")
        L += ["", "### 按来源（全部层平均）", "",
              "| 来源 | prompt 数 | K 误差 | K 加权 | V 误差 | V 加权 | 数字 K / V | 单词 K / V | 答案 token K / V | 答案上的注意力：真 key | 投影 key |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for src in ["all"] + sorted(s for s in rows if s != "all"):
            r = rows[src]
            g = r["groups"]
            kv = lambda name: (f"{np.mean(g[name]['err_k']):.3f} / {np.mean(g[name]['err_v']):.3f}" if name in g else "-")
            L.append(f"| {src} | {r['prompts']} | {np.mean(r['err_k']):.3f} | {np.mean(r['err_k_attnw']):.3f} | {np.mean(r['err_v']):.3f} | "
                     f"{np.mean(r['err_v_attnw']):.3f} | {kv('type:数字')} | {kv('type:单词')} | {kv('type:答案')} | "
                     + (f"{np.mean(r['ans_mass_true']):.4f} | {np.mean(r['ans_mass_proj']):.4f} |" if "ans_mass_true" in r else "- | - |"))
        g = rows["all"]["groups"]
        L += ["", "### 按在中间部分的位置（十等分，全部 prompt，全部层平均）", "", "| 位置 | " + " | ".join(str(i) for i in range(10)) + " |",
              "|---|" + "---|" * 10,
              "| K 误差 | " + " | ".join(fmt(float(np.mean(g[f'pos:{i}']['err_k'])) if f"pos:{i}" in g else None) for i in range(10)) + " |",
              "| V 误差 | " + " | ".join(fmt(float(np.mean(g[f'pos:{i}']['err_v'])) if f"pos:{i}" in g else None) for i in range(10)) + " |", ""]
        L += ["### 逐层（全部 prompt）", "", "| 层 | K 误差 | K 加权 | V 误差 | V 加权 |", "|---|---|---|---|---|"]
        r = rows["all"]
        for l in range(Ll):
            L.append(f"| {l} | {r['err_k'][l]:.3f} | {r['err_k_attnw'][l]:.3f} | {r['err_v'][l]:.3f} | {r['err_v_attnw'][l]:.3f} |")
        L.append("")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", default=os.path.join(C.CKPT_DIR, "r2_mix8k", "projector.pt"))
    ap.add_argument("--configs", default="", help="comma list; default: hyb + 4 layer-group K+V oracles + all-K + all-V")
    ap.add_argument("--kl_configs", default="", help="configs for the kl task (default: --configs + or:all:kv sanity)")
    ap.add_argument("--tasks", default="kl,tokerr,ruler,qa")
    ap.add_argument("--n_kl", type=int, default=8, help="kl: sequences per (type, bucket) cell")
    ap.add_argument("--n_ruler", type=int, default=15)
    ap.add_argument("--n_qa", type=int, default=50)
    ap.add_argument("--n_tok_ruler", type=int, default=5, help="tokerr: prompts per RULER task and length")
    ap.add_argument("--n_tok_qa", type=int, default=10, help="tokerr: prompts per QA source and bucket")
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res_path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    dev = torch.device("cuda")
    tok, small, large = load_pair(dev)
    E = DiagEngine(tok, small, large, a.projector, dev, a.n_sink, a.n_tail)
    Ll = E.L.n_layers
    cfgs = [c for c in a.configs.split(",") if c] or default_configs(Ll)
    kl_cfgs = [c for c in a.kl_configs.split(",") if c] or (cfgs + (["or:all:kv"] if any(c.startswith("or:") for c in cfgs) else []))
    run_cfgs = cfgs + (["big"] if "big" not in cfgs else [])  # big: the reference for "recovered" in RULER / QA
    res["meta"] = {"gpu": torch.cuda.get_device_name(0), "configs": run_cfgs, "kl_configs": kl_cfgs, "args": vars(a),
                   "proj_step": E.proj_step, "time": time.strftime("%F %T"), "Ll": Ll}
    log = GenLog(os.path.join(a.out_dir, "generations.jsonl"))
    for task in a.tasks.split(","):
        if task in res and not a.force:
            print(f"[task {task}] already done, skipping")
            continue
        t0 = time.time()
        if task == "kl":
            res["kl"] = task_kl(E, kl_cfgs, a.n_kl)
        elif task == "ruler":
            res["ruler"] = task_ruler(E, run_cfgs, a.n_ruler, [4096, 8192], log)
        elif task == "qa":
            res["qa"] = task_qa(E, run_cfgs, a.n_qa, log)
        elif task == "tokerr":
            res["tokerr"] = task_tokerr(E, a.n_tok_ruler, a.n_tok_qa)
        else:
            raise SystemExit(f"unknown task {task}")
        print(f"[task {task}] {(time.time() - t0) / 60:.1f} min", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)
        write_summary(res, os.path.join(a.out_dir, "summary.md"), Ll)
    print("[done]", a.out_dir)


if __name__ == "__main__":
    main()
