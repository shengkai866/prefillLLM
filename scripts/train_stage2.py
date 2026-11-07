#!/usr/bin/env python
"""
Stage 2: end-to-end output distillation through the frozen large model.
Phase A: self-distillation data -- the large model generates its own continuation for random prompt prefixes.
Phase B: prompt prefilled by the hybrid scheme (true sink | projected middle | true tail), continuation run
         through the frozen large model; loss = KL(teacher || student) + lambda_nll * NLL
         [+ lambda_attn * attention-output matching on the prompt (stage-1 auxiliary loss, true queries, projected
         middle KV vs the teacher's own KV) on random layers from the middle third onwards]. Gradients flow
         through the frozen large model into the projector only. Curriculum on prompt length.
--rc_fracs (recompute-aware stage 2, prefill/rc_train.py): per sequence a fraction drawn from the list of middle tokens
         (small-model attention from the tail, sentence units = inference's "ours") is recomputed by the large model
         inside the forward, so the projector is trained only on the cells the large model does not compute.
--rc_sel exact:0.5,miss:0.25,random:0.25 (rcn runs): the training selection is sometimes deliberately imperfect (top
         sentences excluded / random sentences) so the projector does not rely on the most attended sentences always
         being recomputed.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import MixedSampler, TokenStore, save_store  # noqa: E402
from prefill.kv import kl_teacher_student, split_prompt, student_forward, teacher_forward  # noqa: E402
from prefill.losses import attention_output_loss  # noqa: E402
from prefill.models import eos_ids, load_pair  # noqa: E402
from prefill.projector import Projector  # noqa: E402
from prefill.fast import FastQwen3  # noqa: E402
from prefill.rc_train import select_middle, student_forward_rc  # noqa: E402
from prefill.trainutil import Logger, cosine_lr, validate_hybrid  # noqa: E402


@torch.no_grad()
def generate_selfdistill(large, tok, sampler, n_samples, cont_len, min_prompt, max_prompt, batch, device, temperature):
    """Generate `n_samples` (prompt + large-model continuation) sequences; store prompt lengths separately."""
    name = os.path.basename(C.SELFDISTILL_DIR)
    if os.path.exists(os.path.join(C.SELFDISTILL_DIR, "train_offsets.npy")):
        print("[selfdistill] exists, skip generation")
        return
    seqs, plens, t0 = [], [], time.time()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    while len(seqs) < n_samples:
        prompts = []
        for _ in range(batch):
            L = random.randint(min_prompt, max_prompt)
            ids = sampler.sample(L)
            prompts.append(ids[:L])
        maxL = max(len(p) for p in prompts)
        inp = torch.full((batch, maxL), pad, dtype=torch.long)
        att = torch.zeros((batch, maxL), dtype=torch.long)
        for i, p in enumerate(prompts):  # left padding
            inp[i, maxL - len(p):] = p
            att[i, maxL - len(p):] = 1
        out = large.generate(input_ids=inp.to(device), attention_mask=att.to(device), max_new_tokens=cont_len,
                             min_new_tokens=cont_len, do_sample=temperature > 0, temperature=max(temperature, 1e-3),
                             top_p=0.95, pad_token_id=pad)
        gen = out[:, maxL:].cpu()
        for i, p in enumerate(prompts):
            seqs.append(torch.cat([p, gen[i]]).numpy())
            plens.append(len(p))
        print(f"[selfdistill] {len(seqs)}/{n_samples} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(C.SELFDISTILL_DIR, exist_ok=True)
    np.save(os.path.join(C.SELFDISTILL_DIR, "train_prompt_len.npy"), np.array(plens, dtype=np.int64))
    save_store(name, "train", seqs)


def load_val_set(name, per_type, max_prompt):
    """Fixed held-out validation sequences: the first `per_type` of every data type with prompt <= max_prompt."""
    d = os.path.join(C.DATA_DIR, name)
    if not os.path.exists(os.path.join(d, "meta.jsonl")):
        print(f"[val] {name} not found, falling back to web validation")
        return []
    st = TokenStore(name, "all")
    pl = np.load(os.path.join(d, "prompt_len.npy"))
    meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    out, cnt = [], {}
    for i, m in enumerate(meta):
        # identical set for every run (4k bucket only), so validation curves are comparable across runs
        if m.get("bucket", "4k") == "4k" and pl[i] <= max_prompt and cnt.get(m["type"], 0) < per_type:
            out.append((m["type"], st.get(i), int(pl[i])))
            cnt[m["type"]] = cnt.get(m["type"], 0) + 1
    print(f"[val] {len(out)} fixed validation sequences: {cnt}")
    return out


@torch.no_grad()
def validate_fixed(large, small, projector, val, n_sink, n_tail, device, rc=None):
    """KL(8B || hybrid) on the 8B's own held-out responses, per data type and overall.
    rc = (L, S, boundary, frac): also KL with `frac` of the middle recomputed (val_kl_rc<frac*100>)."""
    projector.eval()
    per, per_rc = {}, {}
    for typ, ids, P in val:
        ids = ids.unsqueeze(0).to(device)
        t = teacher_forward(large, ids, P)
        s = student_forward(large, small, projector, ids, P, n_sink, n_tail)
        per.setdefault(typ, []).append(kl_teacher_student(t, s).mean().item())
        if rc is not None:
            s = student_forward_rc(rc[0], rc[1], projector, ids, P, n_sink, n_tail, rc[3], rc[2], ckpt=False)[0]
            per_rc.setdefault(typ, []).append(kl_teacher_student(t, s.to(t.device)).mean().item())
    projector.train()
    out = {f"val_kl_{k}": sum(v) / len(v) for k, v in per.items()}
    out["val_kl"] = sum(out.values()) / max(1, len(out))
    if rc is not None:
        tag = f"val_kl_rc{round(rc[3] * 100)}"
        r = {f"{tag}_{k}": sum(v) / len(v) for k, v in per_rc.items()}
        out.update(r)
        out[tag] = sum(r.values()) / max(1, len(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(C.CKPT_DIR, "stage1", "projector.pt"))
    ap.add_argument("--out_dir", default=os.path.join(C.CKPT_DIR, "stage2"))
    ap.add_argument("--sources", default="fineweb_edu,slimpajama")
    ap.add_argument("--n_gen", type=int, default=2400)
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--gen_temperature", type=float, default=0.8)
    ap.add_argument("--cont_len", type=int, default=256)
    ap.add_argument("--min_prompt", type=int, default=256)
    ap.add_argument("--max_prompt", type=int, default=C.MAX_PROMPT_LEN - 128)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lambda_nll", type=float, default=0.1)
    ap.add_argument("--lambda_attn", type=float, default=0.0, help="weight of the attention-output matching loss (0 = off)")
    ap.add_argument("--attn_layers", type=int, default=4, help="layers sampled per step for the attention loss")
    ap.add_argument("--n_sink", type=int, default=8)
    ap.add_argument("--n_tail", type=int, default=64)
    ap.add_argument("--curriculum_frac", type=float, default=0.5, help="fraction of steps over which max prompt len grows 512 -> max")
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--val_seqs", type=int, default=8)
    ap.add_argument("--time_budget_hours", type=float, default=22.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sd", default=None, help="comma list of self-distillation stores made by gen_selfdistill.py "
                    "(round 2); when given, the built-in round-1 web generation is skipped")
    ap.add_argument("--val_sd", default="sd_mixtest", help="held-out self-distillation store for per-type validation")
    ap.add_argument("--val_per_type", type=int, default=2)
    ap.add_argument("--save_step_every", type=int, default=2000, help="also keep projector_step<N>.pt every N steps")
    ap.add_argument("--start_step", type=int, default=0, help="resume an interrupted run: --init its projector.pt, continue "
                    "LR schedule and curriculum from this step (optimizer state is not saved, so the LR re-ramps over --warmup)")
    ap.add_argument("--rc_fracs", default="", help="recompute-aware stage 2: comma list of recompute fractions, one drawn "
                    "per sequence (e.g. 0,0.1,0.1,0.2); empty = plain stage 2")
    ap.add_argument("--rc_val_frac", type=float, default=0.1, help="also validate with this recompute fraction (rc runs)")
    ap.add_argument("--rc_sel", default="exact:1", help="selection modes during rc training with weights, e.g. "
                    "exact:0.5,miss:0.25,random:0.25 (prefill/rc_train.perturb_score); validation always uses exact")
    ap.add_argument("--rc_miss_frac", type=float, default=0.5, help="miss mode: share of the budget whose top sentences are excluded")
    ap.add_argument("--rc_ckpt", type=int, default=1, help="gradient-checkpoint the large-model recompute pass per layer")
    a = ap.parse_args()

    torch.manual_seed(a.seed + a.start_step)
    random.seed(a.seed + a.start_step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    log = Logger(os.path.join(a.out_dir, "train_log.jsonl"))
    tok, small, large = load_pair(device)
    rc_fracs = [float(x) for x in a.rc_fracs.split(",") if x != ""]
    rc = None
    if rc_fracs:
        Lf, Sf = FastQwen3(large), FastQwen3(small)
        boundary = torch.tensor([any(ch in tok.decode([i]) for ch in ".!?\n") for i in range(len(tok))], device=device)
        rc = (Lf, Sf, boundary, a.rc_val_frac)
        sel_modes = [x.split(":")[0] for x in a.rc_sel.split(",")]
        sel_w = [float(x.split(":")[1]) for x in a.rc_sel.split(",")]
        print(f"[rc] recompute-aware stage 2: fractions {rc_fracs}, selection {dict(zip(sel_modes, sel_w))}, "
              f"validation also at {a.rc_val_frac}", flush=True)
    sources = a.sources.split(",")
    train_text = MixedSampler(sources, "train", a.min_prompt, seed=a.seed + 7)
    val = MixedSampler(sources, "test", 768, seed=1234)

    t_start = time.time()
    if a.sd:
        stores = [(TokenStore(n, "all"), np.load(os.path.join(C.DATA_DIR, n, "prompt_len.npy"))) for n in a.sd.split(",")]
        items = [(k, i) for k, (st, pl) in enumerate(stores) for i in range(len(st)) if pl[i] <= a.max_prompt]
        sd_get = lambda it: stores[it[0]][0].get(it[1])
        sd_plen = lambda it: int(stores[it[0]][1][it[1]])
        print(f"[sd] {a.sd}: {len(items)} sequences with prompt <= {a.max_prompt}")
    else:
        generate_selfdistill(large, tok, train_text, a.n_gen, a.cont_len, a.min_prompt, a.max_prompt, a.gen_batch, device, a.gen_temperature)
        sd = TokenStore(os.path.basename(C.SELFDISTILL_DIR), "train")
        plens = np.load(os.path.join(C.SELFDISTILL_DIR, "train_prompt_len.npy"))
        items = [(0, i) for i in range(len(sd))]
        sd_get = lambda it: sd.get(it[1])
        sd_plen = lambda it: int(plens[it[1]])
        print(f"[selfdistill] {len(sd)} sequences, generation took {(time.time()-t_start)/60:.1f} min")
    item_plen = np.array([sd_plen(it) for it in items])
    val_fixed = load_val_set(a.val_sd, a.val_per_type, a.max_prompt)

    projector, ck = Projector.load(a.init, device)
    print(f"[init] {a.init} (stage1 step {ck.get('step')}) ; {projector.num_params()/1e6:.1f}M params")
    projector.train()
    opt = torch.optim.AdamW(projector.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)

    if a.start_step and ck.get("step") != a.start_step:
        print(f"[warn] --start_step {a.start_step} but {a.init} was saved at step {ck.get('step')}")
    step = a.start_step
    while step < a.max_steps:
        lr = cosine_lr(step, a.max_steps, a.warmup, a.lr)
        if a.start_step:  # fresh AdamW moments: ramp back up to the scheduled LR
            lr *= min(1.0, (step - a.start_step + 1) / max(1, a.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        frac = min(1.0, step / max(1, a.curriculum_frac * a.max_steps))
        cur_max = int(512 + (a.max_prompt - 512) * frac)
        stats = {}
        for _ in range(a.accum):
            ok = np.nonzero(item_plen <= cur_max)[0]
            it = items[int(ok[random.randrange(len(ok))])] if len(ok) else items[random.randrange(len(items))]
            ids = sd_get(it).unsqueeze(0).to(device)
            P = sd_plen(it)
            if rc is not None:
                frac = random.choice(rc_fracs)
                mode = random.choices(sel_modes, sel_w)[0]
                sel = select_middle(Sf, ids[0], P, a.n_sink, a.n_tail, frac, boundary, mode=mode, miss_frac=a.rc_miss_frac)
                need_h = a.lambda_attn > 0
                with torch.no_grad():
                    if need_h:
                        out = large(input_ids=ids, output_hidden_states=True, use_cache=False, logits_to_keep=ids.shape[1] - P + 1)
                        t_logits = out.logits[:, :-1]
                        Ll = len(large.model.layers)
                        h_true = torch.stack([h[:, :P].to(device) for h in out.hidden_states[:Ll]], 0)
                        del out
                    else:
                        t_logits = teacher_forward(large, ids, P)
                s_logits, idx, h_hat = student_forward_rc(Lf, Sf, projector, ids, P, a.n_sink, a.n_tail, frac, boundary,
                                                          ckpt=bool(a.rc_ckpt), return_hidden=True, sel=sel)
                if not need_h:
                    h_hat = None
                stats["rc_frac"] = stats.get("rc_frac", 0) + frac / a.accum
                stats["rc_n"] = stats.get("rc_n", 0) + idx.numel() / a.accum
                if frac > 0:
                    stats[f"rc_sel_{mode}"] = stats.get(f"rc_sel_{mode}", 0) + 1 / a.accum
            elif a.lambda_attn > 0:
                with torch.no_grad():  # teacher logits and its own residual streams in one pass
                    out = large(input_ids=ids, output_hidden_states=True, use_cache=False, logits_to_keep=ids.shape[1] - P + 1)
                t_logits = out.logits[:, :-1]
                Ll = len(large.model.layers)
                h_true = torch.stack([h[:, :P].to(device) for h in out.hidden_states[:Ll]], 0)  # [Ll,1,P,d]
                del out
                s_logits, h_hat = student_forward(large, small, projector, ids, P, a.n_sink, a.n_tail, return_hidden=True)
            else:
                t_logits = teacher_forward(large, ids, P)
                s_logits = student_forward(large, small, projector, ids, P, a.n_sink, a.n_tail)
                h_hat = None
            kl = kl_teacher_student(t_logits, s_logits).mean()
            nll = F.cross_entropy(s_logits.float().flatten(0, 1), ids[:, P:].flatten().to(s_logits.device))
            loss = kl + a.lambda_nll * nll
            if h_hat is not None:
                n_s, n_m, _ = split_prompt(P, a.n_sink, a.n_tail)
                h_mix = torch.cat([h_true[:, :, :n_s], h_hat.to(h_true.dtype), h_true[:, :, n_s + n_m:]], 2)
                if rc is not None and idx.numel():  # recomputed tokens are true in the cache
                    h_mix = h_mix.index_copy(2, n_s + idx, h_true[:, :, n_s + idx])
                layers = random.sample(range(Ll // 3, Ll), min(a.attn_layers, Ll - Ll // 3))
                loss_a = attention_output_loss(large, h_mix, h_true, n_s, layers=layers)
                loss = loss + a.lambda_attn * loss_a.to(loss.device)
                stats["attn"] = stats.get("attn", 0) + loss_a.item() / a.accum
            (loss / a.accum).backward()
            stats["kl"] = stats.get("kl", 0) + kl.item() / a.accum
            stats["nll"] = stats.get("nll", 0) + nll.item() / a.accum
            stats["P"] = P
        gn = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            log.log(step=step, lr=lr, grad_norm=gn, cur_max_prompt=cur_max, **stats)
        hours = (time.time() - t_start) / 3600
        if step % a.val_every == 0:
            if val_fixed:
                log.log(step=step, **validate_fixed(large, small, projector, val_fixed, a.n_sink, a.n_tail, device, rc))
            else:
                v = validate_hybrid(large, small, projector, val, a.val_seqs, 2048, 256, a.n_sink, a.n_tail, device)
                log.log(step=step, **v)
        if step % a.save_every == 0 or step == a.max_steps or hours > a.time_budget_hours:
            projector.save(os.path.join(a.out_dir, "projector.pt"), step=step, args=vars(a))
            print(f"[ckpt] step {step} saved ({hours:.2f}h)", flush=True)
        if a.save_step_every and step % a.save_step_every == 0:
            projector.save(os.path.join(a.out_dir, f"projector_step{step}.pt"), step=step, args=vars(a))
        if hours > a.time_budget_hours:
            print("[stop] time budget reached")
            break
    print("[done] stage2")


if __name__ == "__main__":
    main()
