#!/usr/bin/env python
"""
Stage 2: end-to-end output distillation through the frozen large model.
Phase A: self-distillation data -- the large model generates its own continuation for random prompt prefixes.
Phase B: prompt prefilled by the hybrid scheme (true sink | projected middle | true tail), continuation run
         through the frozen large model; loss = KL(teacher || student) + lambda_nll * NLL
         [+ lambda_attn * attention-output matching on the prompt (stage-1 auxiliary loss, true queries, projected
         middle KV vs the teacher's own KV) on random layers from the middle third onwards]. Gradients flow
         through the frozen large model into the projector only. Curriculum on prompt length.
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
from prefill.models import eos_ids, load_pair, teacher_of  # noqa: E402
from prefill.xteacher import NativeTeacher, masked_mean  # noqa: E402
from prefill.projector import Projector, ProjectorTokMoE  # noqa: E402
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
            out.append((m["type"], st.get(i), int(pl[i]), i))
            cnt[m["type"]] = cnt.get(m["type"], 0) + 1
    print(f"[val] {len(out)} fixed validation sequences: {cnt}")
    return out


@torch.no_grad()
def validate_fixed(large, small, projector, val, n_sink, n_tail, device, teacher=None, native=None):
    """KL(teacher || hybrid) on the 8B's own held-out responses, per data type and overall (teacher: default large)."""
    projector.eval()
    per = {}
    for typ, ids, P, i in val:
        ids = ids.unsqueeze(0).to(device)
        if native is not None:  # cross-family reverse pair: the teacher reads its own original sequence
            t, msk = native.logits(teacher, i, device)
        else:
            t, msk = teacher_forward(teacher if teacher is not None else large, ids, P), None
        s = student_forward(large, small, projector, ids, P, n_sink, n_tail)
        per.setdefault(typ, []).append(masked_mean(kl_teacher_student(t, s), msk).item())
    projector.train()
    out = {f"val_kl_{k}": sum(v) / len(v) for k, v in per.items()}
    out["val_kl"] = sum(out.values()) / max(1, len(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(C.CKPT_DIR, "stage1", "projector.pt"),
                    help="projector checkpoint, or 'c2c' for a fresh Cache-to-Cache fuser (prefill/c2c.py; no stage 1)")
    ap.add_argument("--c2c_anneal", type=int, default=0, help="C2C gate-temperature annealing steps (0 = max_steps)")
    ap.add_argument("--wd", type=float, default=0.01)
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
    ap.add_argument("--tmoe", type=int, default=0, help="token-routed MoE projector with this many experts (0 = keep --init as it is); a plain Projector --init is converted exactly (every expert = its trunk), a MoE --init is kept")
    ap.add_argument("--tmoe_topk", type=int, default=2, help="experts per (layer, token) cell")
    ap.add_argument("--tmoe_band", type=float, default=0.0, help="initial layer prior of the gate: neighbouring layers prefer the same experts (0 = none)")
    ap.add_argument("--lambda_bal", type=float, default=0.01, help="weight of the MoE load-balance loss")
    ap.add_argument("--probe_longest", action="store_true", help="first run one forward/backward on the longest training "
                    "sequence and print the peak GPU memory (catches an OOM before the length curriculum reaches it)")
    ap.add_argument("--save_step_every", type=int, default=2000, help="also keep projector_step<N>.pt every N steps")
    ap.add_argument("--start_step", type=int, default=0, help="resume an interrupted run: --init its projector.pt, continue "
                    "LR schedule and curriculum from this step (optimizer state is not saved, so the LR re-ramps over --warmup)")
    a = ap.parse_args()

    torch.manual_seed(a.seed + a.start_step)
    random.seed(a.seed + a.start_step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    log = Logger(os.path.join(a.out_dir, "train_log.jsonl"))
    tok, small, large = load_pair(device)
    teacher = teacher_of(small, large)  # PREFILL_TEACHER=small: reverse pair, the prefilling model is the teacher
    print(f"[teacher] {C.TEACHER_NAME} ({C.TEACHER})")
    if a.lambda_attn > 0 and teacher is not large:
        raise SystemExit("--lambda_attn needs the large model as teacher (it matches the large model's own residual streams)")
    sources = a.sources.split(",")
    train_text = MixedSampler(sources, "train", a.min_prompt, seed=a.seed + 7)
    val = MixedSampler(sources, "test", 768, seed=1234)

    t_start = time.time()
    if a.sd:
        stores = [(TokenStore(n, "all"), np.load(os.path.join(C.DATA_DIR, n, "prompt_len.npy"))) for n in a.sd.split(",")]
        items = [(k, i) for k, (st, pl) in enumerate(stores) for i in range(len(st)) if pl[i] <= a.max_prompt]
        sd_get = lambda it: stores[it[0]][0].get(it[1])
        sd_plen = lambda it: int(stores[it[0]][1][it[1]])
        natives = [NativeTeacher.load(n) for n in a.sd.split(",")]  # cross-family reverse pair: teacher's own view
        if any(x is not None for x in natives):
            print(f"[sd] teacher reads its own sequences for: {[n for n, x in zip(a.sd.split(','), natives) if x]}")
        print(f"[sd] {a.sd}: {len(items)} sequences with prompt <= {a.max_prompt}")
    else:
        generate_selfdistill(large, tok, train_text, a.n_gen, a.cont_len, a.min_prompt, a.max_prompt, a.gen_batch, device, a.gen_temperature)
        sd = TokenStore(os.path.basename(C.SELFDISTILL_DIR), "train")
        plens = np.load(os.path.join(C.SELFDISTILL_DIR, "train_prompt_len.npy"))
        items = [(0, i) for i in range(len(sd))]
        sd_get = lambda it: sd.get(it[1])
        sd_plen = lambda it: int(plens[it[1]])
        natives = [None]
        print(f"[selfdistill] {len(sd)} sequences, generation took {(time.time()-t_start)/60:.1f} min")
    item_plen = np.array([sd_plen(it) for it in items])
    val_fixed = load_val_set(a.val_sd, a.val_per_type, a.max_prompt)
    val_native = NativeTeacher.load(a.val_sd) if C.TEACHER == "small" else None

    if a.init == "c2c":
        from prefill.c2c import C2CFuser
        projector, ck = C2CFuser.new(small, large, anneal_steps=a.c2c_anneal or a.max_steps).to(device), {}
        print(f"[init] fresh C2C fuser, source {C.SMALL_NAME} layer l+{projector.offset} -> decoder {C.LARGE_NAME} layer l; "
              f"{projector.num_params()/1e6:.1f}M params, gate annealing over {projector.cfg['hp']['anneal_steps']} steps")
    else:
        projector, ck = Projector.load(a.init, device)
        print(f"[init] {a.init} (stage1 step {ck.get('step')}) ; {projector.num_params()/1e6:.1f}M params")
    is_c2c = getattr(projector, "is_c2c", False)
    if a.tmoe and not is_c2c and projector.cfg.get("kind") != "tmoe":
        projector = ProjectorTokMoE.from_projector(projector, a.tmoe, a.tmoe_topk, a.tmoe_band)
        print(f"[tmoe] converted: {a.tmoe} experts, top-{a.tmoe_topk}, band {a.tmoe_band} ; {projector.num_params()/1e6:.1f}M params")
    projector.train()
    opt = torch.optim.AdamW(projector.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)

    def run_one(it):
        """KL + NLL (+ MoE load balance) of one training sequence, plain path (no attention loss, no native teacher)."""
        ids, P = sd_get(it).unsqueeze(0).to(device), sd_plen(it)
        t_logits = teacher_forward(teacher, ids, P)
        s_logits = student_forward(large, small, projector, ids, P, a.n_sink, a.n_tail)
        loss = kl_teacher_student(t_logits, s_logits).mean()
        loss = loss + a.lambda_nll * F.cross_entropy(s_logits.float().flatten(0, 1), ids[:, P:].flatten().to(s_logits.device)).to(loss.device)
        if getattr(projector, "aux", None) is not None:
            loss = loss + a.lambda_bal * projector.aux.to(loss.device)
        return loss, ids.shape[1]

    if a.probe_longest:
        it = items[int(np.argmax(item_plen))]  # longest prompt
        loss, n = run_one(it)
        loss.backward()
        opt.zero_grad(set_to_none=True)
        peak = [torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(torch.cuda.device_count())]
        print(f"[probe] longest prompt, sequence of {n} tokens: loss {loss.item():.4f}, peak GiB per GPU {[round(x, 1) for x in peak]}", flush=True)
        del loss

    if a.start_step and ck.get("step") != a.start_step:
        print(f"[warn] --start_step {a.start_step} but {a.init} was saved at step {ck.get('step')}")
    step = a.start_step
    while step < a.max_steps:
        lr = cosine_lr(step, a.max_steps, a.warmup, a.lr)
        if a.start_step:  # fresh AdamW moments: ramp back up to the scheduled LR
            lr *= min(1.0, (step - a.start_step + 1) / max(1, a.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        if is_c2c:
            projector.update_temperature(step)
        frac = min(1.0, step / max(1, a.curriculum_frac * a.max_steps))
        cur_max = int(512 + (a.max_prompt - 512) * frac)
        stats = {}
        for _ in range(a.accum):
            ok = np.nonzero(item_plen <= cur_max)[0]
            it = items[int(ok[random.randrange(len(ok))])] if len(ok) else items[random.randrange(len(items))]
            ids = sd_get(it).unsqueeze(0).to(device)
            P = sd_plen(it)
            if a.lambda_attn > 0:
                with torch.no_grad():  # teacher logits and its own residual streams in one pass
                    out = large(input_ids=ids, output_hidden_states=True, use_cache=False, logits_to_keep=ids.shape[1] - P + 1)
                t_logits = out.logits[:, :-1]
                Ll = len(large.model.layers)
                h_true = torch.stack([h[:, :P].to(device) for h in out.hidden_states[:Ll]], 0)  # [Ll,1,P,d]
                del out
                s_logits, h_hat = student_forward(large, small, projector, ids, P, a.n_sink, a.n_tail, return_hidden=True)
                t_mask = None
            elif natives[it[0]] is not None:
                t_logits, t_mask = natives[it[0]].logits(teacher, it[1], device)
                s_logits = student_forward(large, small, projector, ids, P, a.n_sink, a.n_tail)
                h_hat = None
            else:
                t_logits, t_mask = teacher_forward(teacher, ids, P), None
                s_logits = student_forward(large, small, projector, ids, P, a.n_sink, a.n_tail)
                h_hat = None
            kl = masked_mean(kl_teacher_student(t_logits, s_logits), t_mask)
            nll = F.cross_entropy(s_logits.float().flatten(0, 1), ids[:, P:].flatten().to(s_logits.device))
            loss = kl + a.lambda_nll * nll.to(kl.device)  # split teacher: its logits (and kl) sit on its last GPU
            if h_hat is not None:
                n_s, n_m, _ = split_prompt(P, a.n_sink, a.n_tail)
                h_mix = torch.cat([h_true[:, :, :n_s], h_hat.to(h_true.dtype), h_true[:, :, n_s + n_m:]], 2)
                layers = random.sample(range(Ll // 3, Ll), min(a.attn_layers, Ll - Ll // 3))
                loss_a = attention_output_loss(large, h_mix, h_true, n_s, layers=layers)
                loss = loss + a.lambda_attn * loss_a.to(loss.device)
                stats["attn"] = stats.get("attn", 0) + loss_a.item() / a.accum
            if getattr(projector, "aux", None) is not None:  # token-routed MoE: load balance
                loss = loss + a.lambda_bal * projector.aux.to(loss.device)
                stats["bal"] = stats.get("bal", 0) + projector.aux.item() / a.accum
                stats.update(projector.stats)
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
            projector.eval()  # C2C: hard gates, no dropout (as at inference)
            if is_c2c:
                g = projector.gate_logits()
                log.log(step=step, gates_open_k=sum(k > 0 for k, _ in g), gates_open_v=sum(v > 0 for _, v in g),
                        gate_temp=float(projector.projs[0].gate_temperature))
            if val_fixed:
                log.log(step=step, **validate_fixed(large, small, projector, val_fixed, a.n_sink, a.n_tail, device, teacher, val_native))
            else:
                v = validate_hybrid(large, small, projector, val, a.val_seqs, 2048, 256, a.n_sink, a.n_tail, device, teacher)
                log.log(step=step, **v)
            projector.train()
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
