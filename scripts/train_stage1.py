#!/usr/bin/env python
"""
Stage 1: representation alignment. Frozen small + large model, trainable projector.
Loss = per-layer (1 - cos) + normalised MSE between predicted and true large-model residual streams
     + lambda_attn * attention-output matching on a random subset of layers (true queries, projected KV).
Sink positions (first n_sink tokens) are excluded from the loss because they are always recomputed exactly.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.data import MixedSampler  # noqa: E402
from prefill.kv import large_hidden_states, small_hidden_states  # noqa: E402
from prefill.losses import attention_output_loss, hidden_alignment_loss  # noqa: E402
from prefill.models import load_pair, new_projector  # noqa: E402
from prefill.projector import Projector  # noqa: E402
from prefill.trainutil import Logger, cosine_lr, validate_hybrid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(C.CKPT_DIR, "stage1"))
    ap.add_argument("--sources", default="fineweb_edu,slimpajama")
    ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min_len", type=int, default=768)
    ap.add_argument("--max_len", type=int, default=C.MAX_PROMPT_LEN)
    ap.add_argument("--n_sink", type=int, default=4, help="positions excluded from loss / treated as true in aux loss")
    ap.add_argument("--lambda_attn", type=float, default=1.0)
    ap.add_argument("--attn_layers", type=int, default=4)
    ap.add_argument("--d_mid", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--in_norm", action="store_true", help="RMS-normalise small-model hidden states before the projector")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_seqs", type=int, default=8)
    ap.add_argument("--time_budget_hours", type=float, default=22.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    log = Logger(os.path.join(a.out_dir, "train_log.jsonl"))

    tok, small, large = load_pair(device)
    projector = new_projector(small, large, a.d_mid, a.rank, a.in_norm).to(device)
    step0 = 0
    if a.resume and os.path.exists(a.resume):
        projector, ck = Projector.load(a.resume, device)
        step0 = ck.get("step", 0)
        print(f"[resume] {a.resume} @ step {step0}")
    print(f"[projector] {projector.num_params()/1e6:.1f}M params; Ls={projector.cfg['Ls']} Ll={projector.cfg['Ll']}")
    opt = torch.optim.AdamW(projector.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sources = a.sources.split(",")
    train = MixedSampler(sources, "train", a.min_len, seed=a.seed)
    val = MixedSampler(sources, "test", a.min_len, seed=1234)

    t_start = time.time()
    projector.train()
    step = step0
    while step < a.max_steps:
        lr = cosine_lr(step, a.max_steps, a.warmup, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        stats = {}
        for _ in range(a.accum):
            ids = train.sample(a.max_len).unsqueeze(0).to(device)
            with torch.no_grad():
                h_true = large_hidden_states(large, ids)  # [Ll,1,T,d_l]
                hs = small_hidden_states(small, ids)  # [Ls,1,T,d_s]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                h_hat = projector(hs)
            mask = torch.ones(ids.shape, dtype=torch.bool, device=device)
            mask[:, : a.n_sink] = False
            loss_h, info = hidden_alignment_loss(h_hat, h_true, mask)
            loss = loss_h
            if a.lambda_attn > 0:
                loss_a = attention_output_loss(large, h_hat, h_true, a.n_sink, a.attn_layers)
                loss = loss + a.lambda_attn * loss_a
                stats["attn"] = stats.get("attn", 0) + loss_a.item() / a.accum
            (loss / a.accum).backward()
            stats["loss"] = stats.get("loss", 0) + loss.item() / a.accum
            stats["cos"] = stats.get("cos", 0) + info["cos"].mean().item() / a.accum
            stats["nmse"] = stats.get("nmse", 0) + info["nmse"].mean().item() / a.accum
            stats["T"] = ids.shape[1]
        gn = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            log.log(step=step, lr=lr, grad_norm=gn, **stats)
        hours = (time.time() - t_start) / 3600
        if step % a.val_every == 0:
            v = validate_hybrid(large, small, projector, val, a.val_seqs, min(2048, a.max_len), 256, 8, 64, device)
            cos_l = info["cos"].tolist()
            log.log(step=step, **v, cos_first=cos_l[0], cos_mid=cos_l[len(cos_l) // 2], cos_last=cos_l[-1])
        if step % a.save_every == 0 or step == a.max_steps or hours > a.time_budget_hours:
            projector.save(os.path.join(a.out_dir, "projector.pt"), step=step, args=vars(a))
            projector.save(os.path.join(a.out_dir, f"projector_step{step}.pt"), step=step, args=vars(a))
            print(f"[ckpt] step {step} saved ({hours:.2f}h)", flush=True)
        if hours > a.time_budget_hours:
            print("[stop] time budget reached")
            break
    print("[done] stage1")


if __name__ == "__main__":
    main()
