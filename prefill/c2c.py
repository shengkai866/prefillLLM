"""
Cache-to-Cache (C2C; Fu et al., ICLR'26, arXiv 2510.03215, github.com/thu-nics/C2C) inside this code base, for the
reverse ("big model helps small model") pair: the SOURCE model (code name `small`, e.g. Qwen3-8B, the sharer) and the
DECODER (code name `large`, e.g. Qwen3-0.6B, the receiver) BOTH prefill the prompt; per decoder layer a C2CProjector
(the official class, imported from the C2C repo) reads [source K/V | decoder K/V] and adds a gated residual onto the
decoder's own K/V:  K = K_dec + gate * sigmoid(w) * proj(K_src, K_dec)   (V likewise).
As in C2C's collator (generate_kv_cache_index): the prompt minus its last token is fused; the last prompt token and the
continuation are run by the decoder attending to the fused cache. Layer mapping "last_aligned": decoder layer l <-
source layer l + (L_src - L_dec). Same-tokenizer pairs only (no C2C token alignment here).

Two ways to get a fuser:
  * C2CFuser.new(...)            : fresh, trained by scripts/train_stage2.py --init c2c (our KL-to-the-8B objective; E3)
  * C2CFuser.from_rosetta_dir(d) : a checkpoint written by C2C's own script/train/SFT_train.py (official recipe; E2)
Both are saved/loaded in this repo's format (Projector.load dispatches on kind == "c2c").
"""
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from . import config as C

C2C_DIR = os.environ.get("PREFILL_C2C_DIR", os.path.join(C.SCRATCH_ROOT, "ext", "C2C"))
if os.path.isdir(C2C_DIR) and C2C_DIR not in sys.path:
    sys.path.insert(0, C2C_DIR)  # the official clone wins over any other installed `rosetta`
from rosetta.model.projector import C2CProjector  # noqa: E402

DEFAULT_HP = dict(hidden_dim=1024, intermediate_dim=1024, num_layers=3, dropout=0.1, initial_temperature=1.0,
                  final_temperature=0.001, anneal_steps=1929)  # recipe/train_recipe/C2C_0.6+0.5.json


def kv_dims(model):
    a = model.model.layers[0].self_attn
    H = model.config.num_key_value_heads
    return a.k_proj.out_features // H, H, model.config.num_hidden_layers


class C2CFuser(nn.Module):
    is_c2c = True

    def __init__(self, src_dim, src_heads, src_layers, dst_dim, dst_heads, dst_layers, hp=None):
        super().__init__()
        hp = {**DEFAULT_HP, **(hp or {})}
        self.cfg = dict(src_dim=src_dim, src_heads=src_heads, src_layers=src_layers, dst_dim=dst_dim, dst_heads=dst_heads,
                        dst_layers=dst_layers, hp=hp)
        self.offset = src_layers - dst_layers  # last_aligned (K = 1)
        assert self.offset >= 0, "last_aligned mapping needs a source at least as deep as the decoder"
        self.projs = nn.ModuleList([C2CProjector(source_dim=src_dim, target_dim=dst_dim, source_num_heads=src_heads,
                                                 target_num_heads=dst_heads, dtype=torch.float32, **hp)
                                    for _ in range(dst_layers)])

    @classmethod
    def new(cls, src_model, dst_model, **hp):
        sd, sh, sl = kv_dims(src_model)
        dd, dh, dl = kv_dims(dst_model)
        return cls(sd, sh, sl, dd, dh, dl, hp)

    def src_layer(self, l):
        return l + self.offset

    def update_temperature(self, step):
        for p in self.projs:
            p.update_temperature(step)

    def gate_logits(self):
        return [(p.key_gate_logit.item(), p.value_gate_logit.item()) for p in self.projs]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra):
        torch.save({"kind": "c2c", "state_dict": self.state_dict(), "cfg": self.cfg, **extra}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        if os.path.isdir(path):
            return cls.from_rosetta_dir(path, device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**ck["cfg"])
        m.load_state_dict(ck["state_dict"])
        return m.to(device), ck

    @classmethod
    def from_rosetta_dir(cls, d, device="cpu"):
        """A C2C SFT_train.py checkpoint dir (projector_<i>.pt per decoder layer + projector_config.json + config.json)."""
        cfg = None  # the training config: in the dir itself (HF release) or its parent (SFT_train.py output_dir/final)
        for c in (os.path.join(d, "config.json"), os.path.join(os.path.dirname(os.path.abspath(d)), "config.json")):
            if os.path.exists(c):
                j = json.load(open(c))
                if "model" in j and "projector" in j.get("model", {}):
                    cfg = j
                    break
        assert cfg is not None, f"no C2C training config.json in {d} or its parent"
        hp = {k: v for k, v in cfg["model"]["projector"]["params"].items() if k in DEFAULT_HP}
        assert cfg["model"]["projector"]["type"] == "C2CProjector" and cfg["model"].get("mapping") == "last_aligned"
        pc = json.load(open(os.path.join(d, "projector_config.json")))
        # {target_model_idx: {source_model_idx: {target_layer: [[source_layer, projector_idx]]}}}
        pairs = {int(t): v for t, v in pc["0"]["1"].items()}
        n_dst = len(pairs)
        offs = {int(v[0][0]) - t for t, v in pairs.items()}
        assert len(offs) == 1 and all(int(v[0][1]) == t for t, v in pairs.items()), f"unexpected mapping {pairs}"
        sd0 = torch.load(os.path.join(d, "projector_0.pt"), map_location="cpu")
        dst_heads = sd0["key_scalar_head.weight"].shape[0]
        out_dim = sd0["key_proj_out.weight"].shape[0]
        in_dim = sd0["key_in.weight"].shape[1] - out_dim
        src_layers = n_dst + offs.pop()
        # source K/V width = heads * head_dim; split from the source model's config
        src_cfg = json.load(open(os.path.join(cfg["model"]["teacher_model"], "config.json")))
        src_heads = src_cfg["num_key_value_heads"]
        m = cls(in_dim // src_heads, src_heads, src_layers, out_dim // dst_heads, dst_heads, n_dst, hp)
        for i, p in enumerate(m.projs):
            p.load_state_dict(torch.load(os.path.join(d, f"projector_{i}.pt"), map_location="cpu"))
        return m.to(device), {"step": cfg.get("_step"), "rosetta_dir": d}


def _kv(cache, l):
    return cache.key_cache[l], cache.value_cache[l]


def fused_prefix_cache(dec, src, fuser, prefix_ids):
    """prefix_ids [B, n] (the prompt minus its last token). Decoder and source prefill it without grad; returns a
    DynamicCache of the decoder whose every layer holds the fused K/V (grad flows into `fuser`)."""
    with torch.no_grad():
        c_dec = DynamicCache()
        dec(input_ids=prefix_ids, past_key_values=c_dec, use_cache=True, logits_to_keep=1)
        c_src = DynamicCache()
        src.model(input_ids=prefix_ids, past_key_values=c_src, use_cache=True)
    out = DynamicCache()
    dev = prefix_ids.device
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
        for l, proj in enumerate(fuser.projs):
            ks, vs = _kv(c_src, fuser.src_layer(l))
            kd, vd = _kv(c_dec, l)
            if torch.is_grad_enabled() and fuser.training:  # keep one layer's fuser activations at a time (8k prompts)
                k, v = checkpoint(proj, (ks.to(kd.device), vs.to(kd.device)), (kd, vd), use_reentrant=False)
            else:
                k, v = proj((ks.to(kd.device), vs.to(kd.device)), (kd, vd))
            out.update(k.to(kd.dtype), v.to(vd.dtype), l)
    return out


def c2c_student_forward(dec, src, fuser, ids, prompt_len):
    """Decoder logits [B, m, V] predicting ids[:, prompt_len:] when ids[:, :prompt_len-1] is C2C-fused."""
    m = ids.shape[1] - prompt_len
    if prompt_len < 2:
        out = dec(input_ids=ids, use_cache=False, logits_to_keep=m + 1)
        return out.logits[:, :-1]
    cache = fused_prefix_cache(dec, src, fuser, ids[:, : prompt_len - 1])
    out = dec(input_ids=ids[:, prompt_len - 1:], past_key_values=cache, use_cache=True, logits_to_keep=m + 1)
    return out.logits[:, :-1]


@torch.no_grad()
def c2c_prefill(dec, src, fuser, ids):
    """ids [T] (one prompt). -> (DynamicCache of the whole prompt, last-token logits [1, V])."""
    ids = ids[None]
    if ids.shape[1] < 2:
        cache = DynamicCache()
        out = dec(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return cache, out.logits[:, -1]
    cache = fused_prefix_cache(dec, src, fuser, ids[:, :-1])
    out = dec(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache, out.logits[:, -1]


def cache_to_bufs(cache):
    """DynamicCache (B = 1) -> (kbuf, vbuf) [L, Hkv, T, D] as returned by the fast paths."""
    return torch.stack([k[0] for k in cache.key_cache]), torch.stack([v[0] for v in cache.value_cache])
