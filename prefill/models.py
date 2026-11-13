"""Model loading helpers (frozen, bf16, SDPA attention)."""
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import HF_CACHE_DIR, LARGE_DIR, LARGE_MAXMEM, SMALL_DIR
from .projector import Projector, ProjectorMoE, parse_spans
from .tokalign import load_or_build, same_tokenizer


def yarn_config(path, factor):
    """Config of the model at `path` with static YaRN RoPE scaling (Qwen3's recommended way past 32k: factor 4 ->
    131k positions), or None if the model has its own scaling already (Llama 3.1: 131k natively) or `factor` is falsy.
    HF derives the YaRN factor from max_position_embeddings / original_max_position_embeddings, so both are set.
    Every position is affected (low-frequency dims are interpolated, cos/sin get the mscale), so use it only for
    jobs whose prompts exceed the native window."""
    if not factor:
        return None
    cfg = AutoConfig.from_pretrained(path)
    if getattr(cfg, "rope_scaling", None):
        return None
    orig = min(cfg.max_position_embeddings, 32768)
    cfg.rope_scaling = {"rope_type": "yarn", "factor": float(factor), "original_max_position_embeddings": orig}
    cfg.max_position_embeddings = int(orig * factor)
    print(f"[models] {os.path.basename(path.rstrip('/'))}: YaRN x{factor} -> {cfg.max_position_embeddings} positions", flush=True)
    return cfg


def load_frozen(path, device, dtype=torch.bfloat16, yarn=None):
    cfg = yarn_config(path, yarn)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa", config=cfg)
    m.eval().requires_grad_(False)
    return m.to(device)


def load_split(path, maxmem, dtype=torch.bfloat16, yarn=None):
    """Frozen model split layer-wise over GPUs 0..n-1 with per-GPU caps, e.g. maxmem='24GiB,76GiB'."""
    caps = {i: c.strip() for i, c in enumerate(maxmem.split(","))}
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa", device_map="auto", max_memory=caps,
                                             config=yarn_config(path, yarn))
    m.eval().requires_grad_(False)
    print(f"[models] {path} split over GPUs: {sorted(set(str(d) for d in m.hf_device_map.values()))}", flush=True)
    return m


def load_pair(device, small_dir=SMALL_DIR, large_dir=LARGE_DIR, large_maxmem=LARGE_MAXMEM, yarn=None):
    """(large tokenizer, small model, large model). A small model with another tokenizer (Qwen <-> Llama) gets the
    C2C lookup table large id -> small id attached (prefill/tokalign.py; used by FastQwen3 and small_hidden_states).
    yarn: static YaRN factor for models without RoPE scaling of their own (Qwen3; see yarn_config)."""
    tok = AutoTokenizer.from_pretrained(large_dir)
    small = load_frozen(small_dir, device, yarn=yarn)
    large = load_split(large_dir, large_maxmem, yarn=yarn) if large_maxmem else load_frozen(large_dir, device, yarn=yarn)
    tok_small = AutoTokenizer.from_pretrained(small_dir)
    if not same_tokenizer(tok, tok_small):
        name = f"{os.path.basename(large_dir.rstrip('/'))}_to_{os.path.basename(small_dir.rstrip('/'))}"
        table = load_or_build(tok, tok_small, HF_CACHE_DIR, n_rows=large.config.vocab_size, name=name)
        small.prefill_id_map = table.to(device)
        small.prefill_small_eos = eos_ids(tok_small)
        small.prefill_bos = tok_small.bos_token_id if tok_small.bos_token_id is not None and tok.bos_token_id is None else None
        print(f"[models] cross-tokenizer pair: {name} lookup table ({len(table)} rows), small BOS {small.prefill_bos}", flush=True)
    return tok, small, large


def new_projector(small, large, d_mid=2048, rank=256, in_norm=False, moe_spans=""):
    """moe_spans: e.g. "18-26" -> ProjectorMoE with one expert for large layers 18..26 and one for the other layers."""
    Ls = small.config.num_hidden_layers + 1  # embeddings + every layer output
    Ll = large.config.num_hidden_layers
    if moe_spans:
        return ProjectorMoE(Ls, Ll, small.config.hidden_size, large.config.hidden_size, d_mid=d_mid, rank=rank, in_norm=in_norm,
                            spans=parse_spans(moe_spans))
    return Projector(Ls, Ll, small.config.hidden_size, large.config.hidden_size, d_mid=d_mid, rank=rank, in_norm=in_norm)


def eos_ids(tok):
    ids = {tok.eos_token_id}
    vocab = tok.get_vocab()
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"):  # Qwen3, Llama 3
        if t in vocab:
            ids.add(vocab[t])
    return tuple(i for i in ids if i is not None)
