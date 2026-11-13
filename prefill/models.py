"""Model loading helpers (frozen, bf16, SDPA attention)."""
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import HF_CACHE_DIR, LARGE_DIR, LARGE_MAXMEM, SMALL_DIR, SMALL_MAXMEM, TEACHER
from .projector import Projector
from .tokalign import load_or_build, same_tokenizer, shared_vocab


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
    patch_moe(m)
    return m.to(device)


def patch_moe(m):
    """Qwen3-MoE (e.g. Qwen3-30B-A3B): replace HF's per-expert loop (128 host syncs per layer) with prefill/moe.py's
    grouped version (one sync per layer, same per-expert math). No-op for dense models."""
    if getattr(m.config, "model_type", "") == "qwen3_moe":
        from .moe import patch_qwen3_moe
        patch_qwen3_moe()


def load_split(path, maxmem, dtype=torch.bfloat16, yarn=None):
    """Frozen model split layer-wise over GPUs 0..n-1 with per-GPU caps, e.g. maxmem='24GiB,76GiB'."""
    caps = {i: c.strip() for i, c in enumerate(maxmem.split(","))}
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa", device_map="auto", max_memory=caps,
                                             config=yarn_config(path, yarn))
    m.eval().requires_grad_(False)
    patch_moe(m)
    print(f"[models] {path} split over GPUs: {sorted(set(str(d) for d in m.hf_device_map.values()))}", flush=True)
    return m


def load_pair(device, small_dir=SMALL_DIR, large_dir=LARGE_DIR, large_maxmem=LARGE_MAXMEM, yarn=None, small_maxmem=SMALL_MAXMEM):
    """(large tokenizer, small model, large model). A small model with another tokenizer (Qwen <-> Llama) gets the
    C2C lookup table large id -> small id attached (prefill/tokalign.py; used by FastQwen3 and small_hidden_states).
    yarn: static YaRN factor for models without RoPE scaling of their own (Qwen3; see yarn_config)."""
    tok = AutoTokenizer.from_pretrained(large_dir)
    small = load_split(small_dir, small_maxmem, yarn=yarn) if small_maxmem else load_frozen(small_dir, device, yarn=yarn)
    large = load_split(large_dir, large_maxmem, yarn=yarn) if large_maxmem else load_frozen(large_dir, device, yarn=yarn)
    tok_small = AutoTokenizer.from_pretrained(small_dir)
    if not same_tokenizer(tok, tok_small):
        name = f"{os.path.basename(large_dir.rstrip('/'))}_to_{os.path.basename(small_dir.rstrip('/'))}"
        table = load_or_build(tok, tok_small, HF_CACHE_DIR, n_rows=large.config.vocab_size, name=name)
        small.prefill_id_map = table.to(device)
        small.prefill_small_eos = eos_ids(tok_small)
        small.prefill_bos = tok_small.bos_token_id if tok_small.bos_token_id is not None and tok.bos_token_id is None else None
        print(f"[models] cross-tokenizer pair: {name} lookup table ({len(table)} rows), small BOS {small.prefill_bos}", flush=True)
        if TEACHER == "small":  # cross-family reverse pair: KL over the tokens both vocabularies share
            li, si = shared_vocab(tok, tok_small)
            small.teacher_vocab = (torch.from_numpy(li).to(device), torch.from_numpy(si).to(device), large.config.vocab_size)
            print(f"[models] teacher = small, other vocabulary: KL over {len(li)} shared tokens "
                  f"({len(li) / len(tok):.1%} of the decoder's, {len(si) / len(tok_small):.1%} of the teacher's)", flush=True)
    return tok, small, large


def new_projector(small, large, d_mid=2048, rank=256, in_norm=False):
    Ls = small.config.num_hidden_layers + 1  # embeddings + every layer output
    Ll = large.config.num_hidden_layers
    return Projector(Ls, Ll, small.config.hidden_size, large.config.hidden_size, d_mid=d_mid, rank=rank, in_norm=in_norm)


def teacher_of(small, large):
    """The model whose outputs the projected cache is trained / scored against (config.TEACHER)."""
    return small if TEACHER == "small" else large


def eos_ids(tok):
    ids = {tok.eos_token_id}
    vocab = tok.get_vocab()
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"):  # Qwen3, Llama 3
        if t in vocab:
            ids.add(vocab[t])
    return tuple(i for i in ids if i is not None)
