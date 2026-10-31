"""Model loading helpers (frozen, bf16, SDPA attention)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LARGE_DIR, LARGE_MAXMEM, SMALL_DIR
from .projector import Projector


def load_frozen(path, device, dtype=torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa")
    m.eval().requires_grad_(False)
    return m.to(device)


def load_split(path, maxmem, dtype=torch.bfloat16):
    """Frozen model split layer-wise over GPUs 0..n-1 with per-GPU caps, e.g. maxmem='24GiB,76GiB'."""
    caps = {i: c.strip() for i, c in enumerate(maxmem.split(","))}
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa", device_map="auto", max_memory=caps)
    m.eval().requires_grad_(False)
    print(f"[models] {path} split over GPUs: {sorted(set(str(d) for d in m.hf_device_map.values()))}", flush=True)
    return m


def load_pair(device, small_dir=SMALL_DIR, large_dir=LARGE_DIR, large_maxmem=LARGE_MAXMEM):
    tok = AutoTokenizer.from_pretrained(large_dir)
    small = load_frozen(small_dir, device)
    large = load_split(large_dir, large_maxmem) if large_maxmem else load_frozen(large_dir, device)
    return tok, small, large


def new_projector(small, large, d_mid=2048, rank=256, in_norm=False):
    Ls = small.config.num_hidden_layers + 1  # embeddings + every layer output
    Ll = large.config.num_hidden_layers
    return Projector(Ls, Ll, small.config.hidden_size, large.config.hidden_size, d_mid=d_mid, rank=rank, in_norm=in_norm)


def eos_ids(tok):
    ids = {tok.eos_token_id}
    vocab = tok.get_vocab()
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"):  # Qwen3, Llama 3
        if t in vocab:
            ids.add(vocab[t])
    return tuple(i for i in ids if i is not None)
