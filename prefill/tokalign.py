"""
Cross-tokenizer pairs (small and large model from different families, e.g. Qwen3-1.7B -> Llama-3.1-8B): the
Cache-to-Cache token alignment (Fu et al. 2025, github.com/thu-nics/C2C, rosetta/model/aligner.py).

The large model keeps its own tokens (prompt pools, KV positions, decoding are all in its tokenizer). The small model
gets a sequence of the SAME length in which every large token is replaced by ONE token of the small vocabulary:
  * the large token's string is re-encoded with the small tokenizer; one token -> that token;
  * several tokens -> the one with the longest string ("longest" strategy, C2C's default for different tokenizers;
    the rest of the string is dropped, so e.g. Llama's 3-digit tokens keep one digit for Qwen);
  * special tokens map to the small tokenizer's token of the same role (BOS/EOS/PAD and the chat markers:
    <|im_start|> <-> <|start_header_id|>, <|im_end|> <-> <|eot_id|>, <|endoftext|> <-> <|end_of_text|>), otherwise
    their string is re-encoded like text (C2C falls back to <unk>, which Llama/Qwen do not have).
Positions are therefore 1:1 and nothing downstream changes (projector, scoring, recompute, multi-question, CUDA
graphs); the map depends only on the token id, so it is a lookup table over the large vocabulary applied as a gather
on the GPU. Both families use byte-level BPE with the GPT-2 byte alphabet, so a token whose byte string exists in the
small vocabulary maps exactly (no re-encoding needed).

A small model that expects BOS at the start (Llama) gets it at position 0 when the large tokenizer has none (Qwen):
position 0 is a sink token, always computed by the large model itself, so nothing projected is lost.
"""
import json
import os

import numpy as np
import torch
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

ROLE_PAIRS = [("<|im_start|>", "<|start_header_id|>"), ("<|im_end|>", "<|eot_id|>"), ("<|endoftext|>", "<|end_of_text|>"),
              ("<|im_end|>", "<|eom_id|>")]  # Qwen3 <-> Llama 3 chat markers (either direction)


def same_tokenizer(a, b):
    return len(a) == len(b) and a.get_vocab() == b.get_vocab()


def _role_map(tok_large, tok_small):
    """large special-token string -> small id, by role."""
    vs = tok_small.get_vocab()
    out = {}

    def put(ls, ss):
        if ls is not None and ss is not None and ss in vs and ls not in out:
            out[ls] = vs[ss]
    for attr in ("bos_token", "eos_token", "pad_token"):
        put(getattr(tok_large, attr), getattr(tok_small, attr))
    put(tok_large.bos_token, "<|endoftext|>")  # Qwen3 has no BOS: Llama's <|begin_of_text|> -> its document separator
    put(tok_large.bos_token, tok_small.eos_token)
    for a, b in ROLE_PAIRS:
        put(a, b)
        put(b, a)
    if "<|end_header_id|>" not in out:  # Llama's header end -> the newline that ends Qwen's "<|im_start|>role\n"
        nl = tok_small("\n", add_special_tokens=False)["input_ids"]
        if len(nl) == 1:
            out["<|end_header_id|>"] = nl[0]
    return out


def build_table(tok_large, tok_small, n_rows=None):
    """-> (table np.int64 [n_rows]: large id -> small id, stats dict). Rows beyond the large tokenizer get small EOS."""
    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}
    n_l = len(tok_large)
    n_rows = max(n_rows or 0, n_l)
    vocab_s = tok_small.get_vocab()
    added_l = set(tok_large.get_added_vocab())
    added_s = set(tok_small.get_added_vocab())
    role = _role_map(tok_large, tok_small)
    toks = tok_large.convert_ids_to_tokens(list(range(n_l)))
    table = np.full(n_rows, tok_small.eos_token_id, dtype=np.int64)
    how = np.zeros(n_rows, dtype=np.int8)  # 0 exact, 1 role, 2 re-encoded 1:1, 3 re-encoded longest-of-several, 4 fallback
    todo = []  # (id, text) to re-encode in one batch
    for i, t in enumerate(toks):
        if t is None:
            continue
        if t in added_l:  # special / added token: by role, else its literal string as text
            if t in role:
                table[i], how[i] = role[t], 1
            else:
                todo.append((i, t))
            continue
        if t in vocab_s and t not in added_s:
            table[i], how[i] = vocab_s[t], 0
            continue
        raw = bytes(u2b[c] for c in t)
        try:
            todo.append((i, raw.decode("utf-8")))
        except UnicodeDecodeError:  # partial multi-byte token: longest small token that is a prefix of its bytes
            best = None
            for k in range(len(t), 0, -1):
                if t[:k] in vocab_s:
                    best = vocab_s[t[:k]]
                    break
            if best is None:
                how[i] = 4
            else:
                table[i], how[i] = best, 3
    if todo:
        enc = tok_small([s for _, s in todo], add_special_tokens=False)["input_ids"]
        for (i, s), ids in zip(todo, enc):
            if not ids:
                how[i] = 4
                continue
            if len(ids) == 1:
                table[i], how[i] = ids[0], 2
                continue
            strs = [tok_small.decode([j], skip_special_tokens=False, clean_up_tokenization_spaces=False) for j in ids]
            best = max(range(len(ids)), key=lambda j: len(strs[j]))  # first of the longest, as C2C
            table[i], how[i] = ids[best], 3
    stats = {"n_large": n_l, "n_small": len(tok_small), "exact": int((how == 0).sum()), "role": int((how == 1).sum()),
             "reencoded_1to1": int((how == 2).sum()), "reencoded_longest": int((how == 3).sum()), "fallback": int((how == 4).sum())}
    return table, stats, how


def token_bytes(tok):
    """bytes of every token's string (added tokens: their literal utf-8)."""
    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}
    added = set(tok.get_added_vocab())
    out = []
    for t in tok.convert_ids_to_tokens(list(range(len(tok)))):
        if t is None:
            out.append(b"")
        elif t in added:
            out.append(t.encode("utf-8"))
        else:
            out.append(bytes(u2b.get(c, 0) for c in t))
    return out


def coverage(ids, table, bytes_l, bytes_s):
    """On a token sequence of the large model: fraction of its tokens mapped to a small token with the same bytes, and
    fraction of its bytes kept by the small sequence (bytes of the chosen small token that lie inside the large token)."""
    same = kept = total = 0
    for i in ids:
        bl, bs = bytes_l[i], bytes_s[int(table[i])]
        total += len(bl)
        if bl == bs:
            same += 1
            kept += len(bl)
        elif bs and bs in bl:
            kept += len(bs)
    return same / max(1, len(ids)), kept / max(1, total)


def load_or_build(tok_large, tok_small, cache_dir, n_rows=None, name=None):
    """Lookup table (torch.long, CPU) for this pair, cached as <cache_dir>/tokmap_<name>.npy (+ .json stats)."""
    os.makedirs(cache_dir, exist_ok=True)
    name = name or f"{len(tok_large)}_{len(tok_small)}"
    path = os.path.join(cache_dir, f"tokmap_{name}.npy")
    if os.path.exists(path):
        table = np.load(path)
        if n_rows is None or len(table) >= n_rows:
            return torch.from_numpy(table[:n_rows] if n_rows else table)
    table, stats, _ = build_table(tok_large, tok_small, n_rows)
    np.save(path, table)
    json.dump(stats, open(path[:-4] + ".json", "w"), indent=1)
    print(f"[tokalign] built {path}: {stats}", flush=True)
    return torch.from_numpy(table)
