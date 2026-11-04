#!/usr/bin/env python
"""CPU test of the long-context benchmark pieces: mcq letters for 2-5 options (real Qwen3 / Llama tokenizers if
present), cloze parsing / scoring of scripts/eval_bench.py, OOM context halving, middle truncation of
scripts/prepare_bench.Builder, and YaRN config construction (prefill/models.yarn_config) with a tiny Qwen3 model."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from eval_bench import halve_context, parse_cloze, score_item
from prefill import config as C
from prefill.fast import FastQwen3
from prefill.mcq import choose_letter, letter_token_ids
from prepare_bench import Builder

# 1) cloze parsing and scoring
p = parse_cloze('Sure. {"<mask-0>": "James Bond", "<mask-1>": "Washington DC"}')
assert p == {0: "James Bond", 1: "Washington DC"}, p
p = parse_cloze("<mask-0>: James Bond\n<mask-1>: Washington")
assert p == {0: "James Bond", 1: "Washington"}, p
assert abs(score_item("f1_cloze", '{"<mask-0>": "James Bond", "<mask-1>": "Paris"}', ["James Bond", "Washington DC"]) - 0.5) < 1e-9
assert score_item("f1", "Maria Bello\nmore text", ["Maria Bello"]) == 1.0
print("[ok] cloze / f1 scoring")

# 2) OOM fallback keeps the prefix and the question block
ids = np.arange(100)
h = halve_context(ids, 10, 20)
assert h[:10] == list(range(10)) and h[-20:] == list(range(80, 100)) and len(h) == 10 + 2 * (70 // 4) + 20, h
print("[ok] halve_context")

# 3) letters with real tokenizers, 4) middle truncation of Builder
for d in (C.LARGE_DIR, os.path.join(C.MODELS_DIR, "Llama-3.1-8B-Instruct")):
    if not os.path.exists(d):
        print(f"[skip] {d} not present")
        continue
    tok = AutoTokenizer.from_pretrained(d)
    for n in (2, 3, 4, 5):
        lid = letter_token_ids(tok, n)
        assert len(lid) == n and all(lid), lid
        assert len({i for ids in lid for i in ids}) == sum(len(x) for x in lid), "letters share a first token"
        V = len(tok)
        for g in range(n):
            lg = torch.zeros(1, V)
            lg[0, lid[g][0]] = 5.0
            assert choose_letter(lg, lid) == g
    B = Builder(tok, 300)
    long_ctx = " ".join(f"w{i}" for i in range(2000))
    it = B.item("t", "x", "pre ", "k", long_ctx, " post Q?", "f1", answers=["a"])
    assert len(it["ids"]) <= 300 and it["truncated"] and it["n_ctx"] > 300, (len(it["ids"]), it["n_ctx"])
    assert it["ids"][:it["pre_len"]] == B.T.enc(B.head + "pre ") and it["ids"][-it["q_len"]:] == B.T.enc(" post Q?" + B.tail)
    short = B.item("t", "y", "pre ", "k2", "short context", " post Q?", "f1", answers=["a"])
    assert not short["truncated"] and tok.decode(short["ids"]).count("short context") == 1
    print(f"[ok] letters 2-5 and truncation with {os.path.basename(d)}")

# 5) YaRN: a tiny Qwen3 model with a 64-position window scaled x4 -> FastQwen3 table of 256 positions, attention scaling > 1
from transformers import AutoConfig  # noqa: E402
import tempfile  # noqa: E402
from prefill.models import load_frozen  # noqa: E402
cfg = Qwen3Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=16, max_position_embeddings=64, rope_theta=1e4)
with tempfile.TemporaryDirectory() as td:
    Qwen3ForCausalLM(cfg).save_pretrained(td)
    m = load_frozen(td, "cpu", dtype=torch.float32, yarn=4)
    assert m.config.max_position_embeddings == 256 and m.config.rope_scaling["rope_type"] == "yarn"
    F = FastQwen3(m)
    assert F.cos.shape[0] == 256 and m.model.rotary_emb.attention_scaling > 1.0
    lg, _ = F.prefill(torch.randint(0, 100, (200,)))
    assert lg.shape == (1, 100) and torch.isfinite(lg).all()
    ref = m(torch.randint(0, 100, (1, 200))).logits
    assert torch.isfinite(ref).all()
print("[ok] yarn config: 64 -> 256 positions, fast path runs at 200 tokens")
print("all ok")
