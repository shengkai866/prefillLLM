"""
Core mechanics:
  * hidden_to_kv      : predicted residual streams -> K/V through the frozen large model's own
                        input_layernorm -> k_proj/v_proj -> k_norm -> RoPE
  * hybrid_prefill    : sink tokens (true) | middle tokens (small model + projector) | tail tokens (true)
  * student_forward   : logits of the large model on a continuation when the prompt cache is the hybrid one
  * teacher_forward   : logits of the large model with its own full prefill
  * greedy_decode     : shared decode loop used by every configuration in evaluation
"""
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def all_layers(model):
    return model.model.layers


def small_hidden_states(small, input_ids):
    """All hidden states of the small model, stacked: [Ls, B, T, d_s]. Calls the base model: no vocab logits."""
    out = small.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return torch.stack(out.hidden_states, 0)


def large_hidden_states(large, input_ids):
    """Residual-stream inputs of every large-model layer: [Ll, B, T, d_l] (excludes post-final-norm)."""
    out = large(input_ids=input_ids, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    Ll = len(all_layers(large))
    return torch.stack([h.to(input_ids.device) for h in out.hidden_states[:Ll]], 0)  # layers may sit on several GPUs


def rope_cos_sin(large, position_ids, dtype):
    dummy = torch.zeros(1, dtype=dtype, device=position_ids.device)
    cos, sin = large.model.rotary_emb(dummy, position_ids)
    return cos, sin


def layer_kv(large, layer_idx, h, cos, sin):
    """K/V of one layer from residual stream h [B, T, d_l]. Returns K, V as [B, Hkv, T, hd] (K already RoPE'd)."""
    layer = all_layers(large)[layer_idx]
    attn = layer.self_attn
    B, T, _ = h.shape
    d = layer.input_layernorm.weight.device  # multi-GPU large model: compute on this layer's GPU
    h, cos, sin = h.to(d), cos.to(d), sin.to(d)
    x = layer.input_layernorm(h.to(layer.input_layernorm.weight.dtype))
    k = attn.k_proj(x).view(B, T, -1, attn.head_dim)
    k = (attn.k_norm(k) if hasattr(attn, "k_norm") else k).transpose(1, 2)  # Llama has no q/k norm
    v = attn.v_proj(x).view(B, T, -1, attn.head_dim).transpose(1, 2)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    k = (k * c) + (rotate_half(k) * s)
    return k, v


def layer_q(large, layer_idx, h, cos, sin):
    """True queries of one layer from residual stream h [B, T, d_l] -> [B, Hq, T, hd] (RoPE'd)."""
    layer = all_layers(large)[layer_idx]
    attn = layer.self_attn
    B, T, _ = h.shape
    d = layer.input_layernorm.weight.device
    h, cos, sin = h.to(d), cos.to(d), sin.to(d)
    x = layer.input_layernorm(h.to(layer.input_layernorm.weight.dtype))
    q = attn.q_proj(x).view(B, T, -1, attn.head_dim)
    q = (attn.q_norm(q) if hasattr(attn, "q_norm") else q).transpose(1, 2)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * c) + (rotate_half(q) * s)


def hidden_to_kv(large, h_layers, position_ids) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """h_layers: [Ll, B, T, d_l] -> lists of per-layer K and V."""
    dtype = all_layers(large)[0].input_layernorm.weight.dtype
    cos, sin = rope_cos_sin(large, position_ids, dtype)
    ks, vs = [], []
    for l in range(h_layers.shape[0]):
        k, v = layer_kv(large, l, h_layers[l], cos, sin)
        ks.append(k)
        vs.append(v)
    return ks, vs


def cache_from_kv(ks, vs, base: DynamicCache = None) -> DynamicCache:
    cache = base if base is not None else DynamicCache()
    for l, (k, v) in enumerate(zip(ks, vs)):
        cache.update(k, v, l)
    return cache


def split_prompt(T, n_sink, n_tail):
    """Clamp sink/tail sizes to the prompt length. Tail is always >= 1 (we need real logits for the first token)."""
    n_tail = max(1, min(n_tail, T))
    n_sink = max(0, min(n_sink, T - n_tail))
    return n_sink, T - n_sink - n_tail, n_tail


def hybrid_prefill(large, small, projector, input_ids, n_sink=8, n_tail=64, timing=None, autocast=True):
    """
    Build the large model's KV cache for `input_ids` (B=1 or unpadded batch) using
      [true sink KV | projected middle KV | true tail KV]
    and return (cache, logits_of_last_prompt_token [B, V]).
    `timing` (dict) receives a per-stage breakdown in seconds when given.
    Grad flows into `projector` if autograd is enabled by the caller.
    """
    B, T = input_ids.shape
    n_sink, n_mid, n_tail = split_prompt(T, n_sink, n_tail)
    dev = input_ids.device
    cache = DynamicCache()
    t = {}
    t0 = time.perf_counter()
    if n_sink > 0:
        with torch.no_grad():
            large(input_ids=input_ids[:, :n_sink], past_key_values=cache, use_cache=True, logits_to_keep=1)
    _sync(dev, timing is not None); t["sink"] = time.perf_counter() - t0

    if n_mid > 0:
        t0 = time.perf_counter()
        with torch.no_grad():
            hs = small_hidden_states(small, input_ids[:, : n_sink + n_mid])[:, :, n_sink:]  # [Ls,B,n_mid,d_s]
        _sync(dev, timing is not None); t["small"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(autocast and dev.type == "cuda")):
            h_hat = projector(hs)  # [Ll,B,n_mid,d_l]
        _sync(dev, timing is not None); t["projector"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pos = torch.arange(n_sink, n_sink + n_mid, device=dev).unsqueeze(0).expand(B, -1)
        ks, vs = hidden_to_kv(large, h_hat, pos)
        cache_from_kv(ks, vs, cache)
        _sync(dev, timing is not None); t["kv"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = large(input_ids=input_ids[:, n_sink + n_mid :], past_key_values=cache, use_cache=True, logits_to_keep=1)
    _sync(dev, timing is not None); t["tail"] = time.perf_counter() - t0
    if timing is not None:
        timing.update(t)
    return cache, out.logits[:, -1]


def _sync(dev, enabled=True):
    if enabled and dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def true_prefill(model, input_ids):
    cache = DynamicCache()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache, out.logits[:, -1]


def student_forward(large, small, projector, ids, prompt_len, n_sink=8, n_tail=64, return_hidden=False):
    """
    Logits predicting continuation tokens ids[:, prompt_len:] when the prompt is prefilled by the hybrid scheme.
    Returns [B, m, V] with m = T - prompt_len (position j predicts ids[:, prompt_len + j]).
    Tail tokens and continuation are run through the large model in a single forward.
    return_hidden: also return the projected middle residual streams h_hat [Ll, B, n_mid, d_l] (None if no middle).
    """
    B, T = ids.shape
    m = T - prompt_len
    n_sink, n_mid, n_tail = split_prompt(prompt_len, n_sink, n_tail)
    dev = ids.device
    cache = DynamicCache()
    h_hat = None
    if n_sink > 0:
        with torch.no_grad():
            large(input_ids=ids[:, :n_sink], past_key_values=cache, use_cache=True, logits_to_keep=1)
    if n_mid > 0:
        with torch.no_grad():
            hs = small_hidden_states(small, ids[:, : n_sink + n_mid])[:, :, n_sink:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            h_hat = projector(hs)
        pos = torch.arange(n_sink, n_sink + n_mid, device=dev).unsqueeze(0).expand(B, -1)
        ks, vs = hidden_to_kv(large, h_hat, pos)
        cache_from_kv(ks, vs, cache)
    out = large(input_ids=ids[:, n_sink + n_mid :], past_key_values=cache, use_cache=True, logits_to_keep=m + 1)
    return (out.logits[:, :-1], h_hat) if return_hidden else out.logits[:, :-1]


@torch.no_grad()
def teacher_forward(model, ids, prompt_len):
    """Logits predicting ids[:, prompt_len:] with the model's own full prefill: [B, m, V]."""
    m = ids.shape[1] - prompt_len
    out = model(input_ids=ids, use_cache=False, logits_to_keep=m + 1)
    return out.logits[:, :-1]


@torch.no_grad()
def greedy_decode(model, cache, first_logits, max_new_tokens, eos_ids=(), timing=None, pos0=None):
    """Greedy decode from a prefilled cache. Returns list[int] of generated ids (B must be 1).
    pos0: RoPE position of the first fed-back token when it differs from the cache length (compact caches of
    SpecPrefill-style dropping keep the original positions, so decoding continues at the original prompt length)."""
    dev = first_logits.device
    t0 = time.perf_counter()
    nxt = first_logits.argmax(-1)  # [1]
    out_ids = [nxt.item()]
    for _ in range(max_new_tokens - 1):
        if out_ids[-1] in eos_ids:
            break
        extra = {} if pos0 is None else {"position_ids": torch.tensor([[pos0 + len(out_ids) - 1]], device=dev)}
        out = model(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True, logits_to_keep=1, **extra)
        nxt = out.logits[:, -1].argmax(-1)
        out_ids.append(nxt.item())
    _sync(dev)
    if timing is not None:
        timing["decode"] = time.perf_counter() - t0
        timing["n_decoded"] = len(out_ids)
    return out_ids


def kl_teacher_student(t_logits, s_logits):
    """KL(teacher || student) per position, computed in fp32. Inputs [B, m, V]. Returns [B, m]."""
    t = F.log_softmax(t_logits.float(), -1)
    s = F.log_softmax(s_logits.float(), -1)
    return (t.exp() * (t - s)).sum(-1)
