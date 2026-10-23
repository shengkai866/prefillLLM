"""Stage-1 representation-alignment losses."""
import random

import torch
import torch.nn.functional as F

from .kv import layer_kv, layer_q, rope_cos_sin


def hidden_alignment_loss(h_hat, h_true, mask):
    """
    h_hat, h_true: [Ll, B, T, d]; mask: [B, T] bool (positions included in the loss).
    Per layer: (1 - cosine) + MSE normalised by that layer's mean per-token RMS (so deep layers don't dominate).
    Returns (loss, dict of per-layer cos/nmse for logging).
    """
    h_hat = h_hat.float()
    h_true = h_true.float()
    m = mask[None, :, :, None].float()
    denom = m.sum() * 1.0
    cos = F.cosine_similarity(h_hat, h_true, dim=-1)  # [Ll,B,T]
    cos_loss = ((1 - cos) * mask[None].float()).sum((1, 2)) / mask.sum().clamp(min=1)  # [Ll]
    tok_rms = h_true.pow(2).mean(-1).sqrt()  # [Ll,B,T]
    scale = ((tok_rms * mask[None].float()).sum((1, 2)) / mask.sum().clamp(min=1)).pow(2).clamp(min=1e-6)  # [Ll]
    mse = ((h_hat - h_true).pow(2) * m).mean(-1).sum((1, 2)) / mask.sum().clamp(min=1)  # [Ll]
    nmse = mse / scale
    loss = cos_loss.mean() + nmse.mean()
    return loss, {"cos": cos_loss.detach(), "nmse": nmse.detach()}


def attention_output_loss(large, h_hat, h_true, n_sink, n_layers=4, layers=None):
    """
    Auxiliary loss: with the large model's *true* queries, compare attention outputs using
    [true sink KV | projected KV] versus the fully true KV, on a random subset of layers.
    h_hat, h_true: [Ll, B, T, d_l].  Returns scalar loss (normalised MSE of attention outputs).
    """
    Ll, B, T, _ = h_true.shape
    dev = h_true.device
    if layers is None:
        layers = random.sample(range(Ll), min(n_layers, Ll))
    pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
    dtype = large.model.layers[0].input_layernorm.weight.dtype
    cos, sin = rope_cos_sin(large, pos, dtype)
    total = 0.0
    for l in layers:
        attn = large.model.layers[l].self_attn
        with torch.no_grad():
            q = layer_q(large, l, h_true[l], cos, sin)
            k_t, v_t = layer_kv(large, l, h_true[l], cos, sin)
        k_h, v_h = layer_kv(large, l, h_hat[l], cos, sin)
        if n_sink > 0:
            k_h = torch.cat([k_t[:, :, :n_sink], k_h[:, :, n_sink:]], 2)
            v_h = torch.cat([v_t[:, :, :n_sink], v_h[:, :, n_sink:]], 2)
        g = attn.num_key_value_groups
        with torch.no_grad():
            o_t = F.scaled_dot_product_attention(q, k_t.repeat_interleave(g, 1), v_t.repeat_interleave(g, 1), is_causal=True, scale=attn.scaling)
        o_h = F.scaled_dot_product_attention(q, k_h.repeat_interleave(g, 1), v_h.repeat_interleave(g, 1), is_causal=True, scale=attn.scaling)
        o_t, o_h = o_t[:, :, n_sink:].float(), o_h[:, :, n_sink:].float()
        total = total + ((o_h - o_t).pow(2).mean() / o_t.pow(2).mean().clamp(min=1e-6)).to(dev)
    return total / len(layers)
