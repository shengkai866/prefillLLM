"""
Fast inference path (same math as prefill/kv.py, much less overhead). Works with already-trained projectors.

  FastQwen3          lean Qwen3 forward on plain tensors: fused QKV and gate/up weights (shared storage with the
                     HF model, no extra memory), no Cache objects, KV written straight into a preallocated buffer.
  FoldedProjectorKV  exact algebraic folding of  projector tail (Linear d_mid->d_l, per-layer low-rank, bias)
                     -> input_layernorm -> k_proj / v_proj  into ONE batched matmul over all large layers.
                     The RMSNorm scale needs ||x||; it is computed exactly from a QR factor of the folded map.
  FastHybrid         small prefill -> folded projector -> sink + tail in a SINGLE large-model call
                     (sink queries can only see sink keys under the ordinary causal mask, so they can share the
                     call with the tail tokens once the projected middle KV is already in the buffer).
  FastBig            large-model-only prefill on the same lean code (fair baseline).
  Graphed            CUDA-graph capture/replay of any of the above for a fixed prompt length.

Batch size 1 throughout (latency setting).
"""
import time

import torch
import torch.nn.functional as F

from .kv import split_prompt
from .projector import rms_unit


def rms(x, w, eps):
    """Identical to Qwen3RMSNorm."""
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def rope(x, cos, sin):
    d = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., d:], x[..., :d]), -1) * sin


class FastQwen3:
    def __init__(self, hf):
        cfg = hf.config
        m = hf.model
        self.eps = cfg.rms_norm_eps
        self.H, self.Hkv, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.rep = self.H // self.Hkv
        self.n_layers = cfg.num_hidden_layers
        self.d = cfg.hidden_size
        self.scaling = self.D ** -0.5
        self.embed = m.embed_tokens.weight
        self.norm = m.norm.weight
        self.lm_head = hf.lm_head.weight
        self.layers = []
        with torch.no_grad():
            for L in m.layers:
                a, mlp = L.self_attn, L.mlp
                assert a.q_proj.bias is None and a.k_proj.bias is None and a.v_proj.bias is None
                nq, nk = a.q_proj.weight.shape[0], a.k_proj.weight.shape[0]
                wqkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous()
                ng = mlp.gate_proj.weight.shape[0]
                wgu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
                # re-point the HF modules at views of the fused tensors -> no duplicated weights
                a.q_proj.weight = torch.nn.Parameter(wqkv[:nq], requires_grad=False)
                a.k_proj.weight = torch.nn.Parameter(wqkv[nq:nq + nk], requires_grad=False)
                a.v_proj.weight = torch.nn.Parameter(wqkv[nq + nk:], requires_grad=False)
                mlp.gate_proj.weight = torch.nn.Parameter(wgu[:ng], requires_grad=False)
                mlp.up_proj.weight = torch.nn.Parameter(wgu[ng:], requires_grad=False)
                self.layers.append(dict(ln1=L.input_layernorm.weight, wqkv=wqkv, qn=a.q_norm.weight, kn=a.k_norm.weight,
                                        wo=a.o_proj.weight, ln2=L.post_attention_layernorm.weight, wgu=wgu,
                                        wd=mlp.down_proj.weight))
            # RoPE table, computed exactly like Qwen3RotaryEmbedding (fp32, then cast)
            re = m.rotary_emb
            pos = torch.arange(cfg.max_position_embeddings, device=self.embed.device, dtype=torch.float32)
            freqs = pos[:, None] * re.inv_freq.float().to(pos.device)[None, :]
            emb = torch.cat((freqs, freqs), -1)
            self.cos = (emb.cos() * re.attention_scaling).to(self.embed.dtype)
            self.sin = (emb.sin() * re.attention_scaling).to(self.embed.dtype)

    def _qkv(self, lw, x, cos, sin):
        T = x.shape[0]
        q, k, v = F.linear(x, lw["wqkv"]).split([self.H * self.D, self.Hkv * self.D, self.Hkv * self.D], -1)
        q = rms(q.view(T, self.H, self.D), lw["qn"], self.eps).transpose(0, 1)
        k = rms(k.view(T, self.Hkv, self.D), lw["kn"], self.eps).transpose(0, 1)
        v = v.view(T, self.Hkv, self.D).transpose(0, 1)
        return rope(q, cos, sin), rope(k, cos, sin), v

    def _mlp(self, lw, h):
        g, u = F.linear(rms(h, lw["ln2"], self.eps), lw["wgu"]).chunk(2, -1)
        return h + F.linear(F.silu(g) * u, lw["wd"])

    def _out(self, lw, h, o):  # o [H, T, D]
        T = o.shape[1]
        return h + F.linear(o.transpose(0, 1).reshape(T, self.H * self.D), lw["wo"])

    def prefill(self, ids, kbuf=None, vbuf=None, collect=False, logits=True, pos=None, last_q=None):
        """Causal prefill of ids [T] from position 0, or at sorted absolute positions pos [T] (tokens not in pos are
        simply absent, as in SpecPrefill). Optionally writes K/V into kbuf/vbuf [L,Hkv,>=T,D] and appends the last
        token's query [H,1,D] of every layer to the list last_q.
        Returns (last-token logits [1,V] or None, stacked hidden states [L+1,T,d] like HF output_hidden_states or None)."""
        T = ids.shape[0]
        cos, sin = (self.cos[:T], self.sin[:T]) if pos is None else (self.cos[pos], self.sin[pos])
        h = F.embedding(ids, self.embed)
        hs = [h] if collect else None
        for i, lw in enumerate(self.layers):
            q, k, v = self._qkv(lw, rms(h, lw["ln1"], self.eps), cos, sin)
            if kbuf is not None:
                kbuf[i, :, :T] = k
                vbuf[i, :, :T] = v
            if last_q is not None:
                last_q.append(q[:, -1:])
            o = F.scaled_dot_product_attention(q[None], k.repeat_interleave(self.rep, 0)[None],
                                               v.repeat_interleave(self.rep, 0)[None], is_causal=True,
                                               scale=self.scaling)[0]
            h = self._mlp(lw, self._out(lw, h, o))
            if collect:
                hs.append(h)
        out_logits = None
        if collect:
            hs[-1] = rms(h, self.norm, self.eps)  # HF appends the *normed* final state
        if logits:
            out_logits = F.linear(rms(h[-1:], self.norm, self.eps), self.lm_head)
        return out_logits, (torch.stack(hs, 0) if collect else None)

    def extend(self, ids, cos, sin, kbuf, vbuf, segs, mask):
        """
        Run new tokens ids [Q] (RoPE tables cos/sin [Q,D] already gathered for their positions) against a KV buffer.
        segs: [(src_start, dst_start, n)] where the new K/V rows are written into the buffer before attention.
        mask: additive [rep*Q, T] mask over buffer positions (from causal_mask_for(..., rep)). GQA is handled by
        folding the query groups into the query-length dimension, so the KV buffer is never copied/repeated.
        """
        Q = ids.shape[0]
        amask = mask[None, None]
        h = F.embedding(ids, self.embed)
        for i, lw in enumerate(self.layers):
            q, k, v = self._qkv(lw, rms(h, lw["ln1"], self.eps), cos, sin)
            for a, b, n in segs:
                kbuf[i, :, b:b + n] = k[:, a:a + n]
                vbuf[i, :, b:b + n] = v[:, a:a + n]
            qg = q.reshape(self.Hkv, self.rep * Q, self.D)
            o = F.scaled_dot_product_attention(qg[None], kbuf[i][None], vbuf[i][None], attn_mask=amask,
                                               scale=self.scaling)[0]
            h = self._mlp(lw, self._out(lw, h, o.reshape(self.H, Q, self.D)))
        return F.linear(rms(h[-1:], self.norm, self.eps), self.lm_head)


def causal_mask_for(positions, T, dtype, device, rep=1):
    """Additive mask [rep*Q, T]: query at position p sees buffer keys <= p. Row r*Q+q belongs to query q of
    head-group member r. Row stride padded to a multiple of 16 for the mem-efficient SDPA kernel."""
    Tp = (T + 15) // 16 * 16
    keys = torch.arange(Tp, device=device)
    m = torch.zeros(positions.shape[0], Tp, dtype=dtype, device=device)
    m.masked_fill_(keys[None, :] > positions[:, None], float("-inf"))
    return m.repeat(rep, 1)[:, :T]


class FoldedProjectorKV:
    """
    Exact refactoring of  projector(hs) -> hidden_to_kv(large, .)  for a trained prefill.projector.Projector.

    projector:  a_l = silu(W1 · sum_s w_ls h_s + b1)                       (d_mid)
                x_l = (W2 a_l + b2)(I + D_l U_l) + bias_l = a_l M_l + c_l     (d_l)
    large:      k_l = k_norm(RMSNorm(x_l) Wk_l), v_l = RMSNorm(x_l) Wv_l, RMSNorm(x) = ln ⊙ x / rms(x)
    folded:     [a_l P_k + q_k | a_l P_v + q_v | a_l F + f]  in one bmm, with P_k = M_l diag(ln) Wk^T etc.,
                and ||x_l|| = ||[a_l,1] R^T||  where  [M_l; c_l]^T = Q R  (QR, so the norm is a plain sum of squares).
    Cost per token and layer: d_mid x (2*Hkv*D + d_mid) instead of d_mid x d_l + 2 d_l x rank + d_l x 2*Hkv*D.
    """

    def __init__(self, projector, large_fast: FastQwen3, chunk=4096):
        p = projector
        lf = large_fast
        dev, dt = lf.embed.device, lf.embed.dtype
        self.dt, self.chunk, self.eps = dt, chunk, lf.eps
        self.Hkv, self.D, self.d_l = lf.Hkv, lf.D, lf.d
        self.KV = lf.Hkv * lf.D
        f64 = lambda t: t.detach().to(device=dev, dtype=torch.float64)
        with torch.no_grad():
            self.in_norm = bool(p.cfg.get("in_norm"))
            self.mix = p.layer_mix.detach().float().softmax(-1).to(dev).to(dt)  # [Ll, Ls]
            self.W1 = p.trunk[0].weight.detach().to(dev, dt)
            self.b1 = p.trunk[0].bias.detach().to(dev, dt)
            W2T, b2 = f64(p.trunk[2].weight).T, f64(p.trunk[2].bias)  # [d_mid, d_l], [d_l]
            d_mid = W2T.shape[0]
            ncol = 2 * self.KV + d_mid + 1
            ncol_pad = (ncol + 127) // 128 * 128  # zero columns add nothing to the sum of squares
            Wc, bc = [], []
            for l, lw in enumerate(lf.layers):
                Dl, Ul = f64(p.down[l]), f64(p.up[l])
                M = W2T + (W2T @ Dl) @ Ul
                c = b2 + (b2 @ Dl) @ Ul + f64(p.bias[l].reshape(-1))
                wk = f64(lw["wqkv"][lf.H * lf.D: lf.H * lf.D + self.KV])
                wv = f64(lw["wqkv"][lf.H * lf.D + self.KV:])
                ln = f64(lw["ln1"])
                Wk_, Wv_ = ln[:, None] * wk.T, ln[:, None] * wv.T  # [d_l, KV]
                R = torch.linalg.qr(torch.cat([M, c[None]], 0).T, mode="r")[1]  # [d_mid+1, d_mid+1]
                Fm = R.T
                W = torch.zeros(d_mid, ncol_pad, dtype=torch.float64, device=dev)
                b = torch.zeros(ncol_pad, dtype=torch.float64, device=dev)
                W[:, :self.KV], b[:self.KV] = M @ Wk_, c @ Wk_
                W[:, self.KV:2 * self.KV], b[self.KV:2 * self.KV] = M @ Wv_, c @ Wv_
                W[:, 2 * self.KV:ncol], b[2 * self.KV:ncol] = Fm[:d_mid], Fm[d_mid]
                Wc.append(W.to(dt))
                bc.append(b.to(dt))
            self.Wcat = torch.stack(Wc)  # [Ll, d_mid, ncol_pad]
            # token chunk: keep the per-chunk [Ll, chunk, ncol_pad] output (and its fp32 slices) small for deep/wide
            # large models (32B: 64 layers) whose weights already fill most of the GPU; 8B keeps 4096
            if len(lf.layers) * ncol_pad * self.chunk * 2 > 1.25 * 2 ** 30:
                self.chunk = min(self.chunk, 1024)
            self.bcat = torch.stack(bc)[:, None, :]  # [Ll, 1, ncol_pad]
            self.kn = torch.stack([lw["kn"] for lw in lf.layers])[:, None, None, :]  # [Ll,1,1,D]

    def __call__(self, hs, cos, sin, kdst, vdst, n_layers=None):
        """hs [Ls, n, d_s] small-model hidden states of the middle tokens; cos/sin [n, D]; writes kdst/vdst [Ll,Hkv,n,D].
        n_layers: only the first n_layers large layers (kdst/vdst then have n_layers rows)."""
        n = hs.shape[1]
        Ll = self.Wcat.shape[0] if n_layers is None else n_layers
        for c0 in range(0, n, self.chunk):
            c1 = min(n, c0 + self.chunk)
            nc = c1 - c0
            hc = rms_unit(hs[:, c0:c1]) if self.in_norm else hs[:, c0:c1]
            H1 = F.linear(hc, self.W1)  # [Ls, nc, d_mid]; mixing commutes with W1 (rows of mix sum to 1)
            a = F.silu(torch.matmul(self.mix[:Ll], H1.reshape(H1.shape[0], -1)).view(Ll, nc, -1) + self.b1)
            Y = torch.baddbmm(self.bcat[:Ll], a, self.Wcat[:Ll])  # [Ll, nc, ncol_pad]
            inv = torch.rsqrt(Y[..., 2 * self.KV:].float().pow(2).sum(-1, keepdim=True) / self.d_l + self.eps)
            k = (Y[..., :self.KV].float() * inv).to(self.dt).view(Ll, nc, self.Hkv, self.D)
            k = rms(k, self.kn[:Ll], self.eps).transpose(1, 2)
            v = (Y[..., self.KV:2 * self.KV].float() * inv).to(self.dt).view(Ll, nc, self.Hkv, self.D).transpose(1, 2)
            kdst[:, :, c0:c1] = rope(k, cos[c0:c1], sin[c0:c1])
            vdst[:, :, c0:c1] = v


class FastBig:
    def __init__(self, large_fast: FastQwen3):
        self.L = large_fast

    def prefill(self, ids):
        L = self.L
        T = ids.shape[0]
        kbuf = torch.empty(L.n_layers, L.Hkv, T, L.D, dtype=L.embed.dtype, device=ids.device)
        vbuf = torch.empty_like(kbuf)
        logits, _ = L.prefill(ids, kbuf, vbuf)
        return kbuf, vbuf, logits


class FastHybrid:
    def __init__(self, small_fast: FastQwen3, large_fast: FastQwen3, folded: FoldedProjectorKV, n_sink=8, n_tail=64):
        self.S, self.L, self.fold = small_fast, large_fast, folded
        self.n_sink, self.n_tail = n_sink, n_tail

    # --- stages (separately callable so each can be timed / graphed on its own)
    def stage_small(self, ids_prefix):
        return self.S.prefill(ids_prefix, collect=True, logits=False)[1]

    def stage_fold(self, hs_mid, s, T):
        L = self.L
        m = hs_mid.shape[1]
        kbuf = torch.empty(L.n_layers, L.Hkv, T, L.D, dtype=L.embed.dtype, device=hs_mid.device)
        vbuf = torch.empty_like(kbuf)
        self.fold(hs_mid, L.cos[s:s + m], L.sin[s:s + m], kbuf[:, :, s:s + m], vbuf[:, :, s:s + m])
        return kbuf, vbuf

    def stage_tail(self, ids, kbuf, vbuf, s, m):
        L = self.L
        T = ids.shape[0]
        sel = lambda x: torch.cat([x[:s], x[s + m:]])
        pos = sel(torch.arange(T, device=ids.device))
        mask = causal_mask_for(pos, T, L.embed.dtype, ids.device, L.rep)
        return L.extend(sel(ids), sel(L.cos[:T]), sel(L.sin[:T]), kbuf, vbuf, [(0, 0, s), (s, s + m, T - s - m)], mask)

    def prefill(self, ids, patch=None):
        """ids [T]. Returns (kbuf, vbuf, first-token logits). `patch(kbuf, vbuf, s, m)` may edit the middle KV
        before the tail runs (used for diagnostics)."""
        T = ids.shape[0]
        s, m, _ = split_prompt(T, self.n_sink, self.n_tail)
        if m > 0:
            hs = self.stage_small(ids[:s + m])
            kbuf, vbuf = self.stage_fold(hs[:, s:], s, T)
        else:
            kbuf = torch.empty(self.L.n_layers, self.L.Hkv, T, self.L.D, dtype=self.L.embed.dtype, device=ids.device)
            vbuf = torch.empty_like(kbuf)
        if patch is not None:
            patch(kbuf, vbuf, s, m)
        logits = self.stage_tail(ids, kbuf, vbuf, s, m)
        return kbuf, vbuf, logits


class Graphed:
    """Capture fn(*inputs) as a CUDA graph for the given static shapes; call with same-shaped inputs to replay.
    Outputs are views into graph-owned memory and are overwritten by the next replay."""

    def __init__(self, fn, *example_inputs, warmup=2):
        self.static_in = [x.clone() for x in example_inputs]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn(*self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(*self.static_in)

    def __call__(self, *inputs):
        for dst, x in zip(self.static_in, inputs):
            if dst.data_ptr() != x.data_ptr():
                dst.copy_(x)
        self.graph.replay()
        return self.out


def time_call(fn, repeats=5, warmup=2):
    """Median wall time (s) of fn() with a device sync before and after each call."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def kv_to_dynamic_cache(kbuf, vbuf):
    from transformers import DynamicCache
    cache = DynamicCache()
    for l in range(kbuf.shape[0]):
        cache.update(kbuf[l][None].contiguous(), vbuf[l][None].contiguous(), l)
    return cache
