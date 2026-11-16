"""Projector: small-model hidden states (all layers) -> predicted residual stream of every large-model layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_unit(x, eps=1e-6):
    """x / rms(x) over the last dim, computed in fp32, returned in x's dtype."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


class Projector(nn.Module):
    """
    Maps small-model hidden states [Ls, B, T, d_s] to predicted large-model residual streams [Ll, B, T, d_l].
    Ls counts all small-model hidden states returned by HF (embeddings + each layer output).
    Ll is the number of large-model decoder layers; output l is the *input* residual stream of layer l.
    Token-wise only (no attention), as specified in projectTarget.md.
    in_norm: RMS-normalise every small-model hidden state per token (no weight) before mixing. Small-model residual
    streams grow by orders of magnitude with depth and model size (1.7B starts at ~10x the error of 0.6B), which
    this removes. Checkpoints without the key load with in_norm=False.
    """

    def __init__(self, Ls, Ll, d_s, d_l, d_mid=2048, rank=256, in_norm=False):
        super().__init__()
        self.cfg = dict(Ls=Ls, Ll=Ll, d_s=d_s, d_l=d_l, d_mid=d_mid, rank=rank, in_norm=in_norm)
        # 1. learnable soft mixing of small-model layers, initialised towards depth-proportional alignment
        mix = torch.zeros(Ll, Ls)
        for l in range(Ll):
            s = round(l * (Ls - 1) / max(Ll - 1, 1))
            mix[l, s] = 3.0
        self.layer_mix = nn.Parameter(mix)
        # 2. shared trunk small space -> large space
        self.trunk = nn.Sequential(nn.Linear(d_s, d_mid), nn.SiLU(), nn.Linear(d_mid, d_l))
        # 3. per-layer low-rank correction (+ per-layer bias) ; `up` zero-init so the trunk alone acts at init
        self.down = nn.Parameter(torch.randn(Ll, d_l, rank) * 0.02)
        self.up = nn.Parameter(torch.zeros(Ll, rank, d_l))
        self.bias = nn.Parameter(torch.zeros(Ll, 1, 1, d_l))

    def forward(self, hs_small):  # [Ls, B, T, d_s]
        if self.cfg.get("in_norm"):
            hs_small = rms_unit(hs_small)
        hs_small = hs_small.to(self.trunk[0].weight.dtype)  # fp32 params; autocast (if active) picks bf16 matmuls
        w = self.layer_mix.softmax(-1)  # [Ll, Ls]
        mixed = torch.einsum("ls,sbtd->lbtd", w.to(hs_small.dtype), hs_small)
        base = self.trunk(mixed)  # [Ll, B, T, d_l]
        corr = torch.einsum("lbtd,ldr->lbtr", base, self.down.to(base.dtype))
        corr = torch.einsum("lbtr,lre->lbte", corr, self.up.to(base.dtype))
        return base + corr + self.bias.to(base.dtype)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra):
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg, **extra}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Also loads a C2C fuser (prefill/c2c.py): a checkpoint with kind == "c2c", or a C2C SFT_train.py directory."""
        import os
        if os.path.isdir(path):
            from .c2c import C2CFuser
            return C2CFuser.load(path, device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("kind") == "c2c":
            from .c2c import C2CFuser
            return C2CFuser.load(path, device)
        cfg = dict(ckpt["cfg"])
        m = ProjectorTokMoE(**{k: v for k, v in cfg.items() if k != "kind"}) if cfg.get("kind") == "tmoe" else cls(**cfg)
        m.load_state_dict(ckpt["state_dict"])
        return m.to(device), ckpt


class ProjectorTokMoE(nn.Module):
    """
    Token-routed mixture-of-experts projector (branch tok-moe). Projector's single shared trunk becomes n_experts
    independent trunks (W1_e, W2_e); a gate picks top_k of them for every (large layer, token) cell:
      u_l    = sum_s w_ls h_s                                      layer mix, as in Projector
      g_l    = softmax over the top_k entries of  u_l Wg_l^T + bg_l   (weights renormalised over the chosen experts)
      base_l = sum_e g_le (silu(u_l W1_e^T + b1_e) W2_e^T + b2_e)
      x_l    = base_l (I + D_l U_l) + bias_l                        per-layer low-rank correction, as in Projector
    The gate sees the token (through u_l) and the layer (own Wg_l and bias bg_l), so the experts can split the work by
    token type, by layer, or both. band > 0 initialises bg so that layer l prefers the experts next to
    l * (E - 1) / (Ll - 1): neighbouring layers then start on the same experts ("a few layers per expert"); 0 = no prior.
    Only the chosen experts run (rows grouped by expert, as prefill/moe.py), so the cost is ~top_k trunks, not E.
    Routing is deterministic and identical in train and eval mode.
    from_projector(): every expert starts as a copy of the trained trunk, so the output is the Projector's whatever the
    gate does (the weights of a cell sum to 1); the random gate then sends different cells to different experts, which
    is what makes them diverge in training.
    After a forward in train mode: self.aux = load-balance loss E * sum_e f_e P_e - 1 (f_e share of the routed slots,
    P_e mean gate probability, over all layers and tokens; 0 when both are uniform), self.stats = usage numbers.
    Inference: fast.TokMoEKV (no algebraic fold: the RMSNorm scale of a mixture is not a mixture of the experts').
    """

    def __init__(self, Ls, Ll, d_s, d_l, d_mid=2048, rank=256, in_norm=False, n_experts=8, top_k=2, band=0.0, gate_std=0.02):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.cfg = dict(kind="tmoe", Ls=Ls, Ll=Ll, d_s=d_s, d_l=d_l, d_mid=d_mid, rank=rank, in_norm=in_norm,
                        n_experts=n_experts, top_k=top_k, band=band, gate_std=gate_std)
        E = n_experts
        mix = torch.zeros(Ll, Ls)
        for l in range(Ll):
            mix[l, round(l * (Ls - 1) / max(Ll - 1, 1))] = 3.0
        self.layer_mix = nn.Parameter(mix)
        trunks = [nn.Sequential(nn.Linear(d_s, d_mid), nn.SiLU(), nn.Linear(d_mid, d_l)) for _ in range(E)]  # Projector's init
        self.W1 = nn.Parameter(torch.stack([t[0].weight.data for t in trunks]))  # [E, d_mid, d_s]
        self.b1 = nn.Parameter(torch.stack([t[0].bias.data for t in trunks]))    # [E, d_mid]
        self.W2 = nn.Parameter(torch.stack([t[2].weight.data for t in trunks]))  # [E, d_l, d_mid]
        self.b2 = nn.Parameter(torch.stack([t[2].bias.data for t in trunks]))    # [E, d_l]
        self.gate_w = nn.Parameter(torch.randn(Ll, E, d_s) * gate_std)
        centre = torch.arange(Ll, dtype=torch.float32)[:, None] * (E - 1) / max(Ll - 1, 1)
        self.gate_b = nn.Parameter(-band * (torch.arange(E, dtype=torch.float32)[None] - centre).abs())  # [Ll, E]
        self.down = nn.Parameter(torch.randn(Ll, d_l, rank) * 0.02)
        self.up = nn.Parameter(torch.zeros(Ll, rank, d_l))
        self.bias = nn.Parameter(torch.zeros(Ll, 1, 1, d_l))
        self.aux, self.stats = None, {}

    @classmethod
    @torch.no_grad()
    def from_projector(cls, p, n_experts=8, top_k=2, band=0.0, gate_std=0.02):
        """Exact conversion of a trained Projector (same outputs): its trunk is copied into every expert."""
        c = p.cfg
        m = cls(c["Ls"], c["Ll"], c["d_s"], c["d_l"], d_mid=c["d_mid"], rank=c["rank"], in_norm=c.get("in_norm", False),
                n_experts=n_experts, top_k=top_k, band=band, gate_std=gate_std)
        m = m.to(device=p.layer_mix.device, dtype=p.layer_mix.dtype)
        m.W1.copy_(p.trunk[0].weight[None].expand_as(m.W1))
        m.b1.copy_(p.trunk[0].bias[None].expand_as(m.b1))
        m.W2.copy_(p.trunk[2].weight[None].expand_as(m.W2))
        m.b2.copy_(p.trunk[2].bias[None].expand_as(m.b2))
        for name in ("layer_mix", "down", "up", "bias"):
            getattr(m, name).copy_(getattr(p, name))
        return m

    def route(self, mixed):
        """mixed [Ll, B, T, d_s] -> (chosen experts [Ll, B, T, k], their weights (fp32, sum 1), gate logits [Ll, B, T, E] fp32)."""
        logits = torch.einsum("lbtd,led->lbte", mixed, self.gate_w.to(mixed.dtype)).float() + self.gate_b[:, None, None].float()
        top, sel = torch.topk(logits, self.cfg["top_k"], dim=-1)
        return sel, top.softmax(-1), logits

    def forward(self, hs_small):  # [Ls, B, T, d_s] -> [Ll, B, T, d_l]
        if self.cfg.get("in_norm"):
            hs_small = rms_unit(hs_small)
        hs_small = hs_small.to(self.W1.dtype)
        E, k = self.cfg["n_experts"], self.cfg["top_k"]
        mixed = torch.einsum("ls,sbtd->lbtd", self.layer_mix.softmax(-1).to(hs_small.dtype), hs_small)
        Ll, B, T, d_s = mixed.shape
        sel, w, logits = self.route(mixed)
        x = mixed.reshape(-1, d_s)                      # one row per (layer, batch, token) cell
        flat = sel.reshape(-1)                          # expert of every (cell, slot)
        order = torch.argsort(flat, stable=True)
        row, wt = order // k, w.reshape(-1)[order]
        counts = torch.bincount(flat, minlength=E)
        base, start = None, 0
        for e, n in enumerate(counts.tolist()):         # the one host sync
            if n == 0:
                continue
            r = row[start:start + n]
            y = F.linear(F.silu(F.linear(x[r], self.W1[e], self.b1[e])), self.W2[e], self.b2[e])
            if base is None:
                base = y.new_zeros(x.shape[0], y.shape[-1])
            base.index_add_(0, r, y * wt[start:start + n, None].to(y.dtype))
            start += n
        base = base.view(Ll, B, T, -1)
        if self.training:
            f = counts.float() / counts.sum()
            P = logits.softmax(-1).mean((0, 1, 2))
            self.aux = E * (f.detach() * P).sum() - 1.0
            share = torch.zeros(Ll, E, device=sel.device).scatter_add_(1, sel.reshape(Ll, -1), torch.ones(1, device=sel.device).expand(Ll, sel[0].numel()))
            top_layer = (share / share.sum(1, keepdim=True)).max(1).values.mean()  # 1/E: no layer preference; ~1/k: one fixed set per layer
            ent = -(f * (f + 1e-9).log()).sum() / torch.log(torch.tensor(float(E)))
            self.stats = {"gate_ent": ent.item(), "gate_max": f.max().item(), "gate_min": f.min().item(),
                          "gate_layer_top": top_layer.item(), "gate_w1": w[..., 0].mean().item()}
        corr = torch.einsum("lbtd,ldr->lbtr", base, self.down.to(base.dtype))
        corr = torch.einsum("lbtr,lre->lbte", corr, self.up.to(base.dtype))
        return base + corr + self.bias.to(base.dtype)

    num_params = Projector.num_params
    save = Projector.save
