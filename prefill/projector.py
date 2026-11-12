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

    # --- the projector as  a_l = silu(W1_g(l) mix_l(h) + b1_g(l)),  x_l = a_l M_l + c_l  (used by the folds) ---
    def first_stage(self):
        """-> (W1 [G, d_mid, d_s], b1 [G, d_mid], group index of every large layer [Ll]); G = 1 here."""
        return self.trunk[0].weight[None], self.trunk[0].bias[None], torch.zeros(self.cfg["Ll"], dtype=torch.long)

    def layer_affine(self, l, dtype=torch.float64):
        """(M_l [d_mid, d_l], c_l [d_l]) with x_l = a_l M_l + c_l, computed in `dtype`."""
        f = lambda t: t.detach().to(dtype)
        W2T, b2 = f(self.trunk[2].weight).T, f(self.trunk[2].bias)
        Dl, Ul = f(self.down[l]), f(self.up[l])
        return W2T + (W2T @ Dl) @ Ul, b2 + (b2 @ Dl) @ Ul + f(self.bias[l].reshape(-1))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra):
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg, **extra}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["cfg"])
        kind = cfg.pop("kind", None)
        m = (ProjectorWide if kind == "wide" else Projector)(**cfg)
        m.load_state_dict(ckpt["state_dict"])
        return m.to(device), ckpt


class ProjectorWide(nn.Module):
    """
    Same function class as Projector with the capacity that costs nothing (or little) at inference untied:
      a_l = silu(W1_g(l) · sum_s w_ls h_s + b1_g(l))      W1 shared by groups of large layers (w1_groups = 1: all layers)
      x_l = a_l W2_l + b2_l                                W2_l a FULL per-layer [d_mid, d_l] map (Projector: one shared
                                                           W2 times a rank-`rank` per-layer correction)
    After folding, every large layer is still one [d_mid, 2*KV + d_mid] GEMM, so untying W2 is free at inference;
    w1_groups > 1 mixes the layers before W1 instead of after it (Ll instead of Ls d_s x d_mid GEMMs per token).
    from_projector() converts a trained Projector exactly (same outputs), so training can continue from it.
    """

    def __init__(self, Ls, Ll, d_s, d_l, d_mid=2048, in_norm=False, w1_groups=1, rank=None):
        super().__init__()
        assert 1 <= w1_groups <= Ll
        self.cfg = dict(kind="wide", Ls=Ls, Ll=Ll, d_s=d_s, d_l=d_l, d_mid=d_mid, in_norm=in_norm, w1_groups=w1_groups)
        mix = torch.zeros(Ll, Ls)
        for l in range(Ll):
            mix[l, round(l * (Ls - 1) / max(Ll - 1, 1))] = 3.0
        self.layer_mix = nn.Parameter(mix)
        self.W1 = nn.Parameter(torch.randn(w1_groups, d_mid, d_s) * d_s ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(w1_groups, d_mid))
        self.W2 = nn.Parameter(torch.randn(Ll, d_mid, d_l) * d_mid ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(Ll, d_l))
        self.register_buffer("gidx", torch.arange(Ll) * w1_groups // Ll, persistent=False)

    @classmethod
    @torch.no_grad()
    def from_projector(cls, p, w1_groups=1):
        c = p.cfg
        m = cls(c["Ls"], c["Ll"], c["d_s"], c["d_l"], d_mid=c["d_mid"], in_norm=c.get("in_norm", False), w1_groups=w1_groups)
        m = m.to(device=p.layer_mix.device, dtype=p.layer_mix.dtype)
        m.layer_mix.copy_(p.layer_mix)
        m.W1.copy_(p.trunk[0].weight[None].expand_as(m.W1))
        m.b1.copy_(p.trunk[0].bias[None].expand_as(m.b1))
        for l in range(c["Ll"]):
            M, cl = p.layer_affine(l)
            m.W2[l].copy_(M.to(m.W2.dtype))
            m.b2[l].copy_(cl.to(m.b2.dtype))
        return m

    def forward(self, hs_small):  # [Ls, B, T, d_s] -> [Ll, B, T, d_l]
        if self.cfg.get("in_norm"):
            hs_small = rms_unit(hs_small)
        hs_small = hs_small.to(self.W1.dtype)
        w = self.layer_mix.softmax(-1)
        if self.cfg["w1_groups"] == 1:  # W1 is linear: apply it before mixing (Ls instead of Ll GEMMs)
            H1 = torch.einsum("sbtd,md->sbtm", hs_small, self.W1[0])
            a = torch.einsum("ls,sbtm->lbtm", w.to(H1.dtype), H1) + self.b1[0]
        else:
            mixed = torch.einsum("ls,sbtd->lbtd", w.to(hs_small.dtype), hs_small)
            a = torch.einsum("lbtd,lmd->lbtm", mixed, self.W1[self.gidx]) + self.b1[self.gidx][:, None, None]
        a = F.silu(a)
        return torch.einsum("lbtm,lme->lbte", a, self.W2.to(a.dtype)) + self.b2[:, None, None].to(a.dtype)

    def first_stage(self):
        return self.W1, self.b1, self.gidx

    def layer_affine(self, l, dtype=torch.float64):
        return self.W2[l].detach().to(dtype), self.b2[l].detach().to(dtype)

    num_params = Projector.num_params
    save = Projector.save
