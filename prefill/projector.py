"""Projector: small-model hidden states (all layers) -> predicted residual stream of every large-model layer."""
import torch
import torch.nn as nn


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
        m = cls(**ckpt["cfg"])
        m.load_state_dict(ckpt["state_dict"])
        return m.to(device), ckpt
