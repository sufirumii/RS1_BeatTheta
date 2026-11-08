"""Step 2 - The network: a small conditional 1-D U-Net (about 5.2M parameters with default widths).

Layer names are identical to hf/modeling_rs1_beattheta.py so weights are interchangeable
(checked by tests/test_smoke.py).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], 1)


class Block(nn.Module):
    """Residual block whose scale and shift come from the condition (FiLM)."""

    def __init__(self, cin, cout, cdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 5, padding=2)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 5, padding=2)
        self.film = nn.Linear(cdim, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(cond).chunk(2, 1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        return self.c2(F.silu(self.n2(h))) + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n, self.a = nn.GroupNorm(8, ch), nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        h = self.n(x).transpose(1, 2)
        return x + self.a(h, h, h, need_weights=False)[0].transpose(1, 2)


class UNet1D(nn.Module):
    def __init__(self, widths=(48, 96, 128, 192, 256), n_classes=5, cdim=128):
        super().__init__()
        L = len(widths)
        self.L = L
        self.tproj = nn.Sequential(nn.Linear(64, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.cemb = nn.Embedding(n_classes + 1, cdim)      # last index = "no label" (classifier-free guidance)
        self.inc = nn.Conv1d(1, widths[0], 7, padding=3)
        self.down = nn.ModuleList([Block(w, w, cdim) for w in widths])
        self.samp = nn.ModuleList([nn.Conv1d(widths[i], widths[i + 1], 4, stride=4) for i in range(L - 1)])
        self.mid1, self.attn, self.mid2 = Block(widths[-1], widths[-1], cdim), Attention(widths[-1]), Block(widths[-1], widths[-1], cdim)
        self.upsamp = nn.ModuleList([nn.ConvTranspose1d(widths[i + 1], widths[i], 4, stride=4) for i in range(L - 1)])
        self.up = nn.ModuleList([Block(widths[i] * 2, widths[i], cdim) for i in range(L - 1)])
        self.outn, self.outc = nn.GroupNorm(8, widths[0]), nn.Conv1d(widths[0], 1, 7, padding=3)

    def forward(self, x, t, y):
        cond = self.tproj(timestep_embedding(t)) + self.cemb(y)
        h, skips = self.inc(x), []
        for i in range(self.L - 1):
            h = self.down[i](h, cond)
            skips.append(h)
            h = self.samp[i](h)
        h = self.down[self.L - 1](h, cond)
        h = self.mid2(self.attn(self.mid1(h, cond)), cond)
        for i in reversed(range(self.L - 1)):
            h = self.up[i](torch.cat([self.upsamp[i](h), skips[i]], 1), cond)
        return self.outc(F.silu(self.outn(h)))
