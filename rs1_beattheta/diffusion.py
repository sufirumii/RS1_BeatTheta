"""Step 3a - Diffusion maths: noise schedule and the training target.

The model predicts the velocity v (v-prediction) instead of the noise, which is more stable for audio.
"""
import math
import torch

T_STEPS = 1000


def alpha_bar(t_steps=T_STEPS, device="cpu"):
    """Cumulative signal level at every diffusion step (cosine schedule)."""
    s = 0.008
    t = torch.linspace(0, 1, t_steps + 1, dtype=torch.float64)
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float().to(device)


def noisy_batch(x0, t, ac):
    """Mix clean audio x0 with noise at step t. Returns (noisy input, v target)."""
    noise = torch.randn_like(x0)
    ab = ac[t][:, None, None]
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    v = ab.sqrt() * noise - (1 - ab).sqrt() * x0
    return xt, v
