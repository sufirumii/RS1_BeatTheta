"""RS1_BeatTheta: compact diffusion model for ambient sound textures, with an optional exact binaural-beat layer."""
import math
import numpy as np
import scipy.signal as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_rs1_beattheta import RS1BeatThetaConfig


def _t_embed(t, dim=64):
    half = dim // 2
    fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * fr[None]
    return torch.cat([a.sin(), a.cos()], 1)


class _Block(nn.Module):
    def __init__(self, cin, cout, cdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 5, padding=2)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 5, padding=2)
        self.film = nn.Linear(cdim, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        g, b = self.film(c).chunk(2, 1)
        h = h * (1 + g[:, :, None]) + b[:, :, None]
        return self.c2(F.silu(self.n2(h))) + self.skip(x)


class _Attn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n, self.a = nn.GroupNorm(8, ch), nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        h = self.n(x).transpose(1, 2)
        return x + self.a(h, h, h, need_weights=False)[0].transpose(1, 2)


class RS1BeatTheta(PreTrainedModel):
    """Class-conditional 1-D U-Net diffusion model (v-prediction, classifier-free guidance)."""

    config_class = RS1BeatThetaConfig
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        widths, cdim = tuple(config.widths), config.cdim
        L = len(widths)
        self.L = L
        self.sample_rate = config.sr_model
        self.tproj = nn.Sequential(nn.Linear(64, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.cemb = nn.Embedding(config.n_classes + 1, cdim)       # last index = unconditional
        self.inc = nn.Conv1d(1, widths[0], 7, padding=3)
        self.down = nn.ModuleList([_Block(w, w, cdim) for w in widths])
        self.samp = nn.ModuleList([nn.Conv1d(widths[i], widths[i + 1], 4, stride=4) for i in range(L - 1)])
        self.mid1, self.attn, self.mid2 = _Block(widths[-1], widths[-1], cdim), _Attn(widths[-1]), _Block(widths[-1], widths[-1], cdim)
        self.upsamp = nn.ModuleList([nn.ConvTranspose1d(widths[i + 1], widths[i], 4, stride=4) for i in range(L - 1)])
        self.up = nn.ModuleList([_Block(widths[i] * 2, widths[i], cdim) for i in range(L - 1)])
        self.outn, self.outc = nn.GroupNorm(8, widths[0]), nn.Conv1d(widths[0], 1, 7, padding=3)
        self.post_init()

    def forward(self, x, t, y):
        c = self.tproj(_t_embed(t)) + self.cemb(y)
        h, skips = self.inc(x), []
        for i in range(self.L - 1):
            h = self.down[i](h, c)
            skips.append(h)
            h = self.samp[i](h)
        h = self.down[self.L - 1](h, c)
        h = self.mid2(self.attn(self.mid1(h, c)), c)
        for i in reversed(range(self.L - 1)):
            h = self.up[i](torch.cat([self.upsamp[i](h), skips[i]], 1), c)
        return self.outc(F.silu(self.outn(h)))

    def _alpha_bar(self, device):
        s, T = 0.008, self.config.t_steps
        t = torch.linspace(0, 1, T + 1, dtype=torch.float64)
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ac = f / f[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
        return torch.cumprod(1 - betas, 0).float().to(device)

    @torch.no_grad()
    def _sample_chunks(self, ti, n, steps, guidance, gen):
        dev = next(self.parameters()).device
        ac = self._alpha_bar(dev)
        chunk, T, ncls = self.config.chunk, self.config.t_steps, self.config.n_classes
        ts = torch.linspace(T - 1, 0, steps, device=dev).long()
        out = []
        for s in range(0, n, 16):
            b = min(16, n - s)
            x = torch.randn(b, 1, chunk, device=dev, generator=gen)
            yc = torch.full((b,), ti, device=dev, dtype=torch.long)
            yu = torch.full((b,), ncls, device=dev, dtype=torch.long)
            for i in range(steps):
                t = ts[i].expand(b)
                ab = ac[ts[i]]
                ab_prev = ac[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=dev)
                vc, vu = self(x, t, yc), self(x, t, yu)
                v = vu + guidance * (vc - vu)
                x0 = (ab.sqrt() * x - (1 - ab).sqrt() * v).clamp(-1.2, 1.2)
                eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp(min=1e-4)
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            out.append(x.clamp(-1, 1).float().cpu().numpy()[:, 0])
        return np.concatenate(out, 0)

    @torch.no_grad()
    def generate(self, texture="warm_pad", seconds=10.0, steps=50, guidance=2.0, seed=None,
                 beat_hz=None, carrier_hz=200.0, tone_level=0.12, bed_rms=0.10):
        """Return audio at `self.sample_rate`. Mono texture, or stereo with an exact binaural beat if `beat_hz` is set."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        textures = self.config.textures
        ti = textures.index(texture) if isinstance(texture, str) else int(texture)
        sr, xf, chunk = self.sample_rate, 6144, self.config.chunk
        n_out = int(seconds * sr)
        hop = chunk - xf
        count = max(1, math.ceil(n_out / hop))
        gen = torch.Generator(device=dev.type)
        gen.manual_seed(int(seed) if seed is not None else int(torch.randint(0, 2 ** 31 - 1, (1,))))
        chunks = self._sample_chunks(ti, count, steps, guidance, gen)
        ramp = np.linspace(0, np.pi / 2, xf)
        fi, fo = np.sin(ramp).astype(np.float32), np.cos(ramp).astype(np.float32)   # equal-power crossfade
        bed = np.zeros(count * hop + chunk, np.float32)
        for k, c in enumerate(chunks):
            c = c.astype(np.float32).copy()
            pos = k * hop
            if k > 0:
                bed[pos:pos + xf] *= fo
                c[:xf] *= fi
            bed[pos:pos + chunk] += c
        bed = bed[:n_out]
        if was_training:
            self.train()
        if beat_hz is None:
            bed = bed - bed.mean()
            return (bed * (bed_rms / (np.sqrt(np.mean(bed ** 2)) + 1e-9))).astype(np.float32)
        half = beat_hz / 2 + 8.0                                   # keep the bed out of the tone band
        sos = sps.butter(4, [carrier_hz - half, carrier_hz + half], "bandstop", fs=sr, output="sos")
        bed = sps.sosfilt(sos, bed)
        bed = (bed - bed.mean()) * (bed_rms / (np.sqrt(np.mean(bed ** 2)) + 1e-9))
        t = np.arange(n_out) / sr
        env = np.ones(n_out)
        fade = min(int(5 * sr), n_out // 10)
        if fade > 0:
            env[:fade] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
            env[-fade:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fade))
        left = tone_level * np.sin(2 * np.pi * (carrier_hz - beat_hz / 2) * t)    # right minus left = beat_hz exactly
        right = tone_level * np.sin(2 * np.pi * (carrier_hz + beat_hz / 2) * t)
        audio = np.stack([(bed + left) * env, (bed + right) * env], 1)
        peak = float(np.abs(audio).max())
        if peak > 0.9:
            audio *= 0.9 / peak
        return audio.astype(np.float32)
