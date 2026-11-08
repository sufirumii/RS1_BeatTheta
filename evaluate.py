#!/usr/bin/env python3
"""Evaluate a trained model.

    python evaluate.py --model outputs/rs1_beattheta

Three checks are run and written to <model>/eval/results.json, together with listening samples:
    CHECK 1  loudness: generated clips should sit near the training level (rms about 0.2)
    CHECK 2  spectrum: log-spectral distance to fresh procedural audio, compared with the noise floor
             (two independent batches of real procedural audio) so the number can be put in context
    CHECK 3  binaural beat: measured beat of generated stereo audio against the requested beat
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf
import torch
from transformers import AutoModel

from rs1_beattheta.binaural import measure_beat
from rs1_beattheta.data import SR, TARGET_RMS, TEXTURES, reference_clips


def mean_log_spectrum(clips):
    freqs, _ = sps.welch(clips[0], fs=SR, nperseg=1024)
    power = np.mean([sps.welch(c, fs=SR, nperseg=1024)[1] for c in clips], 0)
    return freqs, 10 * np.log10(power + 1e-12)


def log_spectral_distance(clips_a, clips_b):
    f, a = mean_log_spectrum(list(clips_a))
    _, b = mean_log_spectrum(list(clips_b))
    band = (f >= 30) & (f <= 3800)
    return float(np.sqrt(np.mean((a[band] - b[band]) ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/rs1_beattheta", help="local folder or Hugging Face repo id")
    p.add_argument("--clips", type=int, default=24)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--out", default=None, help="folder for results and samples (default: <model>/eval)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True).to(dev).eval()
    out = Path(a.out) if a.out else (Path(a.model) / "eval" if Path(a.model).exists() else Path("eval"))
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=torch.device(dev).type)
    results = {"textures": {}, "beat": {}}

    print(f"{'texture':12s} {'rms':>6s} {'LSD dB':>8s} {'floor dB':>9s}  verdict")
    for ti, name in enumerate(TEXTURES):
        gen.manual_seed(100 + ti)
        clips = model._sample_chunks(ti, a.clips, a.steps, a.guidance, gen)             # CHECK 1 and 2 use the same clips
        rms = float(np.mean([np.sqrt(np.mean(c ** 2)) for c in clips]))
        lsd = log_spectral_distance(clips, reference_clips(name, a.clips, 10_000))
        floor = log_spectral_distance(reference_clips(name, a.clips, 20_000), reference_clips(name, a.clips, 30_000))
        loud_ok = 0.5 * TARGET_RMS <= rms <= 1.5 * TARGET_RMS
        spec_ok = lsd <= 2 * floor + 1.0
        verdict = "ok" if (loud_ok and spec_ok) else ("check loudness" if not loud_ok else "spectrum far from floor")
        results["textures"][name] = dict(rms=rms, lsd_db=lsd, noise_floor_db=floor, verdict=verdict)
        print(f"{name:12s} {rms:6.3f} {lsd:8.1f} {floor:9.1f}  {verdict}")
        gap = np.zeros(int(0.3 * SR), np.float32)
        sf.write(str(out / f"{name}.wav"), np.concatenate([np.concatenate([c.astype(np.float32), gap]) for c in clips[:3]]), SR)

    print(f"\n{'beat Hz':>8s} {'measured':>10s} {'error':>8s}")                       # CHECK 3
    for beat in (4.0, 6.0, 8.0):
        audio = model.generate("warm_pad", seconds=12, steps=min(a.steps, 20), seed=7, beat_hz=beat, carrier_hz=200.0)
        measured = measure_beat(audio, model.sample_rate, 200.0, beat)
        results["beat"][str(beat)] = dict(measured=measured, error=abs(measured - beat))
        print(f"{beat:8.1f} {measured:10.4f} {abs(measured - beat):8.4f}")

    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults and listening samples saved to {out}")


if __name__ == "__main__":
    main()
