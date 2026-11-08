#!/usr/bin/env python3
"""Generate a session and save it as a WAV file.

    python generate.py --model outputs/rs1_beattheta --texture ocean_swell --seconds 30 --beat 6 --out session.wav
    python generate.py --model <user>/RS1_BeatTheta  --texture warm_pad --seconds 20 --out texture.wav   # no beat
"""
import argparse

import soundfile as sf
import torch
from transformers import AutoModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/rs1_beattheta", help="local folder or Hugging Face repo id")
    p.add_argument("--texture", default="ocean_swell")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--beat", type=float, default=None, help="binaural beat in Hz (omit for texture only)")
    p.add_argument("--carrier", type=float, default=200.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="session.wav")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True).to(dev).eval()
    audio = model.generate(a.texture, seconds=a.seconds, steps=a.steps, guidance=a.guidance, seed=a.seed,
                           beat_hz=a.beat, carrier_hz=a.carrier)
    sf.write(a.out, audio, model.sample_rate)
    print(f"Saved {a.out}: {len(audio) / model.sample_rate:.1f} s, {'stereo' if audio.ndim == 2 else 'mono'}, {model.sample_rate} Hz")


if __name__ == "__main__":
    main()
