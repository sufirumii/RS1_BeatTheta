#!/usr/bin/env python3
"""Simple Gradio interface for trying the model.

    python app.py --model outputs/rs1_beattheta
    python app.py --model <user>/RS1_BeatTheta --share
"""
import argparse
import time

import gradio as gr
import matplotlib
import numpy as np
import torch
from transformers import AutoModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_demo(model):
    textures = list(model.config.textures)

    def make(texture, seconds, use_beat, beat_hz, carrier_hz, steps, guidance, seed):
        t0 = time.time()
        audio = model.generate(texture, seconds=float(seconds), steps=int(steps), guidance=float(guidance),
                               seed=None if seed < 0 else int(seed),
                               beat_hz=float(beat_hz) if use_beat else None, carrier_hz=float(carrier_hz))
        mono = audio if audio.ndim == 1 else audio[:, 0]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.specgram(mono, NFFT=1024, Fs=model.sample_rate, noverlap=768, cmap="magma")
        ax.set_xlabel("seconds"); ax.set_ylabel("Hz"); ax.set_title(texture)
        plt.tight_layout()
        info = (f"{len(mono) / model.sample_rate:.1f} s generated in {time.time() - t0:.1f} s | "
                f"peak {np.abs(audio).max():.2f} | rms {np.sqrt((audio ** 2).mean()):.3f} | {model.sample_rate} Hz")
        return (model.sample_rate, audio), fig, info

    with gr.Blocks(title="RS1_BeatTheta") as demo:
        gr.Markdown("# RS1_BeatTheta tester\nChoose a texture, optionally add a binaural beat, then press Generate. Use stereo headphones for the beat.")
        with gr.Row():
            with gr.Column():
                texture = gr.Dropdown(textures, value=textures[0], label="Texture")
                seconds = gr.Slider(4, 60, value=20, step=1, label="Length (seconds)")
                use_beat = gr.Checkbox(value=True, label="Add binaural beat (stereo)")
                beat_hz = gr.Slider(1, 30, value=6, step=0.5, label="Beat (Hz), theta is 4 to 8")
                carrier_hz = gr.Slider(100, 400, value=200, step=10, label="Carrier tone (Hz)")
                steps = gr.Slider(10, 100, value=50, step=5, label="Sampling steps")
                guidance = gr.Slider(0, 4, value=2.0, step=0.5, label="Guidance")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                go = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_audio = gr.Audio(label="Result", type="numpy")
                out_plot = gr.Plot(label="Spectrogram")
                out_info = gr.Textbox(label="Info", interactive=False)
        go.click(make, [texture, seconds, use_beat, beat_hz, carrier_hz, steps, guidance, seed],
                 [out_audio, out_plot, out_info], api_name="make")
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/rs1_beattheta", help="local folder or Hugging Face repo id")
    p.add_argument("--share", action="store_true")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True).to(dev).eval()
    build_demo(model).launch(share=a.share)


if __name__ == "__main__":
    main()
