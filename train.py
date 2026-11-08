#!/usr/bin/env python3
"""Train RS1_BeatTheta from scratch on synthetic audio.

    python train.py                       # full run (12,000 steps; about 36 min on one T4)
    python train.py --smoke               # tiny run that only checks the pipeline works
    python train.py --minutes 45          # stop after 45 minutes; run again to resume

The script is organised in four labelled steps (see README):
    STEP 1  build the synthetic dataset
    STEP 2  build the model and its EMA copy
    STEP 3  training loop (diffusion, AdamW, mixed precision, checkpoints)
    STEP 4  export a folder that loads with transformers AutoModel
"""
import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rs1_beattheta.data import TEXTURES, build_dataset
from rs1_beattheta.diffusion import T_STEPS, alpha_bar, noisy_batch
from rs1_beattheta.export import export_hf
from rs1_beattheta.model import UNet1D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/rs1_beattheta")
    p.add_argument("--n-per-class", type=int, default=600)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--widths", type=int, nargs="+", default=[48, 96, 128, 192, 256])
    p.add_argument("--minutes", type=float, default=None, help="stop after this many minutes (resume by running again)")
    p.add_argument("--ckpt-minutes", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.widths, a.n_per_class, a.batch, a.steps = [16, 32, 32, 48, 64], 6, 4, 20
    return a


def save_state(path, model, ema, opt, step, widths):
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, widths=list(widths)), tmp)
    os.replace(tmp, path)


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"

    # ---------------------------------------------------------------- STEP 1: synthetic dataset
    print("[step 1] generating synthetic dataset from code ...")
    x_all, y_all = build_dataset(a.n_per_class, a.seed)
    x_all, y_all = x_all.to(dev), y_all.to(dev)
    print(f"[step 1] {tuple(x_all.shape)} clips ({len(TEXTURES)} textures x {a.n_per_class})")

    # ---------------------------------------------------------------- STEP 2: model and EMA copy
    model = UNet1D(widths=tuple(a.widths), n_classes=len(TEXTURES)).to(dev)
    ema = UNet1D(widths=tuple(a.widths), n_classes=len(TEXTURES)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    step, state_path = 0, out / "train_state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location=dev)
        if st["widths"] == list(a.widths):
            model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"]); step = st["step"]
            print(f"[step 2] resumed from step {step}")
    if step == 0:
        ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    print(f"[step 2] model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {dev}")

    # ---------------------------------------------------------------- STEP 3: training loop
    ac = alpha_bar(T_STEPS, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0, last_ckpt, losses = time.time(), time.time(), []
    model.train()
    while step < a.steps and (a.minutes is None or time.time() - t0 < a.minutes * 60):
        idx = torch.randint(0, x_all.shape[0], (a.batch,), device=dev)
        x0, labels = x_all[idx], y_all[idx].clone()
        labels[torch.rand(a.batch, device=dev) < 0.1] = len(TEXTURES)          # drop the label 10% of the time (classifier-free guidance)
        t = torch.randint(0, T_STEPS, (a.batch,), device=dev)
        xt, v_target = noisy_batch(x0, t, ac)
        lr = a.lr * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / a.steps, 1.0))))
        for group in opt.param_groups:
            group["lr"] = lr
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            loss = F.mse_loss(model(xt, t, labels).float(), v_target)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        decay = min(0.999, (1 + step) / (10 + step))                           # exponential moving average of the weights
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
        losses.append(loss.item()); step += 1
        if step % 100 == 0 or step == 1:
            print(f"[step 3] step {step:6d}  loss {np.mean(losses[-100:]):.4f}  lr {lr:.2e}  {(time.time() - t0) / 60:.1f} min")
        if time.time() - last_ckpt > a.ckpt_minutes * 60:
            save_state(state_path, model, ema, opt, step, a.widths); last_ckpt = time.time()
    save_state(state_path, model, ema, opt, step, a.widths)

    # ---------------------------------------------------------------- STEP 4: export for transformers AutoModel
    export_hf(out, ema.state_dict(), a.widths, step)
    print(f"[step 4] exported to {out} (load with AutoModel.from_pretrained('{out}', trust_remote_code=True))")


if __name__ == "__main__":
    main()
