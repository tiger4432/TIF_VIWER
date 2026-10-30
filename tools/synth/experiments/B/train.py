"""Train the whole-stack U-Net.

    python train.py --pool pool.npz --out model.pt [--steps 3000]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NL, UNet, n_in  # noqa: E402


def batches(X, Y, bs, rng, flip=False):
    """Random batches from the pool with per-layer sign flips (= polarity; the layout is fixed, so no rot / flip)."""
    while True:
        idx = rng.integers(0, len(X), bs)
        x = torch.from_numpy(X[idx].astype(np.float32)); y = torch.from_numpy(Y[idx].astype(np.int64))
        if flip and rng.random() < 0.5:
            x, y = x.flip(3), y.flip(2)
        sign = torch.where(torch.rand(bs, NL) < 0.5, -1.0, 1.0)
        x[:, :3 * NL] *= sign.repeat(1, 3)[:, :, None, None]
        yield x.contiguous(), y.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool.npz")
    ap.add_argument("--out", default="model.pt"); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=12); ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--void-w", type=float, default=4.0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    P = np.load(a.pool)
    PX, PY, resid = P["X"], P["Y"], bool(P["resid"])
    dl = batches(PX, PY, a.bs, np.random.default_rng(0))
    model = UNet(n_in(resid), a.base)
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    w = torch.tensor([1.0] + [a.void_w] * NL)
    t0, run = time.time(), 0.0
    for step, (x, y) in enumerate(dl):
        if step >= a.steps:
            break
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=w, ignore_index=-1)
        # soft dice on "void present"
        valid = (y >= 0).float()
        pv = (1 - logits.softmax(1)[:, 0]) * valid
        tv = (y > 0).float()
        dice = 1 - (2 * (pv * tv).sum() + 1) / (pv.sum() + tv.sum() + 1)
        (loss + 0.5 * dice).backward()
        opt.step(); opt.zero_grad(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 100 == 0:
            print(f"step {step} ce {run:.4f} dice {dice.item():.3f} {time.time() - t0:.0f}s", flush=True)
            torch.save({"state": model.state_dict(), "resid": resid, "base": a.base}, a.out)
    torch.save({"state": model.state_dict(), "resid": resid, "base": a.base}, a.out)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
