"""
Trains the small U-Net of Detect Voids' "Segmentation" method from the reviewed training data
(Seg Data window, sat_widgets/void_seg.py) and exports it to ONNX for the viewer (CPU,
onnxruntime). See docs/SEGMENTATION.md.

    python tools/train_seg.py                         # data from ~/TIF_Viewer_seg, -> models/void_unet.onnx
    python tools/train_seg.py --data D:/seg --epochs 60 --out models/void_unet.onnx
    python tools/train_seg.py --mode recon --out models/void_recon.onnx

Modes:
  seg    (default) the model marks void pixels. Input: chip, golden chip, difference (the viewer
         builds the same 3 channels). Labels: the reviewed masks (void / unknown band /
         background) plus synthetic voids pasted into the chips during training.
  recon  the model redraws the chip without voids (a golden chip of its own); the viewer compares
         the chip with it. Trained on chips with synthetic voids pasted in, target = the chip
         (void and unknown pixels of the masks left out of the loss).

Training recipe (from the defect-segmentation literature, e.g. DRAEM, FC-EF change detection,
"simulated reference" die-to-die inspection):
- the chip and the golden chip get independent contrast / gamma / blur / sub-pixel shift changes,
  so layout that "shows through" the difference is learnt as normal;
- synthetic faint voids (ellipse / superellipse, soft edge, bright or dark, 1-4x the chip's noise
  level) are pasted into the chip;
- loss: focal + Dice on known pixels (the band around hand markings is left out).
Requires: torch, numpy, Pillow, onnx (pip install torch onnx). Runs on CPU (minutes to hours).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sat_widgets import void_seg # noqa: E402
from sat_widgets.void_detect import _blur, _components # noqa: E402

import torch # noqa: E402
import torch.nn as nn # noqa: E402
import torch.nn.functional as F # noqa: E402


# ----- data -----
def load_data(root):
    """All included samples: [(group, m float32, y uint8, golden, spread)]."""
    out = []
    for folder, index in void_seg.list_sources(root):
        if not index.get("include", True):
            continue
        refs = {}
        for s in index["samples"]:
            if not s.get("include", True):
                continue
            layer = s["layer"]
            if layer not in refs:
                r = np.load(os.path.join(folder, f"L{layer}_ref.npz"))
                refs[layer] = (r["golden"].astype(np.float32), r["spread"].astype(np.float32))
            d = np.load(os.path.join(folder, s["id"] + ".npz"))
            group = f"{index.get('source')}|{s['row']},{s['col']}" # Chip: train / validation split unit
            out.append((group, d["m"].astype(np.float32), d["y"], *refs[layer]))
    return out


def split(samples, val_frac, seed=0):
    val = lambda g: int(hashlib.md5(f"{seed}{g}".encode()).hexdigest(), 16) % 1000 < val_frac * 1000
    return [s for s in samples if not val(s[0])], [s for s in samples if val(s[0])]


def synth_voids(m, y, rng, n_max=3):
    """Pastes faint synthetic voids into m (in place copy); marks them in y. Returns (m, y)."""
    m, y = m.copy(), y.copy()
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(rng.integers(1, n_max + 1)):
        rx, ry = rng.uniform(3, min(30, w / 4)), rng.uniform(3, min(30, h / 4))
        cx, cy = rng.uniform(rx, w - rx), rng.uniform(ry, h - ry)
        p = rng.choice([2.0, 2.0, rng.uniform(1.2, 4.0)])
        t = (np.abs((xx - cx) / rx) ** p + np.abs((yy - cy) / ry) ** p) ** (1.0 / p)
        soft = np.clip((1.25 - t) / 0.5, 0, 1)
        soft = _blur(soft.astype(np.float32), rng.uniform(0.5, 2.5))
        amp = rng.uniform(1.0, 4.0) * rng.choice([-1.0, 1.0])
        m += amp * soft
        y[(t <= void_seg.CORE) & (y != 1)] = 1
        band = (t > void_seg.CORE) & (t <= void_seg.BAND) & (y == 0)
        y[band] = 255
    return m, y


def appearance(a, rng):
    """Independent contrast / gamma-like / blur / sub-pixel shift change of one image (MAD units)."""
    a = a * rng.uniform(0.75, 1.3) + rng.uniform(-0.4, 0.4)
    if rng.random() < 0.5:
        a = np.sign(a) * np.abs(a) ** rng.uniform(0.85, 1.15)
    if rng.random() < 0.6:
        a = _blur(a.astype(np.float32), rng.uniform(0.3, 1.8))
    if rng.random() < 0.6:
        from sat_widgets.void_detect import _translate
        a = _translate(a, rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
    return a.astype(np.float32)


def make_batch(samples, n, crop, mode, rng, synth_p=0.5):
    xs, ys, ts = [], [], []
    for _ in range(n):
        _, m, y, g, sp = samples[rng.integers(len(samples))]
        clean = m
        if rng.random() < synth_p:
            m, y = synth_voids(m, y, rng)
        m_in, g_in = appearance(m, rng), appearance(g, rng)
        x = void_seg.inputs(m_in, g_in, sp)
        tgt = void_seg.inputs(clean, g, sp)[0] if mode == "recon" else None
        h, w = y.shape
        ph, pw = max(0, crop - h), max(0, crop - w)
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            y = np.pad(y, ((0, ph), (0, pw)), constant_values=255)
            if tgt is not None:
                tgt = np.pad(tgt, ((0, ph), (0, pw)), mode="reflect")
            h, w = y.shape
        # Crop: around a void half of the time
        vy, vx = np.nonzero(y == 1)
        if len(vy) and rng.random() < 0.5:
            k = rng.integers(len(vy))
            r0 = int(np.clip(vy[k] - rng.integers(crop), 0, h - crop)); c0 = int(np.clip(vx[k] - rng.integers(crop), 0, w - crop))
        else:
            r0, c0 = int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1))
        sl = (slice(r0, r0 + crop), slice(c0, c0 + crop))
        x, y = x[:, sl[0], sl[1]], y[sl]
        if tgt is not None:
            tgt = tgt[sl]
        k = rng.integers(4)
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        if tgt is not None:
            tgt = np.rot90(tgt, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
            if tgt is not None:
                tgt = tgt[:, ::-1]
        xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
        if tgt is not None:
            ts.append(np.ascontiguousarray(tgt))
    out = [torch.from_numpy(np.stack(xs)).float(), torch.from_numpy(np.stack(ys).astype(np.int64))]
    if mode == "recon":
        out.append(torch.from_numpy(np.stack(ts)).float())
    return out


# ----- model -----
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Small U-Net: 3 poolings (inputs padded to a multiple of 8), base channels -> about 0.5-2 M parameters."""

    def __init__(self, cin=3, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3, self.e4 = block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.d3 = block(c[2] * 2, c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.d2 = block(c[1] * 2, c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.d1 = block(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def seg_loss(logits, y, gamma=2.0):
    """Focal + soft Dice on known pixels (y 0 / 1; 255 = unknown, left out)."""
    known = (y != 255).float()
    t = (y == 1).float()
    lg = logits[:, 0]
    bce = F.binary_cross_entropy_with_logits(lg, t, reduction="none")
    pt = torch.exp(-bce)
    focal = ((1 - pt) ** gamma * bce * known).sum() / known.sum().clamp(min=1)
    p = torch.sigmoid(lg) * known
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * known).sum() + 1)
    return focal * 10 + dice


def recon_loss(out, tgt, y):
    """L1 + gradient L1 on background pixels (voids and unknown left out)."""
    w = (y == 0).float()
    o = out[:, 0]
    l1 = ((o - tgt).abs() * w).sum() / w.sum().clamp(min=1)
    gx = lambda a: a[:, :, 1:] - a[:, :, :-1]
    gy = lambda a: a[:, 1:, :] - a[:, :-1, :]
    wx, wy = w[:, :, 1:] * w[:, :, :-1], w[:, 1:, :] * w[:, :-1, :]
    gl = ((gx(o) - gx(tgt)).abs() * wx).sum() / wx.sum().clamp(min=1) + ((gy(o) - gy(tgt)).abs() * wy).sum() / wy.sum().clamp(min=1)
    return l1 + gl


# ----- validation -----
def evaluate(model, samples, mode, thr=0.5):
    """seg: pixel F1 and void recall / precision (blobs); recon: mean background L1."""
    model.eval()
    tp = fp = fn = 0
    found = total = det = det_ok = 0
    l1 = []
    with torch.no_grad():
        for _, m, y, g, sp in samples:
            x = void_seg.inputs(m, g, sp)
            h, w = y.shape
            ph, pw = (-h) % void_seg.PAD, (-w) % void_seg.PAD
            xt = torch.from_numpy(np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")[None]).float()
            out = model(xt)[0, 0, :h, :w].numpy()
            if mode == "recon":
                bg = y == 0
                l1.append(float(np.abs(out - x[0])[bg].mean()) if bg.any() else 0.0)
                continue
            pred = 1 / (1 + np.exp(-out)) > thr
            known = y != 255
            t = y == 1
            tp += int((pred & t & known).sum()); fp += int((pred & ~t & known).sum()); fn += int((~pred & t).sum())
            comps_t = _components(t) if t.any() else []
            comps_p = _components(pred) if pred.any() else []
            total += len(comps_t)
            for r0, r1, c0, c1, _ in comps_t:
                if pred[r0:r1 + 1, c0:c1].any():
                    found += 1
            for r0, r1, c0, c1, _ in comps_p:
                det += 1
                if (y[r0:r1 + 1, c0:c1] != 0).any():
                    det_ok += 1
    model.train()
    if mode == "recon":
        return {"l1": float(np.mean(l1)) if l1 else 0.0}
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"pixel_f1": f1, "void_recall": found / max(total, 1), "void_precision": det_ok / max(det, 1),
            "voids": total}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=void_seg.DATA_DIR)
    ap.add_argument("--mode", choices=["seg", "recon"], default="seg")
    ap.add_argument("--out", default=None, help="ONNX file (default models/void_unet.onnx / models/void_recon.onnx)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=100, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--base", type=int, default=16, help="U-Net base channels")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val", type=float, default=0.15, help="fraction of chips held out")
    ap.add_argument("--synth", type=float, default=0.5, help="probability of pasting synthetic voids")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = a.out or os.path.join(repo, "models", "void_recon.onnx" if a.mode == "recon" else "void_unet.onnx")

    samples = load_data(a.data)
    if not samples:
        sys.exit(f"No included training samples in {a.data} (Seg Data window in the viewer)")
    train, val = split(samples, a.val, a.seed)
    n_void = sum(int((s[2] == 1).any()) for s in samples)
    print(f"{len(samples)} samples ({n_void} with voids): {len(train)} train, {len(val)} validation chips' samples")

    model = UNet(3, a.base)
    print(f"U-Net base {a.base}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps)
    best, best_state = None, None
    t0 = time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for _ in range(a.steps):
            batch = make_batch(train, a.batch, a.crop, a.mode, rng, a.synth)
            outp = model(batch[0])
            loss = recon_loss(outp, batch[2], batch[1]) if a.mode == "recon" else seg_loss(outp, batch[1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        res = evaluate(model, val or train[:50], a.mode)
        score = -res["l1"] if a.mode == "recon" else res["void_recall"] * 0.5 + res["pixel_f1"] * 0.5
        if best is None or score > best[0]:
            best = (score, ep, res)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / a.steps:.4f} val {json.dumps({k: round(v, 3) for k, v in res.items()})} "
              f"{time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dummy = torch.zeros(1, 3, a.crop, a.crop)
    kw = dict(input_names=["input"], output_names=["output"], opset_version=17,
              dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}})
    try:
        torch.onnx.export(model, dummy, out, dynamo=False, **kw) # Newer torch: the classic exporter
    except TypeError:
        torch.onnx.export(model, dummy, out, **kw) # Older torch (no dynamo argument)
    info = {"mode": a.mode, "inputs": "chip, golden, (chip - golden) / spread; clip +-10, / 5 (void_seg.inputs)",
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "samples": len(samples), "samples_with_voids": n_void,
            "base": a.base, "epochs": a.epochs, "best_epoch": best[1] + 1, "validation": best[2],
            "suggested_threshold": 5}
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(f"best epoch {best[1] + 1}: {best[2]}\nSaved {out} (+ .json)")


if __name__ == "__main__":
    main()
