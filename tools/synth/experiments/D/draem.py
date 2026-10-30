"""
Method D: DRAEM-style void detection on multi-layer synthetic SAT wafers.

    python tools/synth/experiments/D/draem.py train  --wafers ml_201 ml_202 ml_203 --out runs/D
    python tools/synth/experiments/D/draem.py detect --wafer ml_204 --model runs/D/model.pt --out det.json

A chip is handled as one stack of 8 layers (log intensity, per-layer median removed, 140 x 140 at
factor 2). The structure is the same on every layer, so
- R (reconstruction U-Net) redraws all 8 layers without voids from the (possibly voided) stack;
- D (discriminative U-Net) takes [stack, R(stack), stack - R(stack)] and outputs per layer a
  "present" mask and a "first" mask (void appears on this layer, not on the one above).
Training: void-free chips of the training wafers with synthetic faint voids pasted in (ellipse or
blob masks, soft edges, both polarities, appear on a layer and stay deeper with per-layer shade
and size), plus the real truth voids of the training wafers as extra positives.
Boxes come from the "first" head: connected components per layer -> box in full chip px.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))

NL, R, S, F = 8, 14, 140, 2


# ---------------------------------------------------------------- data
def load_wafer(outdir, name, cache=None):
    path = cache and os.path.join(cache, name + ".npy")
    if path and os.path.exists(path):
        X = np.load(path)
    else:
        from tools.synth.chips import load_chips
        X = load_chips(outdir, name, F)
        if path:
            os.makedirs(cache, exist_ok=True)
            np.save(path, X)
    tp = os.path.join(outdir, name + "_truth.json")
    truth = {}
    if os.path.exists(tp):
        for k, v in json.load(open(tp)).items():
            r, c = map(int, k.split(","))
            truth[(r, c)] = v
    return X, truth


def norm(stack):
    """(NL, S, S) raw -> log, per-layer median removed."""
    L = np.log(np.maximum(stack, 0) + 10.0)
    return (L - np.median(L, axis=(1, 2), keepdims=True)).astype(np.float32)


yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)


def blob_d(rng, cx, cy, rx, ry):
    """Squared normalised distance of an ellipse, optionally rotated / perlin-warped."""
    a = rng.uniform(0, np.pi) if rng.random() < 0.4 else 0.0
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    d = (u / rx) ** 2 + (v / ry) ** 2
    if rng.random() < 0.3:  # irregular (low-frequency noise on the radius)
        th = np.arctan2(v, u)
        w = sum(rng.uniform(-0.15, 0.15) * np.cos(k * th + rng.uniform(0, 6)) for k in range(2, 5))
        d = d / (1 + w) ** 2
    return d


def paste_void(rng, raw, l0=None):
    """Synthetic void in raw stack (in place). Returns (first mask, present masks)."""
    if l0 is None:
        l0 = int(rng.integers(0, NL))
    cx, cy = rng.uniform(14, S - 14), rng.uniform(14, S - 14)
    rx, ry = rng.uniform(2.5, 13), rng.uniform(2.5, 13)
    f = rng.uniform(0.04, 0.15)
    pres = np.zeros((NL, S, S), np.float32)
    first = np.zeros((NL, S, S), np.float32)
    for k in range(l0, NL):
        fk = 1 + rng.choice([-1, 1]) * rng.uniform(0.5, 1.3) * f
        sk = 1.0 if k == l0 else float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))
        d = blob_d(rng, cx, cy, rx * sk, ry * sk)
        soft = np.clip((1.3 - d) / 0.6, 0, 1)
        raw[k] *= 1 + (fk - 1) * soft
        pres[k] = d < 1.0
        if k == l0:
            first[k] = d < 1.0
    return first, pres


def real_targets(t):
    """Targets for a real truth void (half-res). Deeper sizes unknown -> ignore region."""
    cx, cy, rx, ry, l0, l1, _ = t
    cx, cy, rx, ry = cx / F, cy / F, rx / F, ry / F
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    first = np.zeros((NL, S, S), np.float32)
    pres = np.zeros((NL, S, S), np.float32)
    ign = np.zeros((NL, S, S), np.float32)  # 1 = ignore present loss
    first[l0] = d < 1.0
    pres[l0] = d < 1.0
    for k in range(l0 + 1, NL):
        ign[k] = d < 2.6 ** 2 * 1.3
    # first head: tolerate the soft rim
    return first, pres, ign, d


# ---------------------------------------------------------------- nets
def build_nets(base=24):
    import torch
    import torch.nn as nn

    def block(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(4, co), nn.SiLU(),
                             nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(4, co), nn.SiLU())

    class UNet(nn.Module):
        def __init__(self, ci, co, b):
            super().__init__()
            self.e1, self.e2, self.e3 = block(ci, b), block(b, 2 * b), block(2 * b, 4 * b)
            self.m = block(4 * b, 4 * b)
            self.d3, self.d2, self.d1 = block(8 * b, 2 * b), block(4 * b, b), block(2 * b, b)
            self.out = nn.Conv2d(b, co, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            up = lambda t, ref: nn.functional.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            m = self.m(self.pool(e3))
            d3 = self.d3(torch.cat([up(m, e3), e3], 1))
            d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
            d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
            return self.out(d1)

    class Draem(nn.Module):
        def __init__(self, use_rec=True):
            super().__init__()
            self.use_rec = use_rec
            self.rec = UNet(NL, NL, base)
            self.dis = UNet(3 * NL if use_rec else NL, 2 * NL, base)

        def forward(self, x):
            if self.use_rec:
                r = x + self.rec(x)  # residual form: R predicts the correction
                return r, self.dis(torch.cat([x, r, x - r], 1))
            return x, self.dis(x)

    return Draem


# ---------------------------------------------------------------- train
def make_pool(outdir, wafers, cache):
    clean, real = [], []
    for w in wafers:
        X, truth = load_wafer(outdir, w, cache)
        for r in range(R):
            for c in range(R):
                if (r, c) in truth:
                    real.append((X[:, r, c].copy(), truth[(r, c)]))
                else:
                    clean.append(X[:, r, c].copy())
    return clean, real


def sample(rng, clean, real, p_real, p_empty):
    """One training stack -> x (NL,S,S), clean target, first, present, ign."""
    ign = np.zeros((NL, S, S), np.float32)
    if real and rng.random() < p_real:
        raw, t = real[rng.integers(len(real))]
        raw = raw.copy()
        first, pres, ign, _ = real_targets(t)
        tgt = None
    else:
        raw = clean[rng.integers(len(clean))].copy()
        tgt = norm(raw)
        if rng.random() < p_empty:
            first = np.zeros((NL, S, S), np.float32); pres = first.copy()
        else:
            first, pres = paste_void(rng, raw)
    x = norm(raw)
    if tgt is None:
        tgt = x  # real voids: no clean target, rec loss masked below
    # augment: flips (layout is symmetric enough; both axes)
    if rng.random() < 0.5:
        x, tgt, first, pres, ign = [a[:, :, ::-1] for a in (x, tgt, first, pres, ign)]
    if rng.random() < 0.5:
        x, tgt, first, pres, ign = [a[:, ::-1, :] for a in (x, tgt, first, pres, ign)]
    return [np.ascontiguousarray(a) for a in (x, tgt, first, pres, ign)]


def seg_loss(logit, tgt, wmask, gamma=2.0, alpha=0.75):
    import torch
    p = torch.sigmoid(logit)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
    pt = p * tgt + (1 - p) * (1 - tgt)
    at = alpha * tgt + (1 - alpha) * (1 - tgt)
    focal = (at * (1 - pt) ** gamma * bce * wmask).sum() / wmask.sum().clamp(min=1) * 20
    inter = (p * tgt * wmask).sum((2, 3))
    den = ((p + tgt) * wmask).sum((2, 3))
    has = (tgt.sum((2, 3)) > 0).float()
    dice = ((1 - (2 * inter + 1) / (den + 1)) * has).sum() / has.sum().clamp(min=1)
    return focal + dice


def train(a):
    import torch
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    clean, real = make_pool(a.outdir, a.wafers, a.cache)
    if a.no_real:
        real = []
    print(f"clean chips {len(clean)}, real void chips {len(real)}", flush=True)
    model = build_nets(a.base)(use_rec=not a.no_rec)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    hist = []
    for step in range(1, a.steps + 1):
        bs = [sample(rng, clean, real, a.p_real, a.p_empty) for _ in range(a.batch)]
        x, tgt, first, pres, ign = [torch.from_numpy(np.stack(z)) for z in zip(*bs)]
        rec, out = model(x)
        lf = seg_loss(out[:, :NL], first, torch.ones_like(first))
        lp = seg_loss(out[:, NL:], pres, 1 - ign)
        loss = 2 * lf + lp
        if not a.no_rec:
            # rec loss only on synthetic samples (they have a clean target)
            synth = (ign.sum((1, 2, 3)) == 0).float()[:, None, None, None]
            # void pixels weigh a.void_w more, else R learns the identity (voids are a tiny area)
            w = (1 + a.void_w * pres) * synth
            lr_ = ((rec - tgt).abs() * w).sum() / w.sum().clamp(min=1) * 10
            loss = loss + lr_
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
        if step % 50 == 0:
            print(f"step {step} loss {np.mean(hist[-50:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "base": a.base, "use_rec": not a.no_rec}, os.path.join(a.out, "model.pt"))
    print(f"train time {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- detect
def components(mask):
    """4-connected components of a bool mask -> list of (ys, xs) arrays."""
    lab = np.zeros(mask.shape, np.int32)
    comps = []
    H, W = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n = len(comps) + 1
        stack = [(y0, x0)]; lab[y0, x0] = n; pts = []
        while stack:
            y, x = stack.pop(); pts.append((y, x))
            for yy_, xx_ in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy_ < H and 0 <= xx_ < W and mask[yy_, xx_] and not lab[yy_, xx_]:
                    lab[yy_, xx_] = n; stack.append((yy_, xx_))
        p = np.array(pts)
        comps.append((p[:, 0], p[:, 1]))
    return comps


def predict(model_path, X, threads=4, flip_tta=False):
    """X (NL,R,R,S,S) raw -> probs (R,R,2NL,S,S)."""
    import torch
    torch.set_num_threads(threads)
    ck = torch.load(model_path, weights_only=False)
    model = build_nets(ck["base"])(use_rec=ck["use_rec"])
    model.load_state_dict(ck["state"]); model.eval()
    out = np.zeros((R, R, 2 * NL, S, S), np.float32)
    with torch.no_grad():
        for r in range(R):
            xb = torch.from_numpy(np.stack([norm(X[:, r, c]) for c in range(R)]))
            p = torch.sigmoid(model(xb)[1])
            if flip_tta:
                p = p + torch.sigmoid(model(xb.flip(-1))[1]).flip(-1)
                p = p + torch.sigmoid(model(xb.flip(-2))[1]).flip(-2)
                p = p / 3
            out[r] = p.numpy()
    return out


def to_boxes(P, thr=0.5, peak=0.7, min_area=6, nms_layers=True, max_per_chip=None):
    dets = []
    for r in range(R):
        for c in range(R):
            cand = []
            for k in range(NL):
                pf = P[r, c, k]
                for ys, xs in components(pf > thr):
                    sc = float(pf[ys, xs].max())
                    if len(ys) < min_area or sc < peak:
                        continue
                    cand.append((sc, k, xs.min() * F, ys.min() * F, (xs.max() + 1) * F, (ys.max() + 1) * F))
            cand.sort(reverse=True)
            keep = []
            for cd in cand:
                if nms_layers and any(cd[2] <= q[4] and q[2] <= cd[4] and cd[3] <= q[5] and q[3] <= cd[5] for q in keep):
                    continue
                keep.append(cd)
            if max_per_chip:
                keep = keep[:max_per_chip]
            for sc, k, x0, y0, x1, y1 in keep:
                dets.append({"layer": int(k), "row": r, "col": c, "x0": float(x0), "y0": float(y0),
                             "x1": float(x1), "y1": float(y1), "score": sc})
    return dets


def to_boxes_present(P, thr=0.5, peak=0.7, min_area=6, frac=0.5, max_per_chip=None):
    """Union of the per-layer present maps -> components; first layer = earliest layer whose
    present prob inside the component reaches frac * the component's peak (and thr)."""
    dets = []
    for r in range(R):
        for c in range(R):
            pp = P[r, c, NL:]
            pm = pp.max(0)
            cand = []
            for ys, xs in components(pm > thr):
                sc = float(pm[ys, xs].max())
                if len(ys) < min_area or sc < peak:
                    continue
                per = pp[:, ys, xs].max(1)
                k = int(np.nonzero(per >= max(frac * sc, thr))[0][0])
                sel = pp[k, ys, xs] > thr
                bx, by = (xs[sel], ys[sel]) if sel.sum() >= 3 else (xs, ys)
                cand.append((sc, k, bx.min() * F, by.min() * F, (bx.max() + 1) * F, (by.max() + 1) * F))
            cand.sort(reverse=True)
            if max_per_chip:
                cand = cand[:max_per_chip]
            for sc, k, x0, y0, x1, y1 in cand:
                dets.append({"layer": k, "row": r, "col": c, "x0": float(x0), "y0": float(y0),
                             "x1": float(x1), "y1": float(y1), "score": sc})
    return dets


def detect(a):
    t0 = time.time()
    X, _ = load_wafer(a.outdir, a.wafer, a.cache)
    t1 = time.time()
    P = predict(a.model, X, a.threads, a.tta)
    t2 = time.time()
    if a.save_probs:
        np.save(a.save_probs, P.astype(np.float16))
    if a.decode == "present":
        dets = to_boxes_present(P, a.thr, a.peak, a.min_area, a.frac, max_per_chip=a.max_per_chip)
    else:
        dets = to_boxes(P, a.thr, a.peak, a.min_area, max_per_chip=a.max_per_chip)
    json.dump(dets, open(a.out, "w"))
    print(f"{len(dets)} detections; load {t1 - t0:.0f}s, net {t2 - t1:.0f}s, total {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "detect"):
        p = sp.add_parser(name)
        p.add_argument("--outdir", default="data")
        p.add_argument("--cache", default="data/D_cache")
        p.add_argument("--threads", type=int, default=4)
    t = sp.choices["train"]
    t.add_argument("--wafers", nargs="+", default=["ml_201", "ml_202", "ml_203"])
    t.add_argument("--out", default="runs/D")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--base", type=int, default=16)
    t.add_argument("--p_real", type=float, default=0.25)
    t.add_argument("--p_empty", type=float, default=0.25)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no_rec", action="store_true")
    t.add_argument("--void_w", type=float, default=30.0)
    t.add_argument("--no_real", action="store_true")
    d = sp.choices["detect"]
    d.add_argument("--wafer", default="ml_204")
    d.add_argument("--model", default="runs/D/model.pt")
    d.add_argument("--out", default="runs/D/det.json")
    d.add_argument("--thr", type=float, default=0.5)
    d.add_argument("--peak", type=float, default=0.7)
    d.add_argument("--min_area", type=int, default=6)
    d.add_argument("--max_per_chip", type=int, default=None)
    d.add_argument("--tta", action="store_true")
    d.add_argument("--decode", choices=["first", "present"], default="present")
    d.add_argument("--frac", type=float, default=0.5)
    d.add_argument("--save_probs", default=None)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else detect(a)


if __name__ == "__main__":
    main()
