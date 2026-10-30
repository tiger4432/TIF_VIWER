"""Shared pieces of method B (whole-stack U-Net): features, model, void pasting, post-processing."""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NL, R, S = 8, 14, 140   # layers, chips per side, chip size at factor 2
FACTOR = 2
PAD = 2                 # 140 -> 144 (divisible by 8)


def load_wafer(outdir, name):
    X = np.load(os.path.join(outdir, name + "_chips2.npy"))          # (NL, R, R, S, S)
    tp = os.path.join(outdir, name + "_truth.json")
    truth = {}
    if os.path.exists(tp):
        for k, v in json.load(open(tp)).items():
            r, c = map(int, k.split(","))
            truth[(r, c)] = v
    return X, truth


def norm_log(stack):
    L = np.log(np.maximum(stack, 1.0))
    med = np.median(L.reshape(L.shape[0], -1), axis=1)[:, None, None]
    mad = np.median(np.abs(L - med).reshape(L.shape[0], -1), axis=1)[:, None, None] * 1.4826 + 1e-3
    return ((L - med) / mad).astype(np.float32)


def wafer_pcs(X, K=40):
    """Structural templates of a wafer: top-K principal images over all chip-layers (normalised log).
    The layout is the same on every chip, so these span the structure (lines, bumps, pad, die edge,
    with the blur variants); voids sit at random places and do not make it into the top components."""
    D = np.stack([norm_log(X[:, r, c]) for r in range(X.shape[1]) for c in range(X.shape[2])]).reshape(-1, S * S)
    D = D - D.mean(1, keepdims=True)
    _, _, Vt = np.linalg.svd(D[::2], full_matrices=False)
    return Vt[:K].reshape(K, S, S).astype(np.float32)


_yy, _xx = np.mgrid[0:S, 0:S].astype(np.float32) / S - 0.5
POLY = [np.ones((S, S), np.float32), _xx, _yy, _xx * _yy, _xx ** 2, _yy ** 2, _xx ** 3, _yy ** 3, _xx ** 2 * _yy, _xx * _yy ** 2]


def _robust(res):
    return res / (np.median(np.abs(res)) * 1.4826 + 1e-3)


def features(stack, pcs):
    """stack (NL, S, S) raw intensities of one chip, pcs (K, S, S) wafer templates -> (3 NL + 2, S, S).
    Per layer: normalised log; residual of a fit by polynomial + wafer templates + all SHALLOWER layers
    of this chip (what is new on this layer: first-appearance cue); residual of a fit by polynomial +
    templates + all OTHER layers (what this chip's stack cannot explain); plus x, y coordinates.
    Least squares by normal equations on one shared Gram matrix (fast)."""
    Z = norm_log(stack)
    M = np.concatenate([np.stack(POLY), pcs, Z]).reshape(-1, S * S).astype(np.float64)
    G = M @ M.T
    nb = len(POLY) + len(pcs)
    pre, oth = np.zeros_like(Z), np.zeros_like(Z)
    for k in range(NL):
        y = nb + k
        for out, cols in ((pre, list(range(nb + k))), (oth, [j for j in range(nb + NL) if j != y])):
            A = G[np.ix_(cols, cols)] + 1e-6 * np.eye(len(cols))
            coef = np.linalg.solve(A, G[cols, y])
            out[k] = _robust(M[y] - coef @ M[cols]).reshape(S, S)
    return np.concatenate([Z, pre, oth, 2 * _xx[None], 2 * _yy[None]]).astype(np.float32)


def n_in(resid=True):
    return NL * 3 + 2


def _cb(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """3-level U-Net; output NL + 1 classes per pixel (0 = no void, 1 + k = void first on layer k)."""

    def __init__(self, cin, base=24):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.e4 = _cb(cin, b), _cb(b, 2 * b), _cb(2 * b, 4 * b), _cb(4 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), _cb(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), _cb(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), _cb(2 * b, b)
        self.out = nn.Conv2d(b, NL + 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def pad(x):
    return np.pad(x, [(0, 0)] * (x.ndim - 2) + [(PAD, PAD), (PAD, PAD)], mode="reflect")


def ellipse_d(cx, cy, rx, ry):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2


def paste_void(stack, rng, cx, cy, rx, ry, l0, f):
    """Multiply a void (generator's model, in factor-2 px) into stack in place; returns per-layer d maps."""
    ds = []
    for k in range(l0, NL):
        fk = 1 + rng.choice([-1, 1]) * rng.uniform(0.5, 1.3) * f
        sk = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5)))) if k > l0 else 1.0
        d = ellipse_d(cx, cy, rx * sk, ry * sk)
        soft = np.clip((1.3 - d) / 0.6, 0, 1)
        stack[k] = stack[k] * (1 + (fk - 1) * soft)
        ds.append(d)
    return ds


def label_for(lab, truth_list):
    """truth_list: [(cx, cy, rx, ry, l0)] in factor-2 px -> lab (S, S) int64: 0 bg, 1+l0 in the base
    ellipse, -1 (ignore) where a deeper, possibly larger extent could be."""
    for cx, cy, rx, ry, l0 in truth_list:
        lab[(ellipse_d(cx, cy, rx * 2.5, ry * 2.5) < 1.3) & (lab == 0)] = -1
    for cx, cy, rx, ry, l0 in truth_list:
        lab[ellipse_d(cx, cy, rx, ry) <= 1.0] = 1 + l0
    return lab


def to_boxes(prob, r, c, thr, min_area=6, peak=None):
    """prob (NL + 1, S, S) softmax of one chip -> detections (full chip px)."""
    from scipy import ndimage
    pv = 1 - prob[0]
    lab, n = ndimage.label(pv > thr)
    dets = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        if m.sum() < min_area:
            continue
        if peak is not None and pv[sl][m].max() < peak:
            continue
        k = int(np.argmax(prob[1:, sl[0], sl[1]][:, m].sum(1)))
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        dets.append({"layer": k, "row": r, "col": c, "x0": x0 * FACTOR, "y0": y0 * FACTOR,
                     "x1": x1 * FACTOR, "y1": y1 * FACTOR, "score": float(pv[sl][m].max())})
    return dets
