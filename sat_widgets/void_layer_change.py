"""
Layer-change void detection (Detect Voids method "layer_change"; no learning).

Every layer of a scan shows the same chip structures, each with its own brightness, contrast and
polarity per structure, and a void appears on some layer and stays on every deeper one. So each
layer of a chip is explained from what is known to be void-free there, and the void is what the
explanation leaves:

1. The layers of each chip are moved onto its first layer (sub-pixel; gradient magnitude, so
   contrast and polarity do not matter).
2. Layer k of a chip = robust least squares (outliers clipped, so the void does not bend the fit) of
   - the chip's own layers above k (value, square, x / y derivative, blur difference; the nearest
     lc_sources): void-free when k is the void's first layer;
   - the golden chips (median over the chips) of the nearest lc_goldens layers (value, blur difference);
   each with 2 x 2 bilinear weights over the chip (the fit varies smoothly across the chip), plus
   lc_regions layout regions (k-means of the golden layer vectors: pad, die, ...) with their own
   offset / coefficients and a quadratic brightness term. Layer 0 has no layer above: goldens only.
3. The remainder is divided by its chip's robust spread and by its chip-to-chip spread per pixel.
4. Centre-surround filter (Gaussian s minus 2 s) at lc_scales (working px); the evidence of a layer
   is the best |score| over the scales, in units of its robust spread.
5. Place: root mean square of the evidence over the searched layers (scaled like the root sum of
   squares / 2 of 8 layers) above lc_threshold, local maxima, weaker overlapping peaks dropped.
6. Layer ("layer_rule"): "first" = change point of the evidence at the place,
   argmax_l sum_{k >= l} (e_k - lc_tlo) (CUSUM: the void stays on the deeper layers);
   "strongest" = largest evidence; "last" = deepest layer whose evidence reaches lc_tlo.
   The size is the best scale there (radius 1.6 x sigma).

Tuned on synthetic 8-layer wafers (tools/synth/experiments/C on branch exp/C-layer-change;
ml_204: 35 of 50 voids on their first layer, 14 false detections; die-to-die: 0 and 443).
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DEFAULTS = {
    "lc_threshold": 10.0, # Place: evidence over the layers (see 5.)
    "lc_tlo": 3.0, # Evidence of a void-free layer (first layer: CUSUM level)
    "lc_scales": [2.0, 3.0, 4.5, 6.5], # Centre-surround sigmas (working px)
    "lc_sources": 7, # At most this many layers above as sources (nearest first)
    "lc_goldens": 8, # Golden chips of the nearest this many layers as sources
    "lc_regions": 4, # Layout regions with their own coefficients (k-means of the goldens)
    "lc_border": 10, # Working px at the chip border not searched
    "lc_register": True, # Layers of each chip moved onto its first layer (sub-pixel)
    "lc_workers": 0, # Chips fitted in parallel (0 = CPU count)
}

_GM = {}


def _gauss_matrix(n, s):
    """(n, n) Gaussian blur matrix, edges extended (scipy mode "nearest", truncate 4)."""
    key = (n, round(float(s), 4))
    g = _GM.get(key)
    if g is None:
        r = int(4.0 * s + 0.5)
        t = np.arange(-r, r + 1)
        k = np.exp(-0.5 * (t / s) ** 2)
        k /= k.sum()
        i = np.arange(n)[:, None]
        g = np.zeros((n, n))
        np.add.at(g, (np.broadcast_to(i, (n, t.size)), np.clip(i + t[None], 0, n - 1)), np.broadcast_to(k, (n, t.size)))
        g = _GM[key] = g.astype(np.float32)
    return g


def _gauss(a, s):
    """Gaussian blur over the last two axes (edges extended)."""
    h, w = a.shape[-2:]
    return _gauss_matrix(h, s) @ np.asarray(a, np.float32) @ _gauss_matrix(w, s).T


def _max_filter(a, k=7):
    """Maximum over k x k (last two axes); outside is 0 (a >= 0)."""
    r = k // 2
    pad = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(r, r), (r, r)])
    v = np.lib.stride_tricks.sliding_window_view(pad, k, axis=-2).max(-1)
    return np.lib.stride_tricks.sliding_window_view(v, k, axis=-1).max(-1)


def _gradmag(a):
    gy, gx = np.gradient(_gauss(a, 1.0))
    g = np.hypot(gx, gy)
    return g - g.mean()


def register_layers(chip, maxs=2, b=6):
    """(NL, h, w): each layer moved onto the first (gradient magnitude correlation, parabolic peak)."""
    from .void_detect import _translate
    ref = _gradmag(chip[0])[b:-b, b:-b]
    out = [np.asarray(chip[0], np.float32)]
    n = 2 * maxs + 1
    for k in range(1, len(chip)):
        g = _gradmag(chip[k])
        cc = np.zeros((n, n))
        for i, dy in enumerate(range(-maxs, maxs + 1)):
            for j, dx in enumerate(range(-maxs, maxs + 1)):
                cc[i, j] = float((ref * np.roll(g, (dy, dx), (0, 1))[b:-b, b:-b]).sum())
        i, j = np.unravel_index(np.argmax(cc), cc.shape)
        i, j = min(max(i, 1), n - 2), min(max(j, 1), n - 2)
        sub = []
        for a, c0, c in ((cc[i - 1, j], cc[i, j], cc[i + 1, j]), (cc[i, j - 1], cc[i, j], cc[i, j + 1])):
            den = a - 2 * c0 + c
            sub.append(float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0)
        sy, sx = i - maxs + sub[0], j - maxs + sub[1] # Content of layer k sits at +(sx, sy) from layer 0
        out.append(_translate(chip[k], -sx, -sy) if (sx or sy) else np.asarray(chip[k], np.float32))
    return np.stack(out)


def _tents(h, w):
    """2 x 2 bilinear weights over the chip (they sum to 1)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v, u = yy / (h - 1), xx / (w - 1)
    return [(1 - v) * (1 - u), (1 - v) * u, v * (1 - u), v * u]


def _bias(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w - 0.5, yy / h - 0.5
    return [u ** i * v ** j for i in range(3) for j in range(3 - i)]


def _regions(gold, k, iters=20):
    """Layout regions: k-means on the golden layer vectors."""
    nl, h, w = gold.shape
    f = ((gold - gold.mean(axis=(1, 2), keepdims=True)) / (gold.std(axis=(1, 2), keepdims=True) + 1e-9)).reshape(nl, -1).T
    cen = f[np.random.default_rng(0).choice(len(f), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((f[:, None, :] - cen[None]) ** 2).sum(-1), 1)
        cen = np.stack([f[lab == j].mean(0) if np.any(lab == j) else cen[j] for j in range(k)])
    lab = lab.reshape(h, w)
    return [(lab == j).astype(np.float32) for j in range(k)]


def _feats(L):
    c = L - L.mean()
    gy, gx = np.gradient(L)
    return [c, c * c - (c * c).mean(), gx, gy, _gauss(L, 1.5) - L]


def _robust_fit(y, X, iters=4, clip=3.0, sub=2):
    """Remainder y - X coef; coef by least squares on every sub-th pixel, outliers (> clip MADs) refitted out."""
    idx = np.arange(0, y.size, sub)
    keep = idx
    for _ in range(iters):
        Xk = X[keep]
        A = Xk.T @ Xk
        A += np.eye(A.shape[0]) * (1e-4 * np.trace(A) / A.shape[0])
        coef = np.linalg.solve(A, Xk.T @ y[keep])
        res = y[idx] - X[idx] @ coef
        mad = float(np.median(np.abs(res))) * 1.4826 + 1e-9
        keep = idx[np.abs(res) < clip * mad]
    return y - X @ coef


class _Context:
    """Per wafer: goldens, region masks, the chip-independent columns per target layer."""

    def __init__(self, X, targets, p):
        nl, n, h, w = X.shape
        self.p, self.shape = p, (h, w)
        gold = np.stack([np.median(X[k].astype(np.float32), axis=0) for k in range(nl)])
        self.tents = _tents(h, w)
        regions = _regions(gold, int(p["lc_regions"])) if int(p["lc_regions"]) > 1 else []
        gfeats = [[g - g.mean(), _gauss(g, 1.5) - g] for g in gold]
        self.regions = regions[1:]
        self.base, self.sources = {}, {}
        for k in targets:
            golds = sorted(range(nl), key=lambda j: (abs(j - k), j))[:int(p["lc_goldens"])]
            cols = _bias(h, w) + [f * t for j in golds for f in gfeats[j] for t in self.tents]
            for m in self.regions:
                cols += [m] + [gfeats[j][0] * m for j in golds]
            self.base[k] = np.stack([c.ravel() for c in cols], 1).astype(np.float64)
            self.sources[k] = list(range(k - 1, -1, -1))[:int(p["lc_sources"])]

    def remainders(self, chip):
        """(NL, h, w) chip -> {target layer: remainder}."""
        h, w = self.shape
        feats = {}
        out = {}
        for k, base in self.base.items():
            cols = []
            for j in self.sources[k]:
                if j not in feats:
                    feats[j] = _feats(chip[j])
                cols += [f * t for f in feats[j] for t in self.tents]
                cols += [feats[j][0] * m for m in self.regions] + [feats[j][1] * m for m in self.regions]
            X = np.concatenate([base, np.stack([c.ravel() for c in cols], 1)], 1) if cols else base
            out[k] = _robust_fit(chip[k].ravel().astype(np.float64), X).reshape(h, w).astype(np.float32)
        return out


def _normalise(R):
    """Remainders of one layer (n, h, w) -> per chip and per pixel (chip-to-chip) robust units."""
    R = R - np.median(R, axis=0)
    R = R / (np.median(np.abs(R), axis=(1, 2), keepdims=True) * 1.4826 + 1e-9)
    spread = _gauss(np.median(np.abs(R), axis=0) * 1.4826, 1.0)
    return R / np.maximum(spread, 0.5 * float(np.median(spread)))


def find(X, searched, p, rule="first", progress=None):
    """
    X: (NL, n, h, w) chips of all layers (index = loaded layer), searched: [bool] per layer.
    Returns {"dets": [{"z", "i" (chip), "x", "y", "r" (working px), "score"}], "evidence": (NL, n, h, w),
    "place": (n, h, w)}, or None when progress(...) returned False.
    """
    p = dict(DEFAULTS, **p)
    nl, n, h, w = X.shape
    targets = [k for k in range(nl) if searched[k]]
    if not targets or n < 3:
        return {"dets": [], "evidence": np.zeros(X.shape, np.float16), "place": np.zeros((n, h, w), np.float32)}
    ctx = _Context(X, targets, p)
    rem = {k: np.zeros((n, h, w), np.float32) for k in targets}
    reg = bool(p["lc_register"]) and nl > 1

    def work(i):
        chip = register_layers(X[:, i]) if reg else X[:, i].astype(np.float32)
        return ctx.remainders(chip)

    workers = int(p["lc_workers"]) or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(work, range(n))):
            for k, r in out.items():
                rem[k][i] = r
            if progress is not None and not progress(i, n, f"Explaining layers: chip {i + 1}/{n}"):
                ex.shutdown(cancel_futures=True)
                return None

    # Evidence per layer: best centre-surround score over the scales
    scales = [float(s) for s in p["lc_scales"]]
    b = int(p["lc_border"])
    S = np.zeros((nl, n, h, w), np.float16)
    SIG = np.zeros((nl, n, h, w), np.uint8)
    for k in targets:
        Z = _normalise(rem.pop(k))
        best = arg = None
        for si, s in enumerate(scales):
            d = np.abs(_gauss(Z, s) - _gauss(Z, 2 * s))
            d /= float(np.median(d[:, b:-b, b:-b])) * 1.4826 + 1e-9
            if best is None:
                best, arg = d, np.zeros(d.shape, np.uint8)
            else:
                arg[d > best] = si
                best = np.maximum(best, d)
        best[:, :b] = 0; best[:, -b:] = 0; best[:, :, :b] = 0; best[:, :, -b:] = 0
        S[k], SIG[k] = best, arg

    # Place: evidence over the searched layers (root mean square x sqrt(2) = root sum of squares / 2 at 8 layers)
    A = np.sqrt((S[targets].astype(np.float32) ** 2).mean(0) * 2.0)
    kbest = np.array(targets)[S[targets].argmax(0)]
    SIGA = np.take_along_axis(SIG, kbest[None], 0)[0]
    thr, tlo = float(p["lc_threshold"]), float(p["lc_tlo"])
    peaks = [(int(i), int(y), int(x), scales[SIGA[i, y, x]], float(A[i, y, x]))
             for i, y, x in zip(*np.nonzero((A == _max_filter(A)) & (A > thr)))]
    peaks.sort(key=lambda d: -d[4])
    kept = []
    for d in peaks: # A weaker peak overlapping a stronger one (same chip) is the same spot
        if all(k[0] != d[0] or math.hypot(k[1] - d[1], k[2] - d[2]) >= 1.5 * (k[3] + d[3]) for k in kept):
            kept.append(d)

    dets = []
    tg = np.array(targets)
    for i, y, x, s, score in kept:
        prof = S[:, i, y, x].astype(np.float32)
        if rule == "strongest":
            first = int(tg[prof[tg].argmax()])
        elif rule == "last":
            hit = [k for k in targets if prof[k] >= tlo]
            first = hit[-1] if hit else int(tg[prof[tg].argmax()])
        else: # CUSUM change point: the void stays on every deeper layer
            tail = np.cumsum((prof - tlo)[::-1])[::-1]
            first = int(tg[tail[tg].argmax()])
        # Place and size on that layer (the void has another size on other layers)
        rad = max(2, int(round(1.5 * s)))
        y0, x0 = max(0, y - rad), max(0, x - rad)
        win = S[first, i, y0:y + rad + 1, x0:x + rad + 1]
        yy, xx = (y0 + int(win.argmax()) // win.shape[1], x0 + int(win.argmax()) % win.shape[1]) \
            if float(win.max()) >= tlo else (y, x)
        dets.append({"z": first, "i": i, "x": xx + 0.5, "y": yy + 0.5,
                     "r": 1.6 * scales[SIG[first, i, yy, xx]], "score": score})
    return {"dets": dets, "evidence": S, "place": A}
