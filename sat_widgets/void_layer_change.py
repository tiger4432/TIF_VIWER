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
5. Place ("lc_place"): "cusum" (default) = max_l sum_{k >= l} (e_k - lc_tlo) over the searched
   layers: the evidence from the layer where it starts to the last one (a void stays on every
   deeper layer; a void-free layer, e < lc_tlo, lowers it). A void that appears late counts like
   one that appears early, a spike on one layer counts once. Below the last searched layer a
   virtual layer is added: the last layer's evidence blurred by lc_pad_sigma (scaled to the same top
   level). Nothing is known below the last layer, but a void is a blob and keeps its evidence
   blurred while thin residues at the chip edge / pad fade; so a void that first appears on the last
   layer is not just a one-layer spike (synthetic wafers: 8 instead of 2 of 33 such voids found at
   0 false detections). "rms" = root mean square of the
   evidence over the layers x sqrt(2) (late voids are averaged away: on the synthetic wafers
   twice the false detections for the same misses). Above lc_threshold, local maxima, weaker
   overlapping peaks dropped.
6. Layer ("layer_rule"): "first" = change point of the evidence at the place,
   argmax_l sum_{k >= l} (e_k - lc_tlo) (CUSUM: the void stays on the deeper layers);
   "strongest" = largest evidence; "last" = deepest layer whose evidence reaches lc_tlo.
   The size is the best scale there (radius 1.6 x sigma).

From tools/synth/experiments/C (branch exp/C-layer-change). Synthetic 8-layer wafers, lc_threshold 26:
training wafers ml_201..203 miss 32 % of the voids (any layer), 4.0 false detections per wafer;
ml_204 miss 24 %, 2 false (die-to-die: 94 % and 443). Lower Thr: more found, more false (README).
"""
import contextlib
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DEFAULTS = {
    "lc_threshold": 26.0, # Place: evidence over the layers (see 5.; about 10 with lc_place "rms")
    "lc_place": "cusum", # "cusum" or "rms" (see 5.)
    "lc_pad_sigma": 5.0, # Virtual layer below the last: its evidence blurred (working px; 0 = none)
    "lc_tlo": 3.0, # Evidence of a void-free layer (first layer: CUSUM level)
    "lc_scales": [2.0, 3.0, 4.5, 6.5, 9.5, 14.0, 20.0, 28.0], # Centre-surround sigmas (working px); those above
                       # 1/7 of the working chip are left out. The large ones find large voids whole (production
                       # 600 px chips, radius 10-150 px: found 73 -> 88 %, split in pieces 33 -> 11 of 77)
    "lc_sources": 7, # At most this many layers above as sources (nearest first)
    "lc_tents": 2, # Fit coefficients vary over the chip on n x n nodes (bilinear between them)
    "lc_src_blurs": [], # Extra source features: blur differences at these sigmas (focus / sharpness changes)
    "lc_top_pad_sigma": 0.0, # Virtual source layer for the top targets (working px; 0 = none), see lc_top_pad_src
    "lc_top_pad_src": "self", # Its image, blurred: "self" (the target layer) or "below" (the layer under it)
    "lc_top_pad_upto": 1, # Targets 1 .. this get it (their layers above show little layout)
    "lc_goldens": 8, # Golden chips of the nearest this many layers as sources
    "lc_regions": 4, # Layout regions with their own coefficients (k-means of the goldens)
    "lc_border": 10, # Working px at the chip border not searched
    "lc_region_level": 0.4, # Void extent: connected region of its layer's evidence >= this x its peak; weaker
                            # peaks inside it are the same void (0 = off: radius 1.6 x the filter sigma). Large
                            # voids: size found / true 0.40 -> about 1, split 11 -> 2 of 77
    "lc_border_zero": 10, # Remainder zeroed this many working px in from the edge before filtering, so the chip edge
                          # does not leak into the searched part (synthetic, layers aligned: missed 11.2 -> 7.5 %; 0 = off)
    "lc_register": False, # Layers of each chip moved onto its first layer (sub-pixel; a scan's layers are aligned)
    "lc_workers": 0, # Chips fitted in parallel (0 = CPU count, 1 without threadpoolctl)
    "lc_cand_threshold": 5.0, # Learned filter: candidates down to this place score (Train; synthetic: 8 -> 5 missed 12.7 -> 11.0 % at 1 false per wafer)
    "lc_layer0_check": "flag", # Voids already visible on layer 0 (bubbles?): "flag" / "reject" / "off" (see presence)
    "lc_layer0_z": 1.0, # Presence on layer 0 from this |z| on (core remainder / ring noise)
    "lc_layer0_type": None, # "flag": also this void type for them (when declared)
    "lc_learned_cut": None, # Learned filter: keep a candidate from this model score on (None = the model's cut, best F1)
}

# Learned filter (sat_widgets/void_learn.py, Train / Learned filter): features of each candidate
FEATURES = ["place", "first_rel", "n_after", "e_first", "e_max", "tail_mean", "tail_min", "tail_frac", "e_before",
            "pad", "scale", "border", "xn", "yn", "common", "z_pixel", "chip_noise", "rank_in_chip", "blob"]

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


def _shift(a, sx, sy, pad=8):
    """a with its content moved by (sx, sy) px, sub-pixel without blurring (Fourier phase ramp;
    mirrored margin so the edges do not wrap)."""
    h, w = a.shape
    ap = np.pad(np.asarray(a, np.float32), pad, mode="reflect")
    H, W = ap.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    out = np.fft.irfft2(np.fft.rfft2(ap) * np.exp(-2j * np.pi * (ky * sy + kx * sx)), s=(H, W))
    return out[pad:pad + h, pad:pad + w].astype(np.float32)


def register_layers(chip, maxs=2, b=6):
    """(NL, h, w): each layer moved onto the first (gradient magnitude correlation, parabolic peak)."""
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
        out.append(_shift(chip[k], -sx, -sy) if (sx or sy) else np.asarray(chip[k], np.float32))
    return np.stack(out)


def _whole(cc, y, x, subpixel=False):
    """Whole-px (dx, dy) of the circular correlation peak at (y, x); a 1 px peak whose sub-pixel
    (parabola) position is under 0.75 px stays 0 (a near-tie between 0 and 1 px). subpixel: the
    parabola position itself."""
    h, w = cc.shape
    out = []
    for lo, mid, hi, d in ((cc[y, (x - 1) % w], cc[y, x], cc[y, (x + 1) % w], x if x <= w // 2 else x - w),
                           (cc[(y - 1) % h, x], cc[y, x], cc[(y + 1) % h, x], y if y <= h // 2 else y - h)):
        den = lo - 2 * mid + hi
        f = d + (float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5)) if den < 0 else 0.0)
        out.append(f if subpixel else (0 if abs(d) == 1 and abs(f) < 0.75 else d))
    return out


def _refine_peak(P, sy, sx):
    """Correlation peak near the whole-px shift (sx, sy) to 0.01 px: the correlation evaluated on a
    sub-pixel grid straight from its Fourier coefficients P (upsampled DFT, matrix products)."""
    h, w = P.shape
    fy, fx = np.fft.fftfreq(h)[None, :], np.fft.fftfreq(w)[None, :]
    for span, step in ((0.6, 0.05), (0.06, 0.005)):
        oy = sy + np.arange(-span, span + step / 2, step)
        ox = sx + np.arange(-span, span + step / 2, step)
        cc = np.real(np.exp(2j * np.pi * oy[:, None] * fy) @ P @ np.exp(2j * np.pi * ox[:, None] * fx).T)
        a, b = np.unravel_index(int(np.argmax(cc)), cc.shape)
        sy, sx = float(oy[a]), float(ox[b])
    return sx, sy


def align_chips(X, max_frac=0.06, subpixel=True, apply=True):
    """
    (NL, n, h, w) chips -> each chip moved (every layer alike) onto the median chip, and the shifts
    [(dx, dy)]. The bbox may sit off the chip. Matched on the gradient magnitude summed over the
    layers: the layout's contrast and polarity differ from chip to chip, which misleads a match on
    the intensities. The search is limited to max_frac of the chip, so a periodic layout cannot pull
    a chip a whole period off. subpixel: moved by the sub-pixel offset with a Fourier phase shift
    (no blur; bboxes placed within a pixel otherwise blur the golden chip and the chip-to-chip
    spread at the layout edges: synthetic wafer, 0.7 px bbox scatter, 2 -> 16 false detections),
    else by whole pixels (a near-tie between 0 and 1 px stays). apply=False: the shifts only.
    """
    from .void_detect import _translate
    nl, n, h, w = X.shape
    G = np.zeros((n, h, w), np.float32)
    for k in range(nl):
        for i in range(n):
            g = _gradmag(X[k, i])
            G[i] += g / (float(np.abs(g).mean()) + 1e-9)
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    G = (G - G.mean(axis=(1, 2), keepdims=True)) * win
    F = np.fft.rfft2(G)
    ref = np.fft.rfft2(np.median(G, axis=0))
    ref_full = np.fft.fft2(np.median(G, axis=0)) if subpixel else None
    m = max(1, int(max_frac * max(h, w)))
    dy = np.r_[0:m + 1, -m:0]
    out, shifts = X.copy(), []
    for i in range(n):
        cc = np.fft.irfft2(ref * np.conj(F[i]), s=(h, w)) # cc[y, x]: chip rolled by (x, y) vs the median chip
        win_cc = cc[np.ix_(dy % h, dy % w)]
        a, b = np.unravel_index(int(np.argmax(win_cc)), win_cc.shape)
        sx, sy = _whole(cc, int(dy[a]) % h, int(dy[b]) % w) # Rolling the chip by (sx, sy) matches
        if subpixel:
            sx, sy = _refine_peak(ref_full * np.conj(np.fft.fft2(G[i])), float(dy[a]), float(dy[b]))
        if not apply:
            pass
        elif subpixel and (abs(sx) > 0.05 or abs(sy) > 0.05):
            for k in range(nl):
                out[k, i] = _shift(X[k, i], sx, sy)
        elif not subpixel and (sx or sy):
            for k in range(nl):
                out[k, i] = _translate(X[k, i], -sx, -sy)
        shifts.append((sx, sy))
    return out, shifts


def presence(X, gold, k, i, cy, cx, r):
    """
    Is the spot visible on layer k of chip i, independently of the layers above? The chip's patch
    around (cx, cy) (working px) is fitted from the golden chips of all layers (+ a plane) on a ring
    around the spot (radius 1.5 r .. max(3 r, 12)), leaving the core (radius r) out; returns the core's
    mean remainder in units of the ring's remainder spread (signed). The local fit follows the chip's
    own layout contrast, so a spot on layer 0 (which has no layer above) is judged like any other.
    """
    L, n, h, w = X.shape
    W = int(max(3 * r, 12))
    y0, y1, x0, x1 = max(0, int(cy) - W), min(h, int(cy) + W + 1), max(0, int(cx) - W), min(w, int(cx) + W + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.hypot(yy - cy, xx - cx).ravel()
    core, ring = d <= r, (d >= 1.5 * r) & (d <= W)
    if core.sum() < 3 or ring.sum() < 20:
        return 0.0
    A = np.stack([gold[j, y0:y1, x0:x1].ravel() for j in range(L)] + [np.ones(d.size), (xx - cx).ravel(), (yy - cy).ravel()], 1)
    y = X[k, i, y0:y1, x0:x1].astype(np.float64).ravel()
    coef, *_ = np.linalg.lstsq(A[ring], y[ring], rcond=None)
    res = y - A @ coef
    return float(res[core].mean()) / (float(np.std(res[ring])) + 1e-6)


def _tents(h, w, n=2):
    """n x n piecewise-bilinear weights over the chip (they sum to 1; n = 2: the four corners)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if n == 2: # The four corners (same floats as before n was a parameter)
        v, u = yy / (h - 1), xx / (w - 1)
        return [(1 - v) * (1 - u), (1 - v) * u, v * (1 - u), v * u]
    v, u = yy / (h - 1) * (n - 1), xx / (w - 1) * (n - 1)
    hv = [np.maximum(0.0, 1.0 - np.abs(v - i)) for i in range(n)]
    hu = [np.maximum(0.0, 1.0 - np.abs(u - i)) for i in range(n)]
    return [a * b for a in hv for b in hu]


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


def _region(E, y, x, level, reach):
    """Connected part of E >= level around (y, x) within reach px (grown 8-connected, no scipy):
    (y0, x0, mask) of its window, or None when the seed is below the level."""
    h, w = E.shape
    y0, y1, x0, x1 = max(0, y - reach), min(h, y + reach + 1), max(0, x - reach), min(w, x + reach + 1)
    ok = E[y0:y1, x0:x1] >= level
    if not ok[y - y0, x - x0]:
        return None
    m = np.zeros(ok.shape, bool)
    m[y - y0, x - x0] = True
    while True:
        g = m.copy() # 3 x 3 dilation, kept inside ok
        g[1:] |= m[:-1]; g[:-1] |= m[1:]
        g[:, 1:] |= g[:, :-1]; g[:, :-1] |= g[:, 1:]
        g &= ok
        if g.sum() == m.sum():
            return y0, x0, m
        m = g


def _feats(L, blurs=()):
    c = L - L.mean()
    gy, gx = np.gradient(L)
    return [c, c * c - (c * c).mean(), gx, gy, _gauss(L, 1.5) - L] + [_gauss(L, float(s)) - L for s in blurs]


def _robust_fit(y, X, iters=4, clip=3.0, sub=2):
    """
    Remainder y - X coef; coef by least squares on every sub-th pixel, outliers (> clip MADs) refitted
    out. Columns scaled to unit size, so the small ridge does not depend on the intensity scale.
    X'X is formed once; each refit only takes the outliers' share off it (a few % of the pixels).
    """
    Xs, ys = X[::sub], y[::sub]
    G, g = Xs.T @ Xs, Xs.T @ ys
    unit = 1.0 / (np.sqrt(np.diag(G) / len(ys)) + 1e-12)
    scale = unit[:, None] * unit[None, :]
    out = None
    for _ in range(iters):
        if out is None or not out.any():
            A, r = G, g
        else:
            Xo = Xs[out]
            A, r = G - Xo.T @ Xo, g - Xo.T @ ys[out]
        As = A * scale
        As += np.eye(As.shape[0]) * (1e-4 * np.trace(As) / As.shape[0])
        coef = unit * np.linalg.solve(As, r * unit)
        res = ys - Xs @ coef
        mad = float(np.median(np.abs(res))) * 1.4826 + 1e-9
        out = np.abs(res) >= clip * mad
    return y - X @ coef


class _Context:
    """
    Per wafer: goldens, region masks and the columns of the layer fit, kept as blocks: bias, region
    masks, one block per golden layer (chip-independent) and one per source layer of a chip. A target
    layer's normal matrix X'X is assembled from block products: the chip-independent ones are formed
    once per wafer, the ones with a source layer once per chip (neighbouring targets share most of
    their sources and goldens), instead of a full X'X per target.
    """

    def __init__(self, X, targets, p):
        nl, n, h, w = X.shape
        self.p, self.shape = p, (h, w)
        gold = np.stack([np.median(X[k].astype(np.float32), axis=0) for k in range(nl)])
        self.gold = gold
        self.tents = _tents(h, w, int(p.get("lc_tents", 2)))
        self.src_blurs = tuple(p.get("lc_src_blurs") or ())
        regions = _regions(gold, int(p["lc_regions"])) if int(p["lc_regions"]) > 1 else []
        self.regions = regions[1:]
        self.sub = int(p.get("lc_fit_sub", 2)) # Least squares on every sub-th pixel (see _block_fit)
        gfeats = [[g - g.mean(), _gauss(g, 1.5) - g] for g in gold]
        fixed = {("bias",): _bias(h, w)}
        if self.regions:
            fixed[("regions",)] = list(self.regions)
        for j in range(nl):
            fixed[("gold", j)] = [f * t for f in gfeats[j] for t in self.tents] + [gfeats[j][0] * m for m in self.regions]
        names = list(fixed)
        cols = [np.stack([c.ravel() for c in fixed[k]], 1) for k in names]
        edges = np.cumsum([0] + [c.shape[1] for c in cols])
        span = {k: np.arange(edges[i], edges[i + 1]) for i, k in enumerate(names)}
        # All chip-independent columns side by side (full rows / every sub-th row) and their products
        self.Ff = np.concatenate(cols, 1).astype(np.float64)
        self.Fs = self.Ff[::self.sub]
        self.FF = self.Fs.T @ self.Fs
        self.fidx, self.sources = {}, {}
        # Virtual sources (ids nl + k): layer src of the chip blurred by sigma, for the top targets whose
        # layers above show little layout (real top layers: only from the second layer on)
        self.virtual = {}
        tps = float(p.get("lc_top_pad_sigma", 0.0) or 0.0)
        for k in targets:
            golds = sorted(range(nl), key=lambda j: (abs(j - k), j))[:int(p["lc_goldens"])]
            self.sources[k] = list(range(k - 1, -1, -1))[:int(p["lc_sources"])]
            if tps > 0 and 1 <= k <= int(p.get("lc_top_pad_upto", 1)):
                src = k + 1 if p.get("lc_top_pad_src", "self") == "below" and k + 1 < nl else k
                self.virtual[nl + k] = (src, tps)
                self.sources[k] = [nl + k] + self.sources[k]
            self.fidx[k] = np.concatenate([span[("bias",)]] + ([span[("regions",)]] if self.regions else []) +
                                          [span[("gold", j)] for j in golds])

    def remainders(self, chip):
        """(NL, h, w) chip -> {target layer: remainder}."""
        h, w = self.shape
        Sf, Ss, cross, ss = {}, {}, {}, {}

        def source(j):
            if j not in Sf:
                if j in self.virtual:
                    src, sig = self.virtual[j]
                    f = _feats(_gauss(chip[src].astype(np.float32), sig), self.src_blurs)
                else:
                    f = _feats(chip[j], self.src_blurs)
                c = [a * t for a in f for t in self.tents] + [f[0] * m for m in self.regions] + [f[1] * m for m in self.regions]
                Sf[j] = np.stack([x.ravel() for x in c], 1).astype(np.float64)
                Ss[j] = Sf[j][::self.sub]
                cross[j] = Ss[j].T @ self.Fs # Source block x every chip-independent column
            return Sf[j]

        def SS(i, j):
            a, b = min(i, j), max(i, j)
            if (a, b) not in ss:
                ss[(a, b)] = Ss[a].T @ Ss[b]
            return ss[(a, b)] if i <= j else ss[(a, b)].T

        out = {}
        for k, idx in self.fidx.items():
            srcs = self.sources[k]
            for j in srcs:
                source(j)
            G = self.FF[np.ix_(idx, idx)]
            if srcs:
                SF = np.concatenate([cross[j][:, idx] for j in srcs])
                G = np.block([[G, SF.T], [SF, np.block([[SS(i, j) for j in srcs] for i in srcs])]])
            y = chip[k].ravel().astype(np.float64)
            out[k] = _block_fit(y, self.sub, self.Ff, self.Fs, idx, [Sf[j] for j in srcs], [Ss[j] for j in srcs],
                                G).reshape(h, w).astype(np.float32)
        return out


def _block_fit(y, step, Ff, Fs, idx, src_full, src_sub, G, iters=4, clip=3.0):
    """
    _robust_fit with the columns as the chip-independent ones (all of them Ff / Fs, this target's
    idx) and the source blocks, X'X = G assembled from cached block products: the same remainder
    without building X (the chip-independent part is multiplied with a zero-padded coefficient).
    """
    ys = y[::step]
    g = np.concatenate([(Fs.T @ ys)[idx]] + [B.T @ ys for B in src_sub])
    unit = 1.0 / (np.sqrt(np.diag(G) / len(ys)) + 1e-12)
    scale = unit[:, None] * unit[None, :]
    nf = len(idx)
    edges = np.cumsum([nf] + [B.shape[1] for B in src_sub])
    cf = np.zeros(Fs.shape[1])

    def fitted(Fx, blocks, coef):
        cf[idx] = coef[:nf]
        v = Fx @ cf
        for i, B in enumerate(blocks):
            v += B @ coef[edges[i]:edges[i + 1]]
        return v

    out = None
    for _ in range(iters):
        if out is None or not out.any():
            A, r = G, g
        else:
            Xo = np.concatenate([Fs[out][:, idx]] + [B[out] for B in src_sub], 1)
            A, r = G - Xo.T @ Xo, g - Xo.T @ ys[out]
        As = A * scale
        As += np.eye(As.shape[0]) * (1e-4 * np.trace(As) / As.shape[0])
        coef = unit * np.linalg.solve(As, r * unit)
        res = ys - fitted(Fs, src_sub, coef)
        mad = float(np.median(np.abs(res))) * 1.4826 + 1e-9
        out = np.abs(res) >= clip * mad
    return y - fitted(Ff, src_full, coef)


def _normalise(R):
    """Remainders of one layer (n, h, w) -> per chip and per pixel (chip-to-chip) robust units."""
    R = R - np.median(R, axis=0)
    R = R / (np.median(np.abs(R), axis=(1, 2), keepdims=True) * 1.4826 + 1e-9)
    spread = _gauss(np.median(np.abs(R), axis=0) * 1.4826, 1.0)
    return R / np.maximum(spread, 0.5 * float(np.median(spread)))


@contextlib.contextmanager
def _single_blas():
    """BLAS limited to one thread (threadpoolctl); yields False when that is not possible."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield False
        return
    with threadpool_limits(1, "blas"):
        yield True


def find(X, searched, p, rule="first", progress=None):
    """
    X: (NL, n, h, w) chips of all layers (index = loaded layer), searched: [bool] per layer.
    p["learned_model"] (void_learn.Model on FEATURES): candidates down to its cand_threshold, kept when
    its score reaches its cut (det["learned"]); p["learn"] (training): every candidate down to
    lc_cand_threshold, with det["features"].
    Returns {"dets": [{"z", "i" (chip), "x", "y", "r" (working px), "score"}], "evidence": (NL, n, h, w),
    "scale": (NL, n, h, w) index into lc_scales of the best scale, "place": (n, h, w)}, or None when
    progress(...) returned False.
    """
    p = dict(DEFAULTS, **p)
    nl, n, h, w = X.shape
    targets = [k for k in range(nl) if searched[k]]
    if not targets or n < 3:
        return {"dets": [], "evidence": np.zeros(X.shape, np.float16), "scale": np.zeros(X.shape, np.uint8),
                "place": np.zeros((n, h, w), np.float32)}
    ctx = _Context(X, targets, p)
    rem = {k: np.zeros((n, h, w), np.float32) for k in targets}
    reg = bool(p["lc_register"]) and nl > 1

    def work(i):
        chip = register_layers(X[:, i]) if reg else X[:, i].astype(np.float32)
        return ctx.remainders(chip)

    with _single_blas() as limited:
        # One chip per core; BLAS itself on one thread (both multithreaded: 3x slower)
        workers = int(p["lc_workers"]) or ((os.cpu_count() or 1) if limited else 1)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, out in enumerate(ex.map(work, range(n))):
                for k, r in out.items():
                    rem[k][i] = r
                if progress is not None and not progress(i, n, f"Explaining layers: chip {i + 1}/{n}"):
                    ex.shutdown(cancel_futures=True)
                    return None

    # Evidence per layer: best centre-surround score over the scales
    scales = [float(s) for s in p["lc_scales"]]
    scales = [s for s in scales if s <= max(h, w) / 7.0] or scales[:1] # Not larger than the chip allows
    b = int(p["lc_border"])
    S = np.zeros((nl, n, h, w), np.float16)
    SIG = np.zeros((nl, n, h, w), np.uint8)
    zb = int(p.get("lc_border_zero", 0) or 0) # Remainder zeroed this far in before filtering
    for k in targets:
        Z = _normalise(rem.pop(k))
        if zb > 0: # The chip edge (not searched) must not leak into the searched part through the filter
            Z[:, :zb] = 0; Z[:, -zb:] = 0; Z[:, :, :zb] = 0; Z[:, :, -zb:] = 0
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

    # Place: evidence over the searched layers (see 5.)
    thr, tlo = float(p["lc_threshold"]), float(p["lc_tlo"])
    pad = None
    if float(p["lc_pad_sigma"]) > 0 and p["lc_place"] != "rms":
        last = S[targets[-1]].astype(np.float32)
        pad = _gauss(last, float(p["lc_pad_sigma"]))
        pad *= float(np.percentile(last, 99.9)) / max(float(np.percentile(pad, 99.9)), 1e-6)
    if p["lc_place"] == "rms":
        A = np.sqrt((S[targets].astype(np.float32) ** 2).mean(0) * 2.0)
    else: # CUSUM from the deepest layer up: run = sum_{k >= l} (e_k - tlo), A = its maximum over l
        run = A = None
        if pad is not None: # The virtual layer below the last never starts a void on its own
            run = pad - tlo
        for k in reversed(targets):
            run = S[k].astype(np.float32) - tlo if run is None else run + (S[k].astype(np.float32) - tlo)
            A = run.copy() if A is None else np.maximum(A, run)
        del run
    kbest = np.array(targets)[S[targets].argmax(0)]
    SIGA = np.take_along_axis(SIG, kbest[None], 0)[0]
    model, collect = p.get("learned_model"), bool(p.get("learn"))
    if model is not None or collect: # Candidates for the learned filter: lower place scores too
        thr = float(model.cand_threshold) if model is not None else float(p["lc_cand_threshold"])
    peaks = [(int(i), int(y), int(x), scales[SIGA[i, y, x]], float(A[i, y, x]))
             for i, y, x in zip(*np.nonzero((A == _max_filter(np.maximum(A, 0))) & (A > thr)))]
    peaks.sort(key=lambda d: -d[4])
    kept = []
    for d in peaks: # A weaker peak overlapping a stronger one (same chip) is the same spot
        if all(k[0] != d[0] or math.hypot(k[1] - d[1], k[2] - d[2]) >= 1.5 * (k[3] + d[3]) for k in kept):
            kept.append(d)

    dets = []
    tg = np.array(targets)
    feat = _Features(S, A, pad, tg, tlo) if (model is not None or collect) else None
    rlev = float(p.get("lc_region_level", 0.0) or 0.0) # Extent: region of the evidence >= this x its peak
    regions = {} # chip -> [(y0, x0, mask)] of the voids kept: a weaker peak inside one is the same void
    for i, y, x, s, score in kept:
        if rlev > 0 and any(0 <= y - ry0 < m.shape[0] and 0 <= x - rx0 < m.shape[1] and m[y - ry0, x - rx0]
                            for ry0, rx0, m in regions.get(i, ())):
            continue
        prof = S[:, i, y, x].astype(np.float32)
        if rule == "strongest":
            first = int(tg[prof[tg].argmax()])
        elif rule == "last":
            hit = [k for k in targets if prof[k] >= tlo]
            first = hit[-1] if hit else int(tg[prof[tg].argmax()])
        else: # CUSUM change point over the searched layers: the void stays on every deeper layer
            e = prof[tg] if pad is None else np.append(prof[tg], pad[i, y, x])
            tail = np.cumsum((e - tlo)[::-1])[::-1]
            first = int(tg[min(int(tail.argmax()), len(tg) - 1)]) # Starts on the virtual layer: the last
        # Place and size on that layer (the void has another size on other layers)
        rad = max(2, int(round(1.5 * s)))
        y0, x0 = max(0, y - rad), max(0, x - rad)
        win = S[first, i, y0:y + rad + 1, x0:x + rad + 1]
        yy, xx = (y0 + int(win.argmax()) // win.shape[1], x0 + int(win.argmax()) % win.shape[1]) \
            if float(win.max()) >= tlo else (y, x)
        r = 1.6 * scales[SIG[first, i, yy, xx]]
        det = {"z": first, "i": i, "x": xx + 0.5, "y": yy + 0.5, "r": r, "score": score,
               "snr": float(S[first, i, yy, xx])} # Evidence on its layer, in units of the background noise
        if rlev > 0: # Extent from the region of strong evidence (large voids: not just the filter size)
            E = S[first, i].astype(np.float32)
            reg = _region(E, yy, xx, max(tlo, rlev * float(E[yy, xx])), int(max(3 * r, 0.6 * max(E.shape))))
            if reg is not None:
                ry0, rx0, m = reg
                regions.setdefault(i, []).append(reg)
                ys, xs = np.nonzero(m)
                cy, cx = ry0 + (ys.min() + ys.max() + 1) / 2.0, rx0 + (xs.min() + xs.max() + 1) / 2.0
                rx, ry = (xs.max() - xs.min() + 1) / 2.0, (ys.max() - ys.min() + 1) / 2.0
                if max(rx, ry) > r: # Larger than the filter size: the region's box (else the filter's)
                    det.update(x=cx, y=cy, rx=rx, ry=ry, r=max(rx, ry))
        dets.append(det)
        if feat is not None:
            dets[-1]["features"] = feat.row(i, y, x, first, float(SIG[first, i, yy, xx]))
    l0 = p.get("lc_layer0_index") # Index of the scan's layer 0 among the loaded layers (set by void_detect)
    if l0 is not None and p["lc_layer0_check"] != "off" and dets and not searched[l0]:
        # Already visible on layer 0 (not searched there: noisy, bubbles): see presence
        for d in dets:
            d["layer0_z"] = presence(X, ctx.gold, l0, d["i"], d["y"], d["x"], max(2.0, 0.8 * d["r"]))
    if model is not None and dets:
        sc = model.scores([d["features"] for d in dets])
        for d, v in zip(dets, sc):
            d["learned"] = float(v)
        cut = float(p["lc_learned_cut"]) if p.get("lc_learned_cut") is not None else model.cut
        dets = [d for d in dets if d["learned"] >= cut]
    return {"dets": dets, "evidence": S, "scale": SIG, "place": A}


class _Features:
    """FEATURES of a candidate (learned filter): how the evidence runs over the layers from its first
    layer on, how it compares with the same place in the other chips (layout residue shows in many
    chips), where it sits in the chip, whether it is a blob or a thin line."""

    def __init__(self, S, A, pad, tg, tlo):
        self.S, self.A, self.pad, self.tg, self.tlo = S, A, pad, tg, tlo
        self.n, self.h, self.w = A.shape
        self.med = np.median(A, axis=0)
        self.mad = np.median(np.abs(A - self.med), axis=0) * 1.4826 + 1e-3
        self.noise = np.percentile(A.reshape(self.n, -1), 99, axis=1)
        self.rank, self.blur = {}, {}

    def row(self, i, y, x, first, scale):
        S, tg, tlo, h, w = self.S, self.tg, self.tlo, self.h, self.w
        e = S[tg, i, y, x].astype(np.float32)
        pv = float(self.pad[i, y, x]) if self.pad is not None else 0.0
        fi = int(np.nonzero(tg == first)[0][0]) if first in tg else len(tg) - 1
        after, before = e[fi:], e[:fi]
        a = float(self.A[i, y, x])
        common = float((self.A[:, max(0, y - 1):y + 2, max(0, x - 1):x + 2].max(axis=(1, 2)) > 0.5 * a).mean())
        rank = self.rank.get(i, 0)
        self.rank[i] = rank + 1
        if (first, i) not in self.blur:
            self.blur[(first, i)] = _gauss(S[first, i].astype(np.float32), 3.0)
        blob = float(self.blur[(first, i)][y, x]) / max(float(S[first, i, y, x]), 1e-3)
        return [a, fi / max(len(tg) - 1, 1), float(len(tg) - fi), float(e[fi]), float(e.max()), float(after.mean()),
                float(after.min()), float((after > tlo).mean()), float(before.max()) if len(before) else 0.0,
                pv, scale, float(min(y, x, h - 1 - y, w - 1 - x)), x / w, y / h, common,
                (a - float(self.med[y, x])) / float(self.mad[y, x]), float(self.noise[i]), float(rank), blob]
