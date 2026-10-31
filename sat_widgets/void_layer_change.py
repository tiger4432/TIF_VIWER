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
    "lc_scales": [2.0, 3.0, 4.5, 6.5], # Centre-surround sigmas (working px)
    "lc_sources": 7, # At most this many layers above as sources (nearest first)
    "lc_goldens": 8, # Golden chips of the nearest this many layers as sources
    "lc_regions": 4, # Layout regions with their own coefficients (k-means of the goldens)
    "lc_border": 10, # Working px at the chip border not searched
    "lc_register": False, # Layers of each chip moved onto its first layer (sub-pixel; a scan's layers are aligned)
    "lc_workers": 0, # Chips fitted in parallel (0 = CPU count, 1 without threadpoolctl)
    "lc_cand_threshold": 8.0, # Learned filter: candidates down to this place score (Train)
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
    """
    Remainder y - X coef; coef by least squares on every sub-th pixel, outliers (> clip MADs) refitted
    out. Columns scaled to unit size first, so the small ridge does not depend on the intensity scale.
    """
    idx = np.arange(0, y.size, sub)
    unit = 1.0 / (np.sqrt((X[idx] ** 2).mean(0)) + 1e-12)
    X = X * unit
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
    for i, y, x, s, score in kept:
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
        dets.append({"z": first, "i": i, "x": xx + 0.5, "y": yy + 0.5,
                     "r": 1.6 * scales[SIG[first, i, yy, xx]], "score": score})
        if feat is not None:
            dets[-1]["features"] = feat.row(i, y, x, first, float(SIG[first, i, yy, xx]))
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
