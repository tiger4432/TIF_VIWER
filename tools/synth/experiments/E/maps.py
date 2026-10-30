"""
Method E, stage 1: change maps between the layers of one chip.

R[k, j] = layer k minus its explanation by layer j (per chip-layer intensity regression, the
same model as sat_widgets.void_detect._LayoutFit: layout x contrast, blur differences at 1/2/4 px,
brightness, each with 4 bilinear weights across the chip; robust, 3 passes). The chip's own
structure cancels, what is in k and not in j (or with another shade) stays.
Then normalised per wafer: per (k, j) and pixel, divided by the spread of R over all chips
(line edges / misregistration leave the same residual on every chip, a void does not).
numpy only.
"""
import numpy as np

SIGMAS = (1.0, 2.0, 4.0)


def _gk(s):
    r = max(1, int(3 * s + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-x * x / (2 * s * s))
    return g / g.sum()


def blur(a, s):
    """Separable gaussian over the last two axes (reflect padding)."""
    if s <= 0:
        return a
    g = _gk(s)
    r = len(g) // 2
    a = np.asarray(a, np.float32)
    pad = [(0, 0)] * (a.ndim - 2)
    b = np.pad(a, pad + [(r, r), (0, 0)], mode="reflect")
    out = sum(g[i] * b[..., i:i + a.shape[-2], :] for i in range(len(g)))
    b = np.pad(out, pad + [(0, 0), (r, r)], mode="reflect")
    return sum(g[i] * b[..., :, i:i + a.shape[-1]] for i in range(len(g))).astype(np.float32)


def _bases(o):
    h, w = o.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx / (w - 1)).ravel(), (yy / (h - 1)).ravel()
    wts = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v]
    g = o.ravel() - float(o.mean())
    bs = [g] + [(blur(o, s) - o).ravel() for s in SIGMAS] + [np.ones_like(g)]
    return np.stack([b * wj for b in bs for wj in wts], axis=1).astype(np.float32)


def chip_changes(chip):
    """chip (L, H, W) -> R (L, L, H, W) float32, R[k, j] = k - fit(j -> k); R[k, k] = 0."""
    L, h, w = chip.shape
    R = np.zeros((L, L, h, w), np.float32)
    sub = np.arange(0, h * w, 2)
    for j in range(L):
        X = _bases(chip[j])
        o = chip[j].ravel()
        for k in range(L):
            if k == j:
                continue
            r = chip[k].ravel() - o
            keep = sub
            for _ in range(3):
                Xk = X[keep]
                coef = np.linalg.solve(Xk.T @ Xk + np.eye(X.shape[1], dtype=np.float32) * 1e-3, Xk.T @ r[keep])
                res = r[sub] - X[sub] @ coef
                mad = float(np.median(np.abs(res))) * 1.4826 + 1e-6
                keep = sub[np.abs(res) < 3.0 * mad]
                if keep.size < X.shape[1] * 4:
                    break
            R[k, j] = (r - X @ coef).reshape(h, w)
    return R


def _work(args):
    return chip_changes(args)


def wafer_changes(X, procs=4):
    """X (L, R, C, H, W) -> (R*C, L, L, H, W) float16 change maps (chip order row-major)."""
    L, NR, NC, h, w = X.shape
    chips = [X[:, r, c] for r in range(NR) for c in range(NC)]
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as p:
            out = p.map(_work, chips, chunksize=4)
    else:
        out = [_work(c) for c in chips]
    return np.stack(out).astype(np.float16)


def normalise(Rw):
    """Per (k, j, pixel): divide by a robust spread over chips (and keep a floor)."""
    Rw = Rw.astype(np.float32)
    sp = np.quantile(np.abs(Rw), 0.75, axis=0) / 1.15  # ~ sigma for gaussian, robust to the 30 % void chips
    sp = blur(sp, 1.5)
    L = Rw.shape[1]
    for k in range(L):
        for j in range(L):
            if k != j:
                sp[k, j] = np.maximum(sp[k, j], 0.5 * np.median(sp[k, j]))
    sp[np.arange(L), np.arange(L)] = 1.0
    return (Rw / sp[None]).astype(np.float32)


PAD = 26


def dog_bank(Z, scales, ratio=2.5):
    """Z (..., H, W) -> (S, ..., H, W): blur(Z, s) - blur(Z, ratio * s) per scale (FFT, reflect pad)."""
    h, w = Z.shape[-2:]
    pad = [(0, 0)] * (Z.ndim - 2) + [(PAD, PAD), (PAD, PAD)]
    F = np.fft.rfft2(np.pad(Z, pad, mode="reflect"))
    H, W = h + 2 * PAD, w + 2 * PAD
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    f2 = (fy ** 2 + fx ** 2) * (2 * np.pi ** 2)
    out = []
    for s in scales:
        tf = np.exp(-f2 * s * s) - np.exp(-f2 * (ratio * s) ** 2)
        out.append(np.fft.irfft2(F * tf, s=(H, W))[..., PAD:PAD + h, PAD:PAD + w].astype(np.float32))
    return np.stack(out)


def _robust_fit(X, r, sub, passes=3):
    keep = sub
    coef = None
    for _ in range(passes):
        Xk = X[keep]
        coef = np.linalg.solve(Xk.T @ Xk + np.eye(X.shape[1], dtype=np.float32) * 1e-3, Xk.T @ r[keep])
        res = r[sub] - X[sub] @ coef
        mad = float(np.median(np.abs(res))) * 1.4826 + 1e-6
        keep = sub[np.abs(res) < 3.0 * mad]
        if keep.size < X.shape[1] * 4:
            break
    return r - X @ coef


def _wts(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx / (w - 1)).ravel(), (yy / (h - 1)).ravel()
    return [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v]


def multi_changes(chip, sets):
    """Layer k explained jointly by a set of other layers (each x 4 bilinear weights), brightness,
    the blur difference and x / y gradient of their mean (focus, sub-pixel misregistration).
    sets: {k: [j...]} -> {k: residual (H, W)}."""
    L, h, w = chip.shape
    wts = _wts(h, w)
    sub = np.arange(0, h * w, 2)
    out = {}
    for k, js in sets.items():
        if not js:
            continue
        cols = [chip[j].ravel() - chip[j].mean() for j in js]
        mean = np.mean([chip[j] for j in js], axis=0)
        gy, gx = np.gradient(mean)
        cols += [(blur(mean, 1.0) - mean).ravel(), (blur(mean, 2.0) - mean).ravel(), gx.ravel(), gy.ravel(),
                 np.ones(h * w, np.float32)]
        X = np.stack([c * wj for c in cols for wj in wts], axis=1).astype(np.float32)
        out[k] = _robust_fit(X, chip[k].ravel().astype(np.float32), sub).reshape(h, w)
    return out


def chip_multi(chip):
    """(3, L, H, W): k explained by all other layers (loo), by the layers before it (fwd; k = 0: 0),
    by the layers after it (bwd; k = L-1: 0)."""
    L, h, w = chip.shape
    out = np.zeros((3, L, h, w), np.float32)
    for m, sets in enumerate(({k: [j for j in range(L) if j != k] for k in range(L)},
                              {k: list(range(k)) for k in range(L)},
                              {k: list(range(k + 1, L)) for k in range(L)})):
        for k, r in multi_changes(chip, sets).items():
            out[m, k] = r
    return out


def wafer_multi(X, procs=4):
    L, NR, NC, h, w = X.shape
    chips = [X[:, r, c] for r in range(NR) for c in range(NC)]
    from multiprocessing import Pool
    with Pool(procs) as p:
        out = p.map(chip_multi, chips, chunksize=4)
    return np.stack(out).astype(np.float32)  # (N, 3, L, H, W)
