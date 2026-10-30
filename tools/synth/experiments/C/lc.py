"""
Method C: classical layer-change void detection (no learning).

Per chip, layer k is explained from other layers of the same chip (same structure, other intensities)
by a robust per-chip regression with spatially varying weights; the remainder is normalised per pixel by
its chip-to-chip spread on that layer, blobs are found with a multi-scale centre-surround filter, and each
void is assigned to the first layer where it appears.

    python tools/synth/experiments/C/lc.py data ml_204 out.json [--param key=value ...]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))

FACTOR = 2
DEFAULTS = dict(
    log=0,           # regress in log domain (the layer mix is multiplicative: light x mix ^ gamma)
    src="up",        # sources for layer k: "up" = layers 0..k-1 (void-free when k is the first layer),
                     # "all" = all other layers, "down" = deeper layers
    nsrc=7,          # at most this many source layers (nearest first)
    poly=1,          # squared source terms
    deriv=1,         # x / y derivatives of the sources (sub-pixel misregistration between layers)
    blur=1,          # blurred source (sharpness differs)
    grid=2,          # spatial weights: grid x grid bilinear tent functions over the chip
    bias=2,          # polynomial degree of the smooth brightness term
    l0="gold",       # layer 0: explained from the goldens only (with gold=1); "d2d": from the chip's deeper layers
    scales="2,3,4.5,6.5",  # centre-surround sigmas (working px)
    agg="rss",       # evidence over layers for the void's place: "max" or "rss" (root of sum of squares)
    thr=10.0,        # detection threshold on the normalised blob score
    thr0=None,       # layer-0 threshold (default thr) (per-layer mode only)
    tlo=3.0,         # evidence level of a void-free layer ("cusum": first = argmax_l sum_{k>=l} (e_k - tlo))
    first="cusum",   # first-layer evidence: "win" (max near the place), "pt" (at the place), "cusum"
    pw=0,            # half window (working px) of the "pt" / "cusum" evidence
    mode="agg",      # "agg": detect on evidence over layers, then first layer; "layer": per layer
    supp=1.0,        # a later-layer detection within supp * (r1 + r2) of an earlier one is the same void
    border=10,       # ignore this many working px at the chip border
    chipnorm=1,      # also divide each chip's remainder by its own robust spread
    seg=4,           # k regions from the wafer's golden layers (k-means): source x region terms
    cross=0,         # products of neighbouring sources
    gold=1,          # also the wafer's golden chips of all layers (void-free) as sources
)


def segments(X, k, iters=20):
    """Regions of the shared layout: k-means on the golden (median over chips) layer vectors."""
    nl, R, C, h, w = X.shape
    g = np.median(X.reshape(nl, R * C, h, w), axis=1)
    f = ((g - g.mean(axis=(1, 2), keepdims=True)) / (g.std(axis=(1, 2), keepdims=True) + 1e-9)).reshape(nl, -1).T
    rng = np.random.default_rng(0)
    cen = f[rng.choice(len(f), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((f[:, None, :] - cen[None]) ** 2).sum(-1), 1)
        cen = np.stack([f[lab == j].mean(0) if np.any(lab == j) else cen[j] for j in range(k)])
    lab = lab.reshape(h, w)
    return [(lab == j).astype(np.float32) for j in range(k)]


def _gauss(a, s):
    return ndi.gaussian_filter(a, s, mode="nearest")


def _tents(n, h, w):
    """n x n bilinear tent weights over the chip (partition of unity)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for gy in range(n):
        for gx in range(n):
            if n == 1:
                out.append(np.ones((h, w), np.float32))
                continue
            cy, cx = gy * (h - 1) / (n - 1), gx * (w - 1) / (n - 1)
            wy = np.clip(1 - np.abs(yy - cy) / ((h - 1) / (n - 1)), 0, 1)
            wx = np.clip(1 - np.abs(xx - cx) / ((w - 1) / (n - 1)), 0, 1)
            out.append(wy * wx)
    return out


def _bias(deg, h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w - 0.5, yy / h - 0.5
    return [u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)]


def _feats(L, p):
    f = [L - L.mean()]
    if p["poly"]:
        c = L - L.mean()
        f.append(c * c - (c * c).mean())
    if p["deriv"]:
        gy, gx = np.gradient(L)
        f += [gx, gy]
    if p["blur"]:
        f.append(_gauss(L, 1.5) - L)
        if p["blur"] > 1:
            f.append(_gauss(L, 3.0) - L)
    return f


def robust_fit(y, X, iters=4, clip=3.0, sub=2):
    n = y.size
    idx = np.arange(0, n, sub)
    keep = idx
    coef = None
    for _ in range(iters):
        Xk = X[keep]
        A = Xk.T @ Xk
        A += np.eye(A.shape[0], dtype=A.dtype) * (1e-4 * np.trace(A) / A.shape[0])
        coef = np.linalg.solve(A, Xk.T @ y[keep])
        res = y[idx] - X[idx] @ coef
        mad = float(np.median(np.abs(res))) * 1.4826 + 1e-9
        keep = idx[np.abs(res) < clip * mad]
    return y - X @ coef


def chip_remainders(chip, p, segs=None, gold=None):
    """chip: (NL, h, w) -> remainders (NL, h, w); layer 0 is NaN unless p['src'] allows it."""
    nl, h, w = chip.shape
    L = np.log(np.maximum(chip, 1.0)) if p["log"] else chip / 100.0
    tents = _tents(int(p["grid"]), h, w)
    bias = _bias(int(p["bias"]), h, w)
    feats = [_feats(L[j], p) for j in range(nl)]
    gfeats = []
    if gold is not None:
        G = np.log(np.maximum(gold, 1.0)) if p["log"] else gold / 100.0
        for g in G:
            gfeats += [g - g.mean(), _gauss(g, 1.5) - g]
    out = np.full((nl, h, w), np.nan, np.float32)
    for k in (p["only"] if p.get("only") is not None else range(nl)):
        if p["src"] == "up":
            srcs = list(range(k - 1, -1, -1))[: int(p["nsrc"])]
        elif p["src"] == "all":
            srcs = sorted([j for j in range(nl) if j != k], key=lambda j: abs(j - k))[: int(p["nsrc"])]
        elif p["src"] == "down":
            srcs = list(range(k + 1, nl))[: int(p["nsrc"])]
        if not srcs and not gfeats:
            continue
        cols = [b.ravel() for b in bias]
        for f in gfeats:
            for t in tents:
                cols.append((f * t).ravel())
        for j in srcs:
            for f in feats[j]:
                for t in tents:
                    cols.append((f * t).ravel())
        if segs:
            for m in segs[1:]:
                cols.append(m.ravel())
                for f in gfeats[::2]:
                    cols.append((f * m).ravel())
                for j in srcs:
                    cols.append((feats[j][0] * m).ravel())
                    if p["poly"]:
                        cols.append((feats[j][1] * m).ravel())
        if p["cross"]:
            for a, b in zip(srcs[:-1], srcs[1:]):
                cols.append((feats[a][0] * feats[b][0]).ravel())
        X = np.stack(cols, 1).astype(np.float64)
        out[k] = robust_fit(L[k].ravel().astype(np.float64), X).reshape(h, w)
    return out


def _work(args):
    chip, p, segs, gold = args
    return chip_remainders(chip, p, segs, gold)


def _golden(X, p):
    nl, R, C, h, w = X.shape
    return np.median(X.reshape(nl, R * C, h, w), axis=1) if p["gold"] else None


def remainders(X, p, procs=4):
    """X: (NL, R, C, h, w) -> (NL, R, C, h, w) remainders."""
    nl, R, C, h, w = X.shape
    segs = segments(X, int(p["seg"])) if p["seg"] else None
    gold = _golden(X, p)
    jobs = [(X[:, r, c], p, segs, gold) for r in range(R) for c in range(C)]
    with Pool(procs) as pool:
        res = pool.map(_work, jobs, chunksize=4)
    out = np.stack(res, 1).reshape(nl, R, C, h, w)
    return out


def layer0_d2d(X, p, procs=4):
    """Layer 0 has no layer above: explain it from the chip's own deeper layers ('down'), the void
    leaks only partly (other shade per layer). Returns (R, C, h, w)."""
    q = dict(p, src="down", only=[0])
    nl, R, C, h, w = X.shape
    segs = segments(X, int(p["seg"])) if p["seg"] else None
    gold = _golden(X, p)
    jobs = [(X[:, r, c], q, segs, gold) for r in range(R) for c in range(C)]
    with Pool(procs) as pool:
        res = pool.map(_work, jobs, chunksize=4)
    return np.stack([a[0] for a in res], 0).reshape(R, C, h, w)


def normalise(Rm, p):
    """Remainder (R, C, h, w) of one layer -> z map: chip-to-chip spread per pixel."""
    Rm = Rm - np.median(Rm, axis=(0, 1), keepdims=True)
    if p["chipnorm"]:
        s = np.median(np.abs(Rm), axis=(2, 3), keepdims=True) * 1.4826 + 1e-9
        Rm = Rm / s
    spread = np.median(np.abs(Rm), axis=(0, 1), keepdims=True) * 1.4826
    spread = _gauss(spread[0, 0], 1.0)[None, None]
    floor = np.median(spread) * 0.5
    return Rm / np.maximum(spread, floor)


def blob_scores(Z, p):
    """Z (R, C, h, w) -> per scale centre-surround score maps, each normalised to unit spread."""
    out = []
    for s in [float(v) for v in str(p["scales"]).split(",")]:
        c = ndi.gaussian_filter(Z, (0, 0, s, s), mode="nearest")
        su = ndi.gaussian_filter(Z, (0, 0, 2 * s, 2 * s), mode="nearest")
        d = c - su
        b = int(p["border"])
        core = d[:, :, b:-b, b:-b]
        sd = np.median(np.abs(core)) * 1.4826 + 1e-9
        out.append((s, d / sd))
    return out


def detect(Zs, p, thr):
    """Peaks over scales -> [(r, c, y, x, sigma, score)] (working px)."""
    b = int(p["border"])
    stack = np.stack([np.abs(d) for _, d in Zs], 0)  # (S, R, C, h, w)
    sig = np.array([s for s, _ in Zs])
    best = stack.max(0)
    arg = stack.argmax(0)
    best[:, :, :b, :] = 0; best[:, :, -b:, :] = 0; best[:, :, :, :b] = 0; best[:, :, :, -b:] = 0
    mx = ndi.maximum_filter(best, size=(1, 1, 7, 7), mode="constant")
    pk = (best == mx) & (best > thr)
    dets = []
    for r, c, y, x in zip(*np.nonzero(pk)):
        dets.append((int(r), int(c), int(y), int(x), float(sig[arg[r, c, y, x]]), float(best[r, c, y, x])))
    # suppress weaker peaks overlapping a stronger one (same chip)
    dets.sort(key=lambda d: -d[5])
    keep = []
    for d in dets:
        if all(not (k[0] == d[0] and k[1] == d[1] and np.hypot(k[2] - d[2], k[3] - d[3]) < 1.5 * (k[4] + d[4]))
               for k in keep):
            keep.append(d)
    return keep


def layer_scores(X, p, procs=4, cache=None):
    """Per layer: best |centre-surround score| over scales and its sigma, (NL, R, C, h, w) each."""
    nl = X.shape[0]
    if cache is not None and "rem" in cache:
        rem, rem0 = cache["rem"], cache["rem0"]
    else:
        rem = remainders(X, p, procs)
        rem0 = layer0_d2d(X, p, procs) if p["l0"] == "d2d" else None
        if cache is not None:
            cache["rem"], cache["rem0"] = rem, rem0
    key = ("S", p["scales"], p["chipnorm"], p["border"])
    if cache is not None and key in cache:
        return cache[key]
    S = np.zeros(rem.shape, np.float32)
    SIG = np.zeros(rem.shape, np.float32)
    SS = None
    b = int(p["border"])
    for k in range(nl):
        Rm = rem0 if (k == 0 and rem0 is not None) else rem[k]
        Zs = blob_scores(normalise(Rm, p), p)
        st = np.stack([np.abs(d) for _, d in Zs])
        if SS is None:
            SS = np.zeros((nl,) + st.shape, np.float16)
        SS[k] = st
        S[k] = st.max(0)
        SIG[k] = np.array([s for s, _ in Zs], np.float32)[st.argmax(0)]
    S[..., :b, :] = 0; S[..., -b:, :] = 0; S[..., :b] = 0; S[..., -b:] = 0
    if cache is not None:
        cache[key] = (S, SIG, SS)
    return S, SIG, SS


def run(X, p, procs=4, cache=None):
    p = dict(DEFAULTS, **(p or {}))
    S, SIG, SS = layer_scores(X, p, procs, cache)
    if p["mode"] == "layer":
        per_layer = {}
        for k in range(S.shape[0]):
            thr = p["thr0"] if (k == 0 and p["thr0"] is not None) else p["thr"]
            per_layer[k] = _peaks(S[k], SIG[k], float(thr))
        return assign(per_layer, p)
    return assign_agg(S, SIG, p, SS)


def _peaks(A, SIG, thr):
    """Local maxima of A (R, C, h, w) over thr -> [(r, c, y, x, sigma, score)], overlapping weaker dropped."""
    mx = ndi.maximum_filter(A, size=(1, 1, 7, 7), mode="constant")
    dets = [(int(r), int(c), int(y), int(x), float(SIG[r, c, y, x]), float(A[r, c, y, x]))
            for r, c, y, x in zip(*np.nonzero((A == mx) & (A > thr)))]
    dets.sort(key=lambda d: -d[5])
    keep = []
    for d in dets:
        if all(not (k[0] == d[0] and k[1] == d[1] and np.hypot(k[2] - d[2], k[3] - d[3]) < 1.5 * (k[4] + d[4]))
               for k in keep):
            keep.append(d)
    return keep


def assign_agg(S, SIG, p, SS=None):
    """Place from the evidence over all layers; first layer = first layer whose evidence near the
    place reaches tlo (layers above the first are void-free in the 'up' explanation)."""
    nl = S.shape[0]
    if p["agg"] == "rss":
        A = np.sqrt((S ** 2).mean(0)) * np.sqrt(nl) / 2.0
    else:
        A = S.max(0)
    kbest = S.argmax(0)
    SIGA = np.take_along_axis(SIG, kbest[None], 0)[0]
    out = []
    for r, c, y, x, s, sc in _peaks(A, SIGA, float(p["thr"])):
        rad = max(2, int(round(1.5 * s)))
        prof = S[:, r, c, max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1].reshape(nl, -1).max(1)
        if p["first"] != "win":
            pw = int(p["pw"])
            prof = SS[:, :, r, c, max(0, y - pw):y + pw + 1, max(0, x - pw):x + pw + 1].astype(np.float32)
            prof = prof.reshape(nl, -1).max(1)
        kmax = int(prof.argmax())
        if p["first"] == "cusum":
            tail = np.cumsum((prof - p["tlo"])[::-1])[::-1]
            first = int(tail.argmax())
        else:
            first = next((k for k in range(nl) if prof[k] >= p["tlo"]), kmax)
            first = min(first, kmax)
        # size and place at the first layer (later layers have other sizes)
        win = S[first, r, c, max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1]
        dy, dx = np.unravel_index(win.argmax(), win.shape)
        yy, xx = max(0, y - rad) + dy, max(0, x - rad) + dx
        if win.max() < p["tlo"]:
            yy, xx = y, x
        sf = float(SIG[first, r, c, yy, xx])
        rr = 1.6 * sf * FACTOR
        cx, cy = (xx + 0.5) * FACTOR, (yy + 0.5) * FACTOR
        out.append({"layer": first, "row": r, "col": c, "x0": cx - rr, "y0": cy - rr, "x1": cx + rr, "y1": cy + rr,
                    "score": sc})
    return out


def assign(per_layer, p):
    """First layer where a void appears: later detections close to an earlier one are dropped."""
    kept = []
    for k in sorted(per_layer):
        for d in per_layer[k]:
            r, c, y, x, s, sc = d
            if any(e[1] == r and e[2] == c and np.hypot(e[3] - y, e[4] - x) < p["supp"] * 2.0 * (e[5] + s)
                   for e in kept):
                continue
            kept.append((k, r, c, y, x, s, sc))
    out = []
    for k, r, c, y, x, s, sc in kept:
        rad = 1.6 * s * FACTOR
        cx, cy = (x + 0.5) * FACTOR, (y + 0.5) * FACTOR
        out.append({"layer": k, "row": r, "col": c, "x0": cx - rad, "y0": cy - rad, "x1": cx + rad, "y1": cy + rad,
                    "score": sc})
    return out


def _gradmag(a):
    gy, gx = np.gradient(ndi.gaussian_filter(a, 1.0))
    g = np.hypot(gx, gy)
    return g - g.mean()


def register_chip(chip, maxs=2, b=12):
    """chip (NL, H, W) at full resolution: sub-pixel shift of each layer onto layer 0 (gradient
    magnitude correlation: polarity / contrast free; parabolic peak), applied with a cubic spline."""
    ref = _gradmag(chip[0])[b:-b, b:-b]
    out = [chip[0]]
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
        for a, bb, c in ((cc[i - 1, j], cc[i, j], cc[i + 1, j]), (cc[i, j - 1], cc[i, j], cc[i, j + 1])):
            den = a - 2 * bb + c
            sub.append(float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0)
        sh = (i - maxs + sub[0], j - maxs + sub[1])
        out.append(ndi.shift(chip[k], sh, order=3, mode="nearest"))
    return np.stack(out)


def _reg_work(chip):
    r = register_chip(chip)
    nl, H, W = r.shape
    return r.reshape(nl, H // FACTOR, FACTOR, W // FACTOR, FACTOR).mean(axis=(2, 4))


def load(outdir, name, register=True, procs=4):
    """Chips (NL, 14, 14, h, w) at FACTOR; register=True: layers aligned per chip at full resolution first."""
    tag = "_reg" if register else ""
    path = os.path.join(outdir, name + f"_chips{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    from tools.synth.chips import load_chips
    if not register:
        return load_chips(outdir, name, FACTOR)
    X1 = load_chips(outdir, name, 1)
    nl, R, C, H, W = X1.shape
    with Pool(procs) as pool:
        res = pool.map(_reg_work, [X1[:, r, c] for r in range(R) for c in range(C)], chunksize=4)
    X = np.stack(res, 1).reshape(nl, R, C, H // FACTOR, W // FACTOR).astype(np.float32)
    np.save(path, X)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("name"); ap.add_argument("out")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    p = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        p[k] = v
    t0 = time.time()
    X = load(a.outdir, a.name)
    t1 = time.time()
    dets = run(X, p, a.procs)
    json.dump(dets, open(a.out, "w"))
    print(f"{len(dets)} detections, load {t1 - t0:.1f}s, detect {time.time() - t1:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
