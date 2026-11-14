"""
Learned convolution filter for Edge Stains: a k x k kernel over the residual map (chip - reference) / spread that
makes the stains stand out and the background (interference fringes, pad beats, scan lines, layout blotches left
after the reference) drop, learned from the painted / reviewed stains.

Matched filter in coloured noise (Fisher LDA): w = (C_bg + ridge)^-1 (mean_stain - mean_bg) over k x k patches.
C_bg^-1 whitens the background: the patterns that the background has (correlated over neighbouring pixels) are
suppressed; the mean difference takes the shape the stains have (thin, wiggly, along the edge: what was painted).
Several weak dark pixels in that shape add up: a stain too faint for Dark per pixel shows in the score.

Score map z = (w * residual - bg mean) / bg std: in background standard deviations (> conv_threshold: a speck).
Training keeps sufficient statistics per chip (count, sum, sum of outer products, per class): chips and files add
up, and a chip reviewed again replaces its old ones.
"""
import os

import numpy as np

K = 9          # Kernel size (working px)
CLIP = 10.0    # Residual clipped to +-CLIP spreads (a bright pad or a void does not dominate the statistics)


def patches(r, ys, xs, k=K):
    """(n, k*k) patches of r centred at (ys, xs) (edges extended)."""
    h = k // 2
    rp = np.pad(np.clip(r, -CLIP, CLIP).astype(np.float32), h, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(rp, (k, k))
    return win[ys, xs].reshape(len(ys), k * k).astype(np.float64)


def chip_stats(r, pos, neg, k=K, max_n=4000, seed=0):
    """Sufficient statistics of one chip: {"p": (n, sum, outer), "n": (...)} of the patches at the pos / neg pixels
    (each subsampled to max_n)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, m in (("p", pos), ("n", neg)):
        ys, xs = np.nonzero(m)
        if len(ys) > max_n:
            i = rng.choice(len(ys), max_n, replace=False)
            ys, xs = ys[i], xs[i]
        X = patches(r, ys, xs, k) if len(ys) else np.zeros((0, k * k))
        out[name] = (float(len(X)), X.sum(0), X.T @ X)
    return out


def save_stats(path, stats):
    """stats {key: chip_stats} -> npz (float32)."""
    arrs = {}
    for key, st in stats.items():
        for c in ("p", "n"):
            n, s, S = st[c]
            arrs[f"{key}#{c}n"] = np.float64(n)
            arrs[f"{key}#{c}s"] = np.asarray(s, np.float64)
            arrs[f"{key}#{c}S"] = np.asarray(S, np.float32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def load_stats(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with np.load(path) as z:
        for name in z.files:
            key, tag = name.rsplit("#", 1)
            st = out.setdefault(key, {"p": [0.0, None, None], "n": [0.0, None, None]})
            i = {"n": 0, "s": 1, "S": 2}[tag[1]]
            st[tag[0]][i] = float(z[name]) if i == 0 else z[name].astype(np.float64)
    return {k_: {c: tuple(v) for c, v in st.items()} for k_, st in out.items()}


def train(stats, k=K, ridge=1e-2):
    """Kernel from the summed statistics. Returns the model dict or raises ValueError (too few examples)."""
    d = k * k
    tot = {c: [0.0, np.zeros(d), np.zeros((d, d))] for c in ("p", "n")}
    for st in stats.values():
        for c in ("p", "n"):
            n, s, S = st[c]
            if n > 0:
                tot[c][0] += n
                tot[c][1] += s
                tot[c][2] += S
    (np_, sp, Sp), (nn, sn, Sn) = tot["p"], tot["n"]
    if np_ < 50 or nn < 500:
        raise ValueError(f"too few pixels: {int(np_)} stain, {int(nn)} background (painted / reviewed stains needed)")
    mp, mn = sp / np_, sn / nn
    Cn = Sn / nn - np.outer(mn, mn)
    Cn += ridge * np.trace(Cn) / d * np.eye(d)
    w = np.linalg.solve(Cn, mp - mn)
    sd = float(np.sqrt(max(w @ Cn @ w, 1e-12)))
    sep = float(w @ (mp - mn)) / sd # Stain mean in background std (the filter)
    c = k * k // 2
    sep_px = float((mp[c] - mn[c]) / np.sqrt(max(Cn[c, c], 1e-12))) # The same for the centre pixel alone
    return {"kernel": (w / sd).reshape(k, k).astype(np.float32), "offset": float(w @ mn / sd), "k": k,
            "sep": sep, "sep_px": sep_px, "n_pos": int(np_), "n_neg": int(nn)}


def save_model(path, model, cfg=""):
    tmp = path + ".tmp.npz"
    np.savez(tmp, kernel=model["kernel"], offset=model["offset"], k=model["k"], sep=model["sep"],
             sep_px=model["sep_px"], n_pos=model["n_pos"], n_neg=model["n_neg"], cfg=np.str_(cfg))
    os.replace(tmp, path)


def load_model(path):
    with np.load(path) as z:
        return {"kernel": z["kernel"].astype(np.float32), "offset": float(z["offset"]), "k": int(z["k"]),
                "sep": float(z["sep"]), "sep_px": float(z["sep_px"]), "cfg": str(z["cfg"])}


def score(r, model):
    """Score map (background std units, > 0: stain-like) of the residual map r: correlation with the kernel (FFT)."""
    kern = model["kernel"]
    k = kern.shape[0]
    h = k // 2
    a = np.pad(np.clip(r, -CLIP, CLIP).astype(np.float32), h, mode="edge")
    H, W = a.shape
    fa = np.fft.rfft2(a)
    fk = np.fft.rfft2(kern[::-1, ::-1], s=(H, W)) # Correlation = convolution with the flipped kernel
    out = np.fft.irfft2(fa * fk, s=(H, W))
    return (out[k - 1:k - 1 + r.shape[0], k - 1:k - 1 + r.shape[1]] - model["offset"]).astype(np.float32)
