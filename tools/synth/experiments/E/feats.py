"""
Method E: candidates + features per wafer (numpy only).

Change maps per chip (maps.py):
  M[m, k]   layer k minus its explanation by other layers of the same chip, jointly (m = 0 all
            other layers "loo", 1 the layers before k "fwd", 2 the layers after k "bwd");
  R[k, j]   layer k minus its explanation by layer j alone (pairwise, optional).
Normalised per (map, pixel) by the spread over all chips of the wafer (layout leftovers sit at the
same place on every chip, a void does not), then blob responses (DoG) at SCALES, normalised again
by their robust spread. Candidates: local maxima of the max over scales / layers of |loo| and |fwd|.
Features per candidate: per-layer profiles (absolute layer index), shape / scale, and per
(candidate, first-layer hypothesis l) rows with relative features (before / at / after l).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import maps

SCALES = (1.5, 3.0, 6.0)
NC = 16
NMS = 6


def _norm_spread(M, axis_px=(-2, -1)):
    sp = np.quantile(np.abs(M), 0.75, axis=0) / 1.15
    sp = maps.blur(sp, 1.5)
    sp = np.maximum(sp, 0.5 * np.median(sp, axis=axis_px, keepdims=True)) + 1e-6
    return M / sp[None]


def _dog(Z):
    """Z (N, ..., H, W) -> (S, N, ..., H, W) float16, each normalised by its robust spread."""
    out = []
    for sc in SCALES:
        d = np.concatenate([maps.dog_bank(Z[b:b + 28], [sc])[0] for b in range(0, len(Z), 28)])
        sub = d[::3, ..., ::4, ::4]
        mad = np.median(np.abs(sub), axis=(0, -2, -1), keepdims=True) * 1.4826 + 1e-6
        out.append((d / mad[0][None]).astype(np.float16))
    return np.stack(out)


def peaks(A, nc=NC, r=NMS, border=4):
    pad = np.pad(A, ((0, 0), (r, r), (r, r)), constant_values=-1)
    mx = sliding_window_view(pad, (2 * r + 1, 2 * r + 1), axis=(1, 2)).max(axis=(-1, -2))
    pk = (A >= mx) & (A > 0)
    pk[:, :border] = pk[:, -border:] = False
    pk[:, :, :border] = pk[:, :, -border:] = False
    out = []
    for i in range(A.shape[0]):
        ys, xs = np.nonzero(pk[i])
        v = A[i, ys, xs]
        o = np.argsort(-v)[:nc]
        out += [(i, int(ys[t]), int(xs[t]), float(v[t]), rank) for rank, t in enumerate(o)]
    return out


def maps_for_wafer(X, pairwise=True, Rw=None, M=None):
    """-> dict of response maps for a wafer X (L, R, C, H, W)."""
    if M is None:
        M = maps.wafer_multi(X)
    M = M.astype(np.float32)
    Z = _norm_spread(M)
    Z[:, 1, 0] = 0  # no layer before the first / after the last
    Z[:, 2, -1] = 0
    Dm = _dog(Z)    # (S, N, 3, L, H, W)
    del Z
    out = {"Dm": Dm}
    if pairwise:
        if Rw is None:
            Rw = maps.wafer_changes(X)
        Zp = maps.normalise(Rw)
        L = Zp.shape[1]
        Dp = _dog(Zp)  # (S, N, L, L, H, W)
        Dp[:, :, np.arange(L), np.arange(L)] = 0
        out["Dp"] = Dp
    A = np.abs(Dm[:, :, :2].astype(np.float32)).max(axis=(0, 2, 3))
    out["A"] = maps.blur(A, 1.0)
    return out


def split_scores(M):
    L = M.shape[0]
    S = M + M.T
    cross, clean, after = np.zeros(L), np.zeros(L), np.zeros(L)
    for l in range(L):
        a = S[l:, :l]
        cross[l] = a.mean() if a.size else 0.0
        clean[l] = S[:l, :l][~np.eye(l, dtype=bool)].mean() if l > 1 else 0.0
        after[l] = S[l:, l:][~np.eye(L - l, dtype=bool)].mean() if L - l > 1 else 0.0
    return cross, clean, after


def candidate_features(cands, mp, X):
    Dm, A = mp["Dm"], mp["A"]
    Dp = mp.get("Dp")
    S, N, _, L, h, w = Dm.shape
    NR = X.shape[1]
    rows, lrows = [], []
    for (i, y, x, score, rank) in cands:
        r, c = divmod(i, NR)
        v = Dm[:, i, :, :, y, x].astype(np.float32)        # (S, 3, L) signed
        av = np.abs(v)
        sb = int(np.argmax(av[:, 0].max(axis=1)))          # best scale on loo
        best = av.max(axis=0)                               # (3, L) max over scales
        a0 = A[i, y, x]
        ext = []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            e = 12
            for t in range(1, 12):
                yy, xx = y + dy * t, x + dx * t
                yy2, xx2 = y - dy * t, x - dx * t
                if not (0 <= yy < h and 0 <= xx < w and 0 <= yy2 < h and 0 <= xx2 < w):
                    break
                if min(A[i, yy, xx], A[i, yy2, xx2]) < 0.5 * a0:
                    e = t
                    break
            ext.append(e)
        loc = X[:, r, c, max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        gstd = float(np.mean(loc.std(axis=(1, 2)) / (X[:, r, c].std(axis=(1, 2)) + 1e-6)))
        loo, fwd, bwd = best
        f = {"score": score, "rank": rank, "best_scale": sb, "ext_min": min(ext), "ext_max": max(ext),
             "ext_ratio": min(ext) / max(ext), "edge": min(y, x, h - 1 - y, w - 1 - x), "layout_std": gstd,
             "loo_max": loo.max(), "loo_med": float(np.median(loo)), "loo_min": loo.min(),
             "fwd_max": fwd[1:].max(), "bwd_max": bwd[:-1].max(),
             "loo_nsig": int((loo > 4).sum()), "loo_last": loo[-1], "loo_first": loo[0],
             "chip_A_med": float(np.median(A[i])), "chip_A_99": float(np.percentile(A[i], 99)),
             "score_rel": score / (float(np.percentile(A[i], 99)) + 1e-6)}
        for si in range(S):
            for m, nm in enumerate(("loo", "fwd", "bwd")):
                for k in range(L):
                    f[f"{nm}_s{si}_k{k}"] = v[si, m, k]
        if Dp is not None:
            P = np.abs(Dp[:, i, :, :, y, x].astype(np.float32))  # (S, L, L)
            Pm = P[sb]
            cross, clean, after = split_scores(Pm)
            pres = (Pm + Pm.T).sum(axis=1) / (2 * (L - 1))
            f.update({"pw_tot": float(Pm.mean()), "pw_cross_max": cross.max(), "pw_cross_arg": int(cross.argmax()),
                      "pw_split_best": float((cross - 0.5 * (clean + after)).max())})
            for k in range(L):
                f[f"pw_pres_k{k}"] = pres[k]
        else:
            cross = clean = after = pres = np.zeros(L)
        rows.append(f)
        for l in range(L):
            g = {"l": l,
                 "loo_l": loo[l], "loo_prev": loo[l - 1] if l else 0.0,
                 "loo_before_max": loo[:l].max() if l else 0.0, "loo_before_mean": loo[:l].mean() if l else 0.0,
                 "loo_after_min": loo[l:].min(), "loo_after_mean": loo[l:].mean(),
                 "fwd_l": fwd[l], "fwd_next": fwd[l + 1] if l + 1 < L else 0.0,
                 "fwd_before_max": fwd[1:l].max() if l > 1 else 0.0,
                 "bwd_prev": bwd[l - 1] if l else 0.0, "bwd_l": bwd[l] if l + 1 < L else 0.0,
                 "bwd_before_max": bwd[:l].max() if l else 0.0,
                 "loo_jump": loo[l] - (loo[:l].max() if l else 0.0),
                 "loo_jump_rel": (loo[l:].mean() - (loo[:l].mean() if l else 0.0)) / (loo.max() + 1e-6),
                 "pw_cross": cross[l], "pw_clean": clean[l], "pw_after": after[l],
                 "pw_split": cross[l] - 0.5 * (clean[l] + after[l]),
                 "pw_cross_rel": cross[l] / (cross.max() + 1e-6),
                 "pw_pres_l": pres[l], "pw_pres_before_max": pres[:l].max() if l else 0.0}
            lrows.append(g)
    return rows, lrows


def wafer_features(X, pairwise=True, Rw=None, M=None):
    mp = maps_for_wafer(X, pairwise, Rw, M)
    cands = peaks(mp["A"])
    rows, lrows = candidate_features(cands, mp, X)
    return cands, rows, lrows
