"""
Method E, stage 2: candidates + features per wafer (numpy only).

From the normalised change maps Z[k, j] (maps.py) blob responses (DoG) at several scales give
D[s, k, j]; normalised again per (s, k, j) by the robust spread over all chips.
Presence of something new on layer k against layer j: |D[s, k, j]|. A void appearing on layer l0 is
in every layer k >= l0 and in none before, so D[k, j] is strong for k >= l0 > j (and for j >= l0 > k)
and weak for k, j < l0.

Candidates: local maxima of A = max over s of (mean over pairs of |D|^2 above the median pair)
per chip; up to NC per chip.
Features (per candidate): the full 8 x 8 |D| matrix at the best scale, per-layer presence at every
scale, split scores for every possible first layer, sign consistency, blob scale, local layout
gradient, die-to-die (chip minus wafer median of the same layer after the layout fit, D2D) response.
"""
import numpy as np

import maps

SCALES = (1.5, 3.0, 6.0)          # DoG sigmas in 140 px chip coordinates
NC = 12                           # candidates per chip
NMS = 6                           # px (140 scale)


def _pool(a):
    h, w = a.shape[-2:]
    return a.reshape(a.shape[:-2] + (h // 2, 2, w // 2, 2)).mean(axis=(-3, -1))


def dog_all(Z, batch=28):
    """Z (N, L, L, H, W) -> D (S, N, L, L, H, W) float16 (scales >= 3 computed at half resolution)."""
    N, L, _, h, w = Z.shape
    D = np.zeros((len(SCALES), N, L, L, h, w), np.float16)
    for b in range(0, N, batch):
        z = Z[b:b + batch]
        full = [s for s in SCALES if s < 3]
        half = [s for s in SCALES if s >= 3]
        if full:
            D[:len(full), b:b + batch] = maps.dog_bank(z, full)
        if half:
            d = maps.dog_bank(_pool(z), [s / 2 for s in half])
            D[len(full):, b:b + batch] = np.repeat(np.repeat(d, 2, axis=-2), 2, axis=-1)
    # normalise per (s, k, j) by the robust spread over chips and pixels
    L = Z.shape[1]
    for si in range(len(SCALES)):
        for k in range(L):
            for j in range(L):
                if k == j:
                    continue
                v = D[si, ::3, k, j, ::4, ::4].astype(np.float32)
                mad = float(np.median(np.abs(v))) * 1.4826 + 1e-6
                D[si, :, k, j] = (D[si, :, k, j].astype(np.float32) / mad).astype(np.float16)
    return D


def die_to_die(X):
    """Chip minus the wafer median chip of the same layer, after fitting the median to the chip."""
    from maps import _bases
    L, NR, NC_, h, w = X.shape
    out = np.zeros((NR * NC_, L, h, w), np.float32)
    sub = np.arange(0, h * w, 2)
    for k in range(L):
        med = np.median(X[k].reshape(-1, h, w), axis=0)
        B = _bases(med)
        o = med.ravel()
        for i in range(NR * NC_):
            r = X[k].reshape(-1, h, w)[i].ravel() - o
            keep = sub
            for _ in range(3):
                Xk = B[keep]
                coef = np.linalg.solve(Xk.T @ Xk + np.eye(B.shape[1], dtype=np.float32) * 1e-3, Xk.T @ r[keep])
                res = r[sub] - B[sub] @ coef
                mad = float(np.median(np.abs(res))) * 1.4826 + 1e-6
                keep = sub[np.abs(res) < 3.0 * mad]
            out[i, k] = (r - B @ coef).reshape(h, w)
    sp = np.quantile(np.abs(out), 0.75, axis=0) / 1.15
    sp = maps.blur(sp, 1.5)
    sp = np.maximum(sp, 0.5 * np.median(sp, axis=(1, 2), keepdims=True))
    out /= sp[None]
    d = np.stack([maps.dog_bank(out, [s]) [0] if s < 3 else
                  np.repeat(np.repeat(maps.dog_bank(_pool(out), [s / 2])[0], 2, -2), 2, -1) for s in SCALES])
    for si in range(len(SCALES)):
        for k in range(L):
            mad = float(np.median(np.abs(d[si, ::3, k, ::4, ::4]))) * 1.4826 + 1e-6
            d[si, :, k] /= mad
    return d.astype(np.float16)  # (S, N, L, H, W)


def presence(D):
    """Per (s, chip, layer k): mean over j != k of |D[s, k, j]| and of |D[s, j, k]| -> (S, N, L, H, W)."""
    S, N, L = D.shape[:3]
    P = np.zeros((S, N, L) + D.shape[-2:], np.float16)
    for si in range(S):
        for k in range(L):
            a = (D[si, :, k].astype(np.float32) ** 2).sum(axis=1) + (D[si, :, :, k].astype(np.float32) ** 2).sum(axis=1)
            P[si, :, k] = np.sqrt(a / (2 * (L - 1))).astype(np.float16)
    return P


def candidates(P):
    """Candidate map per chip (max over scales and layers of the rms presence, each normalised by
    its own median over the wafer) and its local maxima: [(chip, y, x, score)]."""
    S, N, L = P.shape[:3]
    h, w = P.shape[-2:]
    A = np.zeros((N, h, w), np.float32)
    for si in range(S):
        for k in range(L):
            a = P[si, :, k].astype(np.float32)
            A = np.maximum(A, a / (np.median(a[::3, ::4, ::4]) + 1e-6))
    A = maps.blur(A, 1.0)
    # local maxima with NMS
    from numpy.lib.stride_tricks import sliding_window_view
    r = NMS
    pad = np.pad(A, ((0, 0), (r, r), (r, r)), constant_values=-1)
    mx = sliding_window_view(pad, (2 * r + 1, 2 * r + 1), axis=(1, 2)).max(axis=(-1, -2))
    peaks = (A >= mx) & (A > 0)
    peaks[:, :5, :] = peaks[:, -5:, :] = False
    peaks[:, :, :5] = peaks[:, :, -5:] = False
    out = []
    for i in range(N):
        ys, xs = np.nonzero(peaks[i])
        v = A[i, ys, xs]
        o = np.argsort(-v)[:NC]
        out += [(i, int(ys[t]), int(xs[t]), float(v[t])) for t in o]
    return out, A


def split_scores(M):
    """M (L, L) |D| at a point -> for each first layer l: mean of pairs (k >= l > j) both directions,
    mean of pairs before l (clean), mean of pairs after l."""
    L = M.shape[0]
    S = M + M.T
    cross, clean, after = np.zeros(L), np.zeros(L), np.zeros(L)
    for l in range(L):
        a = S[l:, :l]
        cross[l] = a.mean() if a.size else 0.0
        b = S[:l, :l][~np.eye(l, dtype=bool)] if l > 1 else np.zeros(1)
        clean[l] = b.mean()
        c = S[l:, l:][~np.eye(L - l, dtype=bool)] if L - l > 1 else np.zeros(1)
        after[l] = c.mean()
    return cross, clean, after


def candidate_features(cands, D, Dd, P, X, A):
    """One row per candidate (location-level features) and per (candidate, layer) rows."""
    S, N, L = D.shape[:3]
    h, w = D.shape[-2:]
    rows, lrows = [], []
    # layout gradient magnitude (mean over layers of normalised chip)
    NR = X.shape[1]
    for (i, y, x, score) in cands:
        r, c = divmod(i, NR)
        Ms = np.abs(D[:, i, :, :, y, x].astype(np.float32))      # (S, L, L)
        sig = D[:, i, :, :, y, x].astype(np.float32)
        tot = [Ms[si].sum() for si in range(S)]
        sb = int(np.argmax(tot))
        M = Ms[sb]
        cross, clean, after = split_scores(M)
        Pk = P[:, i, :, y, x].astype(np.float32)                # (S, L)
        dd = Dd[:, i, :, y, x].astype(np.float32)               # (S, L)
        # blob extent: radius where A falls to half
        a0 = A[i, y, x]
        prof = [A[i, y, max(0, x - t):x + t + 1].min() for t in range(1, 15)]
        ext = next((t for t, v in enumerate(prof, 1) if v < 0.5 * a0), 15)
        prof2 = [A[i, max(0, y - t):y + t + 1, x].min() for t in range(1, 15)]
        ext2 = next((t for t, v in enumerate(prof2, 1) if v < 0.5 * a0), 15)
        loc = X[:, r, c, max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        lay = X[:, r, c]
        gstd = float(np.mean(loc.std(axis=(1, 2)) / (lay.std(axis=(1, 2)) + 1e-6)))
        f = {"score": score, "logscore": np.log1p(score), "best_scale": sb, "ext_x": ext, "ext_y": ext2,
             "y": y, "x": x, "edge": min(y, x, h - 1 - y, w - 1 - x), "layout_std": gstd,
             "cross_max": cross.max(), "cross_arg": int(cross.argmax()),
             "split_best": float((cross - 0.5 * (clean + after)).max()),
             "pres_max": Pk.max(), "pres_min": Pk.min(), "pres_med": float(np.median(Pk)),
             "dd_max": np.abs(dd).max(), "dd_med": float(np.median(np.abs(dd))),
             "dd_last": float(np.abs(dd[:, -1]).max()),
             "rank": 0}
        for si in range(S):
            f[f"tot_s{si}"] = tot[si] / (L * (L - 1))
            for k in range(L):
                f[f"P_s{si}_k{k}"] = Pk[si, k]
                f[f"dd_s{si}_k{k}"] = abs(dd[si, k])
        for k in range(L):
            for j in range(L):
                if k != j:
                    f[f"M_{k}_{j}"] = M[k, j]
        # sign consistency on the "after first layer" pairs vs layer 0 explanation
        rows.append(f)
        # per-layer rows (first layer hypotheses)
        Pb = Pk[sb]
        for l in range(L):
            g = {"l": l, "cross": cross[l], "clean": clean[l], "after": after[l],
                 "cross_rel": cross[l] / (cross.max() + 1e-6),
                 "split": cross[l] - 0.5 * (clean[l] + after[l]),
                 "split_rel": (cross[l] - clean[l]) / (cross.max() + 1e-6),
                 "pres_l": Pb[l], "pres_prev": Pb[l - 1] if l else 0.0,
                 "pres_before_max": Pb[:l].max() if l else 0.0,
                 "pres_before_mean": Pb[:l].mean() if l else 0.0,
                 "pres_after_min": Pb[l:].min(), "pres_after_mean": Pb[l:].mean(),
                 "jump": Pb[l] - (Pb[:l].max() if l else 0.0),
                 "chg_prev": M[l, l - 1] + M[l - 1, l] if l else 0.0,
                 "chg_from_before_min": min(M[l, j] for j in range(l)) if l else 0.0,
                 "chg_from_before_mean": np.mean([M[l, j] for j in range(l)]) if l else 0.0,
                 "dd_l": float(np.abs(dd[:, l]).max()),
                 "dd_before_max": float(np.abs(dd[:, :l]).max()) if l else 0.0,
                 "dd_after_min": float(np.abs(dd[:, l:]).max(axis=0).min()),
                 }
            for si in range(S):
                g[f"pres_l_s{si}"] = Pk[si, l]
                g[f"pres_before_max_s{si}"] = Pk[si, :l].max() if l else 0.0
            lrows.append(g)
    return rows, lrows


def wafer_features(X, Rw):
    """X (L, R, C, H, W) chips, Rw change maps -> (cands, rows, lrows)."""
    Z = maps.normalise(Rw)
    D = dog_all(Z)
    del Z
    Dd = die_to_die(X)
    P = presence(D)
    cands, A = candidates(P)
    rows, lrows = candidate_features(cands, D, Dd, P, X, A)
    return cands, rows, lrows
