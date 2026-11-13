"""
Edge stain detection: contamination along the inside of the chip edge that the void detectors leave
out on purpose. A stain is the chip edge up to a dark, ragged, sparsely specked line (100 .. 1000 px
along the edge), on part of the chips.

Per loaded layer:
1. Every chip is cut from the raw image as it lies (no rotation, no resampling: a pad grid of about
   1 px at a small angle gives no moire), moved onto the median chip by whole image px only (coarse:
   the blurred window with the street around the chip, the outline decides; fine: the centre within
   1 working px), reduced by whole blocks to work_size and normalised (median / spread per chip).
2. Reference: the median chip of the golden chips picked (golden_cells), else of the chips that
   differ least (clean_ref share, at most 64), the chip-to-chip spread per pixel and the median
   chip's local min..max (3x3 or +- shift_tol_px; next to the chip edge along the edge only and
   +- edge_tol_px across it).
3. Specks: darker than the local min by dark_threshold spreads, and / or (compare) the local SSIM
   against the median chip lower by ssim_threshold spreads (texture changed while the mean brightness
   hardly is); dense ones only (speck_density).
4. Per edge strip (depth_frac of the chip, edges): the specks joined into lines (closing longer
   along the edge: merge_radius_px x merge_along), pieces along the edge within link_gap_px are one
   stain; the stain is the chip edge up to the farthest line pixel (gaps interpolated), kept when
   min_length_px long, min_depth_px deep and elongated (min_aspect). Pieces of edges meeting at a
   corner are one stain. Found on several layers: marked once, on the first.

stain_from_strokes(): the same area from lines painted by hand (Paint line in the viewer).
Parameters: config.json "void_detect": {"edge_band": {...}} (DEFAULTS below). trace={"cell": (r, c)}
in params collects every step's image of that chip (Debug chip in the viewer).

Synthetic single-layer wafers (alike 280 px chips, stains on half): 9 wafers 648 / 704 found, 6 false;
a 1 px pad grid at 0.3 deg: 0 false (101 with the former rotated cut), about 15 s per 196 chips.
"""
import math

import numpy as np

DEFAULTS = {
    "depth_frac": 0.25,     # Edge zone: this share of the chip size from the edge inward
    "edge_skip_px": 4,      # Outermost px of the chip left out (bbox / package edge)
    "min_length_px": 100,   # Shortest stain (image px, its longer side)
    "min_width_px": 12,     # Averaging box (image px); about the narrowest stain
    "min_aspect": 1.5,      # Longer / shorter side, at least (elongated along the edge)
    "min_depth_px": 6,      # Shallowest stain (image px from the edge): thinner lines along the edge are not stains
    "shift_tol_px": 0,      # A chip pixel within the median chip's min..max over +- this (image px) is no
                            # difference (at least 3x3 working px): pads / gaps a little off are not stains
    "edge_along_px": 4,     # Within this (image px) of the chip edge the min..max range is taken along the edge
                            # only (the steep step across the package edge does not hide a stain there; 0 = off)
    "edges": "tblr",        # Edges searched: t(op) b(ottom) l(eft) r(ight), e.g. "tb" for top and bottom only
    "match": "none",        # Per-chip intensity matching onto the median chip, measured on the centre: "none"
                            # (best on the synthetic sets), "linear" (gain / offset), "curve" (monotone curve)
    "register_frac": 0.6,   # Chips aligned on their centre part only (this share of the size): stains at
                            # the top / bottom do not pull the alignment
    "blur_px": 0.0,         # Gaussian blur of every chip before the comparison (sigma, image px; 0 = off)
    "zone_share": 0.5,      # Share of the stain inside the edge zone, at least
    "touch_px": 6,          # A stain is attached to the chip edge: its pixels within this (image px) of the edge
    "contact": 0.3,         # ... along at least this share of its length (else: inside the chip, not a stain)
    "threshold": 5.0,       # Averaged deviation / its noise (finds the stains: seeds)
    "dark_threshold": 3.0,  # Segmentation: a dark speck of the rim, in chip-to-chip spreads
    "bright_threshold": 1.5,  # Segmentation: the bright inside (3x3 average), in spreads
    "merge_radius_px": 8,   # Segmentation: dark specks this close (image px, across the edge) are one line
    "merge_along": 2.0,     # ... and this many times farther along the edge (a sparse rim joined into one line)
    "smooth_px": 40,        # Area boundary fitted: its depth smoothed along the edge over this (image px; 0 = off)
    "min_speck_px": 0,      # Before the merge: specks (connected dark px) smaller than this area (image px2) dropped (0 = off)
    "min_piece_px": 0,      # Before the merge: speck pieces shorter than this along the edge (image px) dropped (0 = off)
    "link_gap_px": 40,      # Pieces of a dark line this far apart along the edge (image px) are one stain
    "texture_px": 2.0,      # Pad grid pitch (image px): every chip averaged over a box of it before the comparison
                            # (0 = off). Pads every other px: 2
    "shading": True,        # Brightness gradient across each chip (plane of chip - median chip) taken off
    "edge_tol_px": 1.0,     # Next to the chip edge a chip may sit this far (image px) across the edge off the
                            # median chip (it is cut on whole px): no dark / bright line along the edge
    "bright_min": 0.0,      # Optional: the area between the edge and the line brighter than the chips by this
                            # (spreads, mean) to be a stain; 0 = not checked (the dark line decides)
    "speck_density": 0.06,  # Segmentation: share of specks around a speck (merge radius box), at least:
                            # the rim's specks are dense, scattered noise / layout specks are not
    "clean_ref": 0.5,       # Reference (median chip, spread) from this share of the chips that differ least
                            # (at most 64; 1 = a sample of all): stains at the same place on many chips
                            # (long ones, corners) do not get into the reference
    "golden_cells": None,   # [(row, col)] golden chips picked by hand for the median chip (None: automatic)
    "compare": "both",     # What differs from the median chip: "intensity" (darker), "ssim" (local structure /
                            # texture changed: pads covered or smeared, the mean brightness hardly), "both"
    "ssim_sigma_px": 3.0,   # SSIM window (Gaussian sigma, image px)
    "ssim_threshold": 8.0,   # SSIM: a speck where the structure differs by this many chip-to-chip spreads
    "work_size": 384,       # Longest side of the working chip image
    "type": None,           # Void type of the marks (None: the selected type)
}


def _box(a, h, w):
    """Mean over an h x w box (edges extended), via cumulative sums."""
    if h <= 1 and w <= 1:
        return a
    ph, pw = h // 2, w // 2
    p = np.pad(a, ((ph, h - 1 - ph), (pw, w - 1 - pw)), mode="edge").astype(np.float64)
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]
    return (s / (h * w)).astype(np.float32)


def _gauss(a, sigma):
    """Gaussian blur approximated by three box blurs (cumulative sums: the cost does not grow with sigma)."""
    if sigma <= 0:
        return a
    w = max(1, int(round(math.sqrt(12.0 * sigma * sigma / 3.0 + 1.0))))
    if w <= 1:
        return a
    out = a.astype(np.float32)
    for _ in range(3):
        out = _box(out, w, w)
    return out


def _match_centre(a, golden, centre, n=24):
    """
    a mapped onto golden by a monotone intensity curve: golden's pixels in the centre part grouped in
    n quantile classes, each class' median of `a` mapped to its median of golden (interpolated).
    """
    g, c = golden[centre], a[centre]
    edges = np.quantile(g, np.linspace(0, 1, n + 1))
    cls = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, n - 1)
    xs, ys = [], []
    for i in range(n):
        m = cls == i
        if m.sum() >= 8:
            xs.append(float(np.median(c[m])))
            ys.append(float(np.median(g[m])))
    if len(xs) < 2:
        return a
    order = np.argsort(xs)
    xs, ys = np.maximum.accumulate(np.asarray(xs)[order]), np.asarray(ys)[order]
    xs = xs + np.arange(len(xs)) * 1e-6 # Strictly increasing for interp
    lo, hi = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-6), (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-6)
    out = np.interp(a, xs, ys)
    out = np.where(a < xs[0], ys[0] + (a - xs[0]) * lo, out) # Linear beyond the ends
    return np.where(a > xs[-1], ys[-1] + (a - xs[-1]) * hi, out).astype(np.float32)


def _match_linear(a, golden, centre):
    """a mapped onto golden by gain / offset measured on the centre part (robust: 20 / 80 % quantile classes)."""
    g, c = golden[centre], a[centre]
    q = np.quantile(g, [0.1, 0.3, 0.7, 0.9])
    lo, hi = (g >= q[0]) & (g < q[1]), (g >= q[2]) & (g < q[3])
    if lo.sum() < 8 or hi.sum() < 8:
        return a
    x0, x1, y0, y1 = np.median(c[lo]), np.median(c[hi]), np.median(g[lo]), np.median(g[hi])
    k = (y1 - y0) / (x1 - x0) if abs(x1 - x0) > 1e-6 else 1.0
    return ((a - x0) * k + y0).astype(np.float32)


def _register_centre(chips, frac=0.6):
    """
    Chips moved onto the median chip (subpixel phase correlation), the shift measured on the centre
    part only (frac of the size): a large stain along an edge does not pull the chip off.
    """
    from .void_detect import _phase_shift, _translate
    keys = list(chips)
    h, w = next(iter(chips.values())).shape
    mh, mw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    cen = lambda a: a[mh:h - mh, mw:w - mw]
    idx = np.linspace(0, len(keys) - 1, min(len(keys), 48)).astype(int)
    out = chips
    for _ in range(2): # Second pass against the median of the moved chips
        ref = np.median(np.stack([cen(out[keys[i]]).astype(np.float32) for i in idx]), axis=0)
        moved = {}
        for k, c in chips.items():
            a = c.astype(np.float32)
            dx, dy = _phase_shift(cen(a), ref)
            if not (math.isfinite(dx) and math.isfinite(dy)) or math.hypot(dx, dy) > 0.1 * max(h, w):
                dx = dy = 0.0
            moved[k] = _translate(a, dx, dy) if (dx or dy) else a
        out = moved
    return out


def _fill_to_edge(part, side):
    """
    Mask filled from the chip edge (`side` of the window) to its farthest pixel at every place along
    the edge; places without pixels between two with pixels take the interpolated depth.
    """
    a = part if side in ("top", "bottom") else part.T
    if side in ("bottom", "right"):
        a = a[::-1]
    n, m = a.shape
    has = a.any(axis=0)
    if not has.any():
        return part
    depth = np.where(has, n - np.argmax(a[::-1], axis=0), 0).astype(np.float32) # Farthest row + 1
    idx = np.nonzero(has)[0]
    cols = np.arange(m)
    inside = (cols >= idx[0]) & (cols <= idx[-1])
    depth = np.where(inside, np.interp(cols, idx, depth[idx]), 0)
    out = np.arange(n)[:, None] < np.round(depth)[None, :]
    if side in ("bottom", "right"):
        out = out[::-1]
    return out if side in ("top", "bottom") else out.T


def _read(stack, z, x, y, w, h):
    """Raw window (x, y, w, h) of layer z as float32, the part outside the image edge-extended; None outside."""
    from .core_data import to_gray
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(stack.width, x + w), min(stack.height, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    a = to_gray(stack.read_region(z, x0, y0, x1 - x0, y1 - y0)).astype(np.float32)
    if a.shape != (y1 - y0, x1 - x0):
        return None
    return np.pad(a, ((y0 - y, y + h - y1), (x0 - x, x + w - x1)), mode="edge")


def _reduce(a, f):
    """Mean over f x f blocks (whole blocks only)."""
    if f <= 1:
        return a
    h, w = a.shape[0] // f, a.shape[1] // f
    return a[:h * f, :w * f].reshape(h, f, w, f).mean(axis=(1, 3))


def _label(mask):
    """Connected components (8-neighbour) of a boolean mask: (label image, area per label); label 0 = background
    (area 0)."""
    h, w = mask.shape
    runs = []
    for r in np.nonzero(mask.any(axis=1))[0]:
        d = np.diff(np.concatenate(([0], mask[r].astype(np.int8), [0])))
        for c0, c1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            runs.append((int(r), int(c0), int(c1)))
    parent = list(range(len(runs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    by_row = {}
    for i, (r, c0, c1) in enumerate(runs):
        for j in by_row.get(r - 1, ()):
            _, d0, d1 = runs[j]
            if c0 <= d1 and d0 <= c1: # Touching diagonally too
                parent[find(i)] = find(j)
        by_row.setdefault(r, []).append(i)
    lab = np.zeros((h, w), np.int32)
    ids = {}
    areas = [0]
    for i, (r, c0, c1) in enumerate(runs):
        root = find(i)
        if root not in ids:
            ids[root] = len(areas)
            areas.append(0)
        lab[r, c0:c1] = ids[root]
        areas[ids[root]] += c1 - c0
    return lab, np.asarray(areas, np.float64)


def _orient(m, side, back=False):
    """m turned so the chip edge `side` is at the top (rows: distance from the edge, columns: along it); back=True undoes it."""
    if side == "top":
        return m
    if side == "bottom":
        return m[::-1]
    if side == "left":
        return m.T
    return m[:, ::-1].T if not back else m.T[:, ::-1]


def _runs(has, gap):
    """[(c0, c1)] runs of True in a 1-D mask, False gaps up to `gap` long between two True joined."""
    idx = np.nonzero(has)[0]
    if len(idx) == 0:
        return []
    out, s0, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > gap:
            out.append((int(s0), int(prev) + 1))
            s0 = i
        prev = i
    out.append((int(s0), int(prev) + 1))
    return out


def _tukey(n, t):
    """1-D window: 1 in the middle, a cosine fade over the outer t samples at both ends."""
    w = np.ones(n, np.float32)
    t = int(min(t, n // 2))
    if t > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(t) + 0.5) / t)
        w[:t], w[n - t:] = ramp, ramp[::-1]
    return w


def _blur1(v, sigma):
    """Gaussian blur of a 1-D profile (edges extended)."""
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(v, r, mode="edge"), k / k.sum(), mode="valid")


def _corr1(g, ref, lim):
    """Shift d (|d| <= lim, subpixel) with g[i + d] best matching ref[i] (correlation of two profiles)."""
    n = len(g)
    best = None
    sc = {}
    for d in range(-lim, lim + 1):
        a, b = max(0, -d), min(n, n - d)
        sc[d] = float(np.dot(g[a + d:b + d], ref[a:b]))
        if best is None or sc[d] > sc[best]:
            best = d
    if -lim < best < lim:
        cm, c0, cp = sc[best - 1], sc[best], sc[best + 1]
        den = cm - 2 * c0 + cp
        if den < 0:
            return best + float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))
    return float(best)


def _peak_near(a, b, dx0, dy0, rad, hann=True):
    """
    Translation of a against b (as _phase_shift), the correlation peak searched within rad px of
    (dx0, dy0) (rad None: anywhere). hann=False: no window (the image border holds what decides).
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32) if hann else 1.0
    R = np.fft.rfft2((a - a.mean()) * win) * np.conj(np.fft.rfft2((b - b.mean()) * win))
    R /= np.abs(R) ** 0.5 + 1e-9
    r = np.fft.irfft2(R, s=a.shape)
    h, w = r.shape
    if rad is None:
        iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
        iy, ix = (iy - h if iy > h // 2 else iy), (ix - w if ix > w // 2 else ix)
    else:
        best = None
        for iy in range(int(round(dy0)) - rad, int(round(dy0)) + rad + 1):
            for ix in range(int(round(dx0)) - rad, int(round(dx0)) + rad + 1):
                v = r[iy % h, ix % w]
                if best is None or v > best[0]:
                    best = (v, ix, iy)
        _, ix, iy = best

    def sub(c0, cm, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (cm - cp) / d, -0.5, 0.5))
    c0 = r[iy % h, ix % w]
    return (ix + sub(c0, r[iy % h, (ix - 1) % w], r[iy % h, (ix + 1) % w]),
            iy + sub(c0, r[(iy - 1) % h, ix % w], r[(iy + 1) % h, ix % w]))


def _raw_chips(stack, z, origin, Wr, Hr, f, M, frac, sample_n=48, keep=None, tick=None):
    """
    Chips cut from the raw image as they lie (no rotation, no resampling: a pad grid of about 1 px at
    a small angle gives no moire), moved onto the median chip by whole image px only (measured by
    phase correlation on the centre part, frac of the size), then reduced by f x f block means (the
    blocks sit on the same layout place in every chip).
    origin: {cell: (x, y)} raw top-left of the chip window (Wr x Hr, nominal). Returns
    {cell: (chip (Hr // f, Wr // f) float32, (x, y) raw top-left actually cut)}, or None when
    tick(stage, i, n) (called per chip: "sample", "cut") returned False.
    """
    from .void_detect import _blur, _phase_shift
    H, W = Hr // f, Wr // f
    mh, mw = int(H * (1 - frac) / 2), int(W * (1 - frac) / 2)
    cen = lambda a: a[mh:H - mh, mw:W - mw]

    def windows(cells, span):
        """(cell, raw window with margin M) read row by row, up to `span` chips per read."""
        rows = {}
        for c in cells:
            rows.setdefault(c[0], []).append(c)
        for r in sorted(rows):
            row = sorted(rows[r], key=lambda c: origin[c][0])
            i = 0
            while i < len(row):
                j = i + 1
                while j < len(row) and origin[row[j]][0] - origin[row[i]][0] < span * Wr:
                    j += 1
                chunk = row[i:j]
                X0 = min(origin[c][0] for c in chunk) - M
                Y0 = min(origin[c][1] for c in chunk) - M
                X1 = max(origin[c][0] for c in chunk) + Wr + M
                Y1 = max(origin[c][1] for c in chunk) + Hr + M
                big = _read(stack, z, X0, Y0, X1 - X0, Y1 - Y0)
                for c in chunk:
                    x, y = origin[c][0] - M - X0, origin[c][1] - M - Y0
                    yield c, (None if big is None else big[y:y + Hr + 2 * M, x:x + Wr + 2 * M])
                i = j

    def crop(win, sx, sy):
        sx, sy = int(np.clip(sx, -M, M)), int(np.clip(sy, -M, M))
        return _reduce(win[M + sy:M + sy + Hr, M + sx:M + sx + Wr], f).astype(np.float32)

    sig = max(1.5, max(H, W) / 100.0) # Working px: blur for the coarse alignment (a pad grid is gone)
    hw, ww = Hr // f + 2 * (M // f), Wr // f + 2 * (M // f)
    taper = np.outer(_tukey(hw, M // f), _tukey(ww, M // f)).astype(np.float32)

    def whole(win):
        """The window (chip and the street around it) reduced and blurred, faded out over the margin only:
        the chip outline and its large structures decide the coarse alignment, a pad grid does not."""
        a = _gauss(_reduce(win, f).astype(np.float32)[:hw, :ww], sig)
        return (a - a.mean()) * taper

    def shift(win, ref):
        refb, refc = ref
        dx, dy = _peak_near(whole(win), refb, 0, 0, None, hann=False)
        sx, sy = int(round(dx * f)), int(round(dy * f))
        a = crop(win, sx, sy)
        fx, fy = _peak_near(cen(a), refc, 0, 0, 1) # Fine: the centre, within 1 working px
        return int(np.clip(round(sx + fx * f), -M, M)), int(np.clip(round(sy + fy * f), -M, M))

    def make_ref(items):
        """items: [(window, (sx, sy))] -> (median whole window, median centre), moved by the shifts."""
        wh = []
        for w, (sx, sy) in items:
            m = np.pad(w, M, mode="edge")[M + sy:M + sy + w.shape[0], M + sx:M + sx + w.shape[1]]
            wh.append(whole(m))
        return (np.median(np.stack(wh), axis=0),
                np.median(np.stack([cen(crop(w, sx, sy)) for w, (sx, sy) in items]), axis=0))

    cells = sorted(origin)
    pick = [cells[i] for i in sorted(set(np.linspace(0, len(cells) - 1, min(len(cells), sample_n)).astype(int)))]
    sample = {}
    for i, (c, w) in enumerate(windows(pick, 1)):
        if w is not None:
            sample[c] = w
        if tick is not None and not tick("sample", i + 1, len(pick)):
            return None
    if len(sample) < 4:
        return {}
    ref = make_ref([(w, (0, 0)) for w in sample.values()])
    ref = make_ref([(w, shift(w, ref)) for w in sample.values()])
    out = {}
    for i, (c, win) in enumerate(windows(cells, 16)):
        if tick is not None and not tick("cut", i + 1, len(cells)):
            return None
        if win is None:
            continue
        sx, sy = shift(win, ref)
        out[c] = (crop(win, sx, sy), (origin[c][0] + sx, origin[c][1] + sy))
        if keep is not None and c == keep.get("cell"): # Debug trace: the raw window and what was cut
            keep["window"] = win.copy()
            keep["shift"] = (sx, sy)
            keep["M"] = M
    stack.release_cache()
    return out


def _range_along(a, k, axis):
    """min, max over k px along one axis (edges extended)."""
    r = k // 2
    p = np.pad(a, [(r, r) if ax == axis else (0, 0) for ax in range(2)], mode="edge")
    n = a.shape[axis]
    parts = np.stack([np.take(p, np.arange(i, i + n), axis=axis) for i in range(k)])
    return parts.min(axis=0), parts.max(axis=0)


def detect(stack, zs, layer_numbers, grid_cfg, coord_transform, refs, cells, params=None, progress=None, masks=None):
    """
    Returns [(layer, gx, gy, rx, ry, cell, score, outline)] (Deskewed centre / half sizes, outline:
    [(gx, gy)] Deskewed points of the segmented area) or None when cancelled.
    masks (dict): filled with the joined dark lines (red) and the stain areas found (cyan) in the edge
    strips (first searched layer): {cell: ((x0, y0, w, h) Deskewed, (H, W), packed line, packed area)}.
    progress(i, n, text) -> False cancels.
    """
    from .void_detect import _blur, _components, _filt, _normalize, chip_rects
    from PIL import ImageFilter
    from .void_layer_change import _contour
    p = dict(DEFAULTS)
    p.update(params or {})
    to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))
    to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))
    rects = chip_rects(grid_cfg, refs, cells)
    if len(rects) < 4:
        return []
    bw = float(np.median([v[2] for v in rects.values()]))
    bh = float(np.median([v[3] for v in rects.values()]))
    f = max(1, int(math.ceil(max(bw, bh) / float(p["work_size"])))) # Whole reduction steps
    W, H = max(16, int(bw) // f), max(16, int(bh) // f)
    Wr, Hr = W * f, H * f # Raw window of a chip (whole blocks)
    M = max(8, int(round(0.1 * max(Wr, Hr)))) # Margin: alignment range, and the street around the chip for it (raw px)
    origin = {}
    for cell, (x0, y0, w, h) in rects.items():
        cx, cy = to_raw(x0 + w / 2.0, y0 + h / 2.0)
        origin[cell] = (int(round(cx - Wr / 2.0)), int(round(cy - Hr / 2.0)))
    sx = sy = float(f) # Image px per working px
    depth_x, depth_y = int(round(p["depth_frac"] * W)), int(round(p["depth_frac"] * H))
    skip_x, skip_y = int(math.ceil(p["edge_skip_px"] / sx)), int(math.ceil(p["edge_skip_px"] / sy))
    yy, xx = np.mgrid[0:H, 0:W]
    inner = (yy >= skip_y) & (yy < H - skip_y) & (xx >= skip_x) & (xx < W - skip_x)
    strips = [("x", inner & (yy < depth_y), "top"), ("x", inner & (yy >= H - depth_y), "bottom"),
              ("y", inner & (xx < depth_x), "left"), ("y", inner & (xx >= W - depth_x), "right")]
    strips = [st for st in strips if st[2][0] in str(p["edges"]).lower()] # Only the edges asked for
    k_dens = max(3, int(round(p["merge_radius_px"] / min(sx, sy))) | 1)
    mh, mw = int(H * 0.2), int(W * 0.2)
    centre = np.zeros((H, W), bool) # Intensity matching measured here (no edge stains)
    centre[mh:H - mh, mw:W - mw] = True
    edges = np.zeros((H, W), bool) # Union of the edge strips
    for _, strip, _ in strips:
        edges |= strip
    k_merge = 2 * max(1, int(round(p["merge_radius_px"] / min(sx, sy)))) + 1 # Corner joining (working px)
    k_al_x = 2 * max(1, int(round(p["merge_along"] * p["merge_radius_px"] / sx))) + 1 # Along x (top / bottom)
    k_al_y = 2 * max(1, int(round(p["merge_along"] * p["merge_radius_px"] / sy))) + 1 # Along y (left / right)
    k_ac_x = 2 * max(1, int(round(p["merge_radius_px"] / sx))) + 1
    k_ac_y = 2 * max(1, int(round(p["merge_radius_px"] / sy))) + 1

    def line_closing(m, along):
        """Dilate then erode by a box longer along the edge (merge_along x merge radius) than across."""
        h, w = (k_ac_y, k_al_x) if along == "x" else (k_al_y, k_ac_x)
        d = _box(m.astype(np.float32), h, w) > 1e-6
        return _box(d.astype(np.float32), h, w) > 1 - 1e-6

    order = sorted(rects)
    found = []
    taken = {} # cell -> [(x0, y0, x1, y1)] Deskewed boxes already marked (first layer wins)
    # Progress per layer: sample (<= 48 chips), cut (every chip), reference (every chip), detect (every chip)
    N = len(order)
    n_samp = min(N, 48)
    per_layer = n_samp + 3 * N + 2
    n_steps = len(zs) * per_layer
    stage_at = {"sample": 0, "cut": n_samp, "reference": n_samp + N, "detect": n_samp + 2 * N + 1}
    stage_txt = {"sample": "aligning (sample chips)", "cut": "cutting chips", "reference": "reference chip",
                 "detect": "finding lines"}
    import time as _time
    last = [0.0]

    def tick(li, layer, stage, i, n, force=False):
        """progress(...) at most every 0.05 s (and at the end of a stage): False cancels."""
        if progress is None:
            return True
        now = _time.monotonic()
        if not force and i < n and now - last[0] < 0.05:
            return True
        last[0] = now
        lay = f"layer {layer} ({li + 1}/{len(zs)}), " if len(zs) > 1 else f"layer {layer}, "
        return progress(li * per_layer + stage_at[stage] + i, n_steps,
                        f"Edge stains: {lay}{stage_txt[stage]} {i} / {n}")
    for li, (z, layer) in enumerate(zip(zs, layer_numbers)):
        if not tick(li, layer, "sample", 0, n_samp, True):
            return None
        trace = p.get("trace") if li == 0 else None # Debug: every step of one chip ({"cell": (row, col)})
        steps = trace.setdefault("steps", []) if trace is not None else None
        tk = trace.get("cell") if trace is not None else None
        cut = _raw_chips(stack, z, origin, Wr, Hr, f, M, p["register_frac"], keep=trace,
                         tick=lambda stage, i, n, li=li, layer=layer: tick(li, layer, stage, i, n))
        if cut is None:
            return None
        if len(cut) < 4:
            continue
        keys = [k for k in order if k in cut]
        at = {k: cut[k][1] for k in keys} # Raw top-left of every chip as cut
        chips = {k: _normalize(cut[k][0]) for k in keys}
        if tk in cut:
            Mw = trace["M"]
            win = trace.pop("window")
            steps.append(("1 raw window (margin %d px, image px)" % Mw, win, "gray"))
            steps.append(("2 cut: moved %+d, %+d px (whole px, no resampling)" % trace["shift"],
                          win[Mw + trace["shift"][1]:Mw + trace["shift"][1] + Hr, Mw + trace["shift"][0]:Mw + trace["shift"][0] + Wr], "gray"))
            steps.append(("3 reduced %dx%d blocks, normalised" % (f, f), chips[tk], "gray"))
        del cut
        use_ssim = p["compare"] in ("ssim", "both")
        # SSIM looks at the texture (pads covered / smeared): it gets the chips before the pad grid is averaged
        tex = {k: c.astype(np.float16) for k, c in chips.items()} if use_ssim and p["texture_px"] > 0 else None
        if p["texture_px"] > 0:
            # Pad grid averaged out (box of the pad pitch): the median chip has the pads smeared (chips sit a fraction
            # of a px apart), a single chip has them sharp, so between its pads it read darker everywhere, in a grid
            kt = max(2, int(round(p["texture_px"] / f)))
            chips = {k: _box(c, kt, kt) for k, c in chips.items()}
            if tk in chips:
                steps.append(("3b pad grid averaged (Texture %g px)" % p["texture_px"], chips[tk], "gray"))
        if p["blur_px"] > 0: # Gaussian blur (noise down; specks of the rim spread, keep it below their spacing)
            sig = p["blur_px"] / min(sx, sy)
            chips = {k: _gauss(c, sig) for k, c in chips.items()}
            if tk in chips:
                steps.append(("4 blurred (Blur %.1f px)" % p["blur_px"], chips[tk], "gray"))
        if p["shading"]:
            # Brightness gradient across a chip (illumination over the wafer): the plane of (chip - median chip),
            # fitted robustly over the whole chip (stains are a small part: trimmed), is taken off every chip
            samp = [keys[i] for i in sorted(set(np.linspace(0, len(keys) - 1, min(len(keys), 64)).astype(int)))]
            g0 = np.median(np.stack([chips[k] for k in samp]), axis=0)
            st_ = max(1, min(H, W) // 64)
            Y, X = np.mgrid[0:H:st_, 0:W:st_]
            A = np.stack([np.ones(X.size), (X.ravel() - W / 2) / W, (Y.ravel() - H / 2) / H], 1)
            Xf = (np.arange(W)[None, :] - W / 2) / W
            Yf = (np.arange(H)[:, None] - H / 2) / H
            for k in keys:
                d = (chips[k] - g0)[::st_, ::st_].ravel()
                keep = np.ones(d.size, bool)
                for _ in range(3):
                    coef = np.linalg.lstsq(A[keep], d[keep], rcond=None)[0]
                    res = d - A @ coef
                    mad = float(np.median(np.abs(res[keep]))) * 1.4826 + 1e-6
                    keep = np.abs(res) <= 2.5 * mad
                plane = coef[0] + coef[1] * Xf + coef[2] * Yf
                chips[k] = (chips[k] - plane).astype(np.float32)
                if tex is not None:
                    tex[k] = (tex[k].astype(np.float32) - plane).astype(np.float16)
            if tk in chips:
                steps.append(("4b brightness gradient taken off (Shading)", chips[tk], "gray"))
        mode = p["match"]
        fn = None if mode == "none" else (_match_centre if mode == "curve" else _match_linear)
        k_tol = max(3, 2 * int(round(p["shift_tol_px"] / min(sx, sy))) + 1)
        e_y = int(round(p["edge_along_px"] / sy)) + skip_y
        e_x = int(round(p["edge_along_px"] / sx)) + skip_x
        near_tb = (yy < e_y) | (yy >= H - e_y)
        near_lr = (xx < e_x) | (xx >= W - e_x)

        def reference(sel, spread_from=None):
            """Median chip of the chips `sel`, the chip-to-chip spread (of `spread_from`, default `sel`) and the
            median chip's min..max."""
            B0 = np.stack([chips[k] for k in sel])
            golden = np.median(B0, axis=0)
            if fn is not None:
                golden = np.median(np.stack([fn(a, golden, centre) for a in B0]), axis=0)
            if spread_from is not None:
                B0 = np.stack([chips[k] for k in spread_from])
            if fn is not None:
                B0 = np.stack([fn(a, golden, centre) for a in B0])
            spread = np.median(np.abs(B0 - golden), axis=0) * 1.4826
            spread = _filt(spread, ImageFilter.MaxFilter(3))
            spread = np.maximum(spread, 0.5 * float(np.median(spread)) + 1e-3)
            if p["shift_tol_px"] <= 0:
                # Chips are cut within +- 0.5 image px of each other: the reference's range is its values moved by
                # +- edge_tol_px (image px, interpolated) in the 8 directions. A 3 x 3 working px min..max spans
                # f image px: with a pad grid of about 1 px it is the pad level everywhere, and a chip a little
                # brighter / darker between the pads (the inside of a stain) stays inside it
                t = min(1.0, p["edge_tol_px"] / f)
                gp = np.pad(golden, 1, mode="edge")
                gmin, gmax = golden.copy(), golden.copy()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx or dy:
                            sh = (1 - t) * golden + t * gp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                            np.minimum(gmin, sh, out=gmin)
                            np.maximum(gmax, sh, out=gmax)
                return golden, spread, gmin, gmax
            gmin = _filt(golden, ImageFilter.MinFilter(k_tol))
            gmax = _filt(golden, ImageFilter.MaxFilter(k_tol))
            if p["edge_along_px"] > 0:
                # Next to the chip edge the median chip changes steeply across the edge (package edge): a
                # min..max over a square spans that step and hides a stain right at the edge. There the
                # range is taken along the edge only; in the corners (both edges near) the narrower of the two
                rmin, rmax = _range_along(golden, k_tol, 1)
                cmin, cmax = _range_along(golden, k_tol, 0)
                t = min(1.0, p["edge_tol_px"] / f) # Across the edge: +- edge_tol_px (whole chips are cut on whole px)
                if t > 0:
                    for axis, (lo, hi) in ((0, (rmin, rmax)), (1, (cmin, cmax))):
                        g = np.pad(golden, [(1, 1) if ax == axis else (0, 0) for ax in range(2)], mode="edge")
                        n = golden.shape[axis]
                        for d in (0, 2):
                            sh = (1 - t) * golden + t * np.take(g, np.arange(d, d + n), axis=axis)
                            np.minimum(lo, sh, out=lo)
                            np.maximum(hi, sh, out=hi)
                both = near_tb & near_lr
                gmin = np.where(both, np.maximum(rmin, cmin), np.where(near_tb, rmin, np.where(near_lr, cmin, gmin)))
                gmax = np.where(both, np.minimum(rmax, cmax), np.where(near_tb, rmax, np.where(near_lr, cmax, gmax)))
            return golden, spread, gmin, gmax

        def dev_of(k, ref):
            golden, spread, gmin, gmax = ref[:4]
            a = chips[k] if fn is None else fn(chips[k], golden, centre)
            return (np.maximum(a - gmax, 0) - np.maximum(gmin - a, 0)) / spread # > 0 brighter, < 0 darker

        ss_sig = max(0.8, p["ssim_sigma_px"] / min(sx, sy))

        def tex_of(k):
            return chips[k] if tex is None else tex[k].astype(np.float32)

        def dissim(k, golden):
            """Texture change: 1 - the contrast part of SSIM (local std of the chip, pad grid not averaged, against the
            median of such chips; Gaussian window ssim_sigma_px). Pads covered or smeared lower the local contrast;
            pads a fraction of a px off (chips are cut on whole px) do not change it, while the structure part of
            SSIM read them as a change everywhere on the pads."""
            a = tex_of(k)
            L = float(np.percentile(golden, 99) - np.percentile(golden, 1)) + 1e-3
            c2 = (0.03 * L) ** 2
            ma, mb = _gauss(a, ss_sig), _gauss(golden, ss_sig)
            va = np.maximum(_gauss(a * a, ss_sig) - ma * ma, 0)
            vb = np.maximum(_gauss(golden * golden, ss_sig) - mb * mb, 0)
            con = (2 * np.sqrt(va * vb) + c2) / (va + vb + c2)
            return (1.0 - con).astype(np.float32)

        def with_ssim(ref, sel, per_strip=None):
            """ref + (median, spread) of the chips' dissimilarity (the SSIM map in chip-to-chip spreads) + the median
            chip of the un-averaged chips (per edge strip from that strip's cleanest chips, per_strip {mask: chips})."""
            if not use_ssim:
                return ref
            g_t = np.median(np.stack([tex_of(k) for k in sel]), axis=0)
            for m, sel_s in (per_strip or []):
                g_t = np.where(m, np.median(np.stack([tex_of(k) for k in sel_s]), axis=0), g_t)
            D = np.stack([dissim(k, g_t) for k in sel])
            med = np.median(D, axis=0)
            spr = np.median(np.abs(D - med), axis=0) * 1.4826
            spr = np.maximum(_filt(spr, ImageFilter.MaxFilter(3)), 0.5 * float(np.median(spr)) + 1e-4)
            return ref + (med, spr, g_t)

        # Reference from a sample of the chips, then again from the cleanest of them (clean_ref share, at
        # most 64): where many chips are stained at the same place (long stains, corners) the median of
        # all would hold the stain itself and hide it
        n_ref = min(len(keys), 64)
        sample = [keys[i] for i in sorted(set(np.linspace(0, len(keys) - 1, n_ref).astype(int)))]
        # Golden chips picked by hand (golden_cells): the median chip is theirs; the spread still comes from
        # the cleanest chips (a few golden chips cannot tell the chip-to-chip spread)
        gold = [k for k in (tuple(c) for c in (p.get("golden_cells") or [])) if k in chips]
        ref = reference(gold or sample, None if gold else sample)
        clean = sample
        strip_clean = {}
        if p["clean_ref"] < 1.0:
            # Cleanest chips per edge strip (a chip stained at the top may be clean at the bottom): each strip's part
            # of the reference comes from the chips cleanest there, the rest from the chips cleanest overall.
            # Twice: the second pass measures against the reference of the first
            n_clean = min(64, max(min(8, len(keys)), int(round(p["clean_ref"] * len(keys)))))
            regions = [(side, strip) for _, strip, side in strips]
            for it in range(2):
                dirt = {k: [] for k in keys}
                for i, k in enumerate(keys):
                    a = np.clip(np.abs(dev_of(k, ref)), 0, 10)
                    dirt[k] = [float(a[edges].mean())] + [float(a[m].mean()) for _, m in regions]
                    if not tick(li, layer, "reference", it * len(keys) // 2 + (i + 1) // 2, len(keys)):
                        return None
                def cleanest(j):
                    """The chips cleanest by dirt[.][j]: those within 1.5 x the 10 % quantile (the clean cluster:
                    stained chips are well above it), at least 8, at most n_clean"""
                    order_ = sorted(keys, key=lambda k: dirt[k][j])
                    q = dirt[order_[max(0, int(0.1 * len(order_)) - 1)]][j]
                    n = sum(1 for k in order_ if dirt[k][j] <= 1.5 * q + 1e-6)
                    return order_[:max(min(8, len(order_)), min(n_clean, n))]
                clean = cleanest(0)
                new = reference(gold or clean, clean)
                strip_clean = {}
                for j, (side, m) in enumerate(regions):
                    sel = cleanest(j + 1)
                    strip_clean[side] = sel
                    if gold:
                        continue # Golden chips picked: their median everywhere, only the spread per strip
                    r_s = reference(sel, sel)
                    new = tuple(np.where(m, a_s, a_all) for a_s, a_all in zip(r_s, new))
                ref = new
        ref = with_ssim(ref, gold or clean, None if gold else
                        [(m_, list(strip_clean[sd])) for _, m_, sd in strips if sd in strip_clean])
        if steps is not None:
            if trace.get("cell") in chips:
                trace["info"] = [f"reduction {f} (image px per working px), chip {Wr} x {Hr} image px",
                                 (f"reference: {len(gold)} golden chips picked, spread from {len(clean)} chips" if gold else
                                  f"reference: per edge the {len(clean)} of {len(keys)} chips cleanest there (clean_ref {p['clean_ref']})") +
                                 f", traced chip {'in' if tk in (gold or clean) else 'not in'} it" +
                                 "".join(f"; {side}: {'in' if tk in sel else 'not in'}" for side, sel in strip_clean.items())]
            steps.append(("5 median chip (reference)", ref[0], "gray"))
            steps.append(("6 chip-to-chip spread", ref[1], "gray"))
            steps.append(("7 reference min (envelope)", ref[2], "gray"))
            steps.append(("8 reference max (envelope)", ref[3], "gray"))
        if not tick(li, layer, "detect", 0, len(keys), True):
            return None
        # Debug of every chip (first searched layer), kept by the viewer for Debug chip after a run: the shared
        # reference once, per chip its images reduced to <= 256 px as uint8 and the masks packed
        store = p.get("store") if li == 0 else None
        if store is not None:
            ds = max(1, int(math.ceil(max(H, W) / 256.0)))

            def small(a):
                return _reduce(np.asarray(a, np.float32), ds) if ds > 1 else np.asarray(a, np.float32)

            def q8(a, lo, hi):
                return np.clip((small(a) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            store.update(layer=layer, f=f, Wr=Wr, Hr=Hr, M=M, H=H, W=W, use_ssim=use_ssim, chips={}, params=dict(p),
                         ref=[(name, q8(a, *np.percentile(a, [1, 99]))) for name, a in
                              (("5 median chip (reference)", ref[0]), ("6 chip-to-chip spread", ref[1]),
                               ("7 reference min", ref[2]), ("8 reference max", ref[3]))],
                         info=[f"reduction {f} (image px per working px), chip {Wr} x {Hr} image px",
                               (f"reference: {len(gold)} golden chips picked, spread from {len(clean)} chips" if gold else
                                f"reference: per edge the cleanest chips there ({len(clean)} of {len(keys)} overall)")],
                         clean=set(gold or clean), strip_clean={sd: set(v) for sd, v in strip_clean.items()})

        dbg = p.get("debug")
        shown = masks is not None and li == 0
        for ki, k in enumerate(keys):
            if not tick(li, layer, "detect", ki + 1, len(keys)):
                return None
            dev = dev_of(k, ref)
            dark = dev <= -p["dark_threshold"] if p["compare"] != "ssim" else np.zeros((H, W), bool)
            if use_ssim: # Structure changed (SSIM), in chip-to-chip spreads, counts as a speck too
                sdev = (dissim(k, ref[6]) - ref[4]) / ref[5]
                dark = dark | (sdev >= p["ssim_threshold"])
            specks = dark.astype(np.float32)
            specks = (specks > 0) & (_box(specks, k_dens, k_dens) >= p["speck_density"]) # Not lone noise specks
            if p["min_speck_px"] > 0: # Before the merge: specks (connected dark pixels) smaller than min_speck_px (image px2) go
                specks &= edges
                lab, areas = _label(specks)
                specks &= (areas * sx * sy >= p["min_speck_px"])[lab]
            if k == tk:
                steps.append(("9 deviation (spreads): blue darker, red brighter", dev, "signed"))
                if use_ssim:
                    lim_s = max(1.0, float(np.percentile(sdev[edges], 99.5)))
                    steps.append(("9b SSIM change: full yellow = %.1f spreads (threshold %.1f)" % (lim_s, p["ssim_threshold"]),
                                  np.clip(sdev / lim_s, 0, 1), "unit"))
                steps.append(("10 dark specks (Dark %.1f%s)" % (p["dark_threshold"], ", SSIM %.1f" % p["ssim_threshold"] if use_ssim else ""), dark & edges, "mask"))
                steps.append(("11 specks kept (Density %.2f, Speck >= %g px2)" % (p["speck_density"], p["min_speck_px"]), specks & edges, "mask"))
            lines = np.zeros((H, W), bool)
            pieces = np.zeros((H, W), bool) # Specks left by Piece >= (debug)
            cands = []
            for along, strip, side in strips:
                # The dark specks of the rim joined into lines, longer along the edge than across: a sparse,
                # ragged rim becomes one line. A line long enough is a stain: the chip edge up to the line
                # (bright parts are not added: the area does not run past the rim into the chip)
                sp = specks & strip
                if p["min_piece_px"] > 0:
                    # Before the merge: speck pieces (connected) shorter along the edge than min_piece_px are dropped
                    # (lone specks, small blobs: they would join into the line and pull the area inward)
                    lab, _areas = _label(sp)
                    n_lab = len(_areas)
                    if n_lab > 1:
                        ys_, xs_ = np.nonzero(lab)
                        cc = xs_ if along == "x" else ys_
                        lo = np.full(n_lab, 1 << 30); hi = np.full(n_lab, -1)
                        np.minimum.at(lo, lab[ys_, xs_], cc)
                        np.maximum.at(hi, lab[ys_, xs_], cc)
                        ext = (hi - lo + 1) * (sx if along == "x" else sy)
                        sp &= (ext >= p["min_piece_px"])[lab]
                    pieces |= sp
                line = line_closing(sp, along) & strip
                lines |= line
                # Pieces of the line are linked along the edge across gaps up to link_gap_px: a stain is a run
                # along the edge with line pixels, its area the chip edge up to the farthest line pixel
                # (gaps take the interpolated depth)
                o = _orient(line, side)
                skip = skip_y if along == "x" else skip_x
                s_al = sx if along == "x" else sy
                for c0, c1 in _runs(o.any(axis=0), int(round(p["link_gap_px"] / s_al))):
                    length = (c1 - c0) * s_al
                    seg = o[:, c0:c1]
                    rows = np.nonzero(seg.any(axis=1))[0]
                    if length < p["min_length_px"]:
                        if dbg is not None and length >= 0.3 * p["min_length_px"]:
                            dbg.append((k, "short", round(length), round((rows[-1] + 1 - skip) * s_al)))
                        continue
                    # Depth per place along the edge: the farthest line pixel, gaps interpolated, then the median
                    # over merge_along x merge radius along the edge (a stray deeper pixel does not make a spike)
                    has = seg.any(axis=0)
                    dep = np.where(has, seg.shape[0] - np.argmax(seg[::-1], axis=0), 0).astype(np.float32)
                    idx = np.nonzero(has)[0]
                    dep = np.interp(np.arange(seg.shape[1]), idx, dep[idx])
                    km = max(1, int(round(p["merge_along"] * p["merge_radius_px"] / s_al)) | 1)
                    if km > 1 and len(dep) > km:
                        pad_ = np.pad(dep, km // 2, mode="edge")
                        dep = np.median(np.lib.stride_tricks.sliding_window_view(pad_, km), axis=1)
                    # Fitted: the depth averaged along the edge over smooth_px (Gaussian), so the boundary does not
                    # jag in and out across the edge
                    if p["smooth_px"] > 0 and len(dep) > 2:
                        dep = _blur1(np.asarray(dep, np.float64), p["smooth_px"] / s_al / 2.0)
                    fill = np.arange(seg.shape[0])[:, None] < np.round(dep)[None, :]
                    fill[:skip] = False
                    d_px = (int(np.nonzero(fill.any(axis=1))[0][-1]) + 1 - skip) * s_al if fill.any() else 0
                    if d_px < p["min_depth_px"]:
                        if dbg is not None:
                            dbg.append((k, "thin", round(length), round(d_px)))
                        continue
                    if length < p["min_aspect"] * d_px:
                        if dbg is not None:
                            dbg.append((k, "not elongated", round(length), round(d_px)))
                        continue
                    area_o = np.zeros(o.shape, bool)
                    area_o[:, c0:c1] = fill
                    area_full = _orient(area_o, side, back=True)
                    if not area_full.any():
                        continue # Nothing between the chip edge and the line (all inside Skip)
                    if p["bright_min"] > 0 and float(np.clip(dev[area_full], 0, None).mean()) < p["bright_min"]:
                        if dbg is not None:
                            dbg.append((k, "not bright inside", round(length), round(d_px)))
                        continue # Optional check: the area between the edge and the line is brighter
                    cands.append((area_full, float(seg.sum()) / max(1.0, (c1 - c0))))
            # A stain wrapping around a corner shows in two (or three) strips: pieces within the merge
            # radius of each other are one stain, their areas joined
            groups = []
            for full, sc in cands:
                full = full.copy()
                grown = _box(full.astype(np.float32), k_merge, k_merge) > 0
                best = sc
                for gi in [gi for gi, g in enumerate(groups) if (grown & g[0]).any()][::-1]:
                    g = groups.pop(gi)
                    full |= g[0]
                    best = max(best, g[1])
                groups.append((full, best))
            if shown or k == tk or store is not None:
                area_all = np.zeros((H, W), bool)
                for full, _ in groups:
                    area_all |= full
            if k == tk:
                if p["min_piece_px"] > 0:
                    steps.append(("11b specks left by Piece >= %g px (before the merge)" % p["min_piece_px"], pieces, "mask"))
                steps.append(("12 joined lines (Merge %g px, x%g along)" % (p["merge_radius_px"], p["merge_along"]), lines, "mask"))
                steps.append(("13 stain areas: chip edge up to the line", area_all, "mask"))
                trace.setdefault("info", []).extend(f"dropped: {d[1]} (length {d[2]}, depth {d[3]} px)" for d in (dbg or []) if d[0] == k)
            if store is not None:
                lim = 2.0 * p["dark_threshold"]
                cl, ch = np.percentile(chips[k], [1, 99])
                store["chips"][k] = {
                    "at": at[k], "chip": q8(chips[k], cl, ch),
                    "dev": q8(dev, -lim, lim), "dev_lim": lim,
                    "sdev": q8(sdev, 0, max(1.0, float(np.percentile(sdev[edges], 99.5)))) if use_ssim else None,
                    "sdev_lim": max(1.0, float(np.percentile(sdev[edges], 99.5))) if use_ssim else None,
                    "dark": np.packbits(dark & edges), "specks": np.packbits(specks & edges),
                    "lines": np.packbits(lines), "area": np.packbits(area_all),
                    "why": [d for d in (dbg or []) if d[0] == k], "found": len(groups)}
            if shown: # Show mask: dark specks (red), joined lines (yellow), stain areas (cyan), as segmented
                mx, my = to_deskew(*at[k])
                masks[k] = ((mx, my, float(Wr), float(Hr)), (H, W), np.packbits(specks & edges), np.packbits(area_all),
                            np.packbits(lines))
            ox, oy = at[k]
            for full, sc in groups:
                rows, cols = np.nonzero(full)
                r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max() + 1
                corners = [to_deskew(ox + c * sx, oy + r * sy) for c, r in ((c0, r0), (c1, r0), (c0, r1 + 1), (c1, r1 + 1))]
                gx0, gx1 = min(q[0] for q in corners), max(q[0] for q in corners)
                gy0, gy1 = min(q[1] for q in corners), max(q[1] for q in corners)
                if any(not (gx1 < a0 or a1 < gx0 or gy1 < b0 or b1 < gy0) for a0, b0, a1, b1 in taken.get(k, ())):
                    continue # The same stain on an earlier layer
                taken.setdefault(k, []).append((gx0, gy0, gx1, gy1))
                part = full[r0:r1 + 1, c0:c1]
                outline = [to_deskew(ox + (c0 + px + 0.5) * sx, oy + (r0 + py + 0.5) * sy) for py, px in _contour(part)]
                found.append((layer, (gx0 + gx1) / 2, (gy0 + gy1) / 2, (gx1 - gx0) / 2, (gy1 - gy0) / 2, k, sc, outline))
    return found


def stain_from_strokes(strokes, rect, width, side=None):
    """
    A stain marked by hand: strokes painted over its dark line ([[(gx, gy)], ...] Deskewed) in the chip
    rect (x0, y0, w, h) Deskewed, brush `width` image px. The stain is the chip edge (`side`, else the
    edge nearest to the strokes) up to the far side of the painted line, along the painted range (gaps
    between strokes take the interpolated depth). Returns (outline [(gx, gy)] Deskewed, (gx0, gy0, gx1,
    gy1), side, (t0, t1) range along the edge) or None.
    """
    x0, y0, w, h = rect
    pts = []
    for st in strokes:
        st = [tuple(q) for q in st]
        for a, b in zip(st, st[1:] + st[-1:]): # Densified (every 0.5 px): a fast stroke has no holes
            n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.5))
            pts.extend((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n))
        if st:
            pts.append(st[-1])
    if not pts:
        return None
    P = np.asarray(pts, np.float64)
    if side is None:
        mx, my = P.mean(axis=0)
        side = min((("top", my - y0), ("bottom", y0 + h - my), ("left", mx - x0), ("right", x0 + w - mx)), key=lambda q: q[1])[0]
    if side in ("top", "bottom"):
        t, n, L, D = P[:, 0] - x0, (P[:, 1] - y0) if side == "top" else (y0 + h - P[:, 1]), w, h
    else:
        t, n, L, D = P[:, 1] - y0, (P[:, 0] - x0) if side == "left" else (x0 + w - P[:, 0]), h, w
    r = max(0.5, width / 2.0)
    t0, t1 = max(0.0, float(t.min()) - r), min(float(L), float(t.max()) + r)
    if t1 - t0 < 1:
        return None
    bins = np.arange(math.floor(t0), math.ceil(t1) + 1, dtype=np.float64)
    depth = np.full(len(bins), -1.0)
    for ti, ni in zip(t, n): # Round brush: the far side of the painted line
        sel = np.abs(bins - ti) <= r
        if sel.any():
            depth[sel] = np.maximum(depth[sel], ni + np.sqrt(np.maximum(r * r - (bins[sel] - ti) ** 2, 0)))
    has = depth >= 0
    depth = np.interp(bins, bins[has], depth[has])
    depth = np.clip(depth, 0, D / 2.0)
    bins = np.clip(bins, t0, t1)
    step = max(1, len(bins) // 400) # At most ~800 outline points
    keep = list(range(0, len(bins), step)) + ([len(bins) - 1] if (len(bins) - 1) % step else [])
    oriented = [(t0, 0.0), (t1, 0.0)] + [(bins[i], depth[i]) for i in reversed(keep)]

    def back(tt, nn):
        if side == "top":
            return (x0 + tt, y0 + nn)
        if side == "bottom":
            return (x0 + tt, y0 + h - nn)
        if side == "left":
            return (x0 + nn, y0 + tt)
        return (x0 + w - nn, y0 + tt)
    outline = [back(tt, nn) for tt, nn in oriented]
    xs, ys = [q[0] for q in outline], [q[1] for q in outline]
    return outline, (min(xs), min(ys), max(xs), max(ys)), side, (t0, t1)
