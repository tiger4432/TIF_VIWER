"""
Automatic reference bbox placement.

For every (bonded) cell of the topmost layer: cut the deskewed cell window, stretch its contrast,
find the package edges from the column / row intensity profiles, and place a bbox of the
declared size centered on the detected box (when the detected size differs from the declared
one, only the center is taken from the image).

When detection is weak (low contrast, damaged chip, debris):
- a detection is rejected when an edge is weak compared to the profile noise, when its size is
  off the declared size, or when its center deviates from the lattice formed by the other
  detections (median offset);
- rejected cells get an estimated bbox at the cell center + median offset of the accepted
  detections, flagged v["auto"] = "estimated" (drawn dashed, reviewed with "Next Estimated");
  editing such a bbox clears the flag.

Layout refinement (refine_bboxes, after Detect BBOX and with Refine BBOX): the package edge is one
weak feature; every chip shows the same layout, so each chip is aligned on the golden chip (median
of the chips) of its whole layout: the gradient magnitude (contrast / polarity free) summed over a
few loaded layers, cross-correlated within a margin around the bbox (sub-pixel), the golden chip
rebuilt from the aligned chips a few times. A chip whose match is poor or that sits off the lattice
of the others is estimated from the lattice. Bboxes placed or edited by hand (v["hand"]) stay and
set the absolute position: the others move by the median of their offset to them.
fit_grid: the grid (start, pitch, angle) fitted to the bbox centers (robust least squares).

Parameters (config.json "auto_bbox", defaults in DEFAULTS):
    contrast_pct     [lo, hi] percentile stretch
    window_scale     window around a cell, in pitches
    size_tolerance   accepted |detected - declared| / declared
    min_edge_ratio   edge gradient / median |gradient| of the profile
    outlier_tol      accepted deviation from the median offset (lattice), in pitches
    method           Detect BBOX: "layout" (default; detect_layout: from the cell centres, aligned on
                     the layout, no edge needed) or "edge" (package edges, then layout_refine)
    layout_refine    "edge": also align the bboxes on the layout afterwards (refine_bboxes)
    coarse_margin    "layout": search range of the first, coarse pass, in bbox sizes
    refine_margin    search range around the bbox, in bbox sizes
    refine_px        longest side of the working chip image
    refine_layers    loaded layers used (spread over the loaded range)
    refine_min_ncc   a match below this correlation is estimated from the lattice
    refine_inner     the outer band of the chip (this fraction of its size) is left out of the match:
                     noisy package edges (one side sticking out, the chip looking larger) do not
                     pull it; the golden chip's box still sets the absolute place
    absolute         "auto" (default): the absolute place from the golden chip's package box (else the
                     locate step); "grid": the bboxes are only aligned with each other around the cell
                     centres of the grid (a grid set well by hand; narrow streets where the package edge
                     of a chip runs into its neighbour's). Viewer: "From grid" next to Detect BBOX
    locate           "layout": when the coarse pass finds no golden chip box (origin off by more than
                     coarse_margin), find the package edge pair (declared size apart) of the median
                     chip over a window of a whole pitch and move all bboxes by it
    locate_px        longest side of that window's working image
    locate_chips     chips in the median (spread over the wafer)
    locate_layers    loaded layers summed for it
"""
import math
import time

import numpy as np

DEFAULTS = {
    "contrast_pct": [2.0, 98.0],
    "window_scale": 1.2,
    "size_tolerance": 0.3,
    "min_edge_ratio": 4.0,
    "outlier_tol": 0.05,
    "method": "layout",
    "layout_refine": True,
    "coarse_margin": 0.25,
    "refine_margin": 0.08,
    "refine_px": 320,
    "refine_layers": 4,
    "refine_min_ncc": 0.3,
    "refine_inner": 0.1,
    "locate": True,
    "locate_px": 400,
    "locate_chips": 32,
    "locate_layers": 2,
}


def _smooth(profile, k=5):
    # Edge padding: zero padding would create false edges at the window border
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(np.pad(profile, (k // 2, k - 1 - k // 2), mode="edge"), kernel, mode="valid")


def stretch(patch, lo_pct=2.0, hi_pct=98.0):
    """Contrast adjustment: percentile stretch to 0..1."""
    a = patch.astype(np.float32)
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        return None
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _peaks(g, lo, hi, k=6):
    """Indices of the k strongest local maxima of g in [lo, hi)."""
    seg = g[lo:hi]
    if len(seg) < 3:
        return []
    idx = np.nonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > 0))[0] + 1
    idx = idx[np.argsort(seg[idx])[::-1][:k]]
    return [int(i) + lo for i in idx]


def _edges(profile, center_brighter, expected=None, tol=0.3):
    """
    (start, end, strength) of the package edge pair (rising then falling), or None.
    With an expected width the pair closest to it wins, so inner edges (e.g. the die inside the
    scribe line) are not mixed with the outer package edges.
    """
    g = np.gradient(_smooth(profile))
    if not center_brighter:
        g = -g
    n = len(g)
    m = max(2, n // 50) # Ignore the window border
    mid = n // 2
    starts = _peaks(g, m, mid)    # dark -> package
    ends = _peaks(-g, mid, n - m) # package -> dark
    if not starts or not ends:
        return None
    best = None
    for a in starts:
        for b in ends:
            strength = min(g[a], -g[b])
            if expected:
                err = abs((b - a + 1) - expected) / expected
                if err > tol:
                    continue
                score = strength * (1.0 - err / tol)
            else:
                score = strength
            if best is None or score > best[0]:
                best = (score, a, b, strength)
    if best is None:
        return None
    noise = float(np.median(np.abs(g))) + 1e-6
    return best[1], best[2], best[3] / noise


def detect_box(patch, params=DEFAULTS, expected=None):
    """
    Package box in a deskewed window: (cx, cy, w, h, edge_ratio) in window pixels, or None.
    expected: declared (w, h) in window pixels (picks the matching edge pair).
    The package may be brighter or darker than the street; the center decides.
    """
    img = stretch(patch, *params.get("contrast_pct", DEFAULTS["contrast_pct"]))
    if img is None:
        return None
    h, w = img.shape
    # Profiles over the central band (avoids corners / neighbouring packages)
    band_r = slice(int(h * 0.3), int(h * 0.7))
    band_c = slice(int(w * 0.3), int(w * 0.7))
    col_profile = img[band_r, :].mean(axis=0)
    row_profile = img[:, band_c].mean(axis=1)

    bw, bh = max(1, int(w * 0.05)), max(1, int(h * 0.05))
    center = img[band_r, band_c].mean()
    border = np.concatenate([img[:, :bw].ravel(), img[:, -bw:].ravel(), img[:bh, :].ravel(), img[-bh:, :].ravel()]).mean()
    brighter = center >= border

    tol = params.get("size_tolerance", DEFAULTS["size_tolerance"])
    ex = _edges(col_profile, brighter, expected[0] if expected else None, tol)
    ey = _edges(row_profile, brighter, expected[1] if expected else None, tol)
    if ex is None or ey is None:
        return None
    x0, x1, sx = ex
    y0, y1, sy = ey
    return ((x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0, float(x1 - x0 + 1), float(y1 - y0 + 1), min(sx, sy))


def place_bboxes(stack, z, layer_number, grid_cfg, coord_transform, void_manager, bbox_type,
                 size, cells, params=None, progress=None):
    """
    Detects and places the bbox of every cell in `cells` [(row, col)] on `layer_number`. All
    existing bboxes (every cell, every layer) are removed first.
    bbox_type: name of the bbox void type. size: declared (w, h); None -> detected size. One undo step.
    Returns {"detected": n, "estimated": [(r, c)], "failed": [(r, c)]}.
    """
    p = dict(DEFAULTS)
    p.update(params or {})
    to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))
    pitch = max(grid_cfg.pitch_x, grid_cfg.pitch_y)
    half = 0.5 * p["window_scale"] * pitch

    # 1. Detect every cell: (r, c) -> (dx, dy offset of the box center from the cell center, w, h)
    found = {}
    for i, (r, c) in enumerate(cells):
        if progress is not None and not progress(i):
            stack.release_cache()
            return None
        cx = grid_cfg.start_x + (c + 0.5) * grid_cfg.pitch_x
        cy = grid_cfg.start_y + (r + 0.5) * grid_cfg.pitch_y
        x0, y0 = cx - half, cy - half
        quad = [to_raw(x0, y0), to_raw(cx + half, y0), to_raw(cx + half, cy + half), to_raw(x0, cy + half)]
        patch = stack.get_poly_crop(z, quad)
        if patch is None:
            continue
        scale = patch.shape[1] / (2 * half)
        expected = (size[0] * scale, size[1] * scale) if size is not None else None
        box = detect_box(patch, p, expected)
        if box is None or box[4] < p["min_edge_ratio"]:
            continue
        bw, bh = box[2] / scale, box[3] / scale
        if size is not None and (abs(bw - size[0]) > p["size_tolerance"] * size[0] or
                                 abs(bh - size[1]) > p["size_tolerance"] * size[1]):
            continue
        found[(r, c)] = (x0 + box[0] / scale - cx, y0 + box[1] / scale - cy, bw, bh)
    stack.release_cache()

    # 2. Lattice consistency: drop detections far from the median offset
    if found:
        offs = np.array([(v[0], v[1]) for v in found.values()])
        med = np.median(offs, axis=0)
        tol = p["outlier_tol"] * pitch
        found = {k: v for k, v in found.items() if abs(v[0] - med[0]) <= tol and abs(v[1] - med[1]) <= tol}
    med_size = None
    if found:
        offs = np.array([(v[0], v[1]) for v in found.values()])
        med = np.median(offs, axis=0)
        med_size = np.median(np.array([(v[2], v[3]) for v in found.values()]), axis=0)

    # 3. Place: detected, else estimated from the lattice (median offset)
    result = {"detected": 0, "estimated": [], "failed": []}
    vm = void_manager
    vm.begin_change()
    try:
        # Start over: every existing bbox (all cells, all layers) is replaced
        for layer, vs in list(vm.voids.items()):
            for v in [v for v in vs if vm.is_bbox(v)]:
                vm.remove_void(layer, v)

        for (r, c) in cells:
            cx = grid_cfg.start_x + (c + 0.5) * grid_cfg.pitch_x
            cy = grid_cfg.start_y + (r + 0.5) * grid_cfg.pitch_y
            if (r, c) in found:
                dx, dy, bw, bh = found[(r, c)]
                flag = None
            elif med_size is not None:
                dx, dy = float(med[0]), float(med[1])
                bw, bh = float(med_size[0]), float(med_size[1])
                flag = "estimated"
            else:
                result["failed"].append((r, c))
                continue
            w, h = size if size is not None else (bw, bh)
            v = vm.add_void(layer_number, *to_raw(cx + dx, cy + dy), w / 2.0, h / 2.0, bbox_type)
            if flag:
                v["auto"] = flag
                result["estimated"].append((r, c))
            else:
                result["detected"] += 1
    finally:
        vm.end_change()
    return result


def _cut(stack, z, to_raw, rect, out_size, factor):
    """Deskewed rect (x0, y0, w, h) of layer z as a float image of out_size (w, h), or None."""
    from .core_data import to_gray
    from .void_detect import _chip
    x0, y0, w, h = rect
    corners = [to_raw(x0, y0), to_raw(x0, y0 + h), to_raw(x0 + w, y0 + h), to_raw(x0 + w, y0)] # PIL QUAD order
    xs, ys = [q[0] for q in corners], [q[1] for q in corners]
    rx0, ry0 = max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys))))
    rx1, ry1 = min(stack.width, int(math.ceil(max(xs)))), min(stack.height, int(math.ceil(max(ys))))
    if rx1 - rx0 < 4 or ry1 - ry0 < 4:
        return None
    region = to_gray(stack.read_region(z, rx0, ry0, rx1 - rx0, ry1 - ry0))
    quad = tuple(v for q in corners for v in (q[0] - rx0, q[1] - ry0))
    return _chip(region, quad, out_size, factor)


class _ReducedLayer:
    """
    One layer box-reduced by `factor`, read once in row bands: with many chips one read of the layer
    instead of one per chip (the windows of neighbouring chips overlap, compressed strips are decoded
    once). cut() returns the same deskewed window as _cut.
    """
    def __init__(self, stack, z, factor):
        from PIL import Image
        from .core_data import to_gray
        self.factor = factor
        W, H = stack.width, stack.height
        band = factor * max(1, 1024 // factor) # Whole blocks per band: the bands join seamlessly
        rows = []
        for y in range(0, H, band):
            a = Image.fromarray(to_gray(stack.read_region(z, 0, y, W, min(band, H - y))).astype(np.float32))
            rows.append(np.asarray(a.reduce(factor) if factor > 1 else a))
        self.img = Image.fromarray(np.vstack(rows))

    def cut(self, to_raw, rect, out_size):
        from PIL import Image
        x0, y0, w, h = rect
        f = self.factor
        corners = [to_raw(x0, y0), to_raw(x0, y0 + h), to_raw(x0 + w, y0 + h), to_raw(x0 + w, y0)] # PIL QUAD order
        quad = tuple(v / f for q in corners for v in q)
        return np.asarray(self.img.transform(out_size, Image.QUAD, quad, resample=Image.BILINEAR), dtype=np.float32)


def _cutter(stack, z, to_raw, factor, n_windows, window_px):
    """cut(rect, out_size) for layer z: from a once-reduced layer when the windows together cover more
    than a third of the image (many chips), else window by window (_cut)."""
    if n_windows * window_px > stack.width * stack.height / 3.0:
        layer = _ReducedLayer(stack, z, factor)
        return lambda rect, out_size: layer.cut(to_raw, rect, out_size)
    return lambda rect, out_size: _cut(stack, z, to_raw, rect, out_size, factor)


def _parabola(lo, mid, hi):
    den = lo - 2 * mid + hi
    return float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5)) if den < 0 else 0.0


def _lattice(cells, pts, tol):
    """Robust affine fit pts ~ [1, col, row]; returns (prediction function, inlier mask)."""
    A = np.array([[1.0, c, r] for r, c in cells])
    P = np.asarray(pts, np.float64)
    keep = np.ones(len(P), bool)
    coef = None
    for _ in range(4):
        if keep.sum() < 3:
            break
        coef, *_ = np.linalg.lstsq(A[keep], P[keep], rcond=None)
        res = np.hypot(*(P - A @ coef).T)
        keep = res <= max(tol, 3.0 * 1.4826 * float(np.median(res[keep])))
    if coef is None:
        return None, keep
    return (lambda r, c: tuple(np.array([1.0, c, r]) @ coef)), keep


def refine_bboxes(stack, zs, layer_number, grid_cfg, coord_transform, void_manager, bbox_type, size, cells,
                  params=None, progress=None):
    """
    Aligns the bboxes of `cells` on the layout (see the module doc). zs: loaded layer indices to use.
    Missing bboxes are added at the lattice. size: declared (w, h) or None (the bboxes' median).
    One undo step. Returns {"refined", "estimated": [(r, c)], "hand", "moved_px", "ncc"} or None.
    """
    from .void_detect import _translate
    from .void_layer_change import _gradmag
    p = dict(DEFAULTS)
    p.update(params or {})
    to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))
    to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))
    vm = void_manager
    want = set(cells)
    boxes = {}
    for layer, vs in vm.voids.items():
        for v in vs:
            if vm.is_bbox(v):
                gx, gy = to_deskew(v["globalCX"], v["globalCY"])
                cell = grid_cfg.cell_at(gx, gy)
                if cell in want:
                    boxes.setdefault(cell, []).append((layer, v, gx, gy))
    if len(boxes) < 3:
        return None
    if size is None:
        size = (float(np.median([2 * b[0][1]["radiusX"] for b in boxes.values()])),
                float(np.median([2 * b[0][1]["radiusY"] for b in boxes.values()])))
    w, h = size
    cell_c = lambda r, c: (grid_cfg.start_x + (c + 0.5) * grid_cfg.pitch_x, grid_cfg.start_y + (r + 0.5) * grid_cfg.pitch_y)
    off = np.median([(b[0][2] - cell_c(*k)[0], b[0][3] - cell_c(*k)[1]) for k, b in boxes.items()], axis=0)
    centers = {k: ((boxes[k][0][2], boxes[k][0][3]) if k in boxes else
                   (cell_c(*k)[0] + off[0], cell_c(*k)[1] + off[1])) for k in cells}
    hand = {k for k, b in boxes.items() if any(e[1].get("hand") for e in b)}

    # Working images: bbox + margin, gradient magnitude summed over a few layers
    m = p["refine_margin"] * max(w, h)
    factor = max(1, int(math.ceil((max(w, h) + 2 * m) / p["refine_px"])))
    wb, hb = max(8, int(round(w / factor))), max(8, int(round(h / factor)))
    mw = max(2, int(round(m / factor)))
    Ww, Hw = wb + 2 * mw, hb + 2 * mw
    zs = list(zs)
    use = [zs[i] for i in np.unique(np.linspace(0, len(zs) - 1, min(len(zs), int(p["refine_layers"]))).astype(int))]
    order = sorted(centers)
    n = len(order)
    G = np.zeros((n, Hw, Ww), np.float32)
    I = np.zeros((n, Hw, Ww), np.float32) # Intensity (each layer robust-normalised): the golden chip's package box
    ok = np.zeros(n, bool)
    step = 0
    for z in use:
        if progress is not None and not progress(step):
            return None
        cut = _cutter(stack, z, to_raw, factor, n, (Ww * factor) * (Hw * factor))
        for i, k in enumerate(order):
            if progress is not None and not progress(step):
                stack.release_cache()
                return None
            step += 1
            cx, cy = centers[k]
            a = cut((cx - w / 2 - mw * factor, cy - h / 2 - mw * factor, Ww * factor, Hw * factor), (Ww, Hw))
            if a is None:
                continue
            g = _gradmag(a)
            G[i] += g / (float(np.abs(g).mean()) + 1e-9)
            sub = a[::3, ::3] # Robust level / spread from every 9th pixel: the same, 9x cheaper
            med_a = float(np.median(sub))
            I[i] += (a - med_a) / (float(np.median(np.abs(sub - med_a))) + 1e-6)
            ok[i] = True
        stack.release_cache()

    # Align on the golden chip, rebuilt from the aligned chips; only its inner part (refine_inner)
    bi, bj = int(round(p["refine_inner"] * hb)), int(round(p["refine_inner"] * wb))
    inner = np.zeros((hb, wb), bool)
    inner[bi:hb - bi, bj:wb - bj] = True
    s = np.zeros((n, 2))
    ncc = np.zeros(n)
    F = np.fft.rfft2(G)
    for _ in range(3):
        crops = np.stack([_translate(G[i], mw + s[i, 0], mw + s[i, 1])[:hb, :wb] for i in range(n)])
        T = np.median(crops[ok], axis=0)
        T = np.where(inner, T - T[inner].mean(), 0.0)
        T /= float(np.sqrt((T ** 2).sum())) + 1e-9
        Tp = np.zeros((Hw, Ww), np.float32)
        Tp[:hb, :wb] = T
        FT = np.conj(np.fft.rfft2(Tp))
        for i in np.nonzero(ok)[0]:
            cc = np.fft.irfft2(F[i] * FT, s=(Hw, Ww))[:2 * mw + 1, :2 * mw + 1] # cc[dy, dx]: bbox at (dx, dy)
            a, b = np.unravel_index(int(np.argmax(cc)), cc.shape)
            sy = a + (_parabola(cc[a - 1, b], cc[a, b], cc[a + 1, b]) if 0 < a < 2 * mw else 0.0)
            sx = b + (_parabola(cc[a, b - 1], cc[a, b], cc[a, b + 1]) if 0 < b < 2 * mw else 0.0)
            s[i] = (sx - mw, sy - mw)
    crops = np.stack([_translate(G[i], mw + s[i, 0], mw + s[i, 1])[:hb, :wb] for i in range(n)])
    for i in range(n):
        c = np.where(inner, crops[i] - crops[i][inner].mean(), 0.0)
        ncc[i] = float((T * c).sum()) / (float(np.sqrt((c ** 2).sum())) + 1e-9) if ok[i] else 0.0
    # Poor matches and chips off the lattice: estimated from the lattice of the others
    med, mad = float(np.median(ncc[ok])), float(np.median(np.abs(ncc[ok] - np.median(ncc[ok])))) * 1.4826
    good = ok & (ncc >= p["refine_min_ncc"]) & (ncc >= med - 4 * mad)
    absolute = None
    if good.any() and p.get("absolute") == "grid":
        s -= s[good].mean(axis=0) # The grid sets the place: aligned with each other, mean place kept
    elif good.any():
        # The alignment fixes the chips relative to each other; the absolute place: hand-placed
        # bboxes (below), else the package box of the golden chip (the noise of single chips' edges
        # averages out), else the chips keep their mean place
        s -= s[good].mean(axis=0)
        gold = np.median(np.stack([_translate(I[i], s[i, 0], s[i, 1]) for i in np.nonzero(good)[0]]), axis=0)
        box = detect_box(gold, p, (wb, hb))
        if box is not None and box[4] >= p["min_edge_ratio"]:
            absolute = (box[0] - Ww / 2.0, box[1] - Hw / 2.0) # Package centre vs the bbox centre (working px)
            if max(abs(absolute[0]), abs(absolute[1])) <= mw:
                s += np.array(absolute)
            else:
                absolute = None
    new = {k: (centers[k][0] + s[i, 0] * factor, centers[k][1] + s[i, 1] * factor) for i, k in enumerate(order)}
    gi = np.nonzero(good)[0]
    pitch = max(grid_cfg.pitch_x, grid_cfg.pitch_y)
    pred, inl = _lattice([order[i] for i in gi], [new[order[i]] for i in gi], p["outlier_tol"] * pitch)
    for j, i in enumerate(gi):
        if not inl[j]:
            good[i] = False
    if pred is None:
        return None
    est = [k for i, k in enumerate(order) if not good[i] and k not in hand]
    for k in est:
        new[k] = pred(*k)
    # Bboxes placed / edited by hand keep their place and set the absolute position
    anchor = [k for k in hand if good[order.index(k)]]
    if anchor:
        d = np.median([(centers[k][0] - new[k][0], centers[k][1] - new[k][1]) for k in anchor], axis=0)
        new = {k: (v[0] + d[0], v[1] + d[1]) for k, v in new.items()}
    for k in hand:
        new[k] = centers[k]

    vm.begin_change()
    try:
        for k in order:
            if k in hand:
                continue
            for layer, v, _, _ in boxes.get(k, []):
                vm.remove_void(layer, v)
            layer = boxes[k][0][0] if k in boxes else layer_number
            v = vm.add_void(layer, *to_raw(*new[k]), w / 2.0, h / 2.0, bbox_type)
            v["placed"] = "layout"
            if k in est:
                v["auto"] = "estimated"
    finally:
        vm.end_change()
    moved = np.hypot(s[good, 0], s[good, 1]) * factor
    return {"refined": int(good.sum()), "estimated": est, "hand": len(hand),
            "moved_px": float(np.median(moved)) if len(moved) else 0.0, "ncc": med,
            "absolute": "hand" if anchor else ("golden chip box" if absolute is not None else
                                               ("grid" if p.get("absolute") == "grid" else "mean place")),
            "ncc_by_cell": {k: float(ncc[i]) for i, k in enumerate(order)}}


def fit_grid(void_manager, grid_cfg, coord_transform, img_w, img_h):
    """
    Grid fitted to the bbox centers (not the estimated ones): angle and pitches from a robust affine
    fit of the raw centers over (col, row), start so that the cells are centred on the chips.
    Returns {"start_x", "start_y", "pitch_x", "pitch_y", "angle", "rms", "n"} or None (< 4 bboxes).
    """
    from .core_data import CoordinateTransform
    vm = void_manager
    to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))
    cells, raw = [], []
    for vs in vm.voids.values():
        for v in vs:
            if vm.is_bbox(v) and v.get("auto") != "estimated":
                cell = grid_cfg.cell_at(*to_deskew(v["globalCX"], v["globalCY"]))
                if cell is not None:
                    cells.append(cell)
                    raw.append((v["globalCX"], v["globalCY"]))
    if len(cells) < 4 or len({c[1] for c in cells}) < 2 or len({c[0] for c in cells}) < 2:
        return None
    pred, keep = _lattice(cells, raw, 0.02 * max(grid_cfg.pitch_x, grid_cfg.pitch_y))
    A = np.array([[1.0, c, r] for r, c in cells])[keep]
    coef, *_ = np.linalg.lstsq(A, np.asarray(raw, np.float64)[keep], rcond=None)
    col, row = coef[1], coef[2] # Raw step of one column / row
    angle = math.degrees(math.atan2(col[1], col[0]))
    ct = CoordinateTransform(angle, img_w, img_h)
    pitch_x, pitch_y = float(np.hypot(*col)), float(np.hypot(*row))
    ds = np.array([ct.raw_to_deskew(x, y) for (x, y), k in zip(raw, keep) if k])
    cr = np.array([c for c, k in zip(cells, keep) if k])
    sx = float(np.median(ds[:, 0] - (cr[:, 1] + 0.5) * pitch_x))
    sy = float(np.median(ds[:, 1] - (cr[:, 0] + 0.5) * pitch_y))
    res = np.hypot(ds[:, 0] - sx - (cr[:, 1] + 0.5) * pitch_x, ds[:, 1] - sy - (cr[:, 0] + 0.5) * pitch_y)
    return {"start_x": sx, "start_y": sy, "pitch_x": pitch_x, "pitch_y": pitch_y, "angle": angle,
            "rms": float(np.sqrt((res ** 2).mean())), "n": int(keep.sum())}


def _edge_pair(profile, width, expected, k=3):
    """
    Position x0 of a package edge pair `width` apart in a profile longer than a pitch: the package edges
    are a step up at x0 and down at x0 + width (or the reverse, darker package). Returns (x0, strength
    / noise) with x0 the candidate nearest `expected` among those nearly as strong as the best (the
    neighbouring chips repeat it a pitch away), or None.
    """
    g = np.gradient(_smooth(np.asarray(profile, np.float64), k))
    n, wd = len(g), int(round(width))
    m = max(2, n // 50)
    if n - wd - 2 * m < 3:
        return None
    noise = float(np.median(np.abs(g))) + 1e-9
    x0 = np.arange(m, n - wd - m)
    end = np.maximum.reduce([g[x0 + wd + d] for d in (-1, 0, 1)]), np.minimum.reduce([g[x0 + wd + d] for d in (-1, 0, 1)])
    score = np.maximum(np.minimum(g[x0], -end[1]), np.minimum(-g[x0], end[0])) / noise
    best = float(score.max())
    if best <= 0:
        return None
    # Local maxima as strong as the best: the chip and its neighbours (a pitch apart) -> nearest to expected
    peak = np.r_[False, (score[1:-1] >= score[:-2]) & (score[1:-1] >= score[2:]), False]
    cand = x0[peak & (score >= 0.8 * best)]
    if not len(cand):
        cand = x0[[int(np.argmax(score))]]
    pick = int(cand[np.argmin(np.abs(cand - expected))])
    return pick, float(score[pick - m])


def locate_chip(stack, zs, to_raw, centers, size, pitch, params=None, progress=None):
    """
    Absolute place of the chips after they are aligned with each other: the median of the chips over a
    window of a whole pitch more than the bbox (so a chip is inside whatever the start offset), its
    package edges found as a pair of steps the declared size apart. centers: aligned bbox centres
    (deskewed); size (w, h); pitch (px, py). Returns (dx, dy, strength) to add to every centre, or None
    when the edges are not clear (strength below min_edge_ratio) or when cancelled (progress(i) False).
    """
    p = dict(DEFAULTS)
    p.update(params or {})
    w, h = size
    Wd, Hd = w + pitch[0] * 1.1, h + pitch[1] * 1.1 # Deskewed window
    factor = max(1, int(math.ceil(max(Wd, Hd) / p.get("locate_px", 400))))
    Ww, Hw = int(round(Wd / factor)), int(round(Hd / factor))
    pts = list(centers)
    if len(pts) > p.get("locate_chips", 48): # A spread subset is enough for the median
        pts = [pts[i] for i in np.unique(np.linspace(0, len(pts) - 1, int(p.get("locate_chips", 48))).astype(int))]
    zs = list(zs)
    use = [zs[i] for i in np.unique(np.linspace(0, len(zs) - 1, min(len(zs), int(p.get("locate_layers", 2)))).astype(int))]
    acc = np.zeros((len(pts), Hw, Ww), np.float32)
    got = np.zeros(len(pts), bool)
    j = 0
    for z in use:
        cut = _cutter(stack, z, to_raw, factor, len(pts), (Ww * factor) * (Hw * factor))
        for i, (cx, cy) in enumerate(pts):
            if progress is not None and not progress((j * len(pts) + i) / len(use)): # 0 .. len(pts)
                stack.release_cache()
                return None
            a = cut((cx - Wd / 2, cy - Hd / 2, Ww * factor, Hw * factor), (Ww, Hw))
            if a is None or a.shape != (Hw, Ww):
                continue
            sub = a[::3, ::3]
            med = float(np.median(sub))
            acc[i] += (a - med) / (float(np.median(np.abs(sub - med))) + 1e-6)
            got[i] = True
        j += 1
        stack.release_cache()
    acc = acc[got]
    if len(acc) < 3:
        return None
    gold = np.median(np.stack(acc), axis=0)
    wb, hb = w / factor, h / factor
    ex = _edge_pair(gold.mean(axis=0), wb, (Ww - wb) / 2.0)
    ey = _edge_pair(gold.mean(axis=1), hb, (Hw - hb) / 2.0)
    if ex is None or ey is None:
        return None
    strength = min(ex[1], ey[1])
    if strength < p["min_edge_ratio"]:
        return None
    dx = (ex[0] + wb / 2.0 - Ww / 2.0) * factor
    dy = (ey[0] + hb / 2.0 - Hw / 2.0) * factor
    return float(dx), float(dy), strength


def detect_layout(stack, zs, layer_number, grid_cfg, coord_transform, void_manager, bbox_type, size, cells,
                  params=None, progress=None):
    """
    Detect BBOX without package edges (blurred / noisy edges): every bbox of `cells` placed at its
    cell (the existing bboxes' median offset in the cell, else centred), then aligned on the layout
    twice: coarse (coarse_margin, low resolution: the declared origin / angle may be a little off;
    chips beyond reach are taken from the lattice of the others, which follows the angle) and fine.
    Between the two, locate_chip (locate) finds the package of the median chip over a whole pitch and
    moves every bbox by the same amount: a declared origin off by up to almost half a pitch (beyond
    coarse_margin) no longer leaves all bboxes shifted alike. Off by more than half a pitch, the chips
    are taken for the neighbouring cells' (the cell numbering shifts by one: set the origin).
    Every existing bbox is replaced (hand ones too). One undo step. Returns refine_bboxes' result.
    """
    p = dict(DEFAULTS)
    p.update(params or {})
    to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))
    to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))
    vm = void_manager
    cell_c = lambda r, c: (grid_cfg.start_x + (c + 0.5) * grid_cfg.pitch_x, grid_cfg.start_y + (r + 0.5) * grid_cfg.pitch_y)
    offs, sizes = [], []
    for vs in vm.voids.values():
        for v in vs:
            if vm.is_bbox(v):
                gx, gy = to_deskew(v["globalCX"], v["globalCY"])
                k = grid_cfg.cell_at(gx, gy)
                if k is not None:
                    offs.append((gx - cell_c(*k)[0], gy - cell_c(*k)[1]))
                    sizes.append((2 * v["radiusX"], 2 * v["radiusY"]))
    # Start: the existing bboxes' place in their cells, or the cell centres (absolute "grid": always)
    off = np.median(offs, axis=0) if offs and p.get("absolute") != "grid" else (0.0, 0.0)
    if size is None:
        size = tuple(np.median(sizes, axis=0)) if sizes else (0.8 * grid_cfg.pitch_x, 0.8 * grid_cfg.pitch_y)
    n = max(1, len(cells))
    vm.begin_change()
    try:
        for layer, vs in list(vm.voids.items()):
            for v in [v for v in vs if vm.is_bbox(v)]:
                vm.remove_void(layer, v)
        for k in cells:
            cx, cy = cell_c(*k)
            v = vm.add_void(layer_number, *to_raw(cx + off[0], cy + off[1]), size[0] / 2.0, size[1] / 2.0, bbox_type)
            v["placed"] = "layout"
        # Coarse: the whole chip (its package edges fix which period of a repeating layout it is);
        # fine: the inner part only, within less than half a period (noisy edges do not pull)
        coarse = dict(p, refine_margin=p["coarse_margin"], refine_px=max(96, p["refine_px"] * 2 // 3), refine_inner=0.0)
        # Progress over n * layers steps (the caller's range): coarse 0-45 %, locate 45-55 %, fine 55-100 %
        layers = min(len(list(zs)), int(p["refine_layers"]))
        total = n * layers
        t0 = time.monotonic()
        res = refine_bboxes(stack, zs, layer_number, grid_cfg, coord_transform, vm, bbox_type, size, cells, coarse,
                            (lambda i: progress(0.45 * i)) if progress else None)
        t1 = time.monotonic()
        located = None
        if res is not None and p.get("locate", True) and res.get("absolute") == "mean place" and p.get("absolute") != "grid":
            # The coarse pass aligns the chips with each other but reaches only coarse_margin: with the
            # declared origin farther off its golden chip box is not found and they would all keep the
            # same shift. Only then the package edges of the median chip over a whole pitch set the
            # absolute place (a good grid keeps the golden chip box: an edge pair of the declared size
            # elsewhere in the chip must not move every bbox)
            boxes = [(layer, v) for layer, vs in vm.voids.items() for v in vs if vm.is_bbox(v)]
            want = set(cells)
            cen = [to_deskew(v["globalCX"], v["globalCY"]) for _, v in boxes if v.get("auto") != "estimated"]
            nloc = max(1, min(len(cen), int(p["locate_chips"])))
            located = locate_chip(stack, zs, to_raw, cen, size, (grid_cfg.pitch_x, grid_cfg.pitch_y), p,
                                  (lambda j: progress(total * (0.45 + 0.1 * j / nloc))) if progress else None)
            if progress is not None and not progress(0.55 * total): # Cancelled during the locate step
                res = None
            if located is not None and math.hypot(located[0], located[1]) > 0.02 * max(size):
                # Every bbox moves by the same amount; one per cell (near half a pitch two can land in a
                # cell: the one nearest the common place in the cell stays, the fine pass fills gaps)
                moved = {}
                for layer, v in boxes:
                    gx, gy = to_deskew(v["globalCX"], v["globalCY"])
                    vm.remove_void(layer, v)
                    nx, ny = gx + located[0], gy + located[1]
                    k = grid_cfg.cell_at(nx, ny)
                    if k in want: # Off the wanted cells (beyond the bonded ones): dropped
                        moved.setdefault(k, []).append((layer, v, nx, ny))
                ref = np.median([(e[2] - cell_c(*k)[0], e[3] - cell_c(*k)[1]) for k, es in moved.items() for e in es], axis=0)
                for k, es in moved.items():
                    layer, v, nx, ny = min(es, key=lambda e: math.hypot(e[2] - cell_c(*k)[0] - ref[0], e[3] - cell_c(*k)[1] - ref[1]))
                    nv = vm.add_void(layer, *to_raw(nx, ny), size[0] / 2.0, size[1] / 2.0, bbox_type)
                    nv.update({key: v[key] for key in ("placed", "auto") if key in v})
        t2 = time.monotonic()
        if res is not None:
            res = refine_bboxes(stack, zs, layer_number, grid_cfg, coord_transform, vm, bbox_type, size, cells, p,
                                (lambda i: progress(total * 0.55 + 0.45 * i)) if progress else None)
        print(f"[Detect BBOX] {n} cells: coarse {t1 - t0:.1f} s, locate {t2 - t1:.1f} s, fine {time.monotonic() - t2:.1f} s")
    finally:
        vm.end_change()
    if res is not None:
        res["located"] = None if located is None else {"dx": located[0], "dy": located[1], "strength": located[2]}
    return res
