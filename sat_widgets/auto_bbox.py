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
    layout_refine    Detect BBOX also aligns the bboxes on the layout (refine_bboxes)
    refine_margin    search range around the bbox, in bbox sizes
    refine_px        longest side of the working chip image
    refine_layers    loaded layers used (spread over the loaded range)
    refine_min_ncc   a match below this correlation is estimated from the lattice
"""
import math

import numpy as np

DEFAULTS = {
    "contrast_pct": [2.0, 98.0],
    "window_scale": 1.2,
    "size_tolerance": 0.3,
    "min_edge_ratio": 4.0,
    "outlier_tol": 0.05,
    "layout_refine": True,
    "refine_margin": 0.08,
    "refine_px": 320,
    "refine_layers": 4,
    "refine_min_ncc": 0.3,
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
    ok = np.zeros(n, bool)
    step = 0
    for z in use:
        for i, k in enumerate(order):
            if progress is not None and not progress(step):
                stack.release_cache()
                return None
            step += 1
            cx, cy = centers[k]
            a = _cut(stack, z, to_raw, (cx - w / 2 - mw * factor, cy - h / 2 - mw * factor, Ww * factor, Hw * factor),
                     (Ww, Hw), factor)
            if a is None:
                continue
            g = _gradmag(a)
            G[i] += g / (float(np.abs(g).mean()) + 1e-9)
            ok[i] = True
        stack.release_cache()

    # Align on the golden chip, rebuilt from the aligned chips
    s = np.zeros((n, 2))
    ncc = np.zeros(n)
    F = np.fft.rfft2(G)
    for _ in range(3):
        crops = np.stack([_translate(G[i], mw + s[i, 0], mw + s[i, 1])[:hb, :wb] for i in range(n)])
        T = np.median(crops[ok], axis=0)
        T = T - T.mean()
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
        c = crops[i] - crops[i].mean()
        ncc[i] = float((T * c).sum()) / (float(np.sqrt((c ** 2).sum())) + 1e-9) if ok[i] else 0.0
    # Poor matches and chips off the lattice: estimated from the lattice of the others
    med, mad = float(np.median(ncc[ok])), float(np.median(np.abs(ncc[ok] - np.median(ncc[ok])))) * 1.4826
    good = ok & (ncc >= p["refine_min_ncc"]) & (ncc >= med - 4 * mad)
    if good.any():
        # The golden chip's frame drifts while it is rebuilt: the chips keep their mean place (the
        # alignment fixes them relative to each other; hand-placed bboxes, below, set the absolute place)
        s -= s[good].mean(axis=0)
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
            "moved_px": float(np.median(moved)) if len(moved) else 0.0, "ncc": med}


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
