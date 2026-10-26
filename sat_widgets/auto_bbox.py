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

Parameters (config.json "auto_bbox", defaults in DEFAULTS):
    contrast_pct     [lo, hi] percentile stretch
    window_scale     window around a cell, in pitches
    size_tolerance   accepted |detected - declared| / declared
    min_edge_ratio   edge gradient / median |gradient| of the profile
    outlier_tol      accepted deviation from the median offset, in pitches
"""
import numpy as np

DEFAULTS = {
    "contrast_pct": [2.0, 98.0],
    "window_scale": 1.2,
    "size_tolerance": 0.3,
    "min_edge_ratio": 4.0,
    "outlier_tol": 0.05,
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
