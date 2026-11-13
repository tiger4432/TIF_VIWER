"""
Edge stain detection: contamination along the inside of the chip edge that the void detectors leave
out on purpose (the chip edge varies from chip to chip; they look for round blobs). The stains are
long (100 .. 1000 px along the edge, tens of px or more across), with a dark, ragged rim and a bright
inside, on part of the chips.

Per loaded layer:
1. Every chip (its reference bbox) is cut at a working size, normalised (median / spread) and moved
   onto the median chip (subpixel): a bbox a few px off does not show as a dark edge.
2. Per pixel the median chip and the chip-to-chip spread (MAD).
3. Deviation |chip - median| / spread, both ways: the dark rim and the bright inside both count, so
   the stain shows as one area whatever its ragged outline.
4. Averaged over a min_width_px box (pixel noise and thin layout residue drop, the stain stays) and
   kept where above threshold x the noise of the averaged map (all chips).
5. Segmentation, each edge on its own (a stain near a corner is not joined with the next edge):
   in the edge's strip the dark specks of the rim (darker than dark_threshold, dense enough:
   speck_density) and the bright inside between the rim and the edge (brighter than
   bright_threshold, 3x3 average), joined when closer than merge_radius_px (closing): a rim of
   separate dots becomes one line and the stain one area, from the chip edge to its rim. Each area
   touching a found place (4.) is a stain; its box is the box of the segmented area and its
   outline is kept (Shapes, debug). Stains of two edges overlapping at a corner: the larger kept.
   A stain is attached to the chip edge: its pixels come within touch_px of the edge (after the
   edge_skip_px left out) along at least `contact` of its length; areas inside the chip near the
   edge (layout residue, other defects) are not stains.
6. Each stain is kept when it lies mostly in the edge zone (depth_frac of the chip from the
   edge inward, the outermost edge_skip_px left out), is at least min_length_px long and elongated
   (aspect >= min_aspect). Found on several layers: marked once, on the first.

Parameters: config.json "void_detect": {"edge_band": {...}} (DEFAULTS below).

Viewer: "Edge Stains" (Void Detection). Synthetic single-layer wafer with alike chips, crescent
stains on half of them (100 .. 270 px of a 280 px chip, 20 .. 60 px deep, specked rim): 97 % found,
5 false over 3 wafers (0 on the clean one), about 3 s per 196 chips. Chips must look alike (it is a
chip-to-chip comparison).
"""
import math

import numpy as np

DEFAULTS = {
    "depth_frac": 0.25,     # Edge zone: this share of the chip size from the edge inward
    "edge_skip_px": 4,      # Outermost px of the chip left out (bbox / package edge)
    "min_length_px": 100,   # Shortest stain (image px, its longer side)
    "min_width_px": 12,     # Averaging box (image px); about the narrowest stain
    "min_aspect": 1.5,      # Longer / shorter side, at least (elongated along the edge)
    "zone_share": 0.5,      # Share of the stain inside the edge zone, at least
    "touch_px": 6,          # A stain is attached to the chip edge: its pixels within this (image px) of the edge
    "contact": 0.3,         # ... along at least this share of its length (else: inside the chip, not a stain)
    "threshold": 5.0,       # Averaged deviation / its noise (finds the stains: seeds)
    "dark_threshold": 3.0,  # Segmentation: a dark speck of the rim, in chip-to-chip spreads
    "bright_threshold": 1.5,  # Segmentation: the bright inside (3x3 average), in spreads
    "merge_radius_px": 8,   # Segmentation: specks / bright parts this close (image px) are one stain
    "speck_density": 0.12,  # Segmentation: share of specks around a speck (merge radius box), at least:
                            # the rim's specks are dense, scattered noise / layout specks are not
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


def detect(stack, zs, layer_numbers, grid_cfg, coord_transform, refs, cells, params=None, progress=None):
    """
    Returns [(layer, gx, gy, rx, ry, cell, score, outline)] (Deskewed centre / half sizes, outline:
    [(gx, gy)] Deskewed points of the segmented area) or None when cancelled.
    progress(i, n, text) -> False cancels.
    """
    from .auto_bbox import _cutter
    from .void_detect import _components, _normalize, _register, chip_rects
    from .void_layer_change import _contour
    p = dict(DEFAULTS)
    p.update(params or {})
    to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))
    rects = chip_rects(grid_cfg, refs, cells)
    if len(rects) < 4:
        return []
    bw = float(np.median([v[2] for v in rects.values()]))
    bh = float(np.median([v[3] for v in rects.values()]))
    f = max(1, int(math.ceil(max(bw, bh) / float(p["work_size"])))) # Whole reduction steps
    W, H = max(16, int(round(bw / f))), max(16, int(round(bh / f)))
    sx, sy = bw / W, bh / H # Image px per working px
    depth_x, depth_y = int(round(p["depth_frac"] * W)), int(round(p["depth_frac"] * H))
    skip_x, skip_y = int(math.ceil(p["edge_skip_px"] / sx)), int(math.ceil(p["edge_skip_px"] / sy))
    T_x, T_y = max(1, int(round(p["min_width_px"] / sx))), max(1, int(round(p["min_width_px"] / sy)))
    yy, xx = np.mgrid[0:H, 0:W]
    inner = (yy >= skip_y) & (yy < H - skip_y) & (xx >= skip_x) & (xx < W - skip_x)
    zone = inner & ((yy < depth_y) | (yy >= H - depth_y) | (xx < depth_x) | (xx >= W - depth_x))
    t_x = max(1, int(math.ceil(p["touch_px"] / sx))) + skip_x
    t_y = max(1, int(math.ceil(p["touch_px"] / sy))) + skip_y
    strips = [("x", inner & (yy < depth_y), inner & (yy < t_y), "top"),
              ("x", inner & (yy >= H - depth_y), inner & (yy >= H - t_y), "bottom"),
              ("y", inner & (xx < depth_x), inner & (xx < t_x), "left"),
              ("y", inner & (xx >= W - depth_x), inner & (xx >= W - t_x), "right")]
    # (direction along, strip mask, contact band along the chip edge, edge)
    k_dens = max(3, int(round(p["merge_radius_px"] / min(sx, sy))) | 1)
    k_merge = 2 * max(1, int(round(p["merge_radius_px"] / min(sx, sy)))) + 1 # Closing window (working px)

    def closing(m):
        """Dilate then erode by a k_merge square (box sums: fast whatever the radius)."""
        d = _box(m.astype(np.float32), k_merge, k_merge) > 1e-6
        return _box(d.astype(np.float32), k_merge, k_merge) > 1 - 1e-6
    order = sorted(rects)
    found = []
    taken = {} # cell -> [(x0, y0, x1, y1)] Deskewed boxes already marked (first layer wins)
    n_steps = len(zs) * 2
    for li, (z, layer) in enumerate(zip(zs, layer_numbers)):
        if progress is not None and not progress(2 * li, n_steps, f"Edge stains: layer {layer}"):
            return None
        cut = _cutter(stack, z, to_raw, f, len(order), bw * bh)
        chips = {}
        for cell in order:
            x0, y0, w, h = rects[cell]
            a = cut((x0, y0, bw, bh), (W, H))
            if a is not None and a.shape == (H, W):
                chips[cell] = _normalize(a.astype(np.float32))
        stack.release_cache()
        if len(chips) < 4:
            continue
        chips, _ = _register(chips)
        keys = list(chips)
        A = np.stack([chips[k].astype(np.float32) for k in keys])
        med = np.median(A, axis=0)
        spread = np.median(np.abs(A - med), axis=0) * 1.4826
        spread = np.maximum(spread, 0.5 * float(np.median(spread)) + 1e-3)
        if progress is not None and not progress(2 * li + 1, n_steps, f"Edge stains: layer {layer}"):
            return None
        S = {k: _box(np.abs(A[i] - med) / spread, T_y, T_x) for i, k in enumerate(keys)}
        vals = np.concatenate([S[k][zone][::7] for k in keys])
        m0 = float(np.median(vals))
        noise = float(np.median(np.abs(vals - m0))) * 1.4826 + 1e-6
        level = m0 + p["threshold"] * noise
        for i, k in enumerate(keys):
            seeds = (S[k] >= level) & inner
            dbg = p.get("debug")
            if not seeds.any():
                if dbg is not None:
                    dbg.append((k, "no seed", float(S[k][zone].max() / max(level, 1e-6))))
                continue
            dev = (A[i] - med) / spread
            specks = (dev <= -p["dark_threshold"]).astype(np.float32)
            specks = (specks > 0) & (_box(specks, k_dens, k_dens) >= p["speck_density"]) # Dense specks only
            bright = _box(np.clip(dev, 0, None), 3, 3) >= p["bright_threshold"]
            x0, y0 = rects[k][0], rects[k][1]
            cands = []
            for along, strip, band, side in strips:
                seg = closing((specks | bright) & strip) & strip
                seg |= seeds & strip & (_box(np.abs(dev), 3, 3) >= p["bright_threshold"]) # Faint parts the averaging found
                for r0, r1, c0, c1, area in _components(seg):
                    part = seg[r0:r1 + 1, c0:c1]
                    if not (seeds[r0:r1 + 1, c0:c1] & part).any():
                        continue # Segmented, but no stain found there (specks of noise / layout)
                    w_px, h_px = (c1 - c0) * sx, (r1 + 1 - r0) * sy
                    length, depth = (w_px, h_px) if along == "x" else (h_px, w_px)
                    if length < p["min_length_px"] or length < p["min_aspect"] * depth:
                        if dbg is not None:
                            dbg.append((k, "size", round(w_px), round(h_px)))
                        continue
                    touch = part & band[r0:r1 + 1, c0:c1] # Attached to the chip edge, along how much of it
                    share = (touch.any(axis=0).sum() / max(1, c1 - c0)) if along == "x" else \
                        (touch.any(axis=1).sum() / max(1, r1 + 1 - r0))
                    if share < p["contact"]:
                        if dbg is not None:
                            dbg.append((k, "not on the edge", round(float(share), 2)))
                        continue
                    # The stain = the chip edge up to its dark rim, filled: at every place along the edge
                    # from the edge to the farthest segmented pixel (the rim); gaps of the rim bridged
                    part = _fill_to_edge(part, side)
                    cands.append((int(part.sum()), r0, r1, c0, c1, part))
            # A corner: the same stain may show in both strips -> the larger one
            cands.sort(key=lambda c: -c[0])
            kept = []
            for area, r0, r1, c0, c1, part in cands:
                if any(not (c1 <= q[3] or q[4] <= c0 or r1 < q[1] or q[2] < r0) for q in kept):
                    continue
                kept.append((area, r0, r1, c0, c1, part))
            for area, r0, r1, c0, c1, part in kept:
                gx0, gy0 = x0 + c0 * sx, y0 + r0 * sy
                gx1, gy1 = x0 + c1 * sx, y0 + (r1 + 1) * sy
                if any(not (gx1 < a0 or a1 < gx0 or gy1 < b0 or b1 < gy0) for a0, b0, a1, b1 in taken.get(k, ())):
                    continue # The same stain on an earlier layer
                taken.setdefault(k, []).append((gx0, gy0, gx1, gy1))
                score = float(S[k][r0:r1 + 1, c0:c1].max() / max(level, 1e-6))
                outline = [(gx0 + (px + 0.5) * sx, gy0 + (py + 0.5) * sy) for py, px in _contour(part)]
                found.append((layer, (gx0 + gx1) / 2, (gy0 + gy1) / 2, (gx1 - gx0) / 2, (gy1 - gy0) / 2, k, score, outline))
    return found
