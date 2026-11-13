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
5. Each kept area is a stain when it lies mostly in the edge zone (depth_frac of the chip from the
   edge inward, the outermost edge_skip_px left out), is at least min_length_px long and elongated
   (aspect >= min_aspect). Found on several layers: marked once, on the first.

Parameters: config.json "void_detect": {"edge_band": {...}} (DEFAULTS below).

Status: experimental, not wired into the viewer yet. On the synthetic wafers (chips that differ
from each other by design) it finds 45 % of injected stains with ~300 false marks per clean wafer;
to be validated on a wafer whose chips look alike (a single-layer image can only be compared chip
to chip) before a button is added.
"""
import math

import numpy as np

DEFAULTS = {
    "depth_frac": 0.2,      # Edge zone: this share of the chip size from the edge inward
    "edge_skip_px": 4,      # Outermost px of the chip left out (bbox / package edge)
    "min_length_px": 100,   # Shortest stain (image px, its longer side)
    "min_width_px": 12,     # Averaging box (image px); about the narrowest stain
    "min_aspect": 1.5,      # Longer / shorter side, at least (elongated along the edge)
    "zone_share": 0.5,      # Share of the stain inside the edge zone, at least
    "threshold": 5.0,       # Averaged deviation / its noise
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


def detect(stack, zs, layer_numbers, grid_cfg, coord_transform, refs, cells, params=None, progress=None):
    """
    Returns [(layer, gx, gy, rx, ry, cell, score)] (Deskewed centre / half sizes) or None when cancelled.
    progress(i, n, text) -> False cancels.
    """
    from .auto_bbox import _cutter
    from .void_detect import _components, _normalize, _register, chip_rects
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
        for k in keys:
            mask = (S[k] >= level) & inner
            if not mask.any():
                continue
            x0, y0 = rects[k][0], rects[k][1]
            for r0, r1, c0, c1, area in _components(mask):
                # The averaging box widens an area by about half a box on each side
                br0, br1 = r0 + T_y // 2, r1 + 1 - T_y // 2
                bc0, bc1 = c0 + T_x // 2, c1 - T_x // 2
                if br1 <= br0 or bc1 <= bc0:
                    continue
                w_px, h_px = (bc1 - bc0) * sx, (br1 - br0) * sy
                if max(w_px, h_px) < p["min_length_px"] or max(w_px, h_px) < p["min_aspect"] * min(w_px, h_px):
                    continue
                if float(zone[r0:r1 + 1, c0:c1][mask[r0:r1 + 1, c0:c1]].mean()) < p["zone_share"]:
                    continue # Mostly inside the chip, not along its edge
                gx0, gy0 = x0 + bc0 * sx, y0 + br0 * sy
                gx1, gy1 = x0 + bc1 * sx, y0 + br1 * sy
                if any(not (gx1 < a0 or a1 < gx0 or gy1 < b0 or b1 < gy0) for a0, b0, a1, b1 in taken.get(k, ())):
                    continue # The same stain on an earlier layer
                taken.setdefault(k, []).append((gx0, gy0, gx1, gy1))
                score = float(S[k][r0:r1 + 1, c0:c1].max() / max(level, 1e-6))
                found.append((layer, (gx0 + gx1) / 2, (gy0 + gy1) / 2, (gx1 - gx0) / 2, (gy1 - gy0) / 2, k, score))
    return found
