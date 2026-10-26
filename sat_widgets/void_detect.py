"""
Rough automatic void detection (die-to-die comparison).

The background inside a chip is not uniform (layout structures), but it repeats from chip to
chip. So every chip is compared with a "golden chip" of the same layer instead of with a flat
background:

1. Each (bonded) cell is cut out aligned on its reference bbox (the drawn bbox is the reference;
   the cell rect when there is none), reduced to a working size and normalized (median / MAD).
   For the comparison, each chip's gain and offset are fitted to the golden chip robustly
   (defect pixels excluded), so big defects do not make the whole layout look different.
2. Per layer, the golden chip is the per-pixel median over (a sample of) all chips, and the
   per-pixel spread is the median absolute deviation from it. Layout edges vary more from chip
   to chip (registration), so they get a larger spread and do not trigger.
3. |chip - golden| / spread above the threshold, brighter OR darker, cleaned with a small
   opening, gives the candidate blobs; each blob becomes an ellipse (its bounding box).
4. Layers are processed from the top: a void is marked once, on the first layer where it
   appears; on later layers the same spot is skipped. Spots overlapping an existing marking
   (any layer) are skipped too.

Detected voids carry v["auto"] = "detected" (drawn dashed, reviewed with "Next Auto";
editing one clears the flag). The whole run is one undo step.

Parameters (config.json "void_detect", defaults in DEFAULTS):
    threshold        deviation from the golden chip, in spreads
    min_size_px      smallest void diameter kept (full-resolution pixels)
    max_area_frac    blobs larger than this fraction of the chip are ignored (misaligned / broken chip)
    work_size        longest side of the working chip image (pixels)
    template_cells   chips used to build the golden chip
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .core_data import to_gray

DEFAULTS = {
    "threshold": 6.0,
    "min_size_px": 8.0,
    "max_area_frac": 0.2,
    "work_size": 256,
    "template_cells": 200,
}


def _components(mask):
    """Connected components (4-neighbour) of a small boolean mask, from row runs + union-find."""
    runs = [] # (row, c0, c1)
    for r in np.nonzero(mask.any(axis=1))[0]:
        row = np.concatenate(([False], mask[r], [False]))
        d = np.diff(row.astype(np.int8))
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
            if c0 < d1 and d0 < c1:
                parent[find(i)] = find(j)
        by_row.setdefault(r, []).append(i)
    comps = {}
    for i, (r, c0, c1) in enumerate(runs):
        b = comps.setdefault(find(i), [r, r, c0, c1, 0])
        b[0], b[1] = min(b[0], r), max(b[1], r)
        b[2], b[3] = min(b[2], c0), max(b[3], c1)
        b[4] += c1 - c0
    return list(comps.values()) # [r0, r1, c0, c1 (exclusive), area]


def _merge_close(comps, gap):
    """Merges blobs whose boxes are within gap pixels (a void cut in two by a layout line)."""
    comps = [list(c) for c in comps]
    merged = True
    while merged:
        merged = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i], comps[j]
                if a[0] - gap <= b[1] and b[0] - gap <= a[1] and a[2] - gap <= b[3] and b[2] - gap <= a[3]:
                    comps[i] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    del comps[j]
                    merged = True
                    break
            if merged:
                break
    return comps


def _chip(frame_region, quad, size, factor):
    """Aligned chip from a raw region: box-reduced by factor, then QUAD-warped at working size."""
    img = Image.fromarray(frame_region.astype(np.float32))
    if factor > 1:
        img = img.reduce(factor)
        quad = tuple(v / factor for v in quad)
    return np.asarray(img.transform(size, Image.QUAD, quad, resample=Image.BILINEAR), dtype=np.float32)


def _normalize(a):
    sub = a[::3, ::3] # Robust statistics from a subsample (speed)
    med = float(np.median(sub))
    mad = float(np.median(np.abs(sub - med))) * 1.4826 + 1e-3
    return (a - med) / mad


def _fit_gain(chip, golden):
    """
    chip ~ a * golden + b, fitted robustly (pixels of large residual, i.e. defects, are dropped
    and the fit repeated), so a big defect does not skew the brightness match of the whole chip.
    """
    x = golden.ravel()
    y = chip.ravel()
    keep = np.ones(x.shape, bool)
    a, b = 1.0, 0.0
    for _ in range(3):
        xs, ys = x[keep], y[keep]
        if xs.size < 16:
            break
        vx = float(xs.var())
        if vx < 1e-9:
            a, b = 1.0, float(np.median(ys - xs))
        else:
            a = float(((xs - xs.mean()) * (ys - ys.mean())).mean() / vx)
            b = float(ys.mean() - a * xs.mean())
        res = y - (a * x + b)
        mad = float(np.median(np.abs(res[keep][::5]))) * 1.4826 + 1e-6
        keep = np.abs(res) < 3.0 * mad
    return a, b


def _open(mask):
    """Binary opening 3x3 (removes isolated pixels / thin lines)."""
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))) > 0


def detect_voids(stack, grid_cfg, coord_transform, refs, cells, layer_numbers, void_manager, type_name,
                 params=None, progress=None):
    """
    Detects voids on all loaded layers (index z -> absolute layer_numbers[z]) and adds them to
    void_manager with type_name. cells: [(row, col)] to inspect. refs: reference bboxes.
    progress(i, n, text) -> False cancels. Returns {"added": n, "per_layer": {layer: n}} or None.
    """
    p = dict(DEFAULTS)
    p.update(params or {})
    to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))
    to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))

    # Chip rectangles (Deskewed): reference bbox, else the cell. One working size for all chips.
    rects = {}
    for (r, c) in cells:
        if (r, c) in refs:
            rects[(r, c)] = refs[(r, c)]
        else:
            x0, y0 = grid_cfg.cell_origin(r, c)
            rects[(r, c)] = (x0, y0, grid_cfg.pitch_x, grid_cfg.pitch_y)
    if not rects:
        return {"added": 0, "per_layer": {}}
    bw = float(np.median([v[2] for v in rects.values()]))
    bh = float(np.median([v[3] for v in rects.values()]))
    factor = max(1, int(math.ceil(max(bw, bh) / p["work_size"])))
    size = (max(8, int(round(bw / factor))), max(8, int(round(bh / factor))))

    # Existing markings (cell -> [(gx, gy, rx, ry)]) and the voids found so far (first layer wins)
    taken = {}
    for vs in void_manager.voids.values():
        for v in vs:
            if void_manager.is_bbox(v):
                continue
            gx, gy = to_deskew(v["globalCX"], v["globalCY"])
            taken.setdefault(grid_cfg.cell_at(gx, gy), []).append((gx, gy, v["radiusX"], v["radiusY"]))

    def overlaps(cell, gx, gy, rx, ry):
        for (ox, oy, orx, ory) in taken.get(cell, ()):
            if abs(gx - ox) < max(rx, orx) and abs(gy - oy) < max(ry, ory):
                return True
        return False

    order = sorted(rects)
    n_steps = len(layer_numbers) * len(order)
    step = 0
    per_layer = {}
    found = [] # (layer, gx, gy, rx, ry, score)
    min_area = max(2, int((p["min_size_px"] / factor) ** 2 * math.pi / 4))
    max_area = p["max_area_frac"] * size[0] * size[1]
    gap = max(2, int(round(0.05 * max(size))))

    for z, layer in enumerate(layer_numbers):
        # 1. Aligned, normalized chips of this layer
        chips = {}
        for cell in order:
            if progress is not None and not progress(step, n_steps, f"Layer {layer}: reading chips"):
                stack.release_cache()
                return None
            step += 1
            x0, y0, w, h = rects[cell]
            corners = [to_raw(x0, y0), to_raw(x0, y0 + h), to_raw(x0 + w, y0 + h), to_raw(x0 + w, y0)] # PIL QUAD order
            xs = [q[0] for q in corners]
            ys = [q[1] for q in corners]
            rx0, ry0 = max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys))))
            rx1, ry1 = min(stack.width, int(math.ceil(max(xs)))), min(stack.height, int(math.ceil(max(ys))))
            if rx1 - rx0 < 4 or ry1 - ry0 < 4:
                continue
            region = to_gray(stack.read_region(z, rx0, ry0, rx1 - rx0, ry1 - ry0))
            quad = tuple(v for q in corners for v in (q[0] - rx0, q[1] - ry0))
            chips[cell] = _normalize(_chip(region, quad, size, factor)).astype(np.float16)
        stack.release_cache()
        if len(chips) < 3:
            continue

        # 2. Golden chip and per-pixel spread
        keys = list(chips)
        rng = np.random.default_rng(0)
        sample = keys if len(keys) <= p["template_cells"] else \
            [keys[i] for i in rng.choice(len(keys), p["template_cells"], replace=False)]
        stackarr = np.stack([chips[k] for k in sample]).astype(np.float32)
        golden = np.median(stackarr, axis=0)
        spread = np.median(np.abs(stackarr - golden), axis=0) * 1.4826
        del stackarr
        spread = np.asarray(Image.fromarray(spread).filter(ImageFilter.MaxFilter(3))) # Registration jitter
        spread = np.maximum(spread, 0.5 * float(np.median(spread)) + 1e-3)

        # 3. Blobs per chip
        count = 0
        for cell in keys:
            chip = chips[cell].astype(np.float32)
            a, b = _fit_gain(chip, golden)
            dev = np.abs(chip - (a * golden + b)) / (max(abs(a), 0.2) * spread)
            mask = dev > p["threshold"]
            if not mask.any():
                continue
            mask = _open(mask)
            if not mask.any():
                continue
            x0, y0, w, h = rects[cell]
            sx, sy = w / size[0], h / size[1]
            for r0, r1, c0, c1, area in _merge_close(_components(mask), gap):
                if area < min_area or area > max_area:
                    continue
                gx = x0 + (c0 + c1) / 2.0 * sx
                gy = y0 + (r0 + r1 + 1) / 2.0 * sy
                rx = max(1.0, (c1 - c0) / 2.0 * sx)
                ry = max(1.0, (r1 + 1 - r0) / 2.0 * sy)
                if overlaps(cell, gx, gy, rx, ry):
                    continue # Marked already, or found on an upper layer
                score = float(dev[r0:r1 + 1, c0:c1].max())
                taken.setdefault(cell, []).append((gx, gy, rx, ry))
                found.append((layer, gx, gy, rx, ry, score))
                count += 1
        per_layer[layer] = count

    # 4. One undo step for the whole run
    vm = void_manager
    vm.begin_change()
    try:
        for layer, gx, gy, rx, ry, score in found:
            v = vm.add_void(layer, *to_raw(gx, gy), rx, ry, type_name)
            v["auto"] = "detected"
            v["score"] = round(score, 1)
    finally:
        vm.end_change()
    return {"added": len(found), "per_layer": per_layer}
