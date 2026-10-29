"""
Rough automatic void detection (die-to-die comparison).

The background inside a chip is not uniform (layout structures), but it repeats from chip to
chip. So every chip is compared with a "golden chip" of the same layer instead of with a flat
background:

1. Each (bonded) cell is cut out aligned on its reference bbox (the drawn bbox is the reference;
   the cell rect when there is none), reduced to a working size and normalized (median / MAD).
   For the comparison each chip is matched to the golden chip: a monotone intensity curve
   (brightness, contrast, gamma differ from chip to chip) and planar shading across the chip,
   fitted robustly (defect pixels excluded). A pixel inside the 3x3 min..max range of the golden
   chip counts as normal, so 1 px shifts and slightly different edges are tolerated.
2. Per layer, the golden chip is the per-pixel median over (a sample of) all chips, and the
   per-pixel spread is the median absolute deviation from it. Layout edges vary more from chip
   to chip (registration), so they get a larger spread and do not trigger.
3. |chip - golden| / spread above the threshold, brighter OR darker, cleaned with a small
   opening, gives the candidate blobs; each blob becomes an ellipse (its bounding box).
4. A spot found on several layers is marked once, on the layer chosen by layer_rule: the first
   where it appears from the top (default), the last, or the one with the strongest deviation.
   Spots overlapping an existing marking (any layer) are skipped.

Detected voids carry v["auto"] = "detected" (drawn dashed, reviewed with "Next Auto";
editing one clears the flag). The whole run is one undo step.

Parameters (config.json "void_detect", defaults in DEFAULTS):
    threshold        deviation from the golden chip, in spreads
    min_size_px      smallest void diameter kept (full-resolution pixels)
    max_area_frac    blobs larger than this fraction of the chip are ignored (misaligned / broken chip)
    work_size        longest side of the working chip image (pixels)
    template_cells   chips used to build the golden chip
    layer_rule       "first" (default) | "last" | "strongest": layer of a spot found on several layers
    type_rules       void type from the layout position and shape (sat_widgets/void_classify.py);
                     products.json "type_rules" of the selected product win
    method           "die_to_die" (default) or "patchcore" (CNN features, optional: needs
                     onnxruntime and models/*.onnx, see docs/PATCHCORE.md; pc_* parameters)
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .core_data import to_gray
from .void_classify import Classifier

DEFAULTS = {
    "threshold": 6.0,
    "min_size_px": 8.0,
    "max_area_frac": 0.2,
    "work_size": 256,
    "template_cells": 200,
    "method": "die_to_die", # or "patchcore" (optional, sat_widgets/patchcore.py)
    "layer_rule": "first",
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
    """
    Merges blobs whose boxes are within gap pixels (a void cut in two by a layout line).
    Returns [r0, r1, c0, c1, area, number of merged blobs].
    """
    comps = [list(c) + [1] for c in comps]
    merged = True
    while merged:
        merged = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i], comps[j]
                if a[0] - gap <= b[1] and b[0] - gap <= a[1] and a[2] - gap <= b[3] and b[2] - gap <= a[3]:
                    comps[i] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]), a[4] + b[4], a[5] + b[5]]
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


class _Matcher:
    """
    Maps a chip onto the golden chip, so chips that look a bit different compare equal:
    - intensity curve (brightness, contrast, gamma): the golden chip's pixels are grouped in
      classes by value (die, pads, lines, bumps ... each structure level is a class), and each
      class maps the chip's median over the class to the class' golden level. Medians per class:
      defects are a minority in every class, small structures (a pad) keep their own point.
      Piecewise linear between the points, linear beyond the ends (defects are not clipped);
    - planar shading across the chip (offset and gain planes), fitted robustly on what is left.
    Both are estimated twice, alternately.
    """
    LEVELS = 16
    MIN_CLASS = 0.003 # Fraction of the chip

    def __init__(self, golden):
        self.golden = golden
        h, w = golden.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.fx, self.fy = (xx / w - 0.5).ravel(), (yy / h - 0.5).ravel()
        g = golden.ravel()
        self.gc = g - float(g.mean())
        lo, hi = np.percentile(g, [0.5, 99.5])
        cls = np.clip(((g - lo) / max(hi - lo, 1e-9) * self.LEVELS).astype(int), 0, self.LEVELS - 1)
        self.classes = [np.nonzero(cls == k)[0] for k in range(self.LEVELS)]
        # Each class sampled at 2048 evenly spread pixels: all class medians in one call (speed)
        self.classes = np.stack([ix[np.linspace(0, len(ix) - 1, 2048).astype(int)]
                                 for ix in self.classes if len(ix) >= self.MIN_CLASS * g.size]) \
            if any(len(ix) >= self.MIN_CLASS * g.size for ix in self.classes) else np.zeros((0, 2048), int)
        self.sub = np.arange(0, g.size, 4) # Shading fit on a subsample (speed)
        fx, fy, gc = self.fx[self.sub], self.fy[self.sub], self.gc[self.sub]
        self.X = np.stack([np.ones_like(fx), fx, fy, gc, gc * fx, gc * fy], axis=1)
        self.gs = g[self.sub]

    def _curve(self, c, target):
        xs = np.median(c[self.classes], axis=1)
        ys = np.median(target[self.classes], axis=1)
        order = np.argsort(xs)
        xs = xs[order] + np.arange(len(xs)) * 1e-6
        ys = np.maximum.accumulate(ys[order])
        return xs, ys

    @staticmethod
    def _map(v, xs, ys):
        if len(xs) < 2:
            return v - xs[0] + ys[0] if len(xs) else v
        out = np.interp(v, xs, ys)
        s_all = max((ys[-1] - ys[0]) / max(xs[-1] - xs[0], 1e-9), 1e-3)
        hi, lo = v > xs[-1], v < xs[0]
        out[hi] = ys[-1] + (v[hi] - xs[-1]) * s_all # Beyond the ends: overall slope
        out[lo] = ys[0] + (v[lo] - xs[0]) * s_all
        return out

    def _shading(self, r):
        keep = np.ones(r.shape, bool)
        coef = np.zeros(6)
        for _ in range(3):
            if keep.sum() < 32:
                break
            Xk = self.X[keep]
            A = Xk.T @ Xk + np.eye(6) * 1e-6 # Normal equations (6 x 6): fast
            coef = np.linalg.solve(A, Xk.T @ r[keep])
            res = r - self.X @ coef
            mad = float(np.median(np.abs(res[keep]))) * 1.4826 + 1e-6
            keep = np.abs(res) < 3.0 * mad
        return coef

    def apply(self, chip):
        c = chip.ravel()
        g = self.golden.ravel()
        shade = np.zeros_like(g)
        for _ in range(2):
            xs, ys = self._curve(c, g + shade) # Curve towards golden + current shading
            coef = self._shading(self._map(c[self.sub], xs, ys) - self.gs)
            c0, c1, c2, d0, d1, d2 = (float(k) for k in coef)
            shade = c0 + c1 * self.fx + c2 * self.fy + (d0 + d1 * self.fx + d2 * self.fy) * self.gc
        return (self._map(c, xs, ys) - shade).reshape(chip.shape).astype(np.float32)


def _filt(a, f):
    return np.asarray(Image.fromarray(np.asarray(a, np.float32)).filter(f))


def _open(mask):
    """Binary opening 3x3 (removes isolated pixels / thin lines)."""
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))) > 0


def detect_voids(stack, grid_cfg, coord_transform, refs, cells, layer_numbers, void_manager, type_name,
                 params=None, progress=None, diag=None, apply=True):
    """
    Detects voids on all loaded layers (index z -> absolute layer_numbers[z]) and adds them to
    void_manager with type_name. cells: [(row, col)] to inspect. refs: reference bboxes.
    progress(i, n, text) -> False cancels. Returns {"added": n, "per_layer": {layer: n}} or None.
    diag: collector of every stage (sat_widgets/void_eval.py). apply=False: nothing is added and
    existing markings are ignored (evaluation against them); the result has "found".
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

    # Existing markings (cell -> [(gx, gy, rx, ry, layer)]) and the voids found so far per layer
    existing, taken = {}, {}
    for layer, vs in (void_manager.voids.items() if apply else ()):
        for v in vs:
            if void_manager.is_bbox(v):
                continue
            gx, gy = to_deskew(v["globalCX"], v["globalCY"])
            existing.setdefault(grid_cfg.cell_at(gx, gy), []).append((gx, gy, v["radiusX"], v["radiusY"], layer))

    def overlaps(cell, gx, gy, rx, ry, layer):
        """Layer of the existing marking (any layer) / find on this layer the spot overlaps, else None."""
        for (ox, oy, orx, ory, olayer) in existing.get(cell, []) + [t for t in taken.get(cell, []) if t[4] == layer]:
            if abs(gx - ox) < max(rx, orx) and abs(gy - oy) < max(ry, ory):
                return olayer
        return None

    order = sorted(rects)
    n_steps = len(layer_numbers) * len(order)
    step = 0
    per_layer = {}
    found = [] # (layer, gx, gy, rx, ry, score)
    min_area = max(2, int((p["min_size_px"] / factor) ** 2 * math.pi / 4))
    max_area = p["max_area_frac"] * size[0] * size[1]
    gap = max(2, int(round(0.05 * max(size))))
    pc = None
    classifier = Classifier(p.get("type_rules"), void_manager.layout, void_manager.types, void_manager.um_per_px)
    types = {}
    if diag is not None:
        diag.setup(p, size, factor, rects, min_area, max_area, gap)

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
        threshold = float(p.get("pc_threshold", 10.0)) if p["method"] == "patchcore" else p["threshold"]
        if diag is not None:
            diag.layer_start(layer, chips, threshold)

        if p["method"] == "patchcore":
            # Optional scorer (sat_widgets/patchcore.py): CNN features, position-aware nearest neighbour
            if pc is None:
                from . import patchcore
                pc = patchcore.PatchCore(p)
            if progress is not None and not progress(step, n_steps, f"Layer {layer}: PatchCore scoring"):
                return None
            if p.get("pc_match", False):
                # Same brightness / contrast / shading matching as die-to-die before the CNN
                match = _golden(chips, p)[3]
                chips = {k: match.apply(c.astype(np.float32)) for k, c in chips.items()}
            items = ((cell, dev, None, None) for cell, dev in pc.score_layer(chips, size).items())
        else:
            items = _die_to_die(chips, p, diag)

        # 3. Blobs per chip
        count = 0
        for cell, dev, signed, matched in items:
            mask0 = dev > threshold
            mask = _open(mask0) if mask0.any() else mask0
            comps = _merge_close(_components(mask), gap) if mask.any() else []
            info = [] if diag is not None else None
            x0, y0, w, h = rects[cell]
            sx, sy = w / size[0], h / size[1]
            for r0, r1, c0, c1, area, merged in comps:
                gx = x0 + (c0 + c1) / 2.0 * sx
                gy = y0 + (r0 + r1 + 1) / 2.0 * sy
                rx = max(1.0, (c1 - c0) / 2.0 * sx)
                ry = max(1.0, (r1 + 1 - r0) / 2.0 * sy)
                blob = dev[r0:r1 + 1, c0:c1]
                c = {"box": (r0, r1, c0, c1), "area": int(area), "merged": merged, "score": float(blob.max())}
                if info is not None:
                    info.append(c)
                if area < min_area or area > max_area:
                    c["status"] = "small" if area < min_area else "large"
                    continue
                dup = overlaps(cell, gx, gy, rx, ry, layer)
                if dup is not None:
                    c["status"], c["dup_layer"] = "dup", dup
                    continue # Marked already, or found on this layer already
                score = float(blob.max())
                vtype = type_name
                extent = (x0 + c0 * sx, y0 + r0 * sy, x0 + c1 * sx, y0 + (r1 + 1) * sy)
                if classifier:
                    bw_, bh_ = c1 - c0, r1 + 1 - r0
                    feat = {"center": (gx, gy),
                            "extent": extent,
                            "size": max(bw_ * sx, bh_ * sy),
                            "aspect": max(bw_ * sx, bh_ * sy) / max(min(bw_ * sx, bh_ * sy), 1e-6),
                            "fill": area / float(bw_ * bh_),
                            "polarity": None if signed is None else
                            ("bright" if float((signed[r0:r1 + 1, c0:c1] * blob).sum()) > 0 else "dark")}
                    vtype = classifier.type_for(feat, rects[cell], type_name)
                taken.setdefault(cell, []).append((gx, gy, rx, ry, layer))
                found.append((layer, gx, gy, rx, ry, score, vtype, cell, extent, c))
                c["status"], c["type"], c["extent"] = "kept", vtype, extent
                count += 1
            if diag is not None:
                diag.chip(layer, cell, dev, signed, matched, mask0, mask, info)
        per_layer[layer] = count
        if diag is not None:
            diag.layer_end(layer)

    # 4. A spot found on several layers is one void, marked on the layer chosen by layer_rule:
    #    "first" (where it first appears from the top), "last", or "strongest" (largest deviation)
    rule = p.get("layer_rule", "first")
    prefer = {"first": lambda f: (f[0], -f[5]), "last": lambda f: (-f[0], -f[5])}.get(rule, lambda f: (-f[5], f[0]))
    chosen, by_cell = set(), {}
    for f in sorted(found, key=prefer):
        layer, gx, gy, rx, ry, cell, c = f[0], f[1], f[2], f[3], f[4], f[7], f[9]
        hit = next((k for k in by_cell.get(cell, ()) if abs(gx - k[1]) < max(rx, k[3]) and abs(gy - k[2]) < max(ry, k[4])), None)
        if hit is not None:
            c["status"], c["dup_layer"] = "dup", hit[0]
            continue
        by_cell.setdefault(cell, []).append(f)
        chosen.add(id(f))
    found = [f for f in found if id(f) in chosen]
    per_layer = {layer: sum(1 for f in found if f[0] == layer) for layer in per_layer}

    if not apply:
        return {"added": 0, "per_layer": per_layer, "found": found}

    # 4. One undo step for the whole run
    vm = void_manager
    vm.begin_change()
    try:
        for layer, gx, gy, rx, ry, score, vtype, cell, extent, _ in found:
            # Placed like a hand marking (same function as a click). Corner-anchored types attach
            # to the bbox corner nearest the blob, sized to its far extent corner (like drawing
            # it from the corner).
            far = None
            if vm.snap_anchor({"type": vtype}) is not None:
                bx, by, bw, bh = rects[cell]
                kx, ky = min(((bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)),
                             key=lambda k: math.hypot(k[0] - gx, k[1] - gy))
                far = max(((extent[0], extent[1]), (extent[2], extent[1]), (extent[2], extent[3]), (extent[0], extent[3])),
                          key=lambda q: math.hypot(q[0] - kx, q[1] - ky))
            v = vm.place_void(layer, *to_raw(gx, gy), rx, ry, vtype, grid_cfg, coord_transform,
                              hint=(gx, gy), drag_point=far)
            v["auto"] = "detected" # Only the dashed outline until reviewed (any edit clears it)
            v["score"] = round(float(score), 1)
            types[vtype] = types.get(vtype, 0) + 1
    finally:
        vm.end_change()
    return {"added": len(found), "per_layer": per_layer, "types": types}


def _golden(chips, p):
    """Golden chip, per-pixel spread, 3x3 min / max of the golden chip and its matcher."""
    # Golden chip: median of the sample, then again after matching the sample to it
    keys = list(chips)
    rng = np.random.default_rng(0)
    sample = keys if len(keys) <= p["template_cells"] else \
        [keys[i] for i in rng.choice(len(keys), p["template_cells"], replace=False)]
    golden0 = np.median(np.stack([chips[k] for k in sample]).astype(np.float32), axis=0)
    match0 = _Matcher(golden0)
    matched = np.stack([match0.apply(chips[k].astype(np.float32)) for k in sample])
    golden = np.median(matched, axis=0)
    # Per-pixel chip-to-chip spread of the matched chips (larger at layout edges)
    spread = np.median(np.abs(matched - golden), axis=0) * 1.4826
    del matched
    spread = _filt(spread, ImageFilter.MaxFilter(3))
    spread = np.maximum(spread, 0.5 * float(np.median(spread)) + 1e-3)
    # Local range of the golden chip: a pixel inside the 3x3 min..max of the golden chip is
    # not a defect (1 px shifts, slightly different edges)
    gmin = _filt(golden, ImageFilter.MinFilter(3))
    gmax = _filt(golden, ImageFilter.MaxFilter(3))
    return spread, gmin, gmax, _Matcher(golden)


def _die_to_die(chips, p, diag=None):
    """
    Yields (cell, deviation map, sign, matched chip): |chip - golden| / spread after matching,
    see the module doc.
    """
    spread, gmin, gmax, match = _golden(chips, p)
    if diag is not None:
        diag.golden(match.golden, spread)
    for cell in chips:
        m = match.apply(chips[cell].astype(np.float32))
        hi, lo = np.maximum(m - gmax, 0), np.maximum(gmin - m, 0)
        yield cell, (hi + lo) / spread, np.sign(hi - lo), m # Sign: brighter / darker than the golden chip
