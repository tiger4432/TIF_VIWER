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
   With "smooth" (default) the signed difference is first averaged over a small neighbourhood
   and compared with the chip-to-chip spread of that average: large faint voids, invisible pixel
   by pixel, rise above the noise (see _die_to_die).
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
    skip_layers      absolute layer numbers not searched (default [0]: the top layer is noisy)
    smooth           averaging radius (Gaussian sigma, working px) of the deviation; 0 = pixel by
                     pixel (sharper outlines, misses large faint voids)
    type_rules       void type from the layout position and shape (sat_widgets/void_classify.py);
                     products.json "type_rules" of the selected product win
    method           "die_to_die" (default), "patchcore" (CNN features, optional: needs
                     onnxruntime and models/*.onnx, see docs/PATCHCORE.md; pc_* parameters) or
                     "layer_change" (all layers at once: each layer explained from the layers above
                     it and the golden chips; lc_* parameters, see void_layer_change.py)
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
    "method": "die_to_die", # or "patchcore" (optional, sat_widgets/patchcore.py), "layer_change"
    "layer_rule": "first",
    "smooth": 1.5, # Averaging of the deviation (working px) for large faint voids; 0 = per pixel
    "skip_layers": [0], # Absolute layer numbers not searched (top layer: surface echo noise)
    "layout_fit": True, # Fit away per-chip layout contrast / sharpness (see _LayoutFit)
    "register": True, # Move every chip onto the median chip (subpixel) before comparing
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


def _phase_shift(a, b):
    """Translation (dx, dy) of the content of a against b (working px) by phase correlation, subpixel."""
    a = np.nan_to_num(np.asarray(a, np.float32), posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, np.float32), posinf=0.0, neginf=0.0)
    # Hann window: the image border (same place in both) would otherwise pull the peak to 0;
    # partial whitening (|R|^0.5) keeps the peak sharp without amplifying empty frequencies or
    # locking onto a repeated pattern (bump array) one period off
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    A = np.fft.rfft2((a - a.mean()) * win)
    B = np.fft.rfft2((b - b.mean()) * win)
    R = A * np.conj(B)
    R /= np.abs(R) ** 0.5 + 1e-9
    r = np.fft.irfft2(R, s=a.shape)
    h, w = r.shape
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)

    def sub(c0, cm, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d

    dy = iy + sub(r[iy, ix], r[(iy - 1) % h, ix], r[(iy + 1) % h, ix])
    dx = ix + sub(r[iy, ix], r[iy, (ix - 1) % w], r[iy, (ix + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dx), float(dy)


def _translate(a, dx, dy):
    """a sampled at (x + dx, y + dy): undoes a content shift (dx, dy); bilinear, edges extended."""
    h, w = a.shape
    pad = int(math.ceil(max(abs(dx), abs(dy)))) + 1
    ap = np.pad(a.astype(np.float32), pad, mode="edge")
    ix, iy = int(math.floor(dx)), int(math.floor(dy))
    fx, fy = dx - ix, dy - iy
    y0, x0 = pad + iy, pad + ix
    p00 = ap[y0:y0 + h, x0:x0 + w]
    p01 = ap[y0:y0 + h, x0 + 1:x0 + 1 + w]
    p10 = ap[y0 + 1:y0 + 1 + h, x0:x0 + w]
    p11 = ap[y0 + 1:y0 + 1 + h, x0 + 1:x0 + 1 + w]
    return (p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy) + p10 * (1 - fx) * fy + p11 * fx * fy).astype(np.float32)


def _register(chips, max_frac=0.1):
    """
    Chips moved onto the median chip of the layer (subpixel phase correlation): the bbox does not
    have to sit exactly on every chip. Shifts larger than max_frac of the chip are not trusted
    (left as cut). Returns ({cell: registered chip}, {cell: (dx, dy)}).
    """
    keys = list(chips)
    idx = np.linspace(0, len(keys) - 1, min(len(keys), 48)).astype(int)
    ref = np.median(np.stack([chips[keys[i]].astype(np.float32) for i in idx]), axis=0)
    lim = max_frac * max(ref.shape)
    for _ in range(2): # Second pass against the sharper median of the registered chips
        out, shifts = {}, {}
        for k, c in chips.items():
            a = c.astype(np.float32)
            dx, dy = _phase_shift(a, ref)
            if not (math.isfinite(dx) and math.isfinite(dy)) or math.hypot(dx, dy) > lim:
                dx = dy = 0.0 # Not trusted: left as cut
            out[k] = _translate(a, dx, dy).astype(np.float16) if (dx or dy) else c
            shifts[k] = (dx, dy)
        ref = np.median(np.stack([out[keys[i]].astype(np.float32) for i in idx]), axis=0)
    return out, shifts


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


class _LayoutFit:
    """
    Removes what the golden chip explains from a matched chip: per chip, the layout may be drawn
    with another contrast or sharpness (focus), varying across the chip. The difference to the
    golden chip is fitted (robust least squares) with
      golden * w_j                       contrast of the layout,
      (blur_s(golden) - golden) * w_j    sharpness, s = 1, 2, 4 working px,
      w_j                                brightness,
    where w_j are 4 bilinear weights over the chip (corners), so the fit may vary smoothly across
    it. A void is not in the golden chip and is local, so it stays in the remainder.
    """
    SIGMAS = (1.0, 2.0, 4.0)

    def __init__(self, golden):
        self.golden = golden
        h, w = golden.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u, v = (xx / max(w - 1, 1)).ravel(), (yy / max(h - 1, 1)).ravel()
        weights = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v]
        g = golden.ravel() - float(golden.mean())
        bases = [g] + [(_blur(golden, s) - golden).ravel() for s in self.SIGMAS] + [np.ones_like(g)]
        self.X = np.stack([b * wj for b in bases for wj in weights], axis=1).astype(np.float32)
        self.sub = np.arange(0, g.size, 2) # Fit on a subsample (speed)

    def apply(self, m):
        r = (m - self.golden).ravel()
        X, sub = self.X, self.sub
        keep = sub
        coef = None
        for _ in range(3):
            Xk = X[keep]
            A = Xk.T @ Xk + np.eye(X.shape[1], dtype=np.float32) * 1e-3
            coef = np.linalg.solve(A, Xk.T @ r[keep])
            res = r[sub] - X[sub] @ coef
            mad = float(np.median(np.abs(res))) * 1.4826 + 1e-6
            keep = sub[np.abs(res) < 3.0 * mad] # Voids and other outliers do not steer the fit
            if keep.size < X.shape[1] * 4:
                break
        return (m.ravel() - X @ coef).reshape(m.shape).astype(np.float32)


def _filt(a, f):
    return np.asarray(Image.fromarray(np.asarray(a, np.float32)).filter(f))


def _open(mask):
    """Binary opening 3x3 (removes isolated pixels / thin lines)."""
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))) > 0


def detect_voids(stack, grid_cfg, coord_transform, refs, cells, layer_numbers, void_manager, type_name,
                 params=None, progress=None, diag=None, apply=True, on_found=None, on_cells=None):
    """
    Detects voids on all loaded layers (index z -> absolute layer_numbers[z]) and adds them to
    void_manager with type_name. cells: [(row, col)] to inspect. refs: reference bboxes.
    progress(i, n, text) -> False cancels. Returns {"added": n, "per_layer": {layer: n}} or None.
    diag: collector of every stage (sat_widgets/void_eval.py). apply=False: nothing is added and
    existing markings are ignored (evaluation against them); the result has "found".
    Background runs (the GUI thread places the voids): on_found(finds) receives the kept finds
    instead of placing them (place_found places them), per layer as soon as they are final (layer_rule
    "first", layers from the top), else at the end; on_cells(cells, frac): share of the work done on
    these cells (1.0 = done). void_manager is then only read (a copy taken on the GUI thread).
    """
    p = dict(DEFAULTS)
    p.update(params or {})
    to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))
    to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))

    rects = chip_rects(grid_cfg, refs, cells)
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
    skip = {int(x) for x in (p.get("skip_layers") or [])}
    layer_numbers = list(layer_numbers)
    layer_z = [(z, layer) for z, layer in enumerate(layer_numbers) if layer not in skip]
    n_steps = len(layer_z) * len(order)
    step = 0
    per_layer = {}
    found = [] # (layer, gx, gy, rx, ry, score)
    min_area = max(2, int((p["min_size_px"] / factor) ** 2 * math.pi / 4))
    max_area = p["max_area_frac"] * size[0] * size[1]
    gap = max(2, int(round(0.05 * max(size))))
    pc = None
    # Learned filter (sat_widgets/void_learn.py): candidates scored by a model trained from hand
    # markings replace the threshold; "learn" collects the candidates for training instead
    model, collect = p.get("learned_model"), p.get("learn")
    learned = (model is not None or collect is not None) and p["method"] not in ("patchcore", "layer_change")
    if learned:
        from . import void_learn
        cand_thr = float(p.get("cand_threshold", model.cand_threshold if model is not None else 1.0))
    classifier = Classifier(p.get("type_rules"), void_manager.layout, void_manager.types, void_manager.um_per_px)
    types = {}
    if diag is not None:
        diag.setup(p, size, factor, rects, min_area, max_area, gap)
    layer_diffs = None
    reg_shifts = None # Chip registration, measured once (first searched layer), used for all layers
    seg_model = p.get("seg_model")
    if seg_model is not None and getattr(seg_model, "neighbors", 0) and p.get("dump") is None:
        # Multi-layer model: the chips of all layers first (the neighbours of each layer)
        layer_diffs = _layer_chips(stack, grid_cfg, coord_transform, refs, cells, layer_numbers, void_manager, p,
                                   (lambda i, n, t: progress(i // 2, n, "Neighbour layers: " + t)) if progress else None)
        if layer_diffs is None:
            return None

    def read_layer(z, layer):
        """Aligned, normalized chips {cell: float16} of one layer; None when cancelled."""
        nonlocal step
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
            chips[cell] = np.clip(_normalize(_chip(region, quad, size, factor)), -6e4, 6e4).astype(np.float16)
        stack.release_cache()
        return chips

    lc = p["method"] == "layer_change"
    if lc:
        # All layers at once (sat_widgets/void_layer_change.py): each layer explained from the layers above
        n_steps = 2 * len(layer_numbers) * len(order)

        def accept(f):
            """Skipped when marked already; void type from the rules."""
            layer, gx, gy, rx, ry, cell, ext, c = f[0], f[1], f[2], f[3], f[4], f[7], f[8], f[9]
            dup = overlaps(cell, gx, gy, rx, ry, layer)
            if dup is not None:
                c["status"], c["dup_layer"] = "dup", dup
                return None
            vtype = type_name
            if classifier:
                bw_, bh_ = ext[2] - ext[0], ext[3] - ext[1]
                vtype = classifier.type_for({"center": (gx, gy), "extent": ext, "size": max(bw_, bh_),
                                             "aspect": max(bw_, bh_) / max(min(bw_, bh_), 1e-6),
                                             "fill": math.pi / 4, "polarity": None}, rects[cell], type_name)
            c["type"] = vtype
            taken.setdefault(cell, []).append((gx, gy, rx, ry, layer))
            return f[:6] + (vtype,) + f[7:]

        found = _layer_change(read_layer, layer_numbers, skip, order, rects, size, p, progress, diag,
                              lambda: step, accept, on_cells)
        if found is None:
            return None
        per_layer = {layer: sum(1 for f in found if f[0] == layer) for _, layer in layer_z}

    # Finds handed over per layer (on_found): final once their layer is done when the first layer wins
    # and the layers come from the top (a deeper layer cannot take a spot over)
    rule = p.get("layer_rule", "first")
    early = on_found is not None and apply and rule == "first" and \
        [layer for _, layer in layer_z] == sorted(layer for _, layer in layer_z)
    sent, settled = set(), {}
    for li, (z, layer) in enumerate([] if lc else layer_z):
        # 1. Aligned, normalized chips of this layer
        chips = read_layer(z, layer)
        if chips is None:
            return None
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
            # Each chip moved onto the median chip first (the bbox may be off by a few px)
            if p.get("register", True):
                # One shift per chip for every layer: the layers of a scan are aligned with each
                # other, only the bbox may sit off the chip (measured on the first searched layer)
                if reg_shifts is None:
                    src, reg_shifts = _register(chips)
                else:
                    src = {c: (_translate(a.astype(np.float32), *reg_shifts[c]).astype(np.float16)
                               if reg_shifts.get(c, (0.0, 0.0)) != (0.0, 0.0) else a) for c, a in chips.items()}
                shifts = reg_shifts
            else:
                src, shifts = chips, {}
            dctx = {"shifts": shifts}
            items = _die_to_die(src, p, diag, dctx)
            dump = p.get("dump")
            if dump is not None:
                # Training data (sat_widgets/void_seg.py): the chips as the detector sees them
                for cell, _, _, matched in items:
                    dump(layer, cell, matched, dctx["golden"], dctx["spread"], dctx["shifts"].get(cell, (0.0, 0.0)),
                         rects[cell], size)
                continue
            seg = p.get("seg_model")
            if seg is not None:
                items = _learned_maps(items, dctx, seg, p, diag, layer, layer_diffs)

        # 3. Blobs per chip
        count = 0
        for cell, dev, signed, matched in items:
            probs = None
            if learned:
                cands = void_learn.candidates(void_learn.ChipContext(
                    dev * signed, matched, dctx["golden"], dctx["spread"], dctx["sp_med"]), cand_thr)
                if collect is not None:
                    bx, by, bw, bh = rects[cell]
                    ux, uy = bw / size[0], bh / size[1]
                    collect(layer, cell, [((bx + b[2] * ux, by + b[0] * uy, bx + b[3] * ux, by + (b[1] + 1) * uy), area, f)
                                          for b, area, f in cands])
                    if diag is None:
                        continue
                mask0 = dev > cand_thr
                mask = mask0
                comps = [(b[0], b[1], b[2], b[3], area, 1) for b, area, _ in cands]
                probs = list(model.scores([f for _, _, f in cands])) if (model is not None and cands) else [None] * len(cands)
            else:
                mask0 = dev > threshold
                mask = _open(mask0) if mask0.any() else mask0
                comps = _merge_close(_components(mask), gap) if mask.any() else []
            info = [] if diag is not None else None
            x0, y0, w, h = rects[cell]
            sx, sy = w / size[0], h / size[1]
            for ci, (r0, r1, c0, c1, area, merged) in enumerate(comps):
                gx = x0 + (c0 + c1) / 2.0 * sx
                gy = y0 + (r0 + r1 + 1) / 2.0 * sy
                rx = max(1.0, (c1 - c0) / 2.0 * sx)
                ry = max(1.0, (r1 + 1 - r0) / 2.0 * sy)
                blob = dev[r0:r1 + 1, c0:c1]
                c = {"box": (r0, r1, c0, c1), "area": int(area), "merged": merged, "score": float(blob.max())}
                if info is not None:
                    info.append(c)
                if probs is not None:
                    if probs[ci] is None or probs[ci] < model.cut:
                        c["status"] = "cls" # Rejected by the learned filter (or collecting only)
                        continue
                    c["learned"] = float(probs[ci])
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
        if early:
            batch = _settle([f for f in found if f[0] == layer], rule, settled)
            sent.update(id(f) for f in batch)
            if batch:
                on_found(batch)
        if on_cells is not None:
            on_cells(order, (li + 1) / len(layer_z))

    # 4. A spot found on several layers is one void, marked on the layer chosen by layer_rule:
    #    "first" (where it first appears from the top), "last", or "strongest" (largest deviation)
    found = _settle(found, rule, {})
    per_layer = {layer: sum(1 for f in found if f[0] == layer) for layer in per_layer}
    if on_cells is not None:
        on_cells(order, 1.0)

    if not apply:
        return {"added": 0, "per_layer": per_layer, "found": found}
    if on_found is not None:
        rest = [f for f in found if id(f) not in sent]
        if rest:
            on_found(rest)
        for f in found:
            types[f[6]] = types.get(f[6], 0) + 1
        return {"added": len(found), "per_layer": per_layer, "types": types}

    # 4. One undo step for the whole run
    vm = void_manager
    vm.begin_change()
    try:
        types = place_found(vm, found, rects, grid_cfg, coord_transform)
    finally:
        vm.end_change()
    return {"added": len(found), "per_layer": per_layer, "types": types}


def chip_rects(grid_cfg, refs, cells):
    """Chip rectangles (Deskewed): reference bbox, else the cell. One working size for all chips."""
    rects = {}
    for (r, c) in cells:
        if (r, c) in refs:
            rects[(r, c)] = refs[(r, c)]
        else:
            x0, y0 = grid_cfg.cell_origin(r, c)
            rects[(r, c)] = (x0, y0, grid_cfg.pitch_x, grid_cfg.pitch_y)
    return rects


def _settle(found, rule, by_cell):
    """Finds kept by layer_rule (a spot on several layers is one void); by_cell: kept so far, updated."""
    prefer = {"first": lambda f: (f[0], -f[5]), "last": lambda f: (-f[0], -f[5])}.get(rule, lambda f: (-f[5], f[0]))
    chosen = set()
    for f in sorted(found, key=prefer):
        gx, gy, rx, ry, cell, c = f[1], f[2], f[3], f[4], f[7], f[9]
        hit = next((k for k in by_cell.get(cell, ()) if abs(gx - k[1]) < max(rx, k[3]) and abs(gy - k[2]) < max(ry, k[4])), None)
        if hit is not None:
            c["status"], c["dup_layer"] = "dup", hit[0]
            continue
        by_cell.setdefault(cell, []).append(f)
        chosen.add(id(f))
    return [f for f in found if id(f) in chosen]


def place_found(vm, found, rects, grid_cfg, coord_transform):
    """Places finds of detect_voids in vm (inside the caller's begin_change / end_change). Returns {type: n}."""
    to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))
    types = {}
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
    return types


def _layer_change(read_layer, layer_numbers, skip, order, rects, size, p, progress, diag, steps, accept, on_cells=None):
    """
    Method "layer_change" (sat_widgets/void_layer_change.py): the chips of every loaded layer
    (skipped layers too: they still explain the layers below them), then each void with its
    first layer. Returns the found tuples that accept(f) kept (None = drop), None when cancelled.
    """
    from . import void_layer_change as vlc
    layers = {}
    for z, layer in enumerate(layer_numbers):
        chips = read_layer(z, layer)
        if chips is None:
            return None
        layers[z] = chips
        if on_cells is not None:
            on_cells(order, 0.5 * (z + 1) / len(layer_numbers))
    cells = [c for c in order if all(c in chips for chips in layers.values())]
    searched = [layer not in skip for layer in layer_numbers]
    if len(cells) < 3 or not any(searched):
        return []
    X = np.stack([np.stack([layers[z][c] for c in cells]) for z in range(len(layer_numbers))])
    if p.get("register", True):
        # The bbox may sit off the chip: whole px (a sub-pixel shift blurs each chip differently,
        # which the layer fit reads as change; its derivative terms take the sub-pixel rest)
        X = vlc.align_chips(X)[0]
    n0, total = steps(), 2 * len(layer_numbers) * len(order)
    def chip_done(i, n, t):
        if on_cells is not None:
            on_cells([cells[i]], 0.9)
        return progress is None or progress(n0 + (total - n0) * i // max(n, 1), total, t)
    res = vlc.find(X, searched, p, p.get("layer_rule", "first"),
                   chip_done if (progress or on_cells) else None)
    if res is None:
        return None
    W, H = size
    found, comps, collected = [], {}, {}
    for d in res["dets"]:
        cell, layer = cells[d["i"]], layer_numbers[d["z"]]
        x0, y0, w, h = rects[cell]
        sx, sy = w / W, h / H
        gx, gy, rx, ry = x0 + d["x"] * sx, y0 + d["y"] * sy, d["r"] * sx, d["r"] * sy
        extent = (gx - rx, gy - ry, gx + rx, gy + ry)
        if p.get("learn") is not None: # Training (Train): candidates with their features, nothing kept
            collected.setdefault((layer, cell), []).append((extent, int(math.pi * d["r"] ** 2), d["features"]))
            continue
        box = (max(0, int(d["y"] - d["r"])), min(H - 1, int(math.ceil(d["y"] + d["r"])) - 1),
               max(0, int(d["x"] - d["r"])), min(W, int(math.ceil(d["x"] + d["r"]))))
        c = {"box": box, "area": int(math.pi * d["r"] ** 2), "merged": 1, "score": d["score"],
             "status": "kept", "extent": extent}
        if "learned" in d:
            c["learned"] = d["learned"]
        comps.setdefault((layer, cell), []).append(c)
        f = accept((layer, gx, gy, rx, ry, d["score"], None, cell, extent, c))
        if f is not None:
            found.append(f)
    for (layer, cell), cands in collected.items():
        p["learn"](layer, cell, cands)
    if diag is not None:
        # Per layer: its evidence map (dev) against the place threshold; the place map as the mask
        thr = float(p.get("lc_threshold", vlc.DEFAULTS["lc_threshold"]))
        for z, layer in enumerate(layer_numbers):
            if not searched[z]:
                continue
            diag.layer_start(layer, layers[z], thr)
            for i, cell in enumerate(cells):
                mask = res["place"][i] > thr
                diag.chip(layer, cell, res["evidence"][z, i].astype(np.float32), None,
                          X[z, i].astype(np.float32), mask, mask, comps.get((layer, cell), []))
            diag.layer_end(layer)
    return found


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
    fit = None
    if p.get("layout_fit", True):
        # Layout drawn with another contrast / sharpness per chip: fitted away (see _LayoutFit)
        fit = _LayoutFit(golden)
        matched = np.stack([fit.apply(mc) for mc in matched])
        golden = np.median(matched, axis=0)
        fit = _LayoutFit(golden)
        matched = np.stack([fit.apply(mc) for mc in matched])
    # Per-pixel chip-to-chip spread of the matched chips (larger at layout edges)
    spread = np.median(np.abs(matched - golden), axis=0) * 1.4826
    del matched
    spread = _filt(spread, ImageFilter.MaxFilter(3))
    spread = np.maximum(spread, 0.5 * float(np.median(spread)) + 1e-3)
    # Local range of the golden chip: a pixel inside the 3x3 min..max of the golden chip is
    # not a defect (1 px shifts, slightly different edges)
    gmin = _filt(golden, ImageFilter.MinFilter(3))
    gmax = _filt(golden, ImageFilter.MaxFilter(3))
    match = _Matcher(golden)
    match.fit = fit
    return spread, gmin, gmax, match


def _layer_chips(stack, grid_cfg, coord_transform, refs, cells, layer_numbers, void_manager, p, progress):
    """{layer: {cell: matched chip}} of every searched layer (neighbours for multi-layer models)."""
    out = {}

    def dump(layer, cell, m, golden, spread, shift, rect, size):
        out.setdefault(layer, {})[cell] = np.asarray(m, np.float16)

    q = {k: v for k, v in p.items() if k not in ("seg_model", "learned_model", "learn")}
    q["dump"] = dump
    if detect_voids(stack, grid_cfg, coord_transform, refs, cells, layer_numbers, void_manager, "", q,
                    progress, apply=False) is None:
        return None
    return out


def _learned_maps(items, dctx, model, p, diag=None, layer=None, layer_diffs=None):
    """
    Deviation maps from a trained U-Net (sat_widgets/void_seg.py), in place of the die-to-die map:
    - "seg": void probability x 10 (Thr 5 = 50 %);
    - "recon": the model redraws the chip without voids (a golden chip of its own); the remainder
      chip - redrawn, averaged like the die-to-die path ("smooth"), is compared with its per-pixel
      chip-to-chip median / spread over the layer.
    Yields (cell, deviation, sign, matched chip) like _die_to_die.
    """
    from . import void_seg
    items = list(items)
    if model.mode == "seg":
        for cell, _, _, m in items:
            nb = None
            if model.neighbors and layer_diffs is not None:
                nb = [(layer_diffs.get(layer + k) or {}).get(cell) for k in (-1, 1)]
                nb = [None if o is None else o.astype(np.float32) for o in nb]
            x = void_seg.inputs(m, dctx["golden"], dctx["spread"], nb)
            prob = model.predict(x)
            yield cell, prob * 10.0, np.sign(m - dctx["golden"]), m
        return
    sigma = float(p.get("smooth", 0.0))
    rest = {}
    for cell, _, _, m in items:
        redrawn = model.predict(void_seg.inputs(m, dctx["golden"], dctx["spread"]))
        r = (m - redrawn) / np.maximum(dctx["spread"], 1e-3)
        rest[cell] = _blur(r, sigma) if sigma > 0 else r
    stack = np.stack(list(rest.values()))
    med = np.median(stack, axis=0)
    spread = np.quantile(np.abs(stack - med), 0.9, axis=0) / 1.6449 * (4.0 / 3.0)
    spread = _filt(spread, ImageFilter.MaxFilter(3))
    spread = np.maximum(spread, 0.5 * float(np.median(spread)) + 1e-3)
    del stack
    for cell, _, _, m in items:
        d = (rest[cell] - med) / spread
        yield cell, np.abs(d), np.sign(d), m


def _blur(a, sigma):
    """Gaussian blur (separable, edges extended) of a small float image."""
    if sigma <= 0:
        return a
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, r, mode="edge")
    h, w = a.shape
    tmp = sum(k[i] * pad[:, i:i + w] for i in range(2 * r + 1))
    return sum(k[i] * tmp[i:i + h, :] for i in range(2 * r + 1)).astype(np.float32)


def _die_to_die(chips, p, diag=None, ctx=None):
    """
    Yields (cell, deviation map, sign, matched chip): (chip - golden) / spread after matching,
    see the module doc. With "smooth" (working px) the signed deviation is averaged over its
    neighbourhood first: pixel noise averages out while a void (many pixels deviating the same
    way) keeps its level, so large faint voids rise above the noise. The averaged deviation is
    then compared like a chip: against its own per-pixel median and chip-to-chip spread over a
    sample of chips, so layout edges (whose residuals do not average out) stay quiet.
    """
    spread, gmin, gmax, match = _golden(chips, p)
    if diag is not None:
        diag.golden(match.golden, spread)
    if ctx is not None: # For the learned filter (void_learn.ChipContext)
        ctx.update(golden=match.golden, spread=spread, sp_med=float(np.median(spread)))
    sigma = float(p.get("smooth", 0.0))

    golden = match.golden

    def signed(cell):
        m = match.apply(chips[cell].astype(np.float32))
        if match.fit is not None:
            m = match.fit.apply(m)
        if sigma > 0:
            # Averaged path: the plain difference (noise stays near-Gaussian when averaged; the
            # 3x3 envelope would leave sparse spikes whose average is heavy-tailed). Layout edges
            # are handled by the spread of the averaged deviation below.
            return m, (m - golden) / spread
        hi, lo = np.maximum(m - gmax, 0), np.maximum(gmin - m, 0)
        return m, (hi - lo) / spread # > 0: brighter than the golden chip, < 0: darker

    if sigma > 0:
        keys = list(chips)
        n = min(len(keys), int(p["template_cells"]), 64)
        sample = np.stack([_blur(signed(keys[i])[1], sigma) for i in np.linspace(0, len(keys) - 1, n).astype(int)])
        s_med = np.median(sample, axis=0)
        # Spread from the 90th percentile (normal-consistent) instead of the median: averaged
        # residuals of matching / registration are heavy-tailed, the median would underrate them.
        # The extra 4/3 keeps the threshold comparable to the pixel path (tested: same false
        # detections at the same threshold on high-contrast data).
        s_spread = np.quantile(np.abs(sample - s_med), 0.9, axis=0) / 1.6449 * (4.0 / 3.0)
        del sample
        s_spread = _filt(s_spread, ImageFilter.MaxFilter(3))
        s_spread = np.maximum(s_spread, 0.5 * float(np.median(s_spread)) + 1e-3)
    for cell in chips:
        m, d = signed(cell)
        if sigma > 0:
            d = (_blur(d, sigma) - s_med) / s_spread
        yield cell, np.abs(d), np.sign(d), m
