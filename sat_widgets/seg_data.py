"""
Training data for the U-Net (sat_widgets/void_seg.py) from marked files, in the current save format:
- extract(): the open file (MainWindow passes its stack, grid, markings);
- extract_saved(): a saved result (voids.json + meta.json, optional bonding_map.txt) whose
  meta.json names the source TIF and the grid, without opening it in the viewer.
The chips are cut, registered and matched exactly as Detect Voids does (void_detect "dump"), the
masks are drawn from the markings (void_seg.rasterize). The samples are reviewed (kept or left out)
in the Seg Data window before training.
"""
import glob
import json
import os

import numpy as np

from . import void_detect, void_seg
from .core_data import TiffStack, GridConfig, CoordinateTransform, BondingMap
from .void_manager import VoidManager


def _void_shapes(vm, grid_cfg, coord_transform, refs):
    """Markings (not bboxes) in Deskewed space: [dict(layer, cell, gx, gy, rx, ry, shape, p, quadrant, hand)]."""
    to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))
    out = []
    for layer, vs in vm.voids.items():
        for v in vs:
            if vm.is_bbox(v):
                continue
            gx, gy = to_deskew(v["globalCX"], v["globalCY"])
            shape = vm.shape_of(v)
            out.append({"layer": layer, "cell": tuple(grid_cfg.cell_at(gx, gy)), "gx": gx, "gy": gy,
                        "rx": v["radiusX"], "ry": v["radiusY"], "shape": shape,
                        "p": vm.p_of(v) if shape == "curve" else None,
                        "quadrant": vm.quadrant_of(v, gx, gy, refs, grid_cfg),
                        "hand": not v.get("auto")})
    return out


def extract(stack, grid_cfg, coord_transform, refs, cells, layer_numbers, vm, params, progress=None):
    """
    Samples of every searched chip and layer: ({layer: (golden, spread)}, [{layer, row, col, m, y}]).
    Mask: a hand marking of this layer = void (core) with an unknown band; markings of other layers
    and unreviewed detections = unknown; the rest = background.
    """
    shapes = _void_shapes(vm, grid_cfg, coord_transform, refs)
    by_cell = {}
    for s in shapes:
        by_cell.setdefault(s["cell"], []).append(s)
    layer_refs, samples = {}, []

    def dump(layer, cell, m, golden, spread, shift, rect, size):
        layer_refs.setdefault(layer, (np.asarray(golden, np.float32), np.asarray(spread, np.float32)))
        x0, y0, w, h = rect
        sx, sy = w / size[0], h / size[1]
        items = []
        for s in by_cell.get(tuple(cell), ()):
            items.append({"cx": (s["gx"] - x0) / sx - shift[0], "cy": (s["gy"] - y0) / sy - shift[1],
                          "rx": s["rx"] / sx, "ry": s["ry"] / sy, "shape": s["shape"], "p": s["p"],
                          "quadrant": s["quadrant"],
                          "kind": "void" if (s["hand"] and s["layer"] == layer) else "unknown"})
        H, W = m.shape
        samples.append({"layer": layer, "row": int(cell[0]), "col": int(cell[1]),
                        "m": np.asarray(m, np.float32), "y": void_seg.rasterize(H, W, items)})

    p = dict(params)
    for k in ("learned_model", "seg_model", "learn"):
        p.pop(k, None)
    p["method"] = "die_to_die"
    p["dump"] = dump
    res = void_detect.detect_voids(stack, grid_cfg, coord_transform, refs, cells, layer_numbers, vm, "",
                                   p, progress, apply=False)
    if res is None:
        return None
    return layer_refs, samples


def find_saved(folder):
    """voids.json files with a meta file beside them under folder (recursive)."""
    out = []
    for meta in sorted(glob.glob(os.path.join(folder, "**", "*meta.json"), recursive=True)):
        name = os.path.basename(meta)
        voids = os.path.join(os.path.dirname(meta), "voids.json" if name == "meta.json" else name[:-len("_meta.json")] + ".json")
        if os.path.exists(voids):
            out.append(voids)
    return out


def extract_saved(voids_path, params, progress=None):
    """
    Samples of a saved result. Returns (source, meta, layer_refs, samples) or raises ValueError
    with the reason (source TIF missing, no grid in meta.json, ...).
    """
    meta_path = VoidManager.meta_path(voids_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    source = meta.get("source_file") or ""
    folder = os.path.dirname(voids_path)
    if not os.path.isfile(source):
        # Moved: a TIF of the same name next to the result
        cand = os.path.join(folder, os.path.basename(source)) if source else ""
        if cand and os.path.isfile(cand):
            source = cand
        else:
            raise ValueError(f"source TIF not found ({source or 'not recorded'})")
    grid = meta.get("grid")
    if not grid:
        raise ValueError("no grid in the meta file (saved before meta.json had it)")
    stack = TiffStack(source)
    try:
        g = GridConfig()
        for k, v in grid.items():
            setattr(g, k, v)
        ct = CoordinateTransform(g.angle, stack.width, stack.height)
        vm = VoidManager(config_path=os.path.join(folder, "_no_config_.json"))
        vm.load_from_file(voids_path, g, ct, 0)
        cells = [(r, c) for r in range(g.rows) for c in range(g.cols)]
        bm_path = os.path.join(folder, "bonding_map.txt")
        if os.path.exists(bm_path):
            reject = []
            rj = os.path.join(folder, "bonding_map_reject.txt")
            if os.path.exists(rj):
                with open(rj, "r", encoding="utf-8") as f:
                    reject = [x.strip() for x in f.read().split(",") if x.strip()]
            bmap = BondingMap()
            with open(bm_path, "r", encoding="utf-8") as f:
                bmap.parse_text(f.read(), reject)
            cells = [rc for rc in cells if bmap.get_key(*rc)]
        refs = vm.reference_bboxes(g, ct)
        layers_all = [stack.layer_number(z) for z in range(len(stack))]
        keep = set(meta.get("layers") or layers_all)
        p = dict(params)
        p["skip_layers"] = sorted(set(p.get("skip_layers") or []) | (set(layers_all) - keep))
        out = extract(stack, g, ct, refs, cells, layers_all, vm, p, progress)
    finally:
        stack.close()
    if out is None:
        return None
    return (source, meta) + out
