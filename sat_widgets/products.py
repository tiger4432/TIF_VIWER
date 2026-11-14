"""
WF product definitions, read from products.json next to the program.
Nothing about a product is defined in code: only the keys present in the file are applied.

{
  "default": "<product name>",
  "products": {
    "<product name>": {
      "grid": {"rows": 10, "cols": 12, "pitch_x": 500, "pitch_y": 500,
               "start_x": 100, "start_y": 100, "angle": 0.0},      # any subset, image pixels
               # or the pitch in um: "pitch_um": [1000, 1000] (or one number, or "pitch_x_um" /
               # "pitch_y_um"): pitch px = pitch_um / Scale, follows Scale changes (wins over
               # pitch_x / pitch_y while a Scale is set)
      "um_per_px": 2.0,                      # optional: pixel size, sets the Scale field
      "chip_size_um": [960, 960],            # real chip (bbox) size in um. Default size of the
                                             # 'bbox' void type = chip_size_um / Scale
      "layout": {                            # in um from the chip top-left (with chip_size_um)
        "SL": [40, 40],                      # scribe line x, y. Void types declaring
                                             # "snap_offset": "SL" are fitted from the bbox corner
                                             # shifted inward by this
        "style": {"color": [0, 200, 255], "alpha": 200, "width": 1, "dash": [4, 4]},
        "boxes": [                           # dashed sub-boxes drawn inside every bbox
          {"name": "die", "inset": "SL"},    # bbox inset by a layout key ("SL") or by [x, y]
          {"name": "pad", "rect": [40, 40, 60, 60]},                 # [x, y, w, h] from bbox top-left
          {"name": "bump", "rect": [100, 100, 20, 20],
           "repeat": {"cols": 4, "rows": 3, "dx": 60, "dy": 60}},    # array of rects
          {"name": "core", "size": [600, 400]},                      # w, h only: centred on the chip
          {"name": "mark", "size": [50, 50], "center": [0, -300]},   # centred, moved from the chip centre
          {"name": "dot", "circle": [25, 25, 8], "mirror": "corners"},  # circle: centre x, y from the top-left
                                             # corner, radius; here at all four corners
          {"name": "fiducial", "rect": [10, 10, 30, 30], "mirror": "corners"}  # the same at all four corners
                                             # (mirrored; "x": left / right only, "y": top / bottom only)
        ]
      }
    }
  }
}
Optional "type_rules": void types of automatic detection from layout position and shape, see
sat_widgets/void_classify.py (overrides config.json "void_detect" -> "type_rules").
The layout is scaled from its design size (chip_size_um, or "design_size" if given) to each
drawn bbox, so it follows the real bbox even when the scale is slightly off.
Legacy: "bbox": {"w", "h"} in pixels and "layout.design_size" in any unit still work.
"""
import json

GRID_KEYS = ("start_x", "start_y", "pitch_x", "pitch_y", "rows", "cols", "angle")


def pitch_um(prod):
    """Declared pitch in um (x, y) of a product's grid, or None (pitch_um [x, y] / number, pitch_x_um / pitch_y_um)."""
    grid = (prod or {}).get("grid") or {}
    if grid.get("pitch_um") is not None:
        v = grid["pitch_um"]
        x, y = (v, v) if isinstance(v, (int, float)) else (v[0], v[1])
    else:
        x, y = grid.get("pitch_x_um"), grid.get("pitch_y_um")
    if not x and not y:
        return None
    return (float(x) if x else None, float(y) if y else None)


def grid_values(prod, um_per_px):
    """
    The declared grid of a product in image pixels: the GRID_KEYS present, with the pitch from
    pitch_um / um_per_px when a pitch in um is declared and um_per_px > 0.
    """
    grid = (prod or {}).get("grid") or {}
    out = {k: grid[k] for k in GRID_KEYS if k in grid}
    um = pitch_um(prod)
    if um and um_per_px and um_per_px > 0:
        if um[0]:
            out["pitch_x"] = um[0] / um_per_px
        if um[1]:
            out["pitch_y"] = um[1] / um_per_px
    return out


def load_products(path):
    """Returns (products: {name: dict}, default_name or None). Missing file -> ({}, None)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}, None
    products = data.get("products") or {}
    default = data.get("default")
    if default not in products:
        default = next(iter(products), None)
    return products, default


def layout_named_boxes(layout):
    """Expands layout['boxes'] to [(name, x, y, w, h)] in layout design units (repeats share the name; a circle
    as its bounding box)."""
    return [b[:5] for b in layout_shapes(layout)]


def layout_shapes(layout):
    """Expands layout['boxes'] to [(name, x, y, w, h, shape)] in layout design units, shape "rect" or "circle"
    (drawn as the ellipse in the box)."""
    if not layout:
        return []
    design_w, design_h = layout.get("design_size", (0, 0))
    out = []
    for i, box in enumerate(layout.get("boxes", [])):
        name = box.get("name", f"box{i}")
        if "inset" in box:
            inset = box["inset"]
            ix, iy = layout[inset] if isinstance(inset, str) else inset
            out.append((name, ix, iy, design_w - 2 * ix, design_h - 2 * iy, "rect"))
            continue
        shape = "rect"
        if "circle" in box: # [x, y, r]: centre x, y from the chip's top-left corner, radius r
            cx_, cy_, r_ = box["circle"]
            x, y, w, h = cx_ - r_, cy_ - r_, 2 * r_, 2 * r_
            shape = "circle"
        elif "rect" in box:
            x, y, w, h = box["rect"]
        else: # "size": [w, h] centred on the chip ("center": [dx, dy] from the chip centre, default 0, 0)
            w, h = box["size"] if isinstance(box["size"], (list, tuple)) else (box["size"], box["size"])
            dx, dy = box.get("center") or (0, 0)
            x, y = (design_w - w) / 2.0 + dx, (design_h - h) / 2.0 + dy
        if box.get("shape") in ("circle", "ellipse"):
            shape = "circle"
        rep = box.get("repeat") or {}
        # "mirror": "corners" / "xy" (the four corners), "x" (left <-> right), "y" (top <-> bottom): copies mirrored
        # about the chip's centre lines, so one corner's boxes are declared once
        mir = str(box.get("mirror") or "").lower()
        mx = mir in ("x", "xy", "corners", "4", "all")
        my = mir in ("y", "xy", "corners", "4", "all")
        seen = set()
        for r in range(int(rep.get("rows", 1))):
            for c in range(int(rep.get("cols", 1))):
                bx, by = x + c * rep.get("dx", 0), y + r * rep.get("dy", 0)
                for fx in ((False, True) if mx else (False,)):
                    for fy in ((False, True) if my else (False,)):
                        ox = design_w - bx - w if fx else bx
                        oy = design_h - by - h if fy else by
                        k = (round(ox, 6), round(oy, 6))
                        if k in seen: # A box on the centre line mirrors onto itself
                            continue
                        seen.add(k)
                        out.append((name, ox, oy, w, h, shape))
    return out


def layout_boxes(layout):
    """Expands layout['boxes'] to a list of (x, y, w, h) in layout design units."""
    return [b[1:] for b in layout_named_boxes(layout)]
