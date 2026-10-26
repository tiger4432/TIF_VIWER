"""
WF product definitions, read from products.json next to the program.
Nothing about a product is defined in code: only the keys present in the file are applied.

{
  "default": "<product name>",
  "products": {
    "<product name>": {
      "grid": {"rows": 10, "cols": 12, "pitch_x": 500, "pitch_y": 500,
               "start_x": 100, "start_y": 100, "angle": 0.0},      # any subset
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
           "repeat": {"cols": 4, "rows": 3, "dx": 60, "dy": 60}}     # array of rects
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
    """Expands layout['boxes'] to [(name, x, y, w, h)] in layout design units (repeats share the name)."""
    if not layout:
        return []
    design_w, design_h = layout.get("design_size", (0, 0))
    out = []
    for i, box in enumerate(layout.get("boxes", [])):
        name = box.get("name", f"box{i}")
        if "inset" in box:
            inset = box["inset"]
            ix, iy = layout[inset] if isinstance(inset, str) else inset
            out.append((name, ix, iy, design_w - 2 * ix, design_h - 2 * iy))
            continue
        x, y, w, h = box["rect"]
        rep = box.get("repeat") or {}
        for r in range(int(rep.get("rows", 1))):
            for c in range(int(rep.get("cols", 1))):
                out.append((name, x + c * rep.get("dx", 0), y + r * rep.get("dy", 0), w, h))
    return out


def layout_boxes(layout):
    """Expands layout['boxes'] to a list of (x, y, w, h) in layout design units."""
    return [b[1:] for b in layout_named_boxes(layout)]
