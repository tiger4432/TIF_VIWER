"""
Type of an automatically detected void from declared rules: position from the product layout,
and shape. Nothing about the types is in the code; the rules name the types.

Declared per product in products.json ("type_rules", next to "layout"), else in config.json
"void_detect" -> "type_rules". The first rule whose conditions all hold gives the type; when no
rule matches (or none is declared), the type selected in Void Marking is used.

    "type_rules": [
        {"type": "CornerDelam", "near": {"box": "die", "part": "corner", "within": 30}},
        {"type": "PadVoid", "in": "pad", "max_size": 40},
        {"type": "LongVoid", "min_aspect": 3},
        {"type": "DieVoid", "in": "die"}
    ]

Rule keys (all optional except "type"):
  Position (layout boxes of the selected product by name as in products.json; "bbox" is the
  whole reference bbox). Box coordinates are scaled to each drawn bbox like the dashed layout.
    "in": name | [names]          blob center inside a box of one of these names
    "not_in": name | [names]      blob center inside none of them
    "overlaps": name | [names]    blob extent intersects such a box
    "near": {"box": name | [names], "part": "corner" | "edge", "within": px}
                                  blob extent within this distance of a corner / of the outline
                                  of such a box ("within_um" with the Scale set)
  Shape (pixels; the same keys with "_um" use the Scale):
    "min_size" / "max_size"       blob diameter (longer side of its extent)
    "min_aspect" / "max_aspect"   longer / shorter side
    "min_fill" / "max_fill"       blob pixels / extent area (ellipse ~0.79, rectangle ~1,
                                  irregular or thin curved shapes lower)
    "polarity": "bright" | "dark" brighter / darker than the golden chip (Die-to-die only)
"""
import math

from .products import layout_named_boxes


def _names(v):
    return {v} if isinstance(v, str) else set(v or ())


class Classifier:
    def __init__(self, rules, layout, known_types, um_per_px=0.0):
        self.um = um_per_px if um_per_px and um_per_px > 0 else None
        self.rules = []
        lower = {str(t).lower(): t for t in known_types}
        for r in rules or ():
            name = lower.get(str(r.get("type", "")).strip().lower()) # Case-insensitive, as declared
            if name is None:
                print(f"[type_rules] unknown void type {r.get('type')!r}: rule skipped")
                continue
            self.rules.append(dict(r, type=name))
        self.boxes = layout_named_boxes(layout) if layout else []
        self.design = (layout or {}).get("design_size")

    def __bool__(self):
        return bool(self.rules)

    def _px(self, rule, key):
        """Rule value in pixels: key in pixels, or key_um converted with the Scale."""
        if key in rule:
            return float(rule[key])
        if key + "_um" in rule and self.um:
            return float(rule[key + "_um"]) / self.um
        return None

    def _boxes(self, names, bbox):
        """Boxes (x0, y0, x1, y1) in Deskewed pixels for the given names, inside this bbox."""
        bx, by, bw, bh = bbox
        out = []
        if "bbox" in names:
            out.append((bx, by, bx + bw, by + bh))
        if self.boxes:
            dw, dh = self.design or (bw, bh)
            sx, sy = bw / dw, bh / dh
            for name, x, y, w, h in self.boxes:
                if name in names:
                    out.append((bx + x * sx, by + y * sy, bx + (x + w) * sx, by + (y + h) * sy))
        return out

    @staticmethod
    def _rect_point(r, px, py):
        x0, y0, x1, y1 = r
        return math.hypot(max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1))

    @staticmethod
    def _rect_outline(r, b):
        """Distance from rect r to the outline of box b (0 if r crosses it)."""
        x0, y0, x1, y1 = r
        X0, Y0, X1, Y1 = b
        inside = x0 >= X0 and y0 >= Y0 and x1 <= X1 and y1 <= Y1
        if inside:
            return min(x0 - X0, y0 - Y0, X1 - x1, Y1 - y1)
        dx = max(X0 - x1, 0, x0 - X1)
        dy = max(Y0 - y1, 0, y0 - Y1)
        return math.hypot(dx, dy) # Outside: gap to the box; overlapping the border: 0

    def _match(self, rule, f, bbox):
        cx, cy = f["center"]
        ext = f["extent"]
        if "in" in rule:
            if not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in self._boxes(_names(rule["in"]), bbox)):
                return False
        if "not_in" in rule:
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in self._boxes(_names(rule["not_in"]), bbox)):
                return False
        if "overlaps" in rule:
            if not any(ext[0] <= b[2] and b[0] <= ext[2] and ext[1] <= b[3] and b[1] <= ext[3]
                       for b in self._boxes(_names(rule["overlaps"]), bbox)):
                return False
        near = rule.get("near")
        if near:
            within = self._px(near, "within")
            boxes = self._boxes(_names(near.get("box", "bbox")), bbox)
            if within is None or not boxes:
                return False
            if near.get("part", "edge") == "corner":
                d = min(self._rect_point(ext, x, y) for b in boxes for x in (b[0], b[2]) for y in (b[1], b[3]))
            else:
                d = min(self._rect_outline(ext, b) for b in boxes)
            if d > within:
                return False
        for key, value, cmp in (("min_size", f["size"], lambda v, t: v >= t), ("max_size", f["size"], lambda v, t: v <= t)):
            t = self._px(rule, key)
            if t is not None and not cmp(value, t):
                return False
        for key, value, cmp in (("min_aspect", f["aspect"], lambda v, t: v >= t), ("max_aspect", f["aspect"], lambda v, t: v <= t),
                                ("min_fill", f["fill"], lambda v, t: v >= t), ("max_fill", f["fill"], lambda v, t: v <= t)):
            if key in rule and not cmp(value, float(rule[key])):
                return False
        if "polarity" in rule and f.get("polarity") != rule["polarity"]:
            return False
        return True

    def type_for(self, f, bbox, default):
        """f: blob features (Deskewed pixels): center, extent (x0, y0, x1, y1), size, aspect, fill, polarity."""
        for rule in self.rules:
            if self._match(rule, f, bbox):
                return rule["type"]
        return default
