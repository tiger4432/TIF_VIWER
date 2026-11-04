import time
import json
import os
import math
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QColorDialog, QInputDialog,
    QDialog, QLabel
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

from . import output_columns
from .products import layout_boxes

# ==================================================================================
# Void Data Manager
# ==================================================================================

class VoidManager:
    """
    Manages void markings for all layers.
    Stores data in Global Coordinates (truth).
    Handles saving/loading relative to Chip Coordinates.
    """
    # Shapes: "ellipse", "rectangle", "curve" (superellipse |x/rx|^p + |y/ry|^p = 1)
    SHAPES = ("ellipse", "rectangle", "curve")
    DEFAULT_P = 4.0 # Curve exponent when neither the void nor its type defines one
    P_MIN, P_MAX = 0.3, 30.0
    UNDO_LIMIT = 200

    @staticmethod
    def superellipse_path(cx, cy, rx, ry, p, n=96):
        """Closed path of |x/rx|^p + |y/ry|^p = 1 around (cx, cy)."""
        path = QPainterPath()
        e = 2.0 / p
        for i in range(n + 1):
            t = 2.0 * math.pi * i / n
            c, s = math.cos(t), math.sin(t)
            x = cx + rx * math.copysign(abs(c) ** e, c)
            y = cy + ry * math.copysign(abs(s) ** e, s)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path

    @staticmethod
    def quadrant_path(shape, cx, cy, rx, ry, p, quadrant):
        """
        One quadrant of a shape centered at (cx, cy): the region between the two axes through the
        center and the outline, on the side quadrant = (sx, sy) (each +1 or -1).
        """
        sx, sy = quadrant
        path = QPainterPath()
        path.moveTo(cx, cy)
        if shape == "rectangle":
            path.lineTo(cx + sx * rx, cy)
            path.lineTo(cx + sx * rx, cy + sy * ry)
            path.lineTo(cx, cy + sy * ry)
        else:
            e = 2.0 / (p if shape == "curve" else 2.0)
            n = int(min(48, max(6, max(rx, ry) / 3.0)))
            for i in range(n + 1):
                t = 0.5 * math.pi * i / n
                path.lineTo(cx + sx * rx * math.cos(t) ** e, cy + sy * ry * math.sin(t) ** e)
        path.closeSubpath()
        return path

    @staticmethod
    def draw_void_shape(painter: QPainter, shape: str, cx: float, cy: float, rx: float, ry: float, p: float = None, quadrant=None):
        """
        Unified drawing logic for Voids.
        All Voids are defined by Center (cx, cy) and Radii (rx, ry).
        - Ellipse: Drawn using Center, Radius.
        - Rectangle: Drawn using Top-Left calculated from Center - Radius.
        - Curve: Superellipse with exponent p.
        quadrant (sx, sy): draw only that quadrant (voids anchored with their center on a corner).
        """
        if max(rx, ry) < 1.5:
            painter.drawPoint(QPointF(cx, cy)) # Too small on screen for an outline
        elif quadrant:
            painter.drawPath(VoidManager.quadrant_path(shape, cx, cy, rx, ry, p or VoidManager.DEFAULT_P, quadrant))
        elif shape == "rectangle":
            # Rectangle: Center/Radius schema
            # cx, cy is Center
            # rx, ry is Radius (Half-Width/Height)
            painter.drawRect(QRectF(cx - rx, cy - ry, rx * 2, ry * 2))
        elif shape == "curve":
            n = int(min(96, max(12, max(rx, ry) / 1.5))) # Fewer points when small on screen
            painter.drawPath(VoidManager.superellipse_path(cx, cy, rx, ry, p or VoidManager.DEFAULT_P, n))
        else:
            # Ellipse: Center, Radius
            painter.drawEllipse(QPointF(cx, cy), rx, ry)

    # Void type name (case-insensitive) that marks the reference bbox of a chip
    BBOX_TYPE_NAME = "bbox"

    def __init__(self, config_path="config.json"):
        self.voids = {} # Dict[absolute layer number: int, List[dict]]
        self.config_path = config_path
        
        # Undo history: snapshots of self.voids taken before each change (see checkpoint)
        self._undo = []
        self._undo_key = None
        self._undo_time = 0.0
        self._txn_depth = 0
        self._version = 0 # Bumped on every change (see touch)
        self._refs_cache = None
        
        # Layout of the selected WF product (products.json); provides named snap offsets like SL
        self.layout = None
        self.um_per_px = 0.0 # Pixel size (um); 0 = not set -> um fields empty
        self.file_um_per_px = None # Scale stored in the last loaded voids.json

        # Output field/column names (see output_columns.py; overridable in config.json)
        self.json_fields, self.csv_columns = output_columns.resolve()
        self.norm_pitch = output_columns.NORM_PITCH_X # Pixel values written as if pitch_x were this
        
        # Void types keyed by name, in declaration order (shortcuts 1..9, 0 follow this order).
        # Declared in config.json (written by the user; the program never writes it).
        self.types = self.normalize_types([
            {"name": "Defect", "color": (255, 0, 0)},
            {"name": "Warning", "color": (255, 165, 0)},
            {"name": "Safe", "color": (0, 255, 0)},
            {"name": "Check", "color": (0, 255, 255)},
        ])
        
        # Load from config
        self.load_config()

    @staticmethod
    def normalize_types(declared):
        """
        {name: properties} from a void_types declaration, in declaration order. Accepted forms:
          {"Edge": {"color": [...], "shape": "ellipse"}, ...}           (keyed by name)
          [{"name": "Edge", "color": [...]}, ...]                        (list)
          {"0": {"name": "Edge", "color": [...]}, ...}                   (legacy, keyed by number)
        """
        if isinstance(declared, dict):
            items = list(declared.items())
            if items and all(isinstance(v, dict) and "name" in v and str(k).lstrip("-").isdigit() for k, v in items):
                entries = [v for _, v in sorted(items, key=lambda kv: int(kv[0]))] # Legacy numeric keys
            else:
                entries = [dict(v, name=k) for k, v in items]
        else:
            entries = list(declared or [])
        types = {}
        for e in entries:
            t = dict(e)
            name = str(t.get("name", "")).strip()
            if not name:
                continue
            t["name"] = name
            t["color"] = tuple(t.get("color", (255, 255, 0)))
            t.setdefault("shape", "ellipse")
            t.setdefault("def_w", 0.0)
            t.setdefault("def_h", 0.0)
            types[name] = t
        return types

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                self.json_fields, self.csv_columns = output_columns.resolve(data)
                self.norm_pitch = output_columns.norm_pitch(data)
                if "void_types" in data:
                    types = self.normalize_types(data["void_types"])
                    if types:
                        self.types = types
                    print(f"Loaded {len(self.types)} void types from {self.config_path}")
        except FileNotFoundError:
            print("Config file not found, using defaults.")
        except Exception as e:
            print(f"Error loading config: {e}")

    # Type edits from the Types dialog apply to this session only: declarations (config.json)
    # are written by the user, never by the program.
    def add_type(self, name, color, shape="ellipse", def_w=0.0, def_h=0.0):
        name = name.strip()
        if not name or name in self.types:
            return None
        self.types[name] = {"name": name, "color": tuple(color), "shape": shape, "def_w": def_w, "def_h": def_h}
        return name

    def update_type(self, name, color=None, shape=None, def_w=None, def_h=None):
        t = self.types.get(name)
        if t is None:
            return
        for k, val in (("color", color), ("shape", shape), ("def_w", def_w), ("def_h", def_h)):
            if val is not None:
                t[k] = tuple(val) if k == "color" else val

    def rename_type(self, old, new):
        """Renames a type and the voids using it (undoable). Returns True if renamed."""
        new = new.strip()
        if old not in self.types or not new or (new in self.types and new != old):
            return False
        affected = [v for vs in self.voids.values() for v in vs if v.get("type") == old]
        if affected:
            self.checkpoint()
        self.types = {(new if k == old else k): (dict(t, name=new) if k == old else t) for k, t in self.types.items()}
        for v in affected:
            v["type"] = new
        return True

    def remove_type(self, name):
        if name in self.types and len(self.types) > 1:
            fallback = next(k for k in self.types if k != name)
            affected = [v for vs in self.voids.values() for v in vs if v.get("type") == name]
            if affected:
                self.checkpoint()
            del self.types[name]
            # Remap existing voids to the first remaining type
            for v in affected:
                v["type"] = fallback

    def paint_layout(self, painter: QPainter, bbox, to_paint):
        """
        Draws the selected product's layout (dashed sub-boxes, see products.json) inside a reference
        bbox (x0, y0, w, h) in Deskewed space; to_paint(x, y) maps Deskewed -> painter coordinates.
        The layout is scaled from its design size to the bbox (the drawn bbox is the reference).
        """
        layout = self.layout
        if not layout:
            return
        boxes = layout_boxes(layout)
        if not boxes:
            return
        style = layout.get("style", {})
        col = style.get("color", [0, 200, 255])
        pen = QPen(QColor(col[0], col[1], col[2], style.get("alpha", 200)))
        pen.setWidthF(style.get("width", 1.0))
        dash = style.get("dash", [4, 4])
        if dash:
            pen.setDashPattern(dash)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        x0, y0, w, h = bbox
        dw, dh = layout.get("design_size", (w, h))
        sx, sy = w / dw, h / dh
        for (bx, by, bw, bh) in boxes:
            p0 = to_paint(x0 + bx * sx, y0 + by * sy)
            p1 = to_paint(x0 + (bx + bw) * sx, y0 + (by + bh) * sy)
            painter.drawRect(QRectF(QPointF(*p0), QPointF(*p1)))

    LAYER0_COLOR = (255, 0, 255) # Badge of voids already visible on layer 0 (see paint_layer0_badge)

    def paint_layer0_badge(self, painter: QPainter, v, cx, cy, rx, ry, ghost=False, quadrant=None):
        """
        Small "0" badge at the top-right of a void flagged v["layer0"] (Layer change: already
        visible on layer 0, bubble?). Fixed size on screen; translucent on ghosts.
        """
        sx, sy = quadrant or (1, -1)
        if quadrant:
            bx, by = cx + sx * rx * 0.85, cy + sy * ry * 0.85 # Outer corner of the drawn quadrant
        else:
            bx, by = cx + rx * 0.71, cy - ry * 0.71 # Top-right of the outline
        size = 11.0
        rect = QRectF(bx - size / 2, by - size / 2, size, size)
        r, g, b = self.LAYER0_COLOR
        a = 150 if ghost else 230
        painter.save()
        painter.setPen(QPen(QColor(255, 255, 255, a), 1.0))
        painter.setBrush(QColor(r, g, b, a))
        painter.drawRoundedRect(rect, 2.5, 2.5)
        font = painter.font()
        font.setPixelSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 140 if ghost else 255))
        painter.drawText(rect, Qt.AlignCenter, "0")
        painter.restore()

    def paint_void(self, painter: QPainter, v, cx, cy, scale=1.0, style="normal", width=1.0, quadrant=None, badges=False,
                   halo=False, min_radius=0.0):
        """
        Single drawing routine for voids (viewer and export).
        (cx, cy): center in the target's pixel space; scale: target pixels per image pixel.
        style: "normal" (type color, solid), "ghost" (other layers: dotted, translucent),
               "active" (selected: red border, translucent type fill),
               "hover" (mouse over: translucent outline and fill).
        quadrant: see quadrant_of.
        badges: review marks (viewer only): "0" on voids flagged as visible on layer 0.
        halo: a dark line under the outline (viewer: visible on bright and dark backgrounds; not on ghosts).
        min_radius: drawn at least this large (target px; viewer zoomed out: small voids stay visible).
        """
        t = self.type_of(v)
        r, g, b = t.get("color", (255, 255, 0))[:3] # Undeclared type: yellow

        if style == "active":
            pen = QPen(QColor(255, 0, 0, 255))
            brush = QColor(r, g, b, 100)
        elif style == "hover":
            pen = QPen(QColor(r, g, b, 150)) # Mouse over: translucent
            brush = QColor(r, g, b, 70)
        elif style == "ghost":
            pen = QPen(QColor(r, g, b, 128))
            pen.setStyle(Qt.DotLine)
            brush = Qt.NoBrush
        else:
            pen = QPen(QColor(r, g, b, 255))
            if v.get("auto"):
                pen.setStyle(Qt.DashLine) # Estimated bbox / detected void: needs a review
            brush = Qt.NoBrush
        pen.setWidthF(width)
        rx, ry = v["radiusX"] * scale, v["radiusY"] * scale
        if min_radius > 0 and max(rx, ry) < min_radius: # Keep the aspect, grow to the minimum
            k = min_radius / max(rx, ry, 1e-6)
            rx, ry = rx * k, ry * k
        shape = t.get("shape", "ellipse")
        if halo and style != "ghost":
            dark = QPen(QColor(0, 0, 0, 150))
            dark.setWidthF(width + 2.0)
            painter.setPen(dark)
            painter.setBrush(Qt.NoBrush)
            self.draw_void_shape(painter, shape, cx, cy, rx, ry, self.p_of(v), quadrant)
        painter.setPen(pen)
        painter.setBrush(brush)
        self.draw_void_shape(painter, shape, cx, cy, rx, ry, self.p_of(v), quadrant)
        if style in ("active", "hover") and self.has_guide_box(v) and max(rx, ry) >= 1.5:
            # Dashed box: the grab / resize area of curve and corner-snap types
            guide = QPen(QColor(r, g, b, 170))
            guide.setStyle(Qt.DashLine)
            guide.setWidthF(1.0)
            painter.setPen(guide)
            painter.setBrush(Qt.NoBrush)
            if quadrant:
                painter.drawRect(QRectF(QPointF(cx, cy), QPointF(cx + quadrant[0] * rx, cy + quadrant[1] * ry)).normalized())
            else:
                painter.drawRect(QRectF(cx - rx, cy - ry, rx * 2, ry * 2))
        if badges and v.get("layer0"):
            self.paint_layer0_badge(painter, v, cx, cy, rx, ry, ghost=style == "ghost", quadrant=quadrant)

    # -----------------------------------------------
    # Type properties
    # -----------------------------------------------
    def resolve_type(self, name):
        """Declared type name for name, ignoring case ("BBOX" -> "bbox"); unknown names unchanged."""
        if name is None or name in self.types:
            return name
        key = str(name).strip().lower()
        return next((t for t in self.types if t.lower() == key), name)

    def type_of(self, v):
        return self.types.get(v.get("type"), {})

    def shape_of(self, v):
        return self.type_of(v).get("shape", "ellipse")

    def hit_shape(self, v):
        """
        Shape used to pick and resize a void: curve and corner-snap types use their box (shown as a
        dashed guide while hovered or edited) so the grab area does not depend on p or the
        quadrant; the curve itself is still drawn. Other shapes use their outline.
        """
        shape = self.shape_of(v)
        return "rectangle" if shape == "curve" or self.snap_anchor(v) is not None else shape

    def has_guide_box(self, v):
        return self.hit_shape(v) != self.shape_of(v)

    def p_of(self, v):
        """Curve exponent of a void (void value, else type default, else DEFAULT_P)."""
        p = v.get("p")
        if p is None:
            p = self.type_of(v).get("def_p", self.DEFAULT_P)
        return float(p)

    def snap_anchor(self, v):
        """
        Declared in the type (config.json): "snap": "bbox_corner" makes the void attach to a
        corner of its cell's reference bbox. "snap_anchor": "center" (default) puts the shape center
        on the (offset) corner and only the inward quadrant is drawn (e.g. a quarter superellipse
        fitted from the SL corner); "edge" puts a corner of the whole shape's bounding box on it.
        Returns None when the type does not snap.
        """
        t = self.type_of(v)
        if t.get("snap") != "bbox_corner":
            return None
        return t.get("snap_anchor", "center")

    def corner_for_box(self, box, grid_cfg, coord_transform=None):
        """
        Corner (x, y, inward x, inward y) of the reference bbox nearest to a drawn Deskewed box
        (x0, y0, x1, y1): the corner a snap-type void drawn as that box attaches to. None without
        a bbox.
        """
        bx, by = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        bbox = self.bbox_at(self.reference_bboxes(grid_cfg, coord_transform), grid_cfg, bx, by)
        if bbox is None:
            return None
        x0, y0, w, h = bbox
        corners = [(x0, y0, 1, 1), (x0 + w, y0, -1, 1), (x0 + w, y0 + h, -1, -1), (x0, y0 + h, 1, -1)]

        def distance(k):
            kx, ky = k[0], k[1]
            gap = math.hypot(max(box[0] - kx, 0.0, kx - box[2]), max(box[1] - ky, 0.0, ky - box[3]))
            return gap, math.hypot(bx - kx, by - ky) # Ties (corner inside the box): the box center
        return min(corners, key=distance)

    def quadrant_of(self, v, gx, gy, refs, grid_cfg):
        """
        Inward quadrant (sx, sy) of a center-anchored void whose Deskewed center is (gx, gy):
        the side of its cell's reference bbox the center lies on (left half -> +x, top half -> +y).
        None for other voids (drawn whole).
        """
        if self.snap_anchor(v) != "center":
            return None
        bbox = self.bbox_at(refs, grid_cfg, gx, gy)
        if bbox is None:
            return None
        x0, y0, w, h = bbox
        return (1 if gx <= x0 + w / 2 else -1, 1 if gy <= y0 + h / 2 else -1)

    def snap_offset(self, v, bbox_w, bbox_h):
        """
        Inward shift (x, y) of the snap corner from the bbox corner, declared in the type as
        "snap_offset": a layout key such as "SL" (taken from the selected product's layout and
        scaled from layout.design_size to the actual bbox) or [x, y] in pixels. Default (0, 0).
        """
        off = self.type_of(v).get("snap_offset")
        if off is None:
            return 0.0, 0.0
        if isinstance(off, str):
            layout = self.layout or {}
            if off not in layout:
                return 0.0, 0.0
            ox, oy = layout[off]
            dw, dh = layout.get("design_size", (bbox_w, bbox_h))
            return ox * bbox_w / dw, oy * bbox_h / dh
        return float(off[0]), float(off[1])

    # -----------------------------------------------
    # Undo
    # -----------------------------------------------
    # Every change of void data goes through one of:
    #  - the mutators below (add_void, remove_void, clear_layer, delete_voids_in_rect,
    #    remove_type, load_from_file), which record a checkpoint themselves, or
    #  - an interaction transaction (begin_change / end_change) around direct field edits
    #    (mouse drag: sizing, move, resize, snap, erase), or
    #  - an explicit checkpoint() (curve p wheel edits, coalesced per void).
    # Replacing the whole data set for a new image resets the history (reset_history).
    def _snapshot(self):
        return {layer: [dict(v) for v in v_list] for layer, v_list in self.voids.items()}

    def checkpoint(self, coalesce_key=None):
        """
        Records the current state for undo. Call BEFORE changing void data.
        Calls with the same coalesce_key within 1.5 s form one undo step (e.g. wheel ticks).
        Inside a transaction the transaction's checkpoint covers the change.
        """
        self.touch() # Called before every change
        if self._txn_depth > 0:
            return
        now = time.monotonic()
        if coalesce_key is not None and coalesce_key == self._undo_key and now - self._undo_time < 1.5:
            self._undo_time = now
            return
        self._undo.append(self._snapshot())
        del self._undo[:-self.UNDO_LIMIT]
        self._undo_key = coalesce_key
        self._undo_time = now

    def begin_change(self):
        """Starts an interaction (e.g. mouse press .. release) that may edit voids directly."""
        if self._txn_depth == 0:
            self.checkpoint()
        self._txn_depth += 1

    def end_change(self):
        """
        Ends an interaction; drops its undo step when nothing changed.
        Returns True if the interaction changed void data.
        """
        if self._txn_depth == 0:
            return False
        self._txn_depth -= 1
        if self._txn_depth > 0:
            return False
        if self._undo and self._undo[-1] == self._snapshot():
            self._undo.pop()
            return False
        return True

    def can_undo(self):
        return bool(self._undo)

    def undo(self):
        """Restores the state before the last change. Returns True if something was undone."""
        if not self._undo or self._txn_depth > 0:
            return False
        state = self._undo.pop()
        self.voids.clear()
        self.voids.update(state)
        self.apply_bbox_size() # The product may have changed since the snapshot
        self.touch()
        self._undo_key = None
        return True

    def reset_history(self):
        self._undo.clear()
        self._undo_key = None
        self._txn_depth = 0

    def replace_all(self, voids=None):
        """Replaces all void data for a new image (not undoable: history is reset)."""
        self.voids.clear()
        if voids:
            self.voids.update(voids)
        self.touch()
        self.reset_history()

    # -----------------------------------------------
    # Mutators
    # -----------------------------------------------
    def add_void(self, layer, gx, gy, rx, ry, type_name=None):
        self.checkpoint()
        if layer not in self.voids:
            self.voids[layer] = []
            
        new_void = {
            "globalCX": gx,
            "globalCY": gy,
            "radiusX": rx,
            "radiusY": ry,
            "layer": layer,
            "type": type_name if type_name is not None else next(iter(self.types), ""),
            "createdAt": int(time.time() * 1000),
            "voidIndex": 0
        }
        if self.type_of(new_void).get("shape") == "curve":
            new_void["p"] = self.p_of(new_void)
        self.voids[layer].append(new_void)
        return new_void

    def place_void(self, layer, gx, gy, rx, ry, type_name, grid_cfg, coord_transform=None, hint=None, drag_point=None,
                   snap=True):
        """
        A marking as drawn by hand: Raw center (gx, gy), half sizes, then attached to the bbox corner
        for snap types (hint / drag_point in Deskewed space, see snap_void). Used for clicks with a
        declared size and for automatic detection alike, so both give the same kind of void.
        snap=False: placed as given (a mouse drag follows; it snaps when the mouse is released).
        """
        v = self.add_void(int(layer), float(gx), float(gy), float(rx), float(ry), type_name)
        if snap:
            self.snap_void(v, grid_cfg, coord_transform, hint=hint, drag_point=drag_point)
        for k in ("globalCX", "globalCY", "radiusX", "radiusY"):
            v[k] = float(v[k])
        return v

    def remove_void(self, layer, void):
        """Removes this exact void object (identity, not value equality). Returns True if removed."""
        v_list = self.voids.get(layer, [])
        for i, v in enumerate(v_list):
            if v is void:
                self.checkpoint()
                del v_list[i]
                return True
        return False

    def move_void_to_layer(self, void, from_layer, to_layer):
        """Moves this void object to another layer (undoable). Returns True if moved."""
        if from_layer == to_layer:
            return False
        v_list = self.voids.get(from_layer, [])
        for i, v in enumerate(v_list):
            if v is void:
                self.checkpoint()
                del v_list[i]
                v["layer"] = to_layer
                self.voids.setdefault(to_layer, []).append(v)
                return True
        return False

    def remove_voids_where(self, predicate):
        """Removes every void (all layers) for which predicate(layer, void) is true. Returns the count."""
        doomed = [(layer, v) for layer, v_list in self.voids.items() for v in v_list if predicate(layer, v)]
        if not doomed:
            return 0
        self.checkpoint()
        ids = {id(v) for _, v in doomed}
        for layer in self.voids:
            self.voids[layer] = [v for v in self.voids[layer] if id(v) not in ids]
        return len(doomed)

    def set_void_type(self, v, type_name, coalesce_key=None):
        """Changes the type of a void (undoable). Returns True if changed."""
        if v.get("type") == type_name:
            return False
        self.checkpoint(coalesce_key=coalesce_key)
        v["type"] = type_name
        if self.type_of(v).get("shape") == "curve" and "p" not in v:
            v["p"] = self.p_of(v)
        return True

    def toggle_layer0(self, v):
        """Sets / clears the "already visible on layer 0" flag of a void (undoable). Returns the new state."""
        self.checkpoint()
        if v.get("layer0"):
            v.pop("layer0", None)
            return False
        v["layer0"] = True
        return True

    def set_p(self, v, p):
        """Sets the curve exponent (clamped); consecutive edits of one void are one undo step."""
        p = min(self.P_MAX, max(self.P_MIN, float(p)))
        if v.get("p") == p:
            return p
        self.checkpoint(coalesce_key=("p", id(v)))
        v["p"] = p
        return p

    def snap_void(self, v, grid_cfg, coord_transform=None, hint=None, drag_point=None):
        """
        Attaches a snap-type void (see snap_anchor) to a corner of its cell's reference bbox.
        hint (Deskewed point): choose the bbox corner nearest to it and look up the bbox in its
            cell (e.g. the press point while drawing); otherwise the corner nearest to the shape.
        drag_point (Deskewed point): the size follows it (shape spans bbox corner -> point).
        Edits the void in place (call inside a change/checkpoint). Returns True if snapped.
        """
        anchor = self.snap_anchor(v)
        if anchor is None:
            return False
        to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))
        to_raw = coord_transform.deskew_to_raw if coord_transform else (lambda x, y: (x, y))

        gx, gy = to_deskew(v["globalCX"], v["globalCY"])
        rx, ry = v["radiusX"], v["radiusY"]
        bbox = self.bbox_at(self.reference_bboxes(grid_cfg, coord_transform), grid_cfg, *(hint or (gx, gy)))
        if bbox is None:
            return False
        x0, y0, w, h = bbox
        # (corner x, corner y, inward direction x, inward direction y)
        corners = [(x0, y0, 1, 1), (x0 + w, y0, -1, 1), (x0 + w, y0 + h, -1, -1), (x0, y0 + h, 1, -1)]

        def distance(corner):
            kx, ky, sx, sy = corner
            if hint is not None:
                px, py = hint
            elif anchor == "edge":
                px, py = gx - sx * rx, gy - sy * ry # Shape's own corner on that side
            else:
                px, py = gx, gy
            return math.hypot(px - kx, py - ky)

        kx, ky, sx, sy = min(corners, key=distance)
        # Shift inward by the declared offset (e.g. scribe line SL)
        ox, oy = self.snap_offset(v, w, h)
        kx, ky = kx + sx * ox, ky + sy * oy
        if anchor == "edge":
            if drag_point is not None:
                rx = max(1.0, abs(drag_point[0] - kx) / 2.0)
                ry = max(1.0, abs(drag_point[1] - ky) / 2.0)
            cx, cy = kx + sx * rx, ky + sy * ry
        else:
            if drag_point is not None:
                rx = max(1.0, abs(drag_point[0] - kx))
                ry = max(1.0, abs(drag_point[1] - ky))
            cx, cy = kx, ky

        v["globalCX"], v["globalCY"] = to_raw(cx, cy)
        v["radiusX"], v["radiusY"] = rx, ry
        self.touch()
        return True

    def snapped_to(self, bbox_void, grid_cfg, coord_transform=None):
        """Snap-type voids (all layers) attached to the reference bbox drawn by bbox_void."""
        to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))
        cx, cy = to_deskew(bbox_void["globalCX"], bbox_void["globalCY"])
        rect = (cx - bbox_void["radiusX"], cy - bbox_void["radiusY"], 2 * bbox_void["radiusX"], 2 * bbox_void["radiusY"])
        refs = self.reference_bboxes(grid_cfg, coord_transform)
        same = lambda b: b is not None and all(abs(b[i] - rect[i]) < 1e-6 for i in range(4))
        if not any(same(b) for b in refs.values()):
            return [] # Not the reference bbox of its cell (e.g. a lower layer's)
        out = []
        for vs in self.voids.values():
            for v in vs:
                if self.snap_anchor(v) is not None and same(self.bbox_at(refs, grid_cfg, *to_deskew(v["globalCX"], v["globalCY"]))):
                    out.append(v)
        return out

    def resnap(self, grid_cfg, coord_transform=None, voids=None):
        """Re-attaches snap-type voids (default: all) to the nearest corner of their bbox (after bboxes moved / resized)."""
        if voids is None:
            voids = [v for vs in self.voids.values() for v in vs if self.snap_anchor(v) is not None]
        for v in voids:
            self.snap_void(v, grid_cfg, coord_transform)
        return len(voids)

    def snap_corner_point(self, v, grid_cfg, coord_transform=None):
        """Deskewed bbox corner a snap-type void is currently attached to (None if not snapping)."""
        if self.snap_anchor(v) is None:
            return None
        to_deskew = coord_transform.raw_to_deskew if coord_transform else (lambda x, y: (x, y))
        gx, gy = to_deskew(v["globalCX"], v["globalCY"])
        bbox = self.bbox_at(self.reference_bboxes(grid_cfg, coord_transform), grid_cfg, gx, gy)
        if bbox is None:
            return None
        x0, y0, w, h = bbox
        rx, ry = v["radiusX"], v["radiusY"]
        if self.snap_anchor(v) == "edge":
            # The shape corner closest to a bbox corner is the anchored one
            pts = [(gx - rx, gy - ry), (gx + rx, gy - ry), (gx + rx, gy + ry), (gx - rx, gy + ry)]
        else:
            pts = [(gx, gy)]
        corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        return min(corners, key=lambda k: min(math.hypot(px - k[0], py - k[1]) for px, py in pts))

    def delete_void_at(self, layer, gx, gy, hit_radius=10.0, coord_transform=None):
        """
        Deletes the topmost void under the point.
        coord_transform: same as hit_test (needed when the query point is in Deskewed space).
        Returns True if deleted.
        """
        target, _ = self.hit_test(layer, gx, gy, hit_radius, coord_transform)
        return target is not None and self.remove_void(layer, target)
        
    @staticmethod
    def boundary_distance(shape, dx, dy, rx, ry, p=2.0):
        """
        Distance from the point (dx, dy) relative to the center to the shape outline, and whether
        it is inside. Rectangle: exact. Ellipse / superellipse: measured along the ray from the
        center (close to the true distance; exact on the axes).
        """
        ax, ay = abs(dx), abs(dy)
        if shape == "rectangle":
            if ax <= rx and ay <= ry:
                return min(rx - ax, ry - ay), True
            return math.hypot(max(ax - rx, 0.0), max(ay - ry, 0.0)), False
        t = ((ax / max(rx, 1e-9)) ** p + (ay / max(ry, 1e-9)) ** p) ** (1.0 / p)
        if t < 1e-9:
            return min(rx, ry), True
        return math.hypot(dx, dy) * abs(1.0 - 1.0 / t), t <= 1.0

    def hit_test(self, layer, gx, gy, margin=5.0, coord_transform=None, quadrant_ctx=None, allow_bbox=True):
        """
        gx, gy: Query Point (usually Mouse in Screen/Deskewed Space)
        coord_transform: Function(cx, cy) -> (tx, ty) to transform Stored Coords to Query Space.
        allow_bbox: False skips reference bbox voids (not recognized outside the bbox layer).
        Returns (void_obj, type_str)
        type_str: 'center' (move), 'edge' (resize), None
        """
        if layer not in self.voids: return None, None
        
        # Among overlapping voids the one whose outline is closest to the point wins
        # (ties: the one drawn on top). The reference bbox only when no other void is hit:
        # corner voids lie on its outline and would otherwise lose to it.
        best = None
        for v in reversed(self.voids[layer]):
            if not allow_bbox and self.is_bbox(v):
                continue
            cx, cy = v["globalCX"], v["globalCY"]
            
            # Apply Transform if provided (Raw -> Deskewed)
            if coord_transform:
                 cx, cy = coord_transform(cx, cy)
            
            rx = v["radiusX"]
            ry = v["radiusY"]
            
            shape = self.hit_shape(v)
            
            # Quadrant voids (center on a bbox corner): only the inward quadrant counts.
            # quadrant_ctx = (reference bboxes, grid_cfg), see quadrant_of
            quadrant = self.quadrant_of(v, cx, cy, *quadrant_ctx) if quadrant_ctx else None
            if quadrant and ((gx - cx) * quadrant[0] < -margin or (gy - cy) * quadrant[1] < -margin):
                continue
            
            dist, inside = self.boundary_distance(shape, gx - cx, gy - cy, rx, ry,
                                                  self.p_of(v) if shape == "curve" else 2.0)
            if not inside and dist > margin:
                continue
            # Edge band: the margin, but at most a third of the shape so small voids stay movable
            band = min(margin, 0.35 * min(rx, ry)) if inside else margin
            key = (self.is_bbox(v), dist)
            if best is None or key < best[0]:
                # Bboxes are only moved: their size is the product's chip size
                best = (key, v, 'edge' if dist <= band and not self.is_bbox(v) else 'center')

        if best is not None:
            return best[1], best[2]
        return None, None

    def hit_test_all(self, gx, gy, margin=1.0, priority_layer=None, coord_transform=None):
        """
        Searches ALL layers for a hit.
        Returns (void_dict, action, layer_index) or (None, None, None).
        Checks priority_layer first.
        coord_transform: Passed to hit_test
        """
        # 1. Check Priority Layer
        if priority_layer is not None:
             v, action = self.hit_test(priority_layer, gx, gy, margin, coord_transform)
             if v:
                 return v, action, priority_layer
                 
        # 2. Check Others
        for layer_idx in self.voids.keys():
            if layer_idx == priority_layer: continue
            
            v, action = self.hit_test(layer_idx, gx, gy, margin, coord_transform)
            if v:
                return v, action, layer_idx
                
        return None, None, None

    def clear_layer(self, layer):
        if self.voids.get(layer):
            self.checkpoint()
            self.voids[layer] = []

    def delete_voids_in_rect(self, layer, x0, y0, x1, y1):
        """
        Deletes all voids in the given layer that are within the rectangle [x0, y0, x1, y1].
        Using void centers.
        """
        if layer not in self.voids: return 0
        
        self.begin_change()
        to_delete = []
        for v in self.voids[layer]:
            vx = v["globalCX"]
            vy = v["globalCY"]
            if x0 <= vx <= x1 and y0 <= vy <= y1:
                to_delete.append(v)
                
        for v in to_delete:
            self.remove_void(layer, v)
        self.end_change()

        return len(to_delete)

    def is_bbox(self, v):
        """True if the void is a reference bbox marking (type name 'bbox', case-insensitive)."""
        return str(v.get("type", "")).strip().lower() == self.BBOX_TYPE_NAME

    def apply_bbox_size(self):
        """
        Sizes every bbox marking to the bbox type's size (the selected product's chip size), keeping
        its center. The size is not edited by hand; without a declared size the markings keep theirs.
        Not an undo step (follows the product). Returns the number of bboxes resized.
        """
        name = self.resolve_type(self.BBOX_TYPE_NAME)
        t = self.types.get(name) or {}
        w, h = float(t.get("def_w") or 0.0), float(t.get("def_h") or 0.0)
        if w <= 0 or h <= 0:
            return 0
        n = 0
        for v_list in self.voids.values():
            for v in v_list:
                if self.is_bbox(v) and (v["radiusX"] != w / 2.0 or v["radiusY"] != h / 2.0):
                    v["radiusX"], v["radiusY"] = w / 2.0, h / 2.0
                    n += 1
        if n:
            self.touch()
        return n

    @staticmethod
    def bbox_at(refs, grid_cfg, gx, gy):
        """
        Reference bbox containing the Deskewed point (looked up in the point's cell and its
        neighbours), else the one of the point's cell (or None).
        """
        r, c = grid_cfg.cell_at(gx, gy)
        for rr in (r, r - 1, r + 1):
            for cc in (c, c - 1, c + 1):
                b = refs.get((rr, cc))
                if b is not None and b[0] <= gx <= b[0] + b[2] and b[1] <= gy <= b[1] + b[3]:
                    return b
        return refs.get((r, c))

    def touch(self):
        """Marks void data as changed (invalidates cached derived data such as reference bboxes)."""
        self._version += 1

    def reference_bboxes(self, grid_cfg, coord_transform=None):
        """
        Reference bbox per grid cell: {(row, col): (x0, y0, w, h)} in Deskewed (grid-aligned)
        space. The bbox on the topmost (lowest-numbered) layer of the cell is used.
        Cached until the voids, the grid or the transform change.
        """
        key = (self._version, grid_cfg.start_x, grid_cfg.start_y, grid_cfg.pitch_x, grid_cfg.pitch_y,
               id(coord_transform), getattr(coord_transform, "angle", None),
               getattr(coord_transform, "img_cx", None), getattr(coord_transform, "img_cy", None))
        if self._refs_cache is not None and self._refs_cache[0] == key:
            return self._refs_cache[1]
        refs = {}
        for layer in sorted(self.voids):
            for v in self.voids[layer]:
                if not self.is_bbox(v):
                    continue
                gx, gy = v["globalCX"], v["globalCY"]
                if coord_transform:
                    gx, gy = coord_transform.raw_to_deskew(gx, gy)
                cell = grid_cfg.cell_at(gx, gy)
                if cell not in refs:
                    rx, ry = v["radiusX"], v["radiusY"]
                    refs[cell] = (gx - rx, gy - ry, 2 * rx, 2 * ry)
        self._refs_cache = (key, refs)
        return refs

    def void_records(self, grid_cfg, bonding_map=None, coord_transform=None):
        """
        One dict per void with the export values, keyed by the internal names of
        output_columns.py. coord_transform is required for correct cells when the grid is rotated.
        """
        refs = self.reference_bboxes(grid_cfg, coord_transform)
        s = self.um_per_px if self.um_per_px and self.um_per_px > 0 else None
        um = lambda px: px * s if s is not None and px is not None else None
        k = self.pixel_scale(grid_cfg)
        records = []
        for layer in sorted(self.voids):
            for v in self.voids[layer]:
                # Raw (storage) -> Deskewed (grid space)
                gx, gy = v["globalCX"], v["globalCY"]
                if coord_transform:
                    gx, gy = coord_transform.raw_to_deskew(gx, gy)
                row, col = grid_cfg.cell_at(gx, gy)
                chip_x0, chip_y0 = grid_cfg.cell_origin(row, col)


                # Bonding Label
                label = (bonding_map.get_key(row, col) if bonding_map else "") or ""

                # Position relative to the cell's reference bbox (top-left origin)
                bbox = refs.get((row, col))

                records.append({
                    "key": f"{col},{row},{layer},0",
                    "col": col,
                    "row": row,
                    "layer": layer,
                    "void_index": v.get("voidIndex", 0),
                    "type": v.get("type", ""),
                    "center_x": gx - chip_x0,
                    "center_y": gy - chip_y0,
                    "radius_x": v["radiusX"],
                    "radius_y": v["radiusY"],
                    "created_at": v.get("createdAt", 0),
                    "patch_label": f"X{col:02d}_Y{row:02d}_L{layer:02d}_{label or 'void'}",
                    "label": label,
                    "global_x": v["globalCX"],
                    "global_y": v["globalCY"],
                    "p": self.p_of(v) if self.shape_of(v) == "curve" else None,
                    "bbox_rel_x": gx - bbox[0] if bbox else None,
                    "bbox_rel_y": gy - bbox[1] if bbox else None,
                    "layer0": bool(v.get("layer0")), # Layer change: already visible on layer 0 (bubble?)
                    "auto_state": v.get("auto") or None, # "detected" / "estimated" until reviewed
                })
                rec = records[-1]
                for key in ("center_x", "center_y", "radius_x", "radius_y", "bbox_rel_x", "bbox_rel_y"):
                    rec[key + "_um"] = um(rec[key])
                # Pixel values: actual pixels in *_raw, scaled to the normalized pitch in the plain fields
                for key in output_columns.PIXEL_KEYS:
                    raw = rec[key]
                    rec[key + "_raw"] = raw
                    rec[key] = None if raw is None else raw * k
        return records

    def pixel_scale(self, grid_cfg):
        """Factor from actual pixels to the written pixel values (norm pitch / grid x pitch), 1 = none."""
        if not self.norm_pitch or not grid_cfg.pitch_x:
            return 1.0
        return self.norm_pitch / float(grid_cfg.pitch_x)

    def meta_data(self, grid_cfg, extra=None):
        """Everything about the markings except the markings themselves (meta.json)."""
        meta = {
            "version": 4,
            "layer_numbering": "absolute",
            "um_per_px": self.um_per_px or None,
            # Plain pixel fields = actual pixels x scale (grid x pitch normalized); *_raw = actual
            "pixel_scale": {"pitch_x": self.norm_pitch, "pitch_x_raw": float(grid_cfg.pitch_x),
                            "scale": self.pixel_scale(grid_cfg)} if self.norm_pitch else None,
            "grid": {k: getattr(grid_cfg, k) for k in ("start_x", "start_y", "pitch_x", "pitch_y", "rows", "cols", "angle")},
            "types": self.types, # name -> {name, color, shape, ...}
            "layer0_voids": sum(1 for vs in self.voids.values() for v in vs if v.get("layer0")),
            "unreviewed_voids": sum(1 for vs in self.voids.values() for v in vs if v.get("auto") == "detected"),
        }
        meta.update(extra or {})
        return meta

    def void_list(self, grid_cfg, bonding_map=None, coord_transform=None):
        """The markings (voids.json): one object per void, field names from output_columns."""
        return [{name: rec[key] for key, name in self.json_fields.items() if name}
                for rec in self.void_records(grid_cfg, bonding_map, coord_transform)]

    def to_data(self, grid_cfg, bonding_map=None, coord_transform=None):
        """meta.json with the voids.json list as "voids" (what the results plugin gets as ctx["voids_json"])."""
        return dict(self.meta_data(grid_cfg), voids=self.void_list(grid_cfg, bonding_map, coord_transform))

    @staticmethod
    def meta_path(path):
        """meta.json next to voids.json; <name>_meta.json next to another name."""
        folder, name = os.path.split(path)
        stem = os.path.splitext(name)[0]
        return os.path.join(folder, "meta.json" if stem.lower() == "voids" else f"{stem}_meta.json")

    def save_to_file(self, path, grid_cfg, bonding_map=None, coord_transform=None, extra_meta=None):
        """
        voids.json: only the markings (a list; cell-relative values from the current grid, field
        names from output_columns, "layer" absolute, "type" by name). Everything else (format,
        scale, grid, void types, extra_meta such as source and wafer ID) goes to meta.json next to
        it (meta_path). coord_transform is required for correct cells when the grid is rotated.
        """
        voids = self.void_list(grid_cfg, bonding_map, coord_transform)
        meta_path = self.meta_path(path)
        try:
            with open(path, 'w') as f:
                json.dump(voids, f, indent=2)
            with open(meta_path, 'w', encoding="utf-8") as f:
                json.dump(self.meta_data(grid_cfg, extra_meta), f, indent=2, ensure_ascii=False)
            print(f"Saved {len(voids)} voids to {path}, {len(self.types)} types and settings to {meta_path}")
        except Exception as e:
            print(f"Save failed: {e}")

    def _field(self, item, key, default=None):
        """Reads a void field by internal key: configured name first, then the default name."""
        for name in (self.json_fields.get(key), output_columns.JSON_FIELDS[key]):
            if name and name in item:
                return item[name]
        return default

    def load_from_file(self, path, grid_cfg, coord_transform=None, layer_offset=0):
        """
        Reads JSON -> Calculates Global using Grid Config.
        Handles V1 (list), V2 and V3 (dict) formats.
        Resolves 'type' string back to ID.
        layer_offset: absolute number of the first loaded layer. Files before V3 stored layers
        relative to the start layer used when saving; they are assumed to have been saved with
        the same start layer as now.
        """
        try:
            with open(path, 'r') as f:
                raw_data = json.load(f)
            if isinstance(raw_data, list):
                # voids.json holds only the markings; meta.json next to it the rest
                meta_path = self.meta_path(path)
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding="utf-8") as f:
                        raw_data = dict(json.load(f), voids=raw_data)
                elif any(isinstance(v, dict) and any(str(k).endswith("_raw") for k in v) for v in raw_data):
                    # Written by this version (has *_raw fields): absolute layers, actual pixels in *_raw
                    print(f"{os.path.basename(meta_path)} not found: void types and scale not restored")
                    raw_data = {"layer_numbering": "absolute", "voids": raw_data}
            count = self.load_from_data(raw_data, grid_cfg, coord_transform, layer_offset)
            print(f"Loaded {count} voids from {path}")
        except Exception as e:
            print(f"Load failed: {e}")

    def load_from_data(self, raw_data, grid_cfg, coord_transform=None, layer_offset=0, replace=True):
        """
        Voids from already parsed voids.json data (a file, or another system through the
        receive plugin, see plugins/results_api.py). replace=False adds them to the current voids.
        One undo step. Returns the number of voids; raises ValueError on an unknown format.
        """
        if not isinstance(raw_data, (list, dict)):
            raise ValueError("Unknown void data format (expected a voids.json object or list)")
        self.checkpoint() # Loading replaces / extends the current voids (undoable)
        if replace:
            self.voids.clear()
        legacy_names = {} # Old files: type number -> name
        count = 0
        legacy_layers = True
        self.file_um_per_px = None

        # Determine format
        if isinstance(raw_data, list):
            # V1 Format
            void_list = raw_data
        elif isinstance(raw_data, dict):
            legacy_layers = raw_data.get("layer_numbering") != "absolute"
            self.file_um_per_px = raw_data.get("um_per_px")
            # Types stored in the file: declarations (config.json) win; types the
            # declaration does not know are added so their voids keep color and shape
            if "types" in raw_data:
                file_types = self.normalize_types(raw_data["types"])
                for name, t in file_types.items():
                    if self.resolve_type(name) == name: # Same name in another case: the declared one
                        self.types.setdefault(name, t)
                # Legacy files reference types by number: number -> name
                if isinstance(raw_data["types"], dict):
                    for k, t in raw_data["types"].items():
                        if str(k).lstrip("-").isdigit() and isinstance(t, dict) and t.get("name"):
                            legacy_names[int(k)] = str(t["name"]).strip()
            void_list = raw_data.get("voids", [])

        if legacy_layers and layer_offset:
            print(f"Legacy void file: relative layers shifted by +{layer_offset}")

        # Pixel values: *_raw when present, else the plain field divided by the recorded scale
        # (files without a scale hold actual pixels)
        scale = ((raw_data.get("pixel_scale") or {}).get("scale") if isinstance(raw_data, dict) else None) or 1.0

        def px(item, key, default=None):
            value = self._field(item, key + "_raw")
            if value is not None:
                return value
            value = self._field(item, key)
            return default if value is None else value / scale

        for item in void_list:
            layer = self._field(item, "layer", 0)
            if legacy_layers:
                layer += layer_offset
            col = self._field(item, "col", 0)
            row = self._field(item, "row", 0)
            rel_cx = px(item, "center_x", 0)
            rel_cy = px(item, "center_y", 0)
            rx = px(item, "radius_x", 10.0)
            ry = px(item, "radius_y", 10.0)
            
            # Resolve Type: by name; old files by number (their own number -> name table,
            # else the declaration order)
            type_name = self.resolve_type(self._field(item, "type"))
            if not type_name:
                t_id = item.get("type_id")
                order = list(self.types)
                if t_id in legacy_names:
                    type_name = self.resolve_type(legacy_names[t_id])
                elif isinstance(t_id, int) and 0 <= t_id < len(order):
                    type_name = order[t_id]
                else:
                    type_name = order[0] if order else ""
            
            gx = px(item, "global_x")
            gy = px(item, "global_y")
            if gx is None or gy is None:
                # Calc Global from Relative (Grid-Dependent)
                chip_x0, chip_y0 = grid_cfg.cell_origin(row, col)
                gx = chip_x0 + rel_cx
                gy = chip_y0 + rel_cy
                
                # Convert Deskewed (gx) -> Raw, because storage expects Raw.
                # Without a transform the grid is assumed unrotated.
                if coord_transform:
                     gx, gy = coord_transform.deskew_to_raw(gx, gy)
            
            if layer not in self.voids: self.voids[layer] = []
            
            v_obj = {
                "globalCX": gx,
                "globalCY": gy,
                "radiusX": rx,
                "radiusY": ry,
                "layer": layer,
                "type": type_name,
                "createdAt": self._field(item, "created_at", 0),
                "voidIndex": self._field(item, "void_index", 0)
            }
            p = self._field(item, "p")
            if p is not None:
                v_obj["p"] = float(p)
            # Optional review fields (files without them: hand / reviewed markings)
            if self._field(item, "layer0") in (True, 1, "1", "true", "True"):
                v_obj["layer0"] = True
            auto = self._field(item, "auto_state")
            if isinstance(auto, str) and auto:
                v_obj["auto"] = auto

            self.voids[layer].append(v_obj)
            count += 1
        # Bboxes: only the center is taken from the data, the size follows the selected product
        if self.apply_bbox_size():
            self.resnap(grid_cfg, coord_transform) # Corner voids follow the resized bboxes
        return count


class VoidTypeDialog(QDialog):
    """
    Dialog to manage Void Types (Add, Remove, Color, Rename, Shape, Default Size).
    Changes apply to this session only: declarations live in config.json, written by the user.
    """
    def __init__(self, void_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Void Types")
        self.resize(420, 420)
        self.setWindowModality(Qt.ApplicationModal)
        
        self.vm = void_manager
        
        layout = QVBoxLayout(self)
        note = QLabel("Changes apply to this session only. To keep them, edit void_types in config.json.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)
        
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        
        h = QHBoxLayout()
        for text, slot in (("Add", self.add_type), ("Remove", self.remove_type), ("Color", self.change_color),
                           ("Rename", self.rename_type), ("Shape (E/R/C)", self.toggle_shape),
                           ("Default Size", self.set_default_size)):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            h.addWidget(btn)
        layout.addLayout(h)
        
        self.refresh()
        
    def refresh(self):
        self.list_widget.clear()
        for name, data in self.vm.types.items():
            color = data["color"] # (r, g, b)
            shape = data.get("shape", "ellipse")
            def_w = data.get("def_w", 0.0)
            def_h = data.get("def_h", 0.0)
            
            s_code = {"ellipse": "E", "rectangle": "R", "curve": "C"}.get(shape, "?")
            if data.get("snap") == "bbox_corner":
                s_code += "+snap"
            size_code = ""
            if def_w > 0 or def_h > 0:
                 size_code = f" [{def_w:.1f}x{def_h:.1f}]"
            
            item = QListWidgetItem(f"[{s_code}] {name}{size_code}")
            c = QColor(color[0], color[1], color[2])
            item.setBackground(c)
            item.setForeground(Qt.white if (c.red()*0.299 + c.green()*0.587 + c.blue()*0.114) < 128 else Qt.black)
            item.setData(Qt.UserRole, name)
            self.list_widget.addItem(item)

    def _selected(self):
        row = self.list_widget.currentRow()
        return None if row < 0 else self.list_widget.item(row).data(Qt.UserRole)
            
    def add_type(self):
        name, ok = QInputDialog.getText(self, "New Type", "Type Name:")
        if ok and name and self.vm.add_type(name, (255, 255, 0)): # Default Yellow
            self.refresh()
            
    def remove_type(self):
        name = self._selected()
        if name is not None:
            self.vm.remove_type(name)
            self.refresh()
        
    def change_color(self):
        name = self._selected()
        if name is None: return
        curr_c = self.vm.types[name]["color"]
        c = QColorDialog.getColor(QColor(curr_c[0], curr_c[1], curr_c[2]), self)
        if c.isValid():
            self.vm.update_type(name, color=(c.red(), c.green(), c.blue()))
            self.refresh()

    def rename_type(self):
        old_name = self._selected()
        if old_name is None: return
        name, ok = QInputDialog.getText(self, "Rename Type", "New Name:", text=old_name)
        if ok and name and self.vm.rename_type(old_name, name):
            self.refresh()

    def toggle_shape(self):
        name = self._selected()
        if name is None: return
        shapes = self.vm.SHAPES
        curr_shape = self.vm.types[name].get("shape", "ellipse")
        new_shape = shapes[(shapes.index(curr_shape) + 1) % len(shapes)] if curr_shape in shapes else shapes[0]
        self.vm.update_type(name, shape=new_shape)
        self.refresh()

    def set_default_size(self):
        name = self._selected()
        if name is None: return
        data = self.vm.types[name]
        
        from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Set Default Size")
        l = QVBoxLayout(dlg)
        l.addWidget(QLabel("Default Width (0 = Manual):"))
        sb_w = QDoubleSpinBox(); sb_w.setRange(0.0, 99999.0); sb_w.setValue(data.get("def_w", 0.0))
        l.addWidget(sb_w)
        l.addWidget(QLabel("Default Height (0 = Manual):"))
        sb_h = QDoubleSpinBox(); sb_h.setRange(0.0, 99999.0); sb_h.setValue(data.get("def_h", 0.0))
        l.addWidget(sb_h)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        l.addWidget(btns)
        if dlg.exec():
            self.vm.update_type(name, def_w=sb_w.value(), def_h=sb_h.value())
            self.refresh()
