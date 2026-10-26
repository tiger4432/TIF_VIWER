import time
import json
import math
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QColorDialog, QInputDialog,
    QDialog
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

from . import output_columns
from .app_paths import shared_dir

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
            n = 48
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
        if quadrant:
            painter.drawPath(VoidManager.quadrant_path(shape, cx, cy, rx, ry, p or VoidManager.DEFAULT_P, quadrant))
        elif shape == "rectangle":
            # Rectangle: Center/Radius schema
            # cx, cy is Center
            # rx, ry is Radius (Half-Width/Height)
            painter.drawRect(QRectF(cx - rx, cy - ry, rx * 2, ry * 2))
        elif shape == "curve":
            painter.drawPath(VoidManager.superellipse_path(cx, cy, rx, ry, p or VoidManager.DEFAULT_P))
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
        
        # Layout of the selected WF product (products.json); provides named snap offsets like SL
        self.layout = None

        # Output field/column names (see output_columns.py; overridable in config.json)
        self.json_fields, self.csv_columns = output_columns.resolve()
        
        # Default Types
        self.types = {
            0: {"name": "Defect", "color": (255, 0, 0)}, 
            1: {"name": "Warning", "color": (255, 165, 0)}, 
            2: {"name": "Safe", "color": (0, 255, 0)}, 
            3: {"name": "Check", "color": (0, 255, 255)} 
        }
        self.next_type_id = 4
        
        # Load from config
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                self.json_fields, self.csv_columns = output_columns.resolve(data)
                if "void_types" in data:
                    self.types = {}
                    max_id = 0
                    for tid_str, tdata in data["void_types"].items():
                        tid = int(tid_str)
                        # Keep every declared key (e.g. snap, snap_anchor, def_p)
                        self.types[tid] = dict(tdata)
                        self.types[tid]["color"] = tuple(tdata["color"])
                        self.types[tid].setdefault("shape", "ellipse")
                        self.types[tid].setdefault("def_w", 0.0)
                        self.types[tid].setdefault("def_h", 0.0)
                        if tid > max_id: max_id = tid
                    self.next_type_id = max_id + 1
                    print(f"Loaded {len(self.types)} void types from {self.config_path}")
        except FileNotFoundError:
            print("Config file not found, using defaults.")
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self):
        # Keep other sections (e.g. output_columns) of an existing config
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
        except Exception:
            data = {}
        data["void_types"] = {}
        for tid, tdata in self.types.items():
            entry = dict(tdata)
            entry["color"] = list(tdata["color"])
            entry.setdefault("shape", "ellipse")
            entry.setdefault("def_w", 0.0)
            entry.setdefault("def_h", 0.0)
            data["void_types"][str(tid)] = entry
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved config to {self.config_path}")
            if shared_dir():
                print("Note: declarations come from the shared folder; this change is local until "
                      "config.json on the share is updated (it is replaced on the next start).")
        except Exception as e:
            print(f"Error saving config: {e}")

    def add_type(self, name, color, shape="ellipse", def_w=0.0, def_h=0.0):
        tid = self.next_type_id
        self.types[tid] = {
            "name": name, 
            "color": color, 
            "shape": shape,
            "def_w": def_w,
            "def_h": def_h
        }
        self.next_type_id += 1
        self.save_config()
        return tid

    def update_type(self, tid, name, color, shape="ellipse", def_w=0.0, def_h=0.0):
        if tid in self.types:
            # Keep other declared keys (snap, def_p, ...)
            self.types[tid].update({
                "name": name, 
                "color": color, 
                "shape": shape,
                "def_w": def_w,
                "def_h": def_h
            })
            self.save_config()

    def remove_type(self, tid):
        if tid in self.types and len(self.types) > 1:
            fallback = min(k for k in self.types if k != tid)
            affected = [v for vs in self.voids.values() for v in vs if v.get("type_id") == tid]
            if affected:
                self.checkpoint()
            del self.types[tid]
            self.save_config()
            # Remap existing voids to the first remaining type
            for v in affected:
                v["type_id"] = fallback

    def paint_void(self, painter: QPainter, v, cx, cy, scale=1.0, style="normal", width=1.0, quadrant=None):
        """
        Single drawing routine for voids (viewer and export).
        (cx, cy): center in the target's pixel space; scale: target pixels per image pixel.
        style: "normal" (type color, solid), "ghost" (other layers: dotted, translucent),
               "active" (selected: red border, translucent type fill).
        quadrant: see quadrant_of.
        """
        tid = v.get("type_id", 0)
        if tid not in self.types:
            tid = min(self.types) if self.types else 0
        r, g, b = self.types.get(tid, {}).get("color", (255, 255, 0))[:3]

        if style == "active":
            pen = QPen(QColor(255, 0, 0, 255))
            brush = QColor(r, g, b, 100)
        elif style == "ghost":
            pen = QPen(QColor(r, g, b, 128))
            pen.setStyle(Qt.DotLine)
            brush = Qt.NoBrush
        else:
            pen = QPen(QColor(r, g, b, 255))
            if v.get("auto") == "estimated":
                pen.setStyle(Qt.DashLine) # Auto BBOX could not detect it: needs a review
            brush = Qt.NoBrush
        pen.setWidthF(width)
        painter.setPen(pen)
        painter.setBrush(brush)
        self.draw_void_shape(painter, self.types.get(tid, {}).get("shape", "ellipse"),
                             cx, cy, v["radiusX"] * scale, v["radiusY"] * scale, self.p_of(v), quadrant)

    # -----------------------------------------------
    # Type properties
    # -----------------------------------------------
    def type_of(self, v):
        return self.types.get(v.get("type_id", 0), {})

    def shape_of(self, v):
        return self.type_of(v).get("shape", "ellipse")

    def p_of(self, v):
        """Curve exponent of a void (void value, else type default, else DEFAULT_P)."""
        p = v.get("p")
        if p is None:
            p = self.type_of(v).get("def_p", self.DEFAULT_P)
        return float(p)

    def snap_anchor(self, v):
        """
        Declared in the type (config.json): "snap": "bbox_corner" makes the void attach to a
        corner of its cell's reference bbox. "snap_anchor": "edge" (default) puts a corner of the
        shape's bounding box on the (offset) corner, "center" puts the shape center on it and only
        the inward quadrant is drawn (e.g. a quarter superellipse fitted from the SL corner).
        Returns None when the type does not snap.
        """
        t = self.type_of(v)
        if t.get("snap") != "bbox_corner":
            return None
        return t.get("snap_anchor", "edge")

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
        self.reset_history()

    # -----------------------------------------------
    # Mutators
    # -----------------------------------------------
    def add_void(self, layer, gx, gy, rx, ry, type_id=0):
        self.checkpoint()
        if layer not in self.voids:
            self.voids[layer] = []
            
        new_void = {
            "globalCX": gx,
            "globalCY": gy,
            "radiusX": rx,
            "radiusY": ry,
            "layer": layer,
            "type_id": type_id,
            "createdAt": int(time.time() * 1000),
            "voidIndex": 0
        }
        if self.types.get(type_id, {}).get("shape") == "curve":
            new_void["p"] = self.p_of(new_void)
        self.voids[layer].append(new_void)
        return new_void

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
        return True

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
        
    def hit_test(self, layer, gx, gy, margin=5.0, coord_transform=None, quadrant_ctx=None):
        """
        gx, gy: Query Point (usually Mouse in Screen/Deskewed Space)
        coord_transform: Function(cx, cy) -> (tx, ty) to transform Stored Coords to Query Space.
        Returns (void_obj, type_str)
        type_str: 'center' (move), 'edge' (resize), None
        """
        if layer not in self.voids: return None, None
        
        # Check reverse order (Top to Bottom visually)
        for v in reversed(self.voids[layer]):
            cx, cy = v["globalCX"], v["globalCY"]
            
            # Apply Transform if provided (Raw -> Deskewed)
            if coord_transform:
                 cx, cy = coord_transform(cx, cy)
            
            rx = v["radiusX"]
            ry = v["radiusY"]
            
            shape = self.shape_of(v)
            
            # Quadrant voids (center on a bbox corner): only the inward quadrant counts.
            # quadrant_ctx = (reference bboxes, grid_cfg), see quadrant_of
            quadrant = self.quadrant_of(v, cx, cy, *quadrant_ctx) if quadrant_ctx else None
            if quadrant and ((gx - cx) * quadrant[0] < -margin or (gy - cy) * quadrant[1] < -margin):
                continue
            
            dx = abs(gx - cx)
            dy = abs(gy - cy)

            # Inner (move) region: at least half of the shape, so small voids stay movable
            # when zoomed out (large margin).
            in_rx = max(rx * 0.5, rx - margin)
            in_ry = max(ry * 0.5, ry - margin)

            if shape == "rectangle":
                # Rectangle Test (Axis Aligned in Query Space)
                if dx <= rx + margin and dy <= ry + margin:
                     on_edge = (dx > in_rx) or (dy > in_ry)
                     return v, ('edge' if on_edge else 'center')
            else:
                # Ellipse (p = 2) / Curve Test (normalized distance)
                p = self.p_of(v) if shape == "curve" else 2.0
                ndx = dx / (rx + margin)
                ndy = dy / (ry + margin)
                if ndx**p + ndy**p <= 1.0:
                    ndx_in = dx / max(0.1, in_rx)
                    ndy_in = dy / max(0.1, in_ry)
                    if ndx_in**p + ndy_in**p > 1.0:
                         return v, 'edge'
                    return v, 'center'
                    
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
        name = self.types.get(v.get("type_id", 0), {}).get("name", "")
        return name.strip().lower() == self.BBOX_TYPE_NAME

    @staticmethod
    def bbox_at(refs, grid_cfg, gx, gy):
        """Reference bbox containing the Deskewed point, else the one of the point's cell (or None)."""
        for (x0, y0, w, h) in refs.values():
            if x0 <= gx <= x0 + w and y0 <= gy <= y0 + h:
                return (x0, y0, w, h)
        return refs.get(grid_cfg.cell_at(gx, gy))

    def reference_bboxes(self, grid_cfg, coord_transform=None):
        """
        Reference bbox per grid cell: {(row, col): (x0, y0, w, h)} in Deskewed (grid-aligned)
        space. The bbox on the topmost (lowest-numbered) layer of the cell is used.
        """
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
        return refs

    def void_records(self, grid_cfg, bonding_map=None, coord_transform=None):
        """
        One dict per void with the export values, keyed by the internal names of
        output_columns.py. coord_transform is required for correct cells when the grid is rotated.
        """
        refs = self.reference_bboxes(grid_cfg, coord_transform)
        records = []
        for layer in sorted(self.voids):
            for v in self.voids[layer]:
                # Raw (storage) -> Deskewed (grid space)
                gx, gy = v["globalCX"], v["globalCY"]
                if coord_transform:
                    gx, gy = coord_transform.raw_to_deskew(gx, gy)
                row, col = grid_cfg.cell_at(gx, gy)
                chip_x0, chip_y0 = grid_cfg.cell_origin(row, col)

                # Resolve Type Name
                tid = v.get("type_id", 0)
                type_name = self.types[tid]["name"] if tid in self.types else "bbox"

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
                    "type": type_name,
                    "type_id": tid,
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
                })
        return records

    def save_to_file(self, path, grid_cfg, bonding_map=None, coord_transform=None):
        """
        Converts Global -> Chip Relative using current Grid Config.
        coord_transform is required for correct cells when the grid is rotated.
        Saves as Dict with 'types' and 'voids'. Field names come from output_columns.
        "type" field uses Type Name (User Request).
        patchLabel uses x_y_layer_bondinglabel format.
        "layer" is the absolute layer number (version 3).
        """
        data = {
            "version": 3,
            "layer_numbering": "absolute",
            "types": self.types, # ID -> {name, color}
            "voids": [
                {name: rec[key] for key, name in self.json_fields.items() if name}
                for rec in self.void_records(grid_cfg, bonding_map, coord_transform)
            ]
        }
                
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data['voids'])} voids and {len(self.types)} types to {path}")
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

            self.checkpoint() # Loading replaces the current voids (undoable)
            self.voids.clear()
            count = 0
            legacy_layers = True

            # Determine format
            if isinstance(raw_data, list):
                # V1 Format
                void_list = raw_data
            elif isinstance(raw_data, dict):
                legacy_layers = raw_data.get("layer_numbering") != "absolute"
                # V2 Format
                if "types" in raw_data:
                    # File types win, but keep keys declared in config.json (snap, def_p, ...)
                    # for types of the same name that the file does not define
                    by_name = {t.get("name"): t for t in self.types.values()}
                    types = {}
                    for k, t in raw_data["types"].items():
                        merged = dict(by_name.get(t.get("name"), {}))
                        merged.update(t)
                        merged["color"] = tuple(merged.get("color", (255, 255, 0)))
                        types[int(k)] = merged
                    self.types = types
                    if self.types:
                        self.next_type_id = max(self.types.keys()) + 1
                    else:
                         self.next_type_id = 0
                void_list = raw_data.get("voids", [])
            else:
                print("Unknown JSON format")
                return

            # Build Name -> ID map for resolution
            name_to_id = {v["name"]: k for k, v in self.types.items()}

            if legacy_layers and layer_offset:
                print(f"Legacy void file: relative layers shifted by +{layer_offset}")

            for item in void_list:
                layer = self._field(item, "layer", 0)
                if legacy_layers:
                    layer += layer_offset
                col = self._field(item, "col", 0)
                row = self._field(item, "row", 0)
                rel_cx = self._field(item, "center_x", 0)
                rel_cy = self._field(item, "center_y", 0)
                rx = self._field(item, "radius_x", 10.0)
                ry = self._field(item, "radius_y", 10.0)
                
                # Resolve Type
                tid = 0
                t_str = self._field(item, "type")
                t_id = self._field(item, "type_id")
                if t_str and t_str in name_to_id:
                    tid = name_to_id[t_str]
                elif t_id is not None:
                    tid = t_id
                    
                # If Type ID not in current types (e.g. from V1 file or mismatch), fallback to 0
                if tid not in self.types:
                    tid = 0
                
                gx = self._field(item, "global_x")
                gy = self._field(item, "global_y")
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
                    "type_id": tid,
                    "createdAt": self._field(item, "created_at", 0),
                    "voidIndex": self._field(item, "void_index", 0)
                }
                p = self._field(item, "p")
                if p is not None:
                    v_obj["p"] = float(p)

                self.voids[layer].append(v_obj)
                count += 1
            
            print(f"Loaded {count} voids from {path}")
            
        except Exception as e:
            print(f"Load failed: {e}")


class VoidTypeDialog(QDialog):
    """
    Dialog to manage Void Types (Add, Remove, Color).
    Inherits QDialog for proper window behavior.
    """
    def __init__(self, void_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Void Types")
        self.resize(300, 400)
        self.setWindowModality(Qt.ApplicationModal)
        
        self.vm = void_manager
        
        layout = QVBoxLayout(self)
        
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        
        h = QHBoxLayout()
        btn_add = QPushButton("Add")
        btn_remove = QPushButton("Remove")
        btn_color = QPushButton("Color")
        btn_rename = QPushButton("Rename")
        btn_shape = QPushButton("Shape (E/R/C)")
        
        h.addWidget(btn_add)
        h.addWidget(btn_remove)
        h.addWidget(btn_color)
        h.addWidget(btn_rename)
        h.addWidget(btn_shape)
        layout.addLayout(h)
        
        btn_add.clicked.connect(self.add_type)
        btn_remove.clicked.connect(self.remove_type)
        btn_color.clicked.connect(self.change_color)
        btn_rename.clicked.connect(self.rename_type)
        btn_rename.clicked.connect(self.rename_type)
        btn_shape.clicked.connect(self.toggle_shape)
        
        btn_def = QPushButton("Default Size")
        h.addWidget(btn_def)
        btn_def.clicked.connect(self.set_default_size)
        
        self.refresh()
        
    def refresh(self):
        self.list_widget.clear()
        for tid, data in self.vm.types.items():
            name = data["name"]
            color = data["color"] # (r, g, b)
            name = data["name"]
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
            # Set background color
            c = QColor(color[0], color[1], color[2])
            item.setBackground(c)
            # Text color contrast?
            if (c.red()*0.299 + c.green()*0.587 + c.blue()*0.114) < 128:
                item.setForeground(Qt.white)
            else:
                item.setForeground(Qt.black)
                
            item.setData(Qt.UserRole, tid)
            self.list_widget.addItem(item)
            
    def add_type(self):
        name, ok = QInputDialog.getText(self, "New Type", "Type Name:")
        if ok and name:
            self.vm.add_type(name, (255, 255, 0)) # Default Yellow
            self.refresh()
            
    def remove_type(self):
        row = self.list_widget.currentRow()
        if row < 0: return
        tid = self.list_widget.item(row).data(Qt.UserRole)
        self.vm.remove_type(tid)
        self.refresh()
        
    def change_color(self):
        row = self.list_widget.currentRow()
        if row < 0: return
        tid = self.list_widget.item(row).data(Qt.UserRole)
        
        curr_c = self.vm.types[tid]["color"]
        c = QColorDialog.getColor(QColor(curr_c[0], curr_c[1], curr_c[2]), self)
        
        if c.isValid():
            self.vm.types[tid]["color"] = (c.red(), c.green(), c.blue())
            self.vm.save_config()
            self.refresh()

    def rename_type(self):
        row = self.list_widget.currentRow()
        if row < 0: return
        tid = self.list_widget.item(row).data(Qt.UserRole)
        old_name = self.vm.types[tid]["name"]
        
        name, ok = QInputDialog.getText(self, "Rename Type", "New Name:", text=old_name)
        if ok and name:
            self.vm.types[tid]["name"] = name
            self.vm.save_config() # Save changes
            self.refresh()

    def toggle_shape(self):
        row = self.list_widget.currentRow()
        if row < 0: return
        tid = self.list_widget.item(row).data(Qt.UserRole)
        
        shapes = self.vm.SHAPES
        curr_shape = self.vm.types[tid].get("shape", "ellipse")
        new_shape = shapes[(shapes.index(curr_shape) + 1) % len(shapes)] if curr_shape in shapes else shapes[0]
        
        self.vm.types[tid]["shape"] = new_shape
        self.vm.save_config()
        self.refresh()

    def set_default_size(self):
        row = self.list_widget.currentRow()
        if row < 0: return
        tid = self.list_widget.item(row).data(Qt.UserRole)
        
        data = self.vm.types[tid]
        curr_w = data.get("def_w", 0.0)
        curr_h = data.get("def_h", 0.0)
        
        # simple dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Set Default Size")
        l = QVBoxLayout(dlg)
        
        from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QDialogButtonBox
        
        l.addWidget(QLabel("Default Width (0 = Manual):"))
        sb_w = QDoubleSpinBox()
        sb_w.setRange(0.0, 99999.0)
        sb_w.setValue(curr_w)
        l.addWidget(sb_w)
        
        l.addWidget(QLabel("Default Height (0 = Manual):"))
        sb_h = QDoubleSpinBox()
        sb_h.setRange(0.0, 99999.0)
        sb_h.setValue(curr_h)
        l.addWidget(sb_h)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        l.addWidget(btns)
        
        if dlg.exec():
            # Update
            new_w = sb_w.value()
            new_h = sb_h.value()
            
            self.vm.types[tid]["def_w"] = new_w
            self.vm.types[tid]["def_h"] = new_h
            self.vm.save_config()
            self.refresh()
