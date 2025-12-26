import sys
import os
import json
import time
import numpy as np
import tifffile
import re

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QSlider, QLabel, QWidget, 
    QVBoxLayout, QHBoxLayout, QToolBar, QSpinBox, QSplitter, 
    QDockWidget, QPlainTextEdit, QPushButton, QGroupBox, QFormLayout, 
    QDoubleSpinBox, QProgressDialog, QScrollArea, QSizePolicy, QCheckBox,
    QComboBox, QListWidgetItem, QListWidget, QRadioButton, QButtonGroup
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QAction, QColor, QImage, QPainter
from PySide6.QtCore import Qt, Signal, QRectF, QTimer, QPointF
import re

from OpenGL.GL import *
from OpenGL.GL import shaders

# ==================================================================================
# 1. Utils
# ==================================================================================
def to_gray2d_uint16(arr: np.ndarray, z_index: int = 0) -> np.ndarray:
    """
    Extracts a single 2D grayscale layer from a multi-dim array.
    Returns (H, W) uint16.
    """
    current_frame = arr
    if arr.ndim == 4:
        # (Z, H, W, C)
        current_frame = arr[z_index]
    elif arr.ndim == 3:
        if arr.shape[-1] not in (3, 4):
            # (Z, H, W) - Gray Z-stack
            current_frame = arr[z_index]
        else:
            # (H, W, RGB) - Single RGB image
            pass  # No Z slicing needed, but z_index should technically be 0
            
    # Now current_frame is (H, W) or (H, W, C)
    
    # 2. Convert RGB to Gray if needed
    if current_frame.ndim == 3 and current_frame.shape[-1] in (3, 4):
        # RGB(A) to Gray
        # Use float32 for precision
        rgb = current_frame[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        
        # Scale if it was uint8
        if current_frame.dtype == np.uint8:
            gray = gray * 257.0 # 0-255 -> 0-65535 map approximately
            
        current_frame = np.clip(gray, 0, 65535).astype(np.uint16)
        
    # 3. Ensure 2D (H, W)
    if current_frame.ndim != 2:
        raise ValueError(f"Could not convert to (H,W) uint16. Shape: {current_frame.shape}")
        
    # Check if we need to scale uint8 -> uint16
    if current_frame.dtype == np.uint8:
        return (current_frame.astype(np.uint16) * 257)

    return current_frame.astype(np.uint16, copy=False)



class LazyTiffStack:
    """
    Wraps a tifffile.TiffFile instance to provide lazy-loading of layers.
    Mimics a 3D numpy array (Z, H, W) or 4D (Z, H, W, C) interface (read-only).
    """
    def __init__(self, tif_instance):
        self.tif = tif_instance
        self.is_series = (len(self.tif.series) > 1)
        
        if self.is_series:
            self.items = self.tif.series
            ref = self.items[0]
            # series.shape is usually (H, W) or (H, W, C) or (Z, H, W)
            # We assume each series is a "Layer" (Z-slice).
            # If each series is 2D: Stack is (N_Series, H, W)
            self.base_shape = ref.shape
            self.dtype = ref.dtype
            self.len = len(self.items)
        else:
            self.items = self.tif.pages
            ref = self.items[0]
            self.base_shape = ref.shape
            self.dtype = ref.dtype
            self.len = len(self.items)
            
        # Construct Shape
        self.shape = (self.len,) + self.base_shape
        self.ndim = 1 + len(self.base_shape)
        
    def __getitem__(self, key):
        # Handle Integer Index [z]
        if isinstance(key, int):
            if key < 0: key += self.len
            if key < 0 or key >= self.len: raise IndexError("Index out of bounds")
            return self.items[key].asarray()
            
        # Handle Slice [a:b]
        if isinstance(key, slice):
            start, stop, step = key.indices(self.len)
            # Return list of arrays? Or stacked array?
            return np.stack([self.items[i].asarray() for i in range(start, stop, step)])
            
        # Handle tuple [z, y, x]
        if isinstance(key, tuple):
            z = key[0]
            # Defer other dims to the array
            layer = self[z] # Get array
            return layer[key[1:]]
            
        return self.items[key].asarray()

    def __len__(self):
        return self.len

# ==================================================================================
# 2. Tiling System
# ==================================================================================
class Tile:
    """
    Represents a single GPU texture tile.
    """
    def __init__(self, x, y, width, height, data):
        self.x = x  # Global X position
        self.y = y  # Global Y position
        self.w = width
        self.h = height
        self.data = np.ascontiguousarray(data, dtype=np.uint16) # CPU copy
        self.tex_id = None
        self.is_uploaded = False

    def upload(self):
        if self.is_uploaded: 
            return
            
        if self.tex_id is None:
            self.tex_id = glGenTextures(1)
            
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        # 16-bit single channel texture
        # Internal format: GL_R16 (16-bit Red/Gray)
        # Format: GL_RED
        # Type: GL_UNSIGNED_SHORT
        
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R16, 
            self.w, self.h, 0, 
            GL_RED, GL_UNSIGNED_SHORT, self.data
        )
        
        self.is_uploaded = True
        
        # MEMORY OPTIMIZATION: Release CPU copy immediately
        # We don't need it after upload. If context is lost, we reload from disk (LazyStack).
        self.data = None

    def cleanup(self):
        if self.tex_id:
            glDeleteTextures([self.tex_id])
            self.tex_id = None
        self.is_uploaded = False
        self.data = None # Ensure clear

class TiledImage:
    """
    Manages the grid of tiles for a single image layer.
    """
    TILE_SIZE = 4096 # Safe for most modern GPUs (limit is usually 8192 or 16384)

    def __init__(self, full_image: np.ndarray):
        """
        full_image: (H, W) uint16 numpy array
        """
        self.h, self.w = full_image.shape
        self.tiles = []
        
        # Create tiles
        for y in range(0, self.h, self.TILE_SIZE):
            for x in range(0, self.w, self.TILE_SIZE):
                h_chunk = min(self.TILE_SIZE, self.h - y)
                w_chunk = min(self.TILE_SIZE, self.w - x)
                
                # Extract sub-region
                sub_img = full_image[y:y+h_chunk, x:x+w_chunk]
                
                tile = Tile(x, y, w_chunk, h_chunk, sub_img)
                self.tiles.append(tile)

    def upload_all(self):
        for t in self.tiles:
            t.upload()

    def cleanup(self):
        for t in self.tiles:
            t.cleanup()



# ==================================================================================
# 2.2 Void Data Manager
# ==================================================================================
class VoidManager:
    """
    Manages void markings for all layers.
    Stores data in Global Coordinates (truth).
    Handles saving/loading relative to Chip Coordinates.
    """
    def __init__(self):
        self.voids = {} # Dict[layer_index: int, List[dict]]
        
        # Void Types
        # ID -> {name, color: (r,g,b)}
        self.types = {
            0: {"name": "Defect", "color": (255, 0, 0)}, # Red
            1: {"name": "Warning", "color": (255, 165, 0)}, # Orange
            2: {"name": "Safe", "color": (0, 255, 0)}, # Green
            3: {"name": "Check", "color": (0, 255, 255)} # Cyan
        }
        self.next_type_id = 4

    def add_type(self, name, color):
        tid = self.next_type_id
        self.types[tid] = {"name": name, "color": color}
        self.next_type_id += 1
        return tid

    def update_type(self, tid, name, color):
        if tid in self.types:
            self.types[tid] = {"name": name, "color": color}

    def remove_type(self, tid):
        if tid in self.types and len(self.types) > 1:
            del self.types[tid]
            # Remap existing voids to default 0?
            for layer in self.voids:
                for v in self.voids[layer]:
                    if v.get("type_id") == tid:
                        v["type_id"] = 0 # Default

    def add_void(self, layer, gx, gy, radius, type_id=0):
        if layer not in self.voids:
            self.voids[layer] = []
            
        new_void = {
            "globalCX": gx,
            "globalCY": gy,
            "radius": radius,
            "layer": layer,
            "type_id": type_id,
            "createdAt": int(time.time() * 1000),
            "voidIndex": 0
        }
        self.voids[layer].append(new_void)
        return new_void

    def delete_void_at(self, layer, gx, gy, hit_radius=10.0):
        """
        Deletes the first void overlapping the point (Simple Hit Test).
        Returns True if deleted.
        """
        if layer not in self.voids: return False
        
        target = None
        for v in self.voids[layer]:
            # Distance check
            dx = gx - v["globalCX"]
            dy = gy - v["globalCY"]
            dist_sq = dx*dx + dy*dy
            
            rad = v["radius"]
            if dist_sq <= rad*rad:
                target = v
                break
                
        if target:
            self.voids[layer].remove(target)
            return True
        return False

    def hit_test(self, layer, gx, gy, margin=5.0):
        """
        Returns (void_obj, type_str)
        type_str: 'center' (move), 'edge' (resize), None
        """
        if layer not in self.voids: return None, None
        
        for v in reversed(self.voids[layer]): # Top-most first
            dx = gx - v["globalCX"]
            dy = gy - v["globalCY"]
            dist = np.sqrt(dx*dx + dy*dy)
            rad = v["radius"]
            
            if abs(dist - rad) <= margin:
                return v, 'edge'
            
            if dist < rad:
                return v, 'center'
                
        return None, None

    def clear_layer(self, layer):
        if layer in self.voids:
            self.voids[layer] = []

    def save_to_file(self, path, grid_cfg):
        """
        Converts Global -> Chip Relative using current Grid Config.
        Saves as Dict with 'types' and 'voids'.
        "type" field uses Type Name (User Request).
        """
        data = {
            "version": 2,
            "types": self.types, # ID -> {name, color}
            "voids": []
        }
        
        # Flatten all layers
        for layer, v_list in self.voids.items():
            for v in v_list:
                col = int(np.floor((v["globalCX"] - grid_cfg.start_x) / grid_cfg.pitch_x))
                row = int(np.floor((v["globalCY"] - grid_cfg.start_y) / grid_cfg.pitch_y))
                
                chip_x0 = grid_cfg.start_x + col * grid_cfg.pitch_x
                chip_y0 = grid_cfg.start_y + row * grid_cfg.pitch_y
                
                rel_cx = v["globalCX"] - chip_x0
                rel_cy = v["globalCY"] - chip_y0
                
                rad = v["radius"]
                key_str = f"{col},{row},{layer},0"
                
                # Resolve Type Name
                tid = v.get("type_id", 0)
                type_name = "bbox"
                if tid in self.types:
                    type_name = self.types[tid]["name"]
                
                item = {
                    "key": key_str,
                    "x": col, "y": row, "layer": layer, "voidIndex": 0,
                    "type": type_name, # Use Name as requested
                    "type_id": tid, # Keep ID for backup
                    "centerX": rel_cx,
                    "centerY": rel_cy,
                    "radiusX": rad,
                    "radiusY": rad,
                    "createdAt": v.get("createdAt", 0),
                    "patchLabel": f"X{col:02d}_Y{row:02d}_L{layer:02d}_void"
                }
                data["voids"].append(item)
                
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data['voids'])} voids and {len(self.types)} types to {path}")
        except Exception as e:
            print(f"Save failed: {e}")

    def load_from_file(self, path, grid_cfg):
        """
        Reads JSON -> Calculates Global using Grid Config.
        Handles V1 (list) and V2 (dict) formats.
        Resolves 'type' string back to ID.
        """
        try:
            with open(path, 'r') as f:
                raw_data = json.load(f)
                
            self.voids.clear()
            count = 0
            
            # Determine format
            if isinstance(raw_data, list):
                # V1 Format
                void_list = raw_data
            elif isinstance(raw_data, dict):
                # V2 Format
                if "types" in raw_data:
                    self.types = {int(k): v for k, v in raw_data["types"].items()}
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

            for item in void_list:
                layer = item.get("layer", 0)
                col = item.get("x", 0)
                row = item.get("y", 0)
                rel_cx = item.get("centerX", 0)
                rel_cy = item.get("centerY", 0)
                rad = item.get("radiusX", 10.0)
                
                # Resolve Type
                tid = 0
                t_str = item.get("type", None)
                if t_str and t_str in name_to_id:
                    tid = name_to_id[t_str]
                elif "type_id" in item:
                    tid = item["type_id"]
                    
                # If Type ID not in current types (e.g. from V1 file or mismatch), fallback to 0
                if tid not in self.types:
                    tid = 0
                
                # Calc Global
                chip_x0 = grid_cfg.start_x + col * grid_cfg.pitch_x
                chip_y0 = grid_cfg.start_y + row * grid_cfg.pitch_y
                
                gx = chip_x0 + rel_cx
                gy = chip_y0 + rel_cy
                
                if layer not in self.voids: self.voids[layer] = []
                
                v_obj = {
                    "globalCX": gx,
                    "globalCY": gy,
                    "radius": rad,
                    "layer": layer,
                    "type_id": tid,
                    "createdAt": item.get("createdAt", 0),
                    "voidIndex": item.get("voidIndex", 0)
                }
                self.voids[layer].append(v_obj)
                count += 1
            
            print(f"Loaded {count} voids from {path}")
            
        except Exception as e:
            print(f"Load failed: {e}")

# ==================================================================================
# 2.3 Bonding Map & Grid System
# ==================================================================================
class VoidTypeDialog(QWidget):
    """
    Dialog to manage Void Types (Add, Remove, Color).
    Embeds in a QDialog usually, but here inheriting QWidget for simplicity if docked,
    or better just QDialog.
    """
    def __init__(self, void_manager, parent=None):
        from PySide6.QtWidgets import QDialog, QListWidget, QListWidgetItem, QColorDialog
        # Dynamic import to avoid top-level circular deps if any, though top level is fine
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
        
        h.addWidget(btn_add)
        h.addWidget(btn_remove)
        h.addWidget(btn_color)
        h.addWidget(btn_rename)
        layout.addLayout(h)
        
        btn_add.clicked.connect(self.add_type)
        btn_remove.clicked.connect(self.remove_type)
        btn_color.clicked.connect(self.change_color)
        btn_rename.clicked.connect(self.rename_type)
        
        self.refresh()
        
    def refresh(self):
        self.list_widget.clear()
        for tid, data in self.vm.types.items():
            name = data["name"]
            color = data["color"] # (r, g, b)
            item = QListWidgetItem(f"{name}")
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
        from PySide6.QtWidgets import QInputDialog
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
            self.refresh()

    def rename_type(self):
        row = self.list_widget.currentRow()
        if row < 0: return
        tid = self.list_widget.item(row).data(Qt.UserRole)
        old_name = self.vm.types[tid]["name"]
        
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Rename Type", "New Name:", text=old_name)
        if ok and name:
            self.vm.types[tid]["name"] = name
            self.refresh()


class BondingMap:
    """Parses and stores the bonding map data (keys/labels in a grid).
    Assigns colors to unique keys.
    """
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.data_map = [] # List of lists (row-major)
        self.unique_keys = {} # key -> QColor
        self.colors_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (255, 128, 0), (128, 255, 0), (0, 128, 255),
            (128, 0, 255), (255, 0, 128), (128, 128, 128)
        ]

    def parse_text(self, text: str):
        """
        Parses tab/newline separated text (Excel copy-paste).
        Handles optional row/column headers (0, 1, 2...).
        """
        lines = text.strip().split('\n')
        if not lines: return

        # Pre-process into list of lists (preserving empty cells)
        raw_grid = []
        for line in lines:
            line = line.rstrip('\r\n')
            # Use split('\t') to preserve empty columns (Excel style)
            parts = line.split('\t')
            raw_grid.append([p.strip() for p in parts])
            
        if not raw_grid: return

        # Header Detection Heuristic
        # Check Row 0: Are they mostly sequential integers?
        has_col_header = False
        try:
            # Check a few items in first row
            # Usually starts with empty or '0' or whatever
            # Let's count how many look like ints
            ints_found = 0
            for cell in raw_grid[0]:
                if cell.isdigit():
                    ints_found += 1
            if ints_found > len(raw_grid[0]) * 0.5: # > 50% are digits
                has_col_header = True
        except:
            pass

        # Check Col 0: Are they mostly sequential integers?
        has_row_header = False
        try:
            ints_found = 0
            for row in raw_grid:
                if row and row[0].isdigit():
                    ints_found += 1
            if ints_found > len(raw_grid) * 0.5:
                has_row_header = True
        except:
            pass
            
        # Determine start indices
        start_r = 1 if has_col_header else 0
        start_c = 1 if has_row_header else 0
        
        # Extract Data
        self.data_map = []
        unique_set = set()
        
        # Scan to find max dimensions
        max_cols = 0
        valid_rows = 0
        
        # Read data
        extracted_rows = []
        for r in range(start_r, len(raw_grid)):
            row_items = raw_grid[r]
            # Slice off header column
            if start_c < len(row_items):
                data_items = row_items[start_c:]
            else:
                data_items = []
                
            extracted_rows.append(data_items)
            max_cols = max(max_cols, len(data_items))
            
        self.rows = len(extracted_rows)
        self.cols = max_cols
        
        # Normalize to rectangular grid
        self.data_map = []
        for r_idx, row_data in enumerate(extracted_rows):
            padded = row_data + [""] * (max_cols - len(row_items))
            self.data_map.append(padded)
            # Collect unique keys
            for val in padded:
                if val: unique_set.add(val)

        # Assign colors
        self.unique_keys = {}
        sorted_keys = sorted(list(unique_set))
        for i, k in enumerate(sorted_keys):
            rgb = self.colors_palette[i % len(self.colors_palette)]
            self.unique_keys[k] = QColor(rgb[0], rgb[1], rgb[2], 100) # Alpha 100 for overlay

    def get_color(self, r, c) -> QColor:
        if r < 0 or r >= len(self.data_map): return None
        row = self.data_map[r]
        if c < 0 or c >= len(row): return None
        key = row[c]
        if not key: return None # Empty string -> No color
        return self.unique_keys.get(key, None)

    def get_key(self, r, c) -> str:
        if r < 0 or r >= len(self.data_map): return ""
        row = self.data_map[r]
        if c < 0 or c >= len(row): return ""
        return row[c]


class GridConfig:
    def __init__(self):
        self.visible = False
        self.start_x = 100.0
        self.start_y = 100.0
        self.pitch_x = 200.0
        self.pitch_y = 200.0
        self.rows = 1
        self.cols = 1
        self.angle = 0.0      # Rotation in degrees
        self.line_width = 1.0 # Reduced from 2.0
        self.opacity = 0.5    # Default 50% opacity
        
        # Calibration
        self.use_calib = False
        self.target_mean = 128.0
        self.target_std = 40.0

# ==================================================================================
# 3. OpenGL Widget
# ==================================================================================
class GLImageWidget(QOpenGLWidget):
    grid_params_changed = Signal()
    layer_wheel_changed = Signal(int) # Delta (+1 or -1)
    
    def __init__(self):
        super().__init__()
        self.tiled_image: TiledImage = None
        
        self.win_lo = 0.0
        self.win_hi = 1.0
        
        # Grid & Map
        self.grid_cfg = GridConfig()
        self.bonding_map = None # Single BondingMap Instance
        self.map_tex = None
        
        # Void Marking
        self.void_manager = None
        self.void_mode = False # "DRAW" | "EDIT" | "ERASE" | False (Off)
        self.active_type_id = 0
        self.current_layer = 0
        
        # Auto-Calibration
        self.calib_tex = None # GL Texture for (Scale, Offset) per cell
        self.calib_data_shape = (0, 0) # (Rows, Cols)
        
        self.setFocusPolicy(Qt.StrongFocus) # Enable keyboard events
        
        # View transform
        # Image coordinates: (0,0) is top-left, (W,H) is bottom-right.
        # We map this to Normalized Device Coords (NDC) in shader.
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        self._last_mouse = None
        

        # Shader vars
        self.prog = None
        
    def update_map_texture(self):
        """Creates/Updates the texture for the bonding map labels"""
        if not self.bonding_map: 
            return
            
        rows = self.bonding_map.rows
        cols = self.bonding_map.cols
        if rows == 0 or cols == 0:
            return
            
        # Create a tiny texture (cols x rows)
        # Format: GL_RGBA, GL_FLOAT? Or GL_UNSIGNED_BYTE
        # We'll use simple RGBA uint8
        
        # Build buffer
        arr = np.zeros((rows, cols, 4), dtype=np.uint8)
        
        for r in range(rows):
            for c in range(cols):
                color = self.bonding_map.get_color(r, c)
                if color:
                    arr[r, c] = [color.red(), color.green(), color.blue(), color.alpha()]
                else:
                    arr[r, c] = [0, 0, 0, 0] # Transparent
                    
        # Upload
        self.makeCurrent()
        if self.map_tex:
            glDeleteTextures([self.map_tex])
            
        self.map_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.map_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Note: rows is height, cols is width. 
        # But OpenGL expects width, height.
        # arr shape is (H, W, 4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr)
        
        self.doneCurrent()
        self.update()

    def update_calib_texture(self, calib_data: np.ndarray):
        """
        Updates the calibration texture.
        calib_data: (rows, cols, 2) float32 array, where [..., 0] is scale, [..., 1] is offset.
        """
        if calib_data is None or calib_data.size == 0:
            if self.calib_tex:
                self.makeCurrent()
                glDeleteTextures([self.calib_tex])
                self.calib_tex = None
                self.doneCurrent()
            self.calib_data_shape = (0, 0)
            self.update()
            return

        rows, cols, _ = calib_data.shape
        self.calib_data_shape = (rows, cols)

        self.makeCurrent()
        if self.calib_tex:
            glDeleteTextures([self.calib_tex])
            
        self.calib_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.calib_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Upload as RG float texture
        # calib_data is (rows, cols, 2)
        # OpenGL expects (width, height) -> (cols, rows)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, cols, rows, 0, GL_RG, GL_FLOAT, calib_data)
        
        self.doneCurrent()
        self.update()

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # Enable basic GL states
        # Enable blending for overlay
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        
        # Create Shader
        self.vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vs, """
        #version 330
        
        layout(location = 0) in vec2 aPos; // 0..1 quad
        layout(location = 1) in vec2 aTex; 
        
        uniform vec2 tileOffset; // In Image Pixels
        uniform vec2 tileSize;   // In Image Pixels
        uniform vec2 imageSize;  // Total Image Size
        
        uniform float zoom;
        uniform vec2 pan;     
        uniform vec2 viewSize; 
        
        out vec2 uv;
        out vec2 pixelPos; // Pass global pixel position to fragment for grid
        
        void main() {
            pixelPos = tileOffset + aPos * tileSize;
            
            vec2 imgCenter = imageSize * 0.5;
            vec2 viewCenter = imgCenter - pan; 
            
            vec2 finalPos = (pixelPos - viewCenter) * zoom / (viewSize * 0.5);
            finalPos.y = -finalPos.y; 
            
            gl_Position = vec4(finalPos, 0.0, 1.0);
            uv = aTex;
        }
        """)
        
        self.fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fs, """
        #version 330
        in vec2 uv;
        in vec2 pixelPos;
        
        out vec4 color;
        
        uniform sampler2D tex;
        uniform float win_lo;
        uniform float win_hi;
        
        // Grid / Map Params
        uniform int show_grid;
        uniform float grid_x;
        uniform float grid_y;
        uniform float pitch_x;
        uniform float pitch_y;
        uniform int grid_rows;
        uniform int grid_cols;
        uniform float grid_angle; // Degrees
        uniform float line_width;
        uniform float zoom; // to adjust line width?
        
        uniform sampler2D map_tex;
        uniform int has_map;
        uniform float grid_opacity;
        
        uniform sampler2D calib_tex;
        uniform int use_calib;
        
        void main() {
            // 1. Base Image
            float val = texture(tex, uv).r; 

            // Apply Window Leveling
            float pixel_val = val * 65535.0;
            float normalized = (pixel_val - win_lo) / (win_hi - win_lo);
            
            // Grid Calculation Wrapper
            
            // Calculate Grid Coords with Rotation
            float rad = radians(grid_angle);
            float s = sin(rad);
            float c = cos(rad);
            
            vec2 rel = pixelPos - vec2(grid_x, grid_y);
            // Rotate by -angle (World to Grid)
            // x_rot = x*cos(-a) - y*sin(-a) = x*c + y*s
            // y_rot = x*sin(-a) + y*cos(-a) = -x*s + y*c
            vec2 rotPos;
            rotPos.x = rel.x * c + rel.y * s;
            rotPos.y = -rel.x * s + rel.y * c;
            
            float gx = rotPos.x / pitch_x;
            float gy = rotPos.y / pitch_y;
            
            int c_idx = int(floor(gx));
            int r_idx = int(floor(gy));
    
            // Auto-Calibration
            if (use_calib == 1) {
                if (c_idx >= 0 && c_idx < grid_cols && r_idx >= 0 && r_idx < grid_rows) {
                     // Sample Calib Texture
                     // Center of the texel
                     // Data array[0] (Row 0, Top) is uploaded to V=0 (Bottom).
                     // So we want r=0 to map to V=0.
                     vec2 calibUV = vec2((float(c_idx) + 0.5) / float(grid_cols), (float(r_idx) + 0.5) / float(grid_rows));
                     vec2 params = texture(calib_tex, calibUV).rg;
                     
                     normalized = normalized * params.r + params.g;
                }
            }
            
            vec3 baseColor = vec3(clamp(normalized, 0.0, 1.0));
            
            // 2. Grid & Overlay
            vec4 overlayColor = vec4(0.0);
            
            if (show_grid == 1) {
                // Determine if we are inside the grid bounding box
                // Check bounds in Grid Space (gx, gy)
                if (gx >= 0.0 && gx <= float(grid_cols) && gy >= 0.0 && gy <= float(grid_rows)) {
                    
                    int c = int(floor(gx));
                    int r = int(floor(gy));
                    
                    // Borders
                    // Distance to nearest integer
                    float dx = abs(gx - round(gx)) * pitch_x;
                    float dy = abs(gy - round(gy)) * pitch_y;
                    
                    // Zoom-dependent width?
                    float lw = line_width / zoom; 
                    if (lw < 1.0) lw = 1.0;
                    
                    if (dx < lw || dy < lw) {
                        overlayColor = vec4(1.0, 1.0, 0.0, 0.6); // Yellow Grid lines
                    }
                    
                    // Map Params (Colors)
                    if (has_map == 1) {
                            if (c >= 0 && c < grid_cols && r >= 0 && r < grid_rows) {
                                  // Map Lookup: Flip Y for texture lookup?
                                  // We previously handled map vertically?
                                  // Lets assume (row, col) matches Map Texture (row, col) with V inverted?
                                  // Re-use logic: map is uploaded row-by-row. R=0 at V=0?
                                  // If BondingMap.data_map[0] is R=0.
                                  // If we upload it, Row 0 goes to V=0 (Bottom).
                                  // So mapUV should be (r+0.5)/rows => V=Low.
                                  // So same as Calib.
                                  vec2 mapUV = vec2((float(c) + 0.5) / float(grid_cols), (float(r) + 0.5) / float(grid_rows));
                                  vec4 mapCol = texture(map_tex, mapUV);
                                  if (mapCol.a > 0.0) {
                                      // Blend map color
                                      overlayColor = mix(overlayColor, mapCol, 0.3); // Tint
                                  }
                            }
                    }
                }
            }
            
            // Blend
            overlayColor.a *= grid_opacity;
            color = vec4(mix(baseColor.rgb, overlayColor.rgb, overlayColor.a), 1.0);
        }
        """)

        glCompileShader(self.vs)
        if not glGetShaderiv(self.vs, GL_COMPILE_STATUS):
            print("VS Compile Error:", glGetShaderInfoLog(self.vs))
            
        glCompileShader(self.fs)
        if not glGetShaderiv(self.fs, GL_COMPILE_STATUS):
            print("FS Compile Error:", glGetShaderInfoLog(self.fs))

        self.prog = glCreateProgram()
        glAttachShader(self.prog, self.vs)
        glAttachShader(self.prog, self.fs)
        glLinkProgram(self.prog)
        
        # Check errors
        if not glGetProgramiv(self.prog, GL_LINK_STATUS):
            print("Link Error:", glGetProgramInfoLog(self.prog))
            
        # Create a simple Quad VAO (0,0) to (1,1)
        # We will scale this in the vertex shader using 'tileOffset' and 'tileSize'
        vertices = np.array([
            0.0, 0.0,  0.0, 0.0,
            1.0, 0.0,  1.0, 0.0,
            0.0, 1.0,  0.0, 1.0,
            1.0, 1.0,  1.0, 1.0,
        ], dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Attr 0: Pos
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Attr 1: UV
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*4, ctypes.c_void_p(2*4))
        glEnableVertexAttribArray(1)
        

    def set_tiled_image(self, tiled_img: TiledImage, cleanup_old: bool = True):
        # Cleanup old if requested
        if cleanup_old and self.tiled_image and self.tiled_image != tiled_img:
            self.tiled_image.cleanup()
        
        self.tiled_image = tiled_img
        
        # Context must be current for upload
        self.makeCurrent()
        self.tiled_image.upload_all()
        self.doneCurrent()
        
        self.update()

    def set_window_level(self, lo, hi):
        self.win_lo = lo
        self.win_hi = hi
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        if not self.tiled_image:
            return
            
        glUseProgram(self.prog)
        
        # Global uniforms
        w = self.width()
        h = self.height()
        img_w = self.tiled_image.w
        img_h = self.tiled_image.h
        
        glUniform1f(glGetUniformLocation(self.prog, "zoom"), self.zoom)
        glUniform2f(glGetUniformLocation(self.prog, "pan"), self.pan_x, self.pan_y)
        glUniform2f(glGetUniformLocation(self.prog, "viewSize"), w, h)
        glUniform2f(glGetUniformLocation(self.prog, "imageSize"), img_w, img_h)
        glUniform1f(glGetUniformLocation(self.prog, "win_lo"), self.win_lo)
        glUniform1f(glGetUniformLocation(self.prog, "win_hi"), self.win_hi)
        

        glUniform1i(glGetUniformLocation(self.prog, "tex"), 0)
        
        # Grid Uniforms
        glUniform1i(glGetUniformLocation(self.prog, "show_grid"), 1 if self.grid_cfg.visible else 0)
        glUniform1f(glGetUniformLocation(self.prog, "grid_x"), self.grid_cfg.start_x)
        glUniform1f(glGetUniformLocation(self.prog, "grid_y"), self.grid_cfg.start_y)
        glUniform1f(glGetUniformLocation(self.prog, "pitch_x"), self.grid_cfg.pitch_x)
        glUniform1f(glGetUniformLocation(self.prog, "pitch_y"), self.grid_cfg.pitch_y)
        glUniform1i(glGetUniformLocation(self.prog, "grid_rows"), self.grid_cfg.rows)
        glUniform1i(glGetUniformLocation(self.prog, "grid_cols"), self.grid_cfg.cols)
        glUniform1f(glGetUniformLocation(self.prog, "grid_angle"), self.grid_cfg.angle)
        glUniform1f(glGetUniformLocation(self.prog, "line_width"), self.grid_cfg.line_width)
        glUniform1f(glGetUniformLocation(self.prog, "grid_opacity"), self.grid_cfg.opacity)
        
        glUniform1i(glGetUniformLocation(self.prog, "use_calib"), 1 if self.grid_cfg.use_calib and self.calib_tex else 0)
        
        if self.calib_tex:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.calib_tex)
            glUniform1i(glGetUniformLocation(self.prog, "calib_tex"), 2)
        
        glUniform1i(glGetUniformLocation(self.prog, "has_map"), 1 if (self.bonding_map is not None and self.map_tex is not None) else 0)
        
        if self.map_tex:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.map_tex)
            glUniform1i(glGetUniformLocation(self.prog, "map_tex"), 1)
        
        glBindVertexArray(self.vao)
        glActiveTexture(GL_TEXTURE0)
        
        # Draw each tile
        for tile in self.tiled_image.tiles:
            if not tile.is_uploaded:
                continue
                
            glBindTexture(GL_TEXTURE_2D, tile.tex_id)
            
            glUniform2f(glGetUniformLocation(self.prog, "tileOffset"), tile.x, tile.y)
            glUniform2f(glGetUniformLocation(self.prog, "tileSize"), tile.w, tile.h)
            
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            
    # -----------------------------------------------
    # Mouse Interaction
    # -----------------------------------------------
    def wheelEvent(self, e):
        # Ctrl + Wheel -> Layer Switch
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            delta = e.angleDelta().y()
            if delta > 0:
                self.layer_wheel_changed.emit(-1) # Prev Layer
            elif delta < 0:
                self.layer_wheel_changed.emit(1)  # Next Layer
            return

        # Normal Zoom
        # Mouse pos in view
        mx = e.position().x()
        my = e.position().y()
        
        # ... standard zoom log ...
        
        delta = e.angleDelta().y()
        factor = 1.1
        if delta < 0:
            factor = 1.0 / 1.1
            
        # Zoom around mouse point
        # Old Image Pos
        # view_center + (mouse - view_center_screen) / zoom
        
        # Simpler approach:
        # zoom_new = zoom * factor
        # But we want to keep the mouse pointing at the same image coordinate
        
        # Image Coords of mouse before zoom
        view_w = self.width()
        view_h = self.height()
        cx = view_w / 2
        cy = view_h / 2
        
        if self.tiled_image:
             img_w = self.tiled_image.w
             img_h = self.tiled_image.h
        else:
             img_w = 1000
             img_h = 1000
             
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        # Current Pan
        # view_center_x = img_cx - pan_x
        # view_center_y = img_cy - pan_y
        
        # Mouse relative to screen center
        dx = mx - cx
        dy = my - cy
        
        # Mouse in Image Relative to View Center
        # m_rel_view = (dx, dy)
        # Mouse in Image Coords
        # m_img_x = view_center_x + dx / zoom
        # m_img_x = (img_cx - pan_x) + dx / zoom
        
        # We want m_img_x to be same after zoom
        # (img_cx - new_pan_x) + dx / new_zoom = (img_cx - pan_x) + dx / zoom
        # new_pan_x = pan_x + dx/zoom - dx/new_zoom
        
        new_zoom = self.zoom * factor
        
        # Limit zoom
        if new_zoom < 0.001: new_zoom = 0.001
        if new_zoom > 1000.0: new_zoom = 1000.0
        
        self.pan_x += (dx / self.zoom) - (dx / new_zoom)
        self.pan_y += (dy / self.zoom) - (dy / new_zoom)
        
        self.zoom = new_zoom
        self.update()

    def mouseDoubleClickEvent(self, e):
        if not self.grid_cfg.visible or self.tiled_image is None:
            super().mouseDoubleClickEvent(e)
            return

        # 1. Map Mouse to Image Coords
        # Screen Center
        view_w = self.width()
        view_h = self.height()
        cx = view_w / 2
        cy = view_h / 2
        
        # Mouse relative to center
        mx = e.position().x()
        my = e.position().y()
        dx = mx - cx
        dy = my - cy
        
        # Current View Center in Image Space
        img_w = self.tiled_image.w
        img_h = self.tiled_image.h
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        # Pan is (imgCenter - viewCenter)
        # viewCenter = imgCenter - pan
        view_center_x = img_cx - self.pan_x
        view_center_y = img_cy - self.pan_y
        
        # Click position in Image Space
        # Screen Delta = Image Delta * Zoom
        # Image Delta = Screen Delta / Zoom
        click_img_x = view_center_x + dx / self.zoom
        click_img_y = view_center_y + dy / self.zoom
        
        # 2. Check Grid intersection
        gx = click_img_x - self.grid_cfg.start_x
        gy = click_img_y - self.grid_cfg.start_y
        
        if gx < 0 or gy < 0: 
            return # Clicked before grid start
            
        col = int(gx / self.grid_cfg.pitch_x)
        row = int(gy / self.grid_cfg.pitch_y)
        
        if col >= self.grid_cfg.cols or row >= self.grid_cfg.rows:
            return # Clicked outside grid
            
        # 3. Calculate target view
        # Target Cell Center
        cell_cx = self.grid_cfg.start_x + (col + 0.5) * self.grid_cfg.pitch_x
        cell_cy = self.grid_cfg.start_y + (row + 0.5) * self.grid_cfg.pitch_y
        
        # New Pan (target view center should be cell center)
        # new_pan = img_cx - cell_cx
        self.pan_x = img_cx - cell_cx
        self.pan_y = img_cy - cell_cy
        
        # New Zoom
        # Fit pitch_x/y into view_w/h
        margin = 0.95
        zoom_x = view_w / self.grid_cfg.pitch_x
        zoom_y = view_h / self.grid_cfg.pitch_y
        self.zoom = min(zoom_x, zoom_y) * margin
        
        self.update()

    # -----------------------------------------------
    # Overlay Rendering (QPainter)
    # -----------------------------------------------
    def paintEvent(self, e):
        # 1. Draw OpenGL content
        super().paintEvent(e)
        
        # 2. Draw 2D Overlay (Voids)
        if self.void_manager and self.tiled_image:
             painter = QPainter(self)
             painter.setRenderHint(QPainter.Antialiasing)
             
             # Current Layer Voids
             layer_voids = self.void_manager.voids.get(self.current_layer, [])
             
             # Draw Helper
             def to_screen(gx, gy):
                 # Inverse of get_image_coords
                 view_w = self.width()
                 view_h = self.height()
                 cx = view_w / 2
                 cy = view_h / 2
                 
                 if self.tiled_image:
                     img_w = self.tiled_image.w
                     img_h = self.tiled_image.h
                 else:
                     img_w = 1000
                     img_h = 1000
                 
                 img_cx = img_w / 2
                 img_cy = img_h / 2
                 
                 screen_x = (gx - img_cx + self.pan_x) * self.zoom + cx
                 screen_y = (gy - img_cy + self.pan_y) * self.zoom + cy
                 return screen_x, screen_y

             # Iterate Voids
             for v in layer_voids:
                 sx, sy = to_screen(v["globalCX"], v["globalCY"])
                 sr = v["radius"] * self.zoom
                 
                 # Determine Color
                 tid = v.get("type_id", 0)
                 # Safety check
                 if tid not in self.void_manager.types: tid = 0
                 type_data = self.void_manager.types.get(tid, {"color": (255, 255, 0)})
                 col_rgb = type_data["color"]
                 
                 if v == getattr(self, 'active_void', None):
                     # Active: Red border, Type fill
                     painter.setPen(QColor(255, 0, 0, 255)) 
                     painter.setBrush(QColor(col_rgb[0], col_rgb[1], col_rgb[2], 100))
                 else:
                     # Normal: Type border, no fill
                     painter.setPen(QColor(col_rgb[0], col_rgb[1], col_rgb[2], 200)) 
                     painter.setBrush(Qt.NoBrush)
                     
                 # Draw Circle
                 painter.drawEllipse(QPointF(sx, sy), sr, sr)

             painter.end()

    # -----------------------------------------------
    # Mouse Interaction
    # -----------------------------------------------
    def get_image_coords(self, pos):
        # Converts screen position to Global Image Coordinates
        mx = pos.x()
        my = pos.y()
        
        view_w = self.width()
        view_h = self.height()
        cx = view_w / 2
        cy = view_h / 2
        
        dx = mx - cx
        dy = my - cy
        
        if self.tiled_image:
             img_w = self.tiled_image.w
             img_h = self.tiled_image.h
        else:
             img_w = 1000
             img_h = 1000
             
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        view_center_x = img_cx - self.pan_x
        view_center_y = img_cy - self.pan_y
        
        gx = view_center_x + dx / self.zoom
        gy = view_center_y + dy / self.zoom
        
        return gx, gy

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._last_mouse = e.position()
            self.setFocus()
            
            # Void Interaction
            if self.void_mode and self.void_manager and self.tiled_image:
                gx, gy = self.get_image_coords(e.position())
                
                if self.void_mode == "DRAW":
                    # Start New Void
                    r = 50.0 / self.zoom # Default radius visual
                    self.active_void = self.void_manager.add_void(self.current_layer, gx, gy, 10.0, self.active_type_id) # Start small
                    self.active_void_action = 'sizing'
                    self.update()
                    return # Consume event
                    
                elif self.void_mode == "EDIT":
                    # Hit Test
                    v, action = self.void_manager.hit_test(self.current_layer, gx, gy, margin=10.0/self.zoom)
                    if v:
                        self.active_void = v
                        self.active_void_action = action # 'center' or 'edge'
                        self.update()
                        return
                        
                elif self.void_mode == "ERASE":
                    # Delete on click
                    deleted = self.void_manager.delete_void_at(self.current_layer, gx, gy)
                    if deleted:
                        self.update()
                        # If drag-erase is desired, set flag?
                        self.active_void_action = 'erasing'
                        return

    def mouseMoveEvent(self, e):
        if self._last_mouse is None:
            return
            
        dx = e.position().x() - self._last_mouse.x()
        dy = e.position().y() - self._last_mouse.y()
        
        # Void Interaction
        if self.void_mode and getattr(self, 'active_void_action', None):
             # Handle Drag
             gx, gy = self.get_image_coords(e.position())
             
             if self.active_void_action == 'sizing' and self.active_void:
                 # Dist from center
                 dcx = gx - self.active_void["globalCX"]
                 dcy = gy - self.active_void["globalCY"]
                 dist = np.sqrt(dcx*dcx + dcy*dcy)
                 self.active_void["radius"] = dist
                 self.update()
                 
             elif self.active_void_action == 'center' and self.active_void:
                 # Move Center
                 # Simple delta
                 img_dx = dx / self.zoom
                 img_dy = dy / self.zoom
                 self.active_void["globalCX"] += img_dx
                 self.active_void["globalCY"] += img_dy
                 self.update()
                 
             elif self.active_void_action == 'edge' and self.active_void:
                 # Resize (Symetric)
                 # Dist from center
                 dcx = gx - self.active_void["globalCX"]
                 dcy = gy - self.active_void["globalCY"]
                 dist = np.sqrt(dcx*dcx + dcy*dcy)
                 self.active_void["radius"] = dist
                 self.update()
                 
             elif self.active_void_action == 'erasing':
                 # Continuous erase?
                 self.void_manager.delete_void_at(self.current_layer, gx, gy)
                 self.update()
                 
             self._last_mouse = e.position()
             return

        # Check modifiers

        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.ControlModifier and self.grid_cfg.visible:
            # Move Grid
            # dx is screen pixels.
            # grid_x/y are Image Pixels.
            # We need to map Screen Delta -> Image Delta.
            # 1 Image Pixel = Zoom Screen Pixels
            # DeltaImage = DeltaScreen / Zoom
            
            img_dx = dx / self.zoom
            img_dy = dy / self.zoom
            
            self.grid_cfg.start_x += img_dx
            self.grid_cfg.start_y += img_dy
            
            # Notify main window
            self.grid_params_changed.emit()
            
        else:
            # Pan
            self.pan_x += dx / self.zoom
            self.pan_y += dy / self.zoom 
        
        self._last_mouse = e.position()
        self.update()

    def mouseReleaseEvent(self, e):
        # Clear interaction state
        self._last_mouse = None
        
        if self.void_mode:
            self.active_void = None
            self.active_void_action = None
            self.update()
        
    def keyPressEvent(self, e):
        if not self.grid_cfg.visible:
            super().keyPressEvent(e)
            return

        # Nudge amount (in pixels)
        step = 1.0 
        if e.modifiers() == Qt.ShiftModifier:
            step = 10.0
            
        if e.key() == Qt.Key_Left:
            self.grid_cfg.start_x -= step
        elif e.key() == Qt.Key_Right:
            self.grid_cfg.start_x += step
        elif e.key() == Qt.Key_Up:
            self.grid_cfg.start_y -= step
        elif e.key() == Qt.Key_Down:
            self.grid_cfg.start_y += step
        else:
            super().keyPressEvent(e)
            return
            
        self.grid_params_changed.emit()
        self.update()


    def mouseReleaseEvent(self, e):
        self._last_mouse = None


# ==================================================================================
# 4. Main Window
# ==================================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tiled TIF Viewer (OpenGl Mosaic)")
        self.resize(1280, 720)
        
        self.calib_timer = QTimer()
        self.calib_timer.setSingleShot(True)
        self.calib_timer.timeout.connect(self.calculate_calibration)
        
        self.full_data = None #(Z, H, W) or (H, W) or LazyTiffStack
        self.current_tif_file = None # Keep file handle open for Lazy Stack
        self.current_z = 0
        self.layer_offset = 0 # Absolute offset of loaded layers
        self.layer_cache = {} # Cache for TiledImage objects to avoid re-uploading
        self.layer_calib_data = {} # Cache for Calibration Data (Rows, Cols, 2)
        
        self.glw = GLImageWidget()
        
        # Controls
        self.spin_layer = QSpinBox()
        self.spin_layer.setRange(0, 0)
        self.spin_layer.setEnabled(False)
        self.spin_layer.valueChanged.connect(self.on_layer_changed)
        
        self.slider_lo = QSlider(Qt.Horizontal)
        self.slider_lo.setRange(0, 65535)
        self.slider_lo.setValue(0)
        self.slider_lo.valueChanged.connect(self.on_window_changed)
        
        self.slider_hi = QSlider(Qt.Horizontal)
        self.slider_hi.setRange(0, 65535)
        self.slider_hi.setValue(65535)
        self.slider_hi.valueChanged.connect(self.on_window_changed)
        
        # Layouts
        ctrl_widget = QWidget()
        h = QHBoxLayout(ctrl_widget)
        h.addWidget(QLabel("Layer:"))
        h.addWidget(self.spin_layer)
        h.addWidget(QLabel("Min:"))
        h.addWidget(self.slider_lo)
        h.addWidget(QLabel("Max:"))
        h.addWidget(self.slider_hi)
        
        root = QWidget()
        v = QVBoxLayout(root)
        v.addWidget(self.glw, 1)
        v.addWidget(ctrl_widget)
        self.setCentralWidget(root)
        
        # Toolbar
        tb = QToolBar()
        self.addToolBar(tb)
        a = QAction("Open TIF", self)
        a.triggered.connect(self.open_file)
        tb.addAction(a)

        # ---------------------------------------------------------
        # Dock Widget for Grid/Map Controls
        # ---------------------------------------------------------
        dock = QDockWidget("Grid & Map Tools", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        
        # 1. Grid Controls
        gb_grid = QGroupBox("Grid Configuration")
        form = QFormLayout(gb_grid)
        
        self.chk_grid_show = QPushButton("Show Grid: OFF")
        self.chk_grid_show.setCheckable(True)
        self.chk_grid_show.toggled.connect(self.toggle_grid)
        form.addRow(self.chk_grid_show)
        
        self.sb_grid_x = QDoubleSpinBox(); self.sb_grid_x.setRange(-10000, 50000); self.sb_grid_x.setValue(100)
        self.sb_grid_y = QDoubleSpinBox(); self.sb_grid_y.setRange(-10000, 50000); self.sb_grid_y.setValue(100)
        form.addRow("Origin X:", self.sb_grid_x)
        form.addRow("Origin Y:", self.sb_grid_y)
        
        self.sb_pitch_x = QDoubleSpinBox(); self.sb_pitch_x.setRange(1, 10000); self.sb_pitch_x.setValue(200)
        self.sb_pitch_y = QDoubleSpinBox(); self.sb_pitch_y.setRange(1, 10000); self.sb_pitch_y.setValue(200)
        form.addRow("Pitch X:", self.sb_pitch_x)
        form.addRow("Pitch Y:", self.sb_pitch_y)
        
        self.sb_rows = QSpinBox(); self.sb_rows.setRange(1, 1000); self.sb_rows.setValue(5)
        self.sb_cols = QSpinBox(); self.sb_cols.setRange(1, 1000); self.sb_cols.setValue(5)
        form.addRow("Rows:", self.sb_rows)
        form.addRow("Cols:", self.sb_cols)
        
        self.sb_angle = QDoubleSpinBox(); self.sb_angle.setRange(-180, 180); self.sb_angle.setValue(0.0)
        self.sb_angle.setSingleStep(0.1)
        form.addRow("Angle:", self.sb_angle)
        
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(50)
        form.addRow("Opacity:", self.slider_opacity)
        
        # Connect signals
        for w in [self.sb_grid_x, self.sb_grid_y, self.sb_pitch_x, self.sb_pitch_y, self.sb_angle]:
            w.valueChanged.connect(self.update_grid_params)
        for w in [self.sb_rows, self.sb_cols]:
            w.valueChanged.connect(self.update_grid_params)
            
        self.slider_opacity.valueChanged.connect(self.update_grid_params)
            
        self.glw.grid_params_changed.connect(self.on_grid_moved_by_input)
        self.glw.layer_wheel_changed.connect(self.on_layer_wheel_scroll)
        
        dock_layout.addWidget(gb_grid)

        # 1.5 Load Config
        gb_load = QGroupBox("Load Configuration (Next File)")
        form_load = QFormLayout(gb_load)
        
        self.sb_start_layer = QSpinBox(); self.sb_start_layer.setRange(0, 100); self.sb_start_layer.setValue(0)
        self.sb_end_layer = QSpinBox(); self.sb_end_layer.setRange(0, 100); self.sb_end_layer.setValue(10)
        form_load.addRow("Start Limit:", self.sb_start_layer)
        form_load.addRow("End Limit:", self.sb_end_layer)
        
        dock_layout.addWidget(gb_load)
        
        # 2. Bonding Map Paste
        gb_map = QGroupBox("Bonding Map (Paste Excel)")
        v_map = QVBoxLayout(gb_map)
        self.txt_map = QPlainTextEdit()
        self.txt_map.setPlaceholderText("Paste grid data here (tab separated)...")
        v_map.addWidget(self.txt_map)
        
        btn_import_map = QPushButton("Import Map Data")
        btn_import_map.clicked.connect(self.import_map_data)
        v_map.addWidget(btn_import_map)
        
        dock_layout.addWidget(gb_map)
        
        # 3. Extraction
        gb_extract = QGroupBox("Extraction")
        v_ext = QVBoxLayout(gb_extract)
        btn_extract = QPushButton("Extract Patches")
        btn_extract.clicked.connect(self.extract_patches)
        v_ext.addWidget(btn_extract)
        v_ext.addWidget(btn_extract)
        dock_layout.addWidget(gb_extract)
        
        # 3.5 Void Marking Control
        self.void_manager = VoidManager()
        self.glw.void_manager = self.void_manager
        
        self.gb_void = QGroupBox("Void Marking")
        v_layout_void = QVBoxLayout(self.gb_void)
        
        # Toggle Mode
        self.chk_void_mode = QCheckBox("Enable Void Mode (M)")
        self.chk_void_mode.toggled.connect(self.on_void_mode_toggled)
        
        # Mode Selection (Draw/Edit/Erase)
        h_modes = QHBoxLayout()
        self.rb_draw = QRadioButton("Draw")
        self.rb_edit = QRadioButton("Edit")
        self.rb_erase = QRadioButton("Erase")
        self.rb_draw.setChecked(True)
        
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_draw)
        self.mode_group.addButton(self.rb_edit)
        self.mode_group.addButton(self.rb_erase)
        
        # Use buttonClicked (pointer to button) or standard signals
        self.mode_group.buttonClicked.connect(self.on_void_mode_changed)
        
        h_modes.addWidget(self.rb_draw)
        h_modes.addWidget(self.rb_edit)
        h_modes.addWidget(self.rb_erase)
        
        # Type Selector
        h_type = QHBoxLayout()
        self.cb_void_type = QComboBox()
        self.cb_void_type.currentIndexChanged.connect(self.on_void_type_changed)
        self.btn_edit_types = QPushButton("Types...")
        self.btn_edit_types.clicked.connect(self.open_type_manager)
        h_type.addWidget(QLabel("Type:"))
        h_type.addWidget(self.cb_void_type, 1)
        h_type.addWidget(self.btn_edit_types)

        
        # Info
        self.lbl_void_count = QLabel("Voids: 0")
        
        # Buttons
        h_layout_void_btns = QHBoxLayout()
        self.btn_load_voids = QPushButton("Load JSON")
        self.btn_save_voids = QPushButton("Save JSON")
        self.btn_clear_voids = QPushButton("Clear Layer")
        
        self.btn_load_voids.clicked.connect(self.load_voids)
        self.btn_save_voids.clicked.connect(self.save_voids)
        self.btn_clear_voids.clicked.connect(self.clear_voids)
        
        h_layout_void_btns.addWidget(self.btn_load_voids)
        h_layout_void_btns.addWidget(self.btn_save_voids)
        
        v_layout_void.addWidget(self.chk_void_mode)
        v_layout_void.addLayout(h_modes)
        v_layout_void.addLayout(h_type)
        v_layout_void.addWidget(self.lbl_void_count)
        v_layout_void.addLayout(h_layout_void_btns)
        v_layout_void.addWidget(self.btn_clear_voids)
        
        dock_layout.addWidget(self.gb_void)
        
        # Init Types
        self.populate_void_types()
        
        # Shortcuts for Void Mode
        self.action_toggle_void = QAction("Toggle Void Mode", self)
        self.action_toggle_void.setShortcut("m")
        self.action_toggle_void.triggered.connect(self.toggle_void_mode_action)
        self.addAction(self.action_toggle_void)
        
        self.action_void_edit = QAction("Void Edit Mode", self)
        self.action_void_edit.setShortcut("e")
        self.action_void_edit.triggered.connect(self.set_void_mode_edit)
        self.addAction(self.action_void_edit)
        
        self.action_void_erase = QAction("Void Erase Mode", self)
        self.action_void_erase.setShortcut("d")
        self.action_void_erase.triggered.connect(self.set_void_mode_erase)
        self.addAction(self.action_void_erase)

        
        # 4. Auto Calibration
        gb_calib = QGroupBox("Auto Calibration")
        form_calib = QFormLayout(gb_calib)
        
        self.chk_use_calib = QPushButton("Enable Auto-Calib: OFF")
        self.chk_use_calib.setCheckable(True)
        self.chk_use_calib.toggled.connect(self.toggle_calib)
        form_calib.addRow(self.chk_use_calib)
        
        self.chk_calib_robust = QPushButton("Rboust Mode (Ignore Voids): OFF")
        self.chk_calib_robust.setCheckable(True)
        self.chk_calib_robust.clicked.connect(lambda c: self.chk_calib_robust.setText(f"Robust Mode (Ignore Voids): {'ON' if c else 'OFF'}"))
        self.chk_calib_robust.clicked.connect(self.calculate_calibration)
        form_calib.addRow(self.chk_calib_robust)
        
        self.chk_calib_all = QPushButton("Apply to All Layers: OFF")
        self.chk_calib_all.setCheckable(True)
        self.chk_calib_all.clicked.connect(lambda c: self.chk_calib_all.setText(f"Apply to All Layers: {'ON' if c else 'OFF'}"))
        form_calib.addRow(self.chk_calib_all)
        
        self.sb_target_mean = QDoubleSpinBox(); self.sb_target_mean.setRange(0, 65535); self.sb_target_mean.setValue(30000)
        self.sb_target_std = QDoubleSpinBox(); self.sb_target_std.setRange(0, 65535); self.sb_target_std.setValue(10000)
        form_calib.addRow("Target Mean:", self.sb_target_mean)
        form_calib.addRow("Target Std:", self.sb_target_std)
        
        btn_recalc = QPushButton("Recalculate")
        btn_recalc.clicked.connect(self.calculate_calibration)
        form_calib.addRow(btn_recalc)
        
        dock_layout.addWidget(gb_calib)
        
        dock_layout.addStretch()
        dock.setWidget(dock_content)

        # Sync initial state
        self.on_window_changed()
        self.update_grid_params()

    def toggle_grid(self, checked):
        self.chk_grid_show.setText(f"Show Grid: {'ON' if checked else 'OFF'}")
        self.glw.grid_cfg.visible = checked
        self.glw.update()

    def update_grid_params(self):
        cfg = self.glw.grid_cfg
        cfg.start_x = self.sb_grid_x.value()
        cfg.start_y = self.sb_grid_y.value()
        cfg.pitch_x = self.sb_pitch_x.value()
        cfg.pitch_y = self.sb_pitch_y.value()
        cfg.rows = self.sb_rows.value()
        cfg.cols = self.sb_cols.value()
        cfg.angle = self.sb_angle.value()
        cfg.opacity = self.slider_opacity.value() / 100.0
        self.glw.update()

    def on_grid_moved_by_input(self):
        # Update spinboxes without triggering recursive updates
        cfg = self.glw.grid_cfg
        self.sb_grid_x.blockSignals(True)
        self.sb_grid_y.blockSignals(True)
        
        self.sb_grid_x.setValue(cfg.start_x)
        self.sb_grid_y.setValue(cfg.start_y)
        self.sb_angle.setValue(cfg.angle)
        
        self.sb_grid_x.blockSignals(False)
        self.sb_grid_y.blockSignals(False)
        
        if self.chk_use_calib.isChecked():
             # Disable auto-recalc on move (User Request)
             # User must click 'Recalculate'
             pass
             # self.calib_timer.start(200) # 200ms debounce

    def on_layer_wheel_scroll(self, delta):
        # delta is +1 or -1
        if not self.spin_layer.isEnabled():
            return
            
        current = self.spin_layer.value()
        new_val = current + delta
        # Spinbox handles clamping but let's be safe
        new_val = max(self.spin_layer.minimum(), min(self.spin_layer.maximum(), new_val))
        self.spin_layer.setValue(new_val)

    def import_map_data(self):
        text = self.txt_map.toPlainText()
        if not text.strip():
            return
            
        bmap = BondingMap()
        bmap.parse_text(text)
        
        if bmap.rows > 0 and bmap.cols > 0:
            self.glw.bonding_map = bmap
            self.glw.update_map_texture()
            
            # Auto-update grid dimensions to match map
            self.sb_rows.setValue(bmap.rows)
            self.sb_cols.setValue(bmap.cols)
            
            # Enable grid if not
            if not self.chk_grid_show.isChecked():
                self.chk_grid_show.setChecked(True)
                
            print(f"Map Imported: {bmap.rows}x{bmap.cols}, Unique Keys: {len(bmap.unique_keys)}")
            
    def extract_patches(self):
        if self.full_data is None:
            print("No image loaded.")
            return

        # Choose directory
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir:
            return
            
        print(f"Extracting to {out_dir}...")
        
        cfg = self.glw.grid_cfg
        win_lo = self.glw.win_lo
        win_hi = self.glw.win_hi
        win_range = max(1.0, win_hi - win_lo)
        use_calib = self.chk_use_calib.isChecked()
        
        # Determine layers
        # User said "all layers"
        if self.full_data.ndim == 2 or (self.full_data.ndim == 3 and self.full_data.shape[-1] in (3,4)):
            layers = [0]
        elif self.full_data.ndim == 3:
            layers = range(self.full_data.shape[0])
        elif self.full_data.ndim == 4:
             layers = range(self.full_data.shape[0])
        else:
            layers = [0]
            
        import os
        
        total_extracted = 0
        
        progress = QProgressDialog("Extracting patches...", "Cancel", 0, len(layers), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        for idx, z in enumerate(layers):
            if progress.wasCanceled(): break
            progress.setValue(idx)
            QApplication.processEvents()
            
            # Create Layer Folder
            # Folder name: Layer_Z
            layer_dir = os.path.join(out_dir, f"Layer_{z}")
            os.makedirs(layer_dir, exist_ok=True)
            
            # Get Image
            try:
                img = to_gray2d_uint16(self.full_data, z)
            except:
                continue
                
            h, w = img.shape
            
            # Get Calibration Data for this layer
            calib_map = None
            if use_calib:
                calib_map = self.layer_calib_data.get(z, None)
                
            # Rotation pre-calc
            rad = np.radians(cfg.angle)
            sin_a = np.sin(rad)
            cos_a = np.cos(rad)
            
            for r in range(cfg.rows):
                for c in range(cfg.cols):
                    # Check Bonding Map
                    label = "chip"
                    if self.glw.bonding_map:
                         key = self.glw.bonding_map.get_key(r, c)
                         if not key:
                             continue # Skip unbonded
                         label = key
                    
                    # Calculate Rotated Center
                    # Unrotated Center relative to Grid Origin
                    cx_rel = (c + 0.5) * cfg.pitch_x
                    cy_rel = (r + 0.5) * cfg.pitch_y
                    
                    # Rotate (Grid Angle)
                    rot_x = cx_rel * cos_a - cy_rel * sin_a
                    rot_y = cx_rel * sin_a + cy_rel * cos_a
                    
                    # Absolute Center
                    center_x = cfg.start_x + rot_x
                    center_y = cfg.start_y + rot_y
                    
                    # Extract Upright Patch using QPainter (Large Crop + Rotate)
                    # 1. Determine safe bounding box for rotation
                    diag = np.sqrt(cfg.pitch_x**2 + cfg.pitch_y**2)
                    r_bound = int(diag / 2.0) + 5 # Extra padding
                    
                    x_min = int(max(0, center_x - r_bound))
                    y_min = int(max(0, center_y - r_bound))
                    x_max = int(min(w, center_x + r_bound))
                    y_max = int(min(h, center_y + r_bound))
                    
                    if x_max <= x_min or y_max <= y_min:
                        continue
                        
                    # Extract Source Crop
                    roi = img[y_min:y_max, x_min:x_max].astype(np.float32)
                    
                    # Apply Window Level
                    roi = (roi - win_lo) / win_range
                    
                    # Apply Calibration
                    if calib_map is not None:
                        if r < calib_map.shape[0] and c < calib_map.shape[1]:
                            scale = calib_map[r, c, 0]
                            offset = calib_map[r, c, 1]
                            roi = roi * scale + offset
                            
                    # Clip to 0-1
                    roi = np.clip(roi, 0.0, 1.0)
                    
                    # Convert to uint8 (Visual)
                    roi_u8 = (roi * 255.0).astype(np.uint8)
                    
                    # Create Source QImage
                    h_roi, w_roi = roi_u8.shape
                    q_src = QImage(roi_u8.data, w_roi, h_roi, w_roi, QImage.Format_Grayscale8)
                    
                    # Create Target QImage (Upright)
                    dst_w = int(cfg.pitch_x)
                    dst_h = int(cfg.pitch_y)
                    q_dst = QImage(dst_w, dst_h, QImage.Format_Grayscale8)
                    q_dst.fill(0)
                    
                    # Paint Rotated
                    p = QPainter(q_dst)
                    # p.setRenderHint(QPainter.SmoothPixmapTransform) # Bilinear - optional, might blur slightly
                    # Using default (Nearest) usually preserves edge sharpness for scientific images.
                    # But user wants "visual appearance", so Smooth might be better for rotation.
                    # Let's use Smooth.
                    p.setRenderHint(QPainter.SmoothPixmapTransform)
                    
                    # Transform: Target Center -> Rotate -> Match Source Coords
                    p.translate(dst_w / 2.0, dst_h / 2.0)
                    p.rotate(-cfg.angle) # Rotate back to upright
                    p.translate(-center_x, -center_y) # Align with Global System
                    
                    # Draw Source at its global position
                    p.drawImage(x_min, y_min, q_src)
                    p.end()
                    
                    # Filename: X00_Y03_L01_LEG_class.png
                    fname = f"X{c:02d}_Y{r:02d}_L{z:02d}_LEG_{label}.png"
                    fpath = os.path.join(layer_dir, fname)
                    
                    q_dst.save(fpath)
                    
                    total_extracted += 1

        progress.setValue(len(layers))
        print(f"Extraction Complete. Total {total_extracted} patches.")

    def toggle_void_mode_action(self):
        self.chk_void_mode.setChecked(not self.chk_void_mode.isChecked())

    def set_void_mode_edit(self):
        self.chk_void_mode.setChecked(True)
        self.rb_edit.setChecked(True)
        self.on_void_mode_changed()

    def set_void_mode_erase(self):
        self.chk_void_mode.setChecked(True)
        self.rb_erase.setChecked(True)
        self.on_void_mode_changed()

    def on_void_mode_changed(self):
        # Called when Radio Button changes
        if not self.chk_void_mode.isChecked():
            return
            
        if self.rb_draw.isChecked():
            self.glw.void_mode = "DRAW"
            print("Void Mode: DRAW")
        elif self.rb_edit.isChecked():
            self.glw.void_mode = "EDIT"
            print("Void Mode: EDIT")
        elif self.rb_erase.isChecked():
            self.glw.void_mode = "ERASE"
            print("Void Mode: ERASE")
            
        self.glw.setFocus()
        self.glw.update()

    def on_void_mode_toggled(self, checked):
        if checked:
            # Re-apply current radio selection
            self.on_void_mode_changed()
        else:
            self.glw.void_mode = False
            print("Void Mode: OFF")
        self.glw.setFocus()
        
    def load_voids(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Voids JSON", "", "JSON (*.json)")
        if path:
            self.void_manager.load_from_file(path, self.glw.grid_cfg)
            self.populate_void_types() # Refresh types
            self.update_void_ui()
            self.glw.update()
            
    def populate_void_types(self):
        self.cb_void_type.blockSignals(True)
        self.cb_void_type.clear()
        
        # Sort by ID
        for tid, data in sorted(self.void_manager.types.items()):
            self.cb_void_type.addItem(data["name"], tid)
            
        self.cb_void_type.blockSignals(False)
        
        # Set first as default if available
        if self.cb_void_type.count() > 0:
            self.cb_void_type.setCurrentIndex(0)
            self.on_void_type_changed(0)
            
    def on_void_type_changed(self, idx):
        if idx < 0: return
        tid = self.cb_void_type.currentData()
        self.glw.active_type_id = tid
        print(f"Active Void Type: {self.void_manager.types[tid]['name']}")
        
    def open_type_manager(self):
        dlg = VoidTypeDialog(self.void_manager, self)
        dlg.exec()
        self.populate_void_types()
        self.glw.update()

    def save_voids(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Voids JSON", "voids.json", "JSON (*.json)")
        if path:
            self.void_manager.save_to_file(path, self.glw.grid_cfg)
            
    def clear_voids(self):
        self.void_manager.clear_layer(self.current_z) # Use current Z
        self.update_void_ui()
        self.glw.update()
        
    def update_void_ui(self):
        # Update label count
        if self.current_z in self.void_manager.voids:
            count = len(self.void_manager.voids[self.current_z])
        else:
            count = 0
        self.lbl_void_count.setText(f"Voids: {count}")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "TIFF (*.tif *.tiff)")
        if not path:
            return
            
        # Get Load Limits
        req_start = self.sb_start_layer.value()
        req_end = self.sb_end_layer.value()
        
        print(f"Loading {path}...")
        print(f"Requested Range: {req_start} to {req_end}")
        
        try:
            # 1. Cleanup previous file handle
            if self.current_tif_file:
                self.current_tif_file.close()
                self.current_tif_file = None
                
            # 2. Open new file (Keep handle for Lazy Load)
            self.current_tif_file = tifffile.TiffFile(path)
            
            # 3. Create Lazy Stack
            # This handles both Multi-Series (fast open) and regular stacks.
            lazy_stack = LazyTiffStack(self.current_tif_file)
            
            print(f"Opened with Lazy Loading (Instant). Shape: {lazy_stack.shape}")
            
            self.full_data = lazy_stack
            self.layer_offset = 0 # Offset handling simplified for now
            
            # Slice if requested?
            # LazyTiffStack doesn't support sophisticated slicing view yet, but we can just use indices.
            # If user requested range 0..10, we just map logic to that.
            # For simplicity, we expose the whole file, but 'open_file' logic usually sets limits.
            # Let's trust LazyStack to expose everything, and user navigates.
            
            # Check limits
            req_start = self.sb_start_layer.value()
            req_end = self.sb_end_layer.value()
            
            # If we want to restrict range, we could wrap LazyStack or just ignore?
            # The previous logic "Loaded Series [s..e]".
            # With Lazy Load, we can just say "Available: 0..N".
            # The 'spin_layer' range will define what is accessible.
            
            # Determine Num Layers
            num_layers = len(lazy_stack)
            
            # Setup layer spinner
            self.spin_layer.blockSignals(True)
            if num_layers > 1:
                self.spin_layer.setRange(0, num_layers - 1)
                self.spin_layer.setValue(0)
                self.spin_layer.setEnabled(True)
            elif num_layers == 1:
                # Could be (1, H, W, C) or (1, H, W)
                # LazyStack ndim is base + 1
                if lazy_stack.ndim == 4: # (1, H, W, C)
                     # Treat C as channels of single layer?
                     # But spin_layer controls 'Z'.
                     self.spin_layer.setRange(0, 0)
                     self.spin_layer.setEnabled(False)
                else:
                     self.spin_layer.setRange(0, 0)
                     self.spin_layer.setEnabled(False)
            
            self.spin_layer.blockSignals(False)
            
            # Clear cache
            for t in self.layer_cache.values():
                t.cleanup()
            self.layer_cache.clear()
            self.layer_calib_data.clear()
            
            # Auto-Leveling (Estimate from first visual layer)
            try:
                # Accessing [0] loads the first layer from disk
                sample_layer = to_gray2d_uint16(self.full_data, 0)
                
                # Subsample for speed (max 10k items)
                h, w = sample_layer.shape
                step = int(max(1, np.sqrt((h*w)/10000)))
                # Ensure 2D slicing works on numpy array
                stats_slice = sample_layer[::step, ::step]
                
                vmin, vmax = np.min(stats_slice), np.max(stats_slice)
                
                # Set sliders
                self.slider_lo.blockSignals(True)
                self.slider_hi.blockSignals(True)
                self.slider_lo.setValue(int(vmin))
                self.slider_hi.setValue(int(vmax))
                self.slider_lo.blockSignals(False)
                self.slider_hi.blockSignals(False)
                
                self.on_window_changed()
                print(f"Auto-Leveled to [{vmin}, {vmax}]")
                
            except Exception as e:
                print(f"Auto-Level failed: {e}")
                # traceback.print_exc()

            # Preload GPU Cache? 
            # With lazy loading, preloading defeats the purpose of "Low RAM", 
            # BUT it puts it in VRAM. VRAM is faster.
            # User output showed "Preloading..."
            self.preload_gpu_cache()
            
            self.load_layer(0)
            
        except Exception as e:
            print(f"Error loading file: {e}")
            import traceback
            traceback.print_exc()
            


    def preload_gpu_cache(self):
        # Preloads all loaded layers into GPU memory to enable instant switching.
        if self.full_data is None:
            return
            
        # Determine number of layers to process
        # If it's a single 2D image, nothing extra to do really, but let's handle consistent logic
        task_count = 0
        if self.full_data.ndim == 2 or (self.full_data.ndim == 3 and self.full_data.shape[-1] in (3,4)):
            task_count = 1
        elif self.full_data.ndim == 3: # (Z, H, W)
            task_count = self.full_data.shape[0]
        elif self.full_data.ndim == 4: # (Z, H, W, C)
            task_count = self.full_data.shape[0]
            
        if task_count <= 1:
            return

        print(f"Preloading {task_count} layers to VRAM...")
        
        progress = QProgressDialog("Preloading layers to GPU...", "Cancel", 0, task_count, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.resize(400, 100)
        progress.show()
        
        # We need a GL context active to upload textures.
        # GLImageWidget.makeCurrent() is called inside set_tiled_image -> upload_all
        # But here we want to upload without setting it as CURRENT display image immediately.
        # We can manually create TiledImage and call upload().
        # Note: TiledImage.upload() does NOT require the widget to be the *current* widget for painting,
        # but it DOES require a valid OpenGL context to be current on the thread.
        # Since we are in the main thread and the widget is initialized, we can use it.
        
        self.glw.makeCurrent()
        
        try:
            for i in range(task_count):
                if progress.wasCanceled():
                    print("Preloading canceled.")
                    break
                    
                progress.setValue(i)
                # progress.setLabelText(f"Uploading Layer {i+1}/{task_count}...") # Optional update
                QApplication.processEvents() 
                
                # Check if already cached (e.g. if we reload or something)
                if i in self.layer_cache:
                    continue
                    
                # Create TiledImage
                img_2d = to_gray2d_uint16(self.full_data, i)
                tiled = TiledImage(img_2d)
                
                # Upload
                # TiledImage.upload() makes GL calls. Context must be current.
                # We made it current above.
                tiled.upload_all()
                
                self.layer_cache[i] = tiled
                
            progress.setValue(task_count)
            
        finally:
            self.glw.doneCurrent()

    def load_layer(self, z_index):
        if self.full_data is None:
            return
            
        tiled = None
        # Check Cache
        if z_index in self.layer_cache:
            # print(f"Cache Hit for Layer {z_index}")
            tiled = self.layer_cache[z_index]
        else:
            print(f"Processing Layer {z_index}...")
            img_2d = to_gray2d_uint16(self.full_data, z_index)
            print("Creating tiles...")
            tiled = TiledImage(img_2d)
            self.layer_cache[z_index] = tiled
            print("Uploading to GPU...")

        # Set Image
        self.glw.set_tiled_image(tiled, cleanup_old=False)
        
        # Update Calibration Texture for this layer
        calib = self.layer_calib_data.get(z_index, None)
        self.glw.update_calib_texture(calib)
        
        self.update()

    def on_layer_changed(self, val):
        self.current_z = val
        
        # Update label to show absolute index
        # Assuming we have access to the label widget? 
        # We constructed it in __init__ locally. 
        # Can't easily access. Let's just print or update window title?
        # Better: Update the spinbox suffix or prefix or tooltip
        
        abs_layer = self.layer_offset + val
        self.spin_layer.setSuffix(f" (Abs: {abs_layer})")
        
        self.load_layer(val)

    def on_window_changed(self):
        self.glw.win_lo = self.slider_lo.value()
        self.glw.win_hi = self.slider_hi.value()
        self.glw.update()
        
    def toggle_calib(self, checked):
        self.chk_use_calib.setText(f"Enable Auto-Calib: {'ON' if checked else 'OFF'}")
        self.glw.grid_cfg.use_calib = checked
        if checked:
            self.calculate_calibration()
        else:
            self.glw.update()

    def calculate_calibration(self):
        if self.full_data is None: return
        
        # Determine layers to calculate
        layers_to_calc = [self.current_z]
        if self.chk_calib_all.isChecked():
            # Calc for all available layers
             if self.full_data.ndim > 2 and (not (self.full_data.ndim==3 and self.full_data.shape[-1] in [3,4])):
                 layers_to_calc = list(range(self.full_data.shape[0]))
        
        # Progress Dialog if multiple
        progress = None
        if len(layers_to_calc) > 1:
            progress = QProgressDialog("Calibrating all layers...", "Cancel", 0, len(layers_to_calc), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
        cfg = self.glw.grid_cfg
        target_mean = self.sb_target_mean.value()
        target_std = self.sb_target_std.value()
        win_lo = self.glw.win_lo
        
        # Loop layers
        for i, z_idx in enumerate(layers_to_calc):
            if progress:
                if progress.wasCanceled(): break
                progress.setValue(i)
                QApplication.processEvents()
                
            img = to_gray2d_uint16(self.full_data, z_idx)
            if img is None: continue
            
            h, w = img.shape
            
            # Grid Loop
            start_x = cfg.start_x
            start_y = cfg.start_y
            pitch_x = cfg.pitch_x
            pitch_y = cfg.pitch_y
            rows = cfg.rows
            cols = cfg.cols
            
            # Win Lo/Hi for normalization
            # Note: win_lo/hi might be shared or per layer?
            # Typically shared global slider.
            win_lo = self.glw.win_lo
            win_hi = self.glw.win_hi
            win_range = win_hi - win_lo
            if win_range < 1: win_range = 1
            
            calib_data = np.zeros((rows, cols, 2), dtype=np.float32)
            
            # Rotation pre-calc
            rad = np.radians(cfg.angle)
            sin_a = np.sin(rad)
            cos_a = np.cos(rad)
            
            for r in range(rows):
                # Optimization? No, need per-cell center
                
                for c in range(cols):
                    # Calculate Rotated Center
                    cx_rel = (c + 0.5) * pitch_x
                    cy_rel = (r + 0.5) * pitch_y
                    
                    rot_x = cx_rel * cos_a - cy_rel * sin_a
                    rot_y = cx_rel * sin_a + cy_rel * cos_a
                    
                    center_x = start_x + rot_x
                    center_y = start_y + rot_y
                    
                    x0 = int(center_x - pitch_x/2)
                    y0 = int(center_y - pitch_y/2)
                    x1 = int(center_x + pitch_x/2)
                    y1 = int(center_y + pitch_y/2)
                    
                    # Clip
                    y0_c = max(0, min(h, y0))
                    y1_c = max(0, min(h, y1))
                    x0_c = max(0, min(w, x0))
                    x1_c = max(0, min(w, x1))
                    
                    if x1_c <= x0_c or y1_c <= y0_c:
                        calib_data[r, c, 0] = 1.0 # Scale
                        calib_data[r, c, 1] = 0.0 # Offset
                        continue
                        
                    # Check Bonding Map (Selective Calibration)
                    if self.glw.bonding_map:
                        key = self.glw.bonding_map.get_key(r, c)
                        if not key:
                            # Not bonded -> Skip calibration
                            calib_data[r, c, 0] = 1.0
                            calib_data[r, c, 1] = 0.0
                            continue
                            
                    roi = img[y0_c:y1_c, x0_c:x1_c]
                    
                    # Performance optimization
                    if roi.size > 10000:
                        step = int(np.sqrt(roi.size / 10000))
                        roi_stats = roi[::step, ::step]
                    else:
                        roi_stats = roi
                    
                    if self.chk_calib_robust.isChecked():
                        mean = np.median(roi_stats)
                        q75, q25 = np.percentile(roi_stats, [75, 25])
                        iqr = q75 - q25
                        std = iqr / 1.35 
                    else:
                        mean = np.mean(roi_stats)
                        std = np.std(roi_stats)
                    
                    if std < 1.0: std = 1.0 
                    
                    scale = target_std / std
                    offset = (target_mean - win_lo - scale * (mean - win_lo)) / win_range
                    
                    calib_data[r, c, 0] = scale
                    calib_data[r, c, 1] = offset
            
            # Store in cache
            self.layer_calib_data[z_idx] = calib_data
            
        if progress:
            progress.setValue(len(layers_to_calc))
            
        # Update current layer View
        if self.current_z in self.layer_calib_data:
            self.glw.update_calib_texture(self.layer_calib_data[self.current_z])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
