import time
import json
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QColorDialog, QInputDialog,
    QDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

# ==================================================================================
# Void Data Manager
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

    def add_void(self, layer, gx, gy, rx, ry, type_id=0):
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
        self.voids[layer].append(new_void)
        return new_void

    def delete_void_at(self, layer, gx, gy, hit_radius=10.0):
        """
        Deletes the first void overlapping the point (Simple Ellipse Test).
        Returns True if deleted.
        """
        if layer not in self.voids: return False
        
        target, _ = self.hit_test(layer, gx, gy, hit_radius)
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
            rx = v["radiusX"]
            ry = v["radiusY"]
            
            # Normalized distance
            # dist = sqrt((dx/rx)^2 + (dy/ry)^2)
            # Boundary is dist=1.0
            
            dist_sq = (dx*dx)/(rx*rx) + (dy*dy)/(ry*ry)
            dist = np.sqrt(dist_sq)
            
            # Margin check is tricky for ellipse. 
            # Approximate: if dist is close to 1.0 (edge)
            # Use 'effective radius' for margin? 
            # Or just check if 0.8 < dist < 1.2?
            # Let's say edge is if dist is between 0.8 and 1.2 (visual approx)
            
            if 0.8 <= dist <= 1.2:
                return v, 'edge'
            
            if dist < 0.8:
                return v, 'center'
                
        return None, None

    def hit_test_all(self, gx, gy, margin=1.0, priority_layer=None):
        """
        Searches ALL layers for a hit.
        Returns (void_dict, action, layer_index) or (None, None, None).
        Checks priority_layer first.
        """
        # 1. Check Priority Layer
        if priority_layer is not None:
             v, action = self.hit_test(priority_layer, gx, gy, margin)
             if v:
                 return v, action, priority_layer
                 
        # 2. Check Others
        for layer_idx in self.voids.keys():
            if layer_idx == priority_layer: continue
            
            v, action = self.hit_test(layer_idx, gx, gy, margin)
            if v:
                return v, action, layer_idx
                
        return None, None, None

    def clear_layer(self, layer):
        if layer in self.voids:
            self.voids[layer] = []

    def delete_voids_in_rect(self, layer, x0, y0, x1, y1):
        """
        Deletes all voids in the given layer that are within the rectangle [x0, y0, x1, y1].
        Using void centers.
        """
        if layer not in self.voids: return 0
        
        to_delete = []
        for v in self.voids[layer]:
            vx = v["globalCX"]
            vy = v["globalCY"]
            if x0 <= vx <= x1 and y0 <= vy <= y1:
                to_delete.append(v)
                
        for v in to_delete:
            self.voids[layer].remove(v)
            
        return len(to_delete)

    def save_to_file(self, path, grid_cfg, bonding_map=None):
        """
        Converts Global -> Chip Relative using current Grid Config.
        Saves as Dict with 'types' and 'voids'.
        "type" field uses Type Name (User Request).
        patchLabel uses x_y_layer_bondinglabel format.
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
                
                rx = v["radiusX"]
                ry = v["radiusY"]
                key_str = f"{col},{row},{layer},0"
                
                # Resolve Type Name
                tid = v.get("type_id", 0)
                type_name = "bbox"
                if tid in self.types:
                    type_name = self.types[tid]["name"]

                # Resolve Bonding Label for patchLabel
                bonding_label = "void"
                if bonding_map:
                    key = bonding_map.get_key(row, col)
                    if key:
                        bonding_label = key
                
                item = {
                    "key": key_str,
                    "x": col, "y": row, "layer": layer, "voidIndex": 0,
                    "type": type_name, # Use Name as requested
                    "type_id": tid, # Keep ID for backup
                    "centerX": rel_cx,
                    "centerY": rel_cy,
                    "radiusX": rx,
                    "radiusY": ry,
                    "createdAt": v.get("createdAt", 0),
                    "patchLabel": f"X{col:02d}_Y{row:02d}_L{layer:02d}_{bonding_label}",
                    "globalCX": v["globalCX"],
                    "globalCY": v["globalCY"]
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
                rx = item.get("radiusX", 10.0)
                ry = item.get("radiusY", 10.0)
                
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
                
                if "globalCX" in item and "globalCY" in item:
                    # Use absolute if available (Grid-Independent)
                    gx = item["globalCX"]
                    gy = item["globalCY"]
                else:
                    # Calc Global from Relative (Grid-Dependent)
                    chip_x0 = grid_cfg.start_x + col * grid_cfg.pitch_x
                    chip_y0 = grid_cfg.start_y + row * grid_cfg.pitch_y
                    
                    gx = chip_x0 + rel_cx
                    gy = chip_y0 + rel_cy
                
                if layer not in self.voids: self.voids[layer] = []
                
                v_obj = {
                    "globalCX": gx,
                    "globalCY": gy,
                    "radiusX": rx,
                    "radiusY": ry,
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
        
        name, ok = QInputDialog.getText(self, "Rename Type", "New Name:", text=old_name)
        if ok and name:
            self.vm.types[tid]["name"] = name
            self.refresh()
