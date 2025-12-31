import os
import csv
import json
import numpy as np
import math
from datetime import datetime

from PySide6.QtCore import QObject, Signal, Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QTransform, QFont

class ExportManager(QObject):
    progress_update = Signal(int, str) # Percentage, Message
    finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, data_stack, void_manager, grid_cfg, bonding_map):
        super().__init__()
        self.data_stack = data_stack
        self.void_manager = void_manager
        self.grid_cfg = grid_cfg
        self.bonding_map = bonding_map
        self.stop_requested = False

    def run_export(self, output_dir, options: dict):
        """
        Main export loop.
        options = {'raw': bool, 'overlay': bool, 'mask': bool, 'merged': bool, 
                   'json': bool, 'csv': bool}
        """
        self.stop_requested = False
        
        try:
            # 1. Create subdirectories
            subdirs = {}
            if options.get('raw'): subdirs['raw'] = os.path.join(output_dir, 'raw')
            if options.get('overlay'): subdirs['overlay'] = os.path.join(output_dir, 'overlay')
            if options.get('mask'): subdirs['mask'] = os.path.join(output_dir, 'mask')
            if options.get('merged'): subdirs['merged'] = os.path.join(output_dir, 'merged')
            
            for p in subdirs.values():
                os.makedirs(p, exist_ok=True)

            # Metadata Exports
            if options.get('json'):
                self._export_json(output_dir)
            if options.get('csv'):
                self._export_csv(output_dir)

            # 2. Determine Layers
            num_layers = 1
            if hasattr(self.data_stack, 'shape'):
                if len(self.data_stack.shape) == 3:
                     num_layers = self.data_stack.shape[0]
                elif len(self.data_stack.shape) == 4:
                     num_layers = self.data_stack.shape[0] # Z, H, W, C
            elif hasattr(self.data_stack, '__len__'):
                 num_layers = len(self.data_stack)

            # 3. Pre-calc Grid Geometry (Rotation)
            cfg = self.grid_cfg
            rad = np.radians(cfg.angle)
            sin_a = np.sin(rad)
            cos_a = np.cos(rad)
            
            w_pitch = int(cfg.pitch_x)
            h_pitch = int(cfg.pitch_y)
            
            total_chips = cfg.rows * cfg.cols * num_layers
            if total_chips == 0: total_chips = 1
            processed = 0
            
            # Accumulators
            merged_accumulators = {} 
            mask_accumulators = {} # Key: (c, r) or label? User said "chip 별로". Using (c,r) is safer for unique identification.

            # Iterate Layers
            for layer_idx in range(num_layers):
                if self.stop_requested: break
                
                # Create Layer Subdirs (Only for Raw/Overlay)
                layer_name = f"Layer_{layer_idx:02d}"
                layer_subdirs = {}
                for k in ['raw', 'overlay']: # Mask/Merged are top-level
                    if k in subdirs:
                        p = os.path.join(subdirs[k], layer_name)
                        os.makedirs(p, exist_ok=True)
                        layer_subdirs[k] = p
                
                # Get Layer Image (Gray uint16)
                try:
                    full_layer = self._get_layer_image(layer_idx)
                except Exception as e:
                    print(f"Failed to load layer {layer_idx}: {e}")
                    continue
                    
                h_img, w_img = full_layer.shape
                
                for r in range(cfg.rows):
                    for c in range(cfg.cols):
                        if self.stop_requested: break
                        
                        # Bonding Map Check
                        label = "chip"
                        if self.bonding_map:
                            key = self.bonding_map.get_key(r, c)
                            if not key: continue # Skip unbonded
                            label = key
                            
                        # Calculate Rotated Center
                        cx_rel = (c + 0.5) * cfg.pitch_x
                        cy_rel = (r + 0.5) * cfg.pitch_y
                        
                        rot_x = cx_rel * cos_a - cy_rel * sin_a
                        rot_y = cx_rel * sin_a + cy_rel * cos_a
                        
                        center_x = cfg.start_x + rot_x
                        center_y = cfg.start_y + rot_y
                        
                        # Extract Patch
                        patch_qimg = self._extract_patch_qimage(full_layer, center_x, center_y, w_pitch, h_pitch, cfg.angle)
                        if patch_qimg is None: continue # Out of bounds
                        
                        # Sanitize Filename & Create Title Text
                        safe_label = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in label])
                        
                        # Titles
                        title_raw_ov = f"X{c:02d}_Y{r:02d}_L{layer_idx:02d}_LEG:{safe_label}"
                        title_mask = f"X{c:02d}_Y{r:02d}_LEG:{safe_label}"
                        
                        basename = f"X{c:02d}_Y{r:02d}_L{layer_idx:02d}_LEG_{safe_label}"
                        
                        # A. Raw Export
                        if options.get('raw'):
                            path = os.path.join(layer_subdirs['raw'], f"{basename}.png")
                            final_img = self._add_title_bar(patch_qimg, title_raw_ov)
                            if not final_img.save(path):
                                print(f"Error saving {path}")
                            
                        # B. Overlay (Burn-in)
                        if options.get('overlay'):
                            ov_img = patch_qimg.convertToFormat(QImage.Format_ARGB32)
                            self._draw_voids(ov_img, layer_idx, center_x, center_y, w_pitch, h_pitch, cfg.angle, mode='overlay')
                            path = os.path.join(layer_subdirs['overlay'], f"{basename}_overlay.png")
                            final_img = self._add_title_bar(ov_img, title_raw_ov)
                            if not final_img.save(path):
                                print(f"Error saving {path}")
                            
                        # C. Mask (Accumulate per Chip)
                        if options.get('mask'):
                            # Key: (c, r) -> {image, label}
                            key = (c, r)
                            if key not in mask_accumulators:
                                img = QImage(w_pitch, h_pitch, QImage.Format_ARGB32)
                                img.fill(Qt.transparent)
                                mask_accumulators[key] = {'img': img, 'label': safe_label, 'title': title_mask}
                            
                            # Draw voids for this layer onto the accumulator
                            self._draw_voids(mask_accumulators[key]['img'], layer_idx, center_x, center_y, w_pitch, h_pitch, cfg.angle, mode='mask')
                            
                        # D. Merged Accumulate
                        if options.get('merged'):
                            safe_label_merged = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in label])
                            # Title for Merged: LEG:{label}_MERGED
                            title_merged = f"LEG:{safe_label_merged}_MERGED"
                            
                            if safe_label_merged not in merged_accumulators:
                                merged_accumulators[safe_label_merged] = {
                                    'img': QImage(w_pitch, h_pitch, QImage.Format_ARGB32),
                                    'title': title_merged
                                }
                                merged_accumulators[safe_label_merged]['img'].fill(Qt.transparent)
                                
                            self._draw_voids(merged_accumulators[safe_label_merged]['img'], layer_idx, center_x, center_y, w_pitch, h_pitch, cfg.angle, mode='mask')

                        processed += 1
                        if processed % 10 == 0:
                            self.progress_update.emit(int(processed / total_chips * 100), f"Processing {r}, {c}...")

            # Save Masks (Flattened)
            if options.get('mask'):
                for (c, r), data in mask_accumulators.items():
                    # Filename: X{c}_Y{r}_LEG_{label}.png (No Layer Info)
                    fname = f"X{c:02d}_Y{r:02d}_LEG_{data['label']}.png"
                    path = os.path.join(subdirs['mask'], fname)
                    
                    final_img = self._add_title_bar(data['img'], data['title'])
                    if not final_img.save(path):
                        print(f"Error saving mask {path}")

            # Save Merged
            if options.get('merged'):
                for label, data in merged_accumulators.items():
                    img = data['img']
                    title = data['title']
                    
                    path = os.path.join(subdirs['merged'], f"Merged_Label_{label}.png")
                    final_img = self._add_title_bar(img, title)
                    final_img.save(path)

            self.finished.emit()
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            import traceback
            traceback.print_exc()

    def _add_title_bar(self, img: QImage, text: str) -> QImage:
        """
        Adds a 40px black title bar at the top with white text.
        """
        w, h = img.width(), img.height()
        title_h = 40
        
        # New Image with extra height
        new_h = h + title_h
        # Use ARGB32 for safety (handles transparency in original img)
        out_img = QImage(w, new_h, QImage.Format_ARGB32)
        out_img.fill(Qt.black) # Background is black
        
        painter = QPainter(out_img)
        
        # Draw Original Image below title
        # Note: If original has transparency (Mask), we want black BG or transparent?
        # User said "black title bar". Usually implies the content area is separate.
        # But if Mask is transparent, drawing it on Black will make it Black.
        # Let's keep the content area transparent if possible?
        # But `out_img.fill(Qt.black)` fills everything.
        
        # Actually, for Mask/Merged (transparent), the user might still want the void outlines to be visible.
        # If I fill with Black, the transparent background becomes black.
        # If I fill title with black and rest transparent? 
        # But the User request showed a GREY background in the screenshot provided? 
        # No, the screenshot shows: "Black bar with white text" on top. Content below.
        # Let's assume the content background should be preserved.
        
        # 1. Clear text area to black, rest to transparent?
        # Better: Create empty image. Fill top rect with black.
        out_img.fill(Qt.transparent)
        
        # Draw Black Bar
        painter.fillRect(0, 0, w, title_h, Qt.black)
        
        # Draw Text
        pen = QPen(Qt.white)
        painter.setPen(pen)
        font = painter.font()
        font.setPixelSize(24) # Adjust size
        font.setBold(True)
        painter.setFont(font)
        
        # Align Left-Center in the bar
        rect = QRectF(10, 0, w-20, title_h)
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, text)
        
        # Draw the original image at (0, 40)
        painter.drawImage(0, title_h, img)
        
        painter.end()
        return out_img

    def _get_layer_image(self, z_index):
        # Local helper or import from core_data
        # Simulating to_gray2d_uint16 logic
        arr = self.data_stack
        
        # Simple slicing for now, assuming standard shape
        # Better: import to_gray2d_uint16 from .core_data
        from .core_data import to_gray2d_uint16
        return to_gray2d_uint16(arr, z_index)

    def _extract_patch_qimage(self, img, cx, cy, w, h, angle):
        # Extract Upright Patch using QPainter (Large Crop + Rotate)
        # 1. Determine safe bounding box for rotation
        diag = np.sqrt(w**2 + h**2)
        r_bound = int(diag / 2.0) + 5
        
        h_img, w_img = img.shape
        
        x_min = int(max(0, cx - r_bound))
        y_min = int(max(0, cy - r_bound))
        x_max = int(min(w_img, cx + r_bound))
        y_max = int(min(h_img, cy + r_bound))
        
        if x_max <= x_min or y_max <= y_min:
             return None
             
        # Extract Source Crop
        roi = img[y_min:y_max, x_min:x_max]
        
        # Normalize (Simple Min/Max for Export? Or use Window settings if passed?)
        # User might want "What I see is what I get".
        # For validation, let's use simple min/max normalization to ensure visibility.
        # Or better: If using MainWindow logic, we should use window level!
        # But Exporter doesn't have reference to Window Level...
        # Let's standardize to full range for now to avoid pitch black images.
        
        c_min, c_max = roi.min(), roi.max()
        if c_max > c_min:
             roi_norm = ((roi - c_min) / (c_max - c_min) * 255).astype(np.uint8)
        else:
             roi_norm = np.zeros_like(roi, dtype=np.uint8)
        
        # Make Contiguous! Critical for QImage
        roi_norm = np.ascontiguousarray(roi_norm)
             
        # Create Source QImage
        # MUST keep reference to data if not copying? 
        # But QImage(..., Format) usually wraps.
        # We should use .copy() on the QImage to deep copy the pixel data immediately.
        h_roi, w_roi = roi_norm.shape
        q_src = QImage(roi_norm.data, w_roi, h_roi, w_roi, QImage.Format_Grayscale8).copy()
        
        # Create Target QImage (Upright)
        q_dst = QImage(w, h, QImage.Format_Grayscale8)
        q_dst.fill(0)
        
        painter = QPainter(q_dst)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Transform: Center -> Rotate -> Center
        # Source coordinate of patch center relative to crop top-left
        src_cx = cx - x_min
        src_cy = cy - y_min
        
        # Target center
        dst_cx = w / 2
        dst_cy = h / 2
        
        painter.translate(dst_cx, dst_cy)
        painter.rotate(-angle) # Un-rotate the grid angle to make chip upright
        painter.translate(-src_cx, -src_cy)
        
        painter.drawImage(0, 0, q_src)
        painter.end()
        
        return q_dst

    def _draw_voids(self, target_img: QImage, current_layer, cx, cy, w, h, angle, mode='overlay'):
        """
        Draws voids relevant to this chip onto the target QImage.
        mode: 'overlay' (draw normally), 'mask' (hollow outlines, specific colors)
        """
        painter = QPainter(target_img)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Setup Transform to map Global -> Chip Local
        pcx = target_img.width() / 2
        pcy = target_img.height() / 2
        
        # Coordinate Transform Setup
        # We transform Raw (Global) -> Patch (Upright/Deskewed)
        # Patch is rotated by '-angle' relative to Raw.
        rad = math.radians(-angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # painter.translate(pcx, pcy)
        # painter.rotate(angle) 
        # painter.translate(-cx, -cy) 
        # REPLACED WITH MANUAL TRANSFORM PER VOID
        # to ensure Upright Shapes stay Upright. 
        
        # Decide which layers to draw
        # If Overlay: Draw ALL layers (Current=Solid, Others=Dot)
        # If Mask/Merged: Draw ONLY current layer (Solid)
        
        layers_to_draw = []
        if mode == 'overlay':
            layers_to_draw = self.void_manager.voids.keys()
        else:
            layers_to_draw = [current_layer]
            
        for layer_key in layers_to_draw:
            voids = self.void_manager.voids.get(layer_key, [])
            
            is_active_layer = (layer_key == current_layer)
            
            for v in voids:
                # Check bounds roughly
                vx, vy = v["globalCX"], v["globalCY"]
                if abs(vx - cx) > w or abs(vy - cy) > h: continue 
                
                # Attributes
                rx = v["radiusX"]
                ry = v["radiusY"]
                tid = v.get("type_id", 0)
                
                # Transform Coordinates Manually
                # Raw Delta from Patch Center (Source Image Center)
                dx = vx - cx
                dy = vy - cy
                
                # Rotate Delta (Raw -> Upright)
                rot_x = dx * cos_a - dy * sin_a
                rot_y = dx * sin_a + dy * cos_a
                
                # Patch Coordinates (relative to Patch Center pcx, pcy)
                px = pcx + rot_x
                py = pcy + rot_y
                
                type_data = self.void_manager.types.get(tid, {"color": (255, 0, 0)})
                col_rgb = type_data["color"]
                color = QColor(col_rgb[0], col_rgb[1], col_rgb[2])
                
                pen = QPen(color)
                pen.setWidth(2)
                
                if mode == 'overlay':
                    if is_active_layer:
                        pen.setStyle(Qt.SolidLine)
                    else:
                        pen.setStyle(Qt.DotLine) 
                        
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    
                elif mode == 'mask':
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    
                # Draw
                shape = type_data.get("shape", "ellipse")
                if shape == "rectangle":
                     # Schema: globalCX=Left, globalCY=Top, radiusX=Width, radiusY=Height
                     # px, py is transformed Left/Top
                     painter.drawRect(QRectF(px, py, rx, ry))
                else:
                     # Ellipse: Center, Radius
                     # Logic check: 'vx, vy' in Ellipse means Center?
                     # Let's check 'add_void' logic.
                     # Yes, for Ellipse, globalCX/CY is Center.
                     # So px, py is Center.
                     painter.drawEllipse(QPointF(px, py), rx, ry)
            
        painter.end()

    def _export_json(self, output_dir):
        path = os.path.join(output_dir, "voids.json")
        self.void_manager.save_to_file(path, self.grid_cfg, self.bonding_map)

    def _export_csv(self, output_dir):
        path = os.path.join(output_dir, "voids.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["Layer", "CellRow", "CellCol", "Label", "VoidType", "GlobalX", "GlobalY", "RelativeX", "RelativeY", "RadiusX", "RadiusY"])
            
            # Iterate
            for layer, v_list in self.void_manager.voids.items():
                for v in v_list:
                    vx, vy = v["globalCX"], v["globalCY"]
                    
                    # Determine which cell it belongs to
                    gx = (vx - self.grid_cfg.start_x) / self.grid_cfg.pitch_x
                    gy = (vy - self.grid_cfg.start_y) / self.grid_cfg.pitch_y
                    c = int(np.floor(gx))
                    r = int(np.floor(gy))
                    
                    # Calculate Chip Relative (ignoring rotation for CSV as per V1 behavior)
                    # Or should we respect rotation?
                    # "chip 내 보이드의 x,y좌표" -> usually implies unrotated relative
                    chip_x0 = self.grid_cfg.start_x + c * self.grid_cfg.pitch_x
                    chip_y0 = self.grid_cfg.start_y + r * self.grid_cfg.pitch_y
                    rel_x = vx - chip_x0
                    rel_y = vy - chip_y0
                    
                    label = ""
                    if self.bonding_map:
                        label = self.bonding_map.get_key(r, c)
                        
                    tid = v.get("type_id", 0)
                    tname = self.void_manager.types.get(tid, {}).get("name", "Unknown")
                    
                    writer.writerow([layer, r, c, label, tname, vx, vy, rel_x, rel_y, v["radiusX"], v["radiusY"]])
