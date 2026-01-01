
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

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.stop_requested = False
        
        # Shortcuts for cleaner access (optional, but good for read)
        self.data_stack = self.main_window.full_data
        self.void_manager = self.main_window.void_manager
        # self.grid_cfg = self.main_window.glw.grid_cfg 
        # Access dynamically to ensure freshness during run_export

    def run_export(self, output_dir, options: dict, win_lo=0, win_hi=65535):
        """
        Main export loop.
        options = {'raw': bool, 'overlay': bool, 'mask': bool, 'merged': bool, 
                   'json': bool, 'csv': bool}
        win_lo, win_hi: Normalization range (from UI)
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

            # 3. Access Configs & ROIs
            cfg = self.main_window.glw.grid_cfg
            rois = self.main_window.rois
            
            if not rois:
                # Should not happen if triggered from main window
                # But just in case
                self.main_window.update_rois()
                rois = self.main_window.rois
            
            w_pitch = int(cfg.pitch_x)
            h_pitch = int(cfg.pitch_y)
            
            total_chips = len(rois) * num_layers
            if total_chips == 0: total_chips = 1
            processed = 0
            
            # Accumulators
            merged_accumulators = {} 
            mask_accumulators = {} 
            
            # Patch Metadata Collection (List of Dicts)
            patch_metadata_list = []

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
                
                # Get Calibration Map for this layer
                calib_map = None
                if self.main_window.layer_calib_data:
                    calib_map = self.main_window.layer_calib_data.get(layer_idx, None)
                
                # Iterate ROIs
                for item in rois:
                    if self.stop_requested: break
                    
                    r = item['y']
                    c = item['x']
                    label = item.get('label', f"{r}_{c}")
                    bbox_raw = item.get('bbox', [])
                    
                    if not bbox_raw: continue

                    # Calibration Params
                    scale, offset = 1.0, 0.0
                    if calib_map is not None:
                         if r < calib_map.shape[0] and c < calib_map.shape[1]:
                             scale = calib_map[r, c, 0]
                             offset = calib_map[r, c, 1]
                    
                    # Extract Patch (Unified)
                    # use layer_idx, bbox_raw
                    patch_qimg = self._extract_patch_unified(layer_idx, bbox_raw, win_lo, win_hi, scale, offset)
                    
                    if patch_qimg is None: continue 
                    
                    # For drawing voids, we need Center in Raw Coords.
                    gx = cfg.start_x + (c + 0.5) * cfg.pitch_x
                    gy = cfg.start_y + (r + 0.5) * cfg.pitch_y
                    
                    # Using GLWidget's descale logic
                    center_x, center_y = self.main_window.glw.deskew_to_raw(gx, gy)
                    
                    # Deskewed BBox (for visual alignment/mosaic)
                    x0 = cfg.start_x + c * cfg.pitch_x
                    y0 = cfg.start_y + r * cfg.pitch_y
                    x1 = x0 + cfg.pitch_x
                    y1 = y0 + cfg.pitch_y
                    bbox_deskewed = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    
                    # Sanitize Filename & Create Title Text
                    safe_label = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in label])
                    
                    # Titles
                    title_raw_ov = f"X{c:02d}_Y{r:02d}_L{layer_idx:02d}_LEG:{safe_label}"
                    title_mask = f"X{c:02d}_Y{r:02d}_LEG:{safe_label}"
                    
                    basename = f"X{c:02d}_Y{r:02d}_L{layer_idx:02d}_LEG_{safe_label}"
                    
                    # Default filename for metadata (prefer raw, then overlay)
                    saved_rel_path = ""

                    # A. Raw Export
                    if options.get('raw'):
                        fname = f"{basename}.png"
                        path = os.path.join(layer_subdirs['raw'], fname)
                        final_img = self._add_title_bar(patch_qimg, title_raw_ov)
                        if not final_img.save(path):
                            print(f"Error saving {path}")
                        else:
                            # Save relative path for metadata
                            # raw/Layer_XX/filename.png
                            saved_rel_path = f"raw/{layer_name}/{fname}"
                        
                    # B. Overlay (Burn-in)
                    if options.get('overlay'):
                        ov_img = patch_qimg.convertToFormat(QImage.Format_ARGB32)
                        # Pass center_x, center_y (Raw Center)
                        self._draw_voids(ov_img, layer_idx, center_x, center_y, w_pitch, h_pitch, cfg.angle, mode='overlay')
                        fname = f"{basename}_overlay.png"
                        path = os.path.join(layer_subdirs['overlay'], fname)
                        final_img = self._add_title_bar(ov_img, title_raw_ov)
                        if not final_img.save(path):
                            print(f"Error saving {path}")
                        else:
                            if not saved_rel_path:
                                saved_rel_path = f"overlay/{layer_name}/{fname}"
                        
                    # C. Mask (Accumulate per Chip)
                    if options.get('mask'):
                        key = (c, r)
                        if key not in mask_accumulators:
                            img = QImage(w_pitch, h_pitch, QImage.Format_ARGB32)
                            img.fill(Qt.transparent)
                            mask_accumulators[key] = {'img': img, 'label': safe_label, 'title': title_mask}
                        
                        self._draw_voids(mask_accumulators[key]['img'], layer_idx, center_x, center_y, w_pitch, h_pitch, cfg.angle, mode='mask')
                        
                    # D. Merged Accumulate
                    if options.get('merged'):
                        safe_label_merged = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in label])
                        title_merged = f"LEG:{safe_label_merged}_MERGED"
                        
                        if safe_label_merged not in merged_accumulators:
                            merged_accumulators[safe_label_merged] = {
                                'img': QImage(w_pitch, h_pitch, QImage.Format_ARGB32),
                                'title': title_merged
                            }
                            merged_accumulators[safe_label_merged]['img'].fill(Qt.transparent)
                            
                        self._draw_voids(merged_accumulators[safe_label_merged]['img'], layer_idx, center_x, center_y, w_pitch, h_pitch, cfg.angle, mode='mask')

                    # Collect Metadata
                    if saved_rel_path:
                        meta = {
                            "layer": layer_idx,
                            "x": c,
                            "y": r,
                            "label": safe_label,
                            "bbox_raw": bbox_raw, # List of (x,y) tuples
                            "bbox_deskewed": bbox_deskewed,
                            "filename": saved_rel_path
                        }
                        patch_metadata_list.append(meta)

                    processed += 1
                    if processed % 10 == 0:
                        self.progress_update.emit(int(processed / total_chips * 100), f"Processing {r}, {c}...")

            # Save Masks
            if options.get('mask'):
                for (c, r), data in mask_accumulators.items():
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
            
            # Save Patch Metadata (patches.json)
            if options.get('json'):
                p_path = os.path.join(output_dir, "patches.json")
                with open(p_path, 'w') as f:
                    out_data = {
                        "grid_config": {
                            "start_x": cfg.start_x,
                            "start_y": cfg.start_y,
                            "pitch_x": cfg.pitch_x,
                            "pitch_y": cfg.pitch_y,
                            "rows": cfg.rows,
                            "cols": cfg.cols,
                            "angle": cfg.angle
                        },
                        "patches": patch_metadata_list
                    }
                    json.dump(out_data, f, indent=2)

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
        # Use ARGB32 to capture transparency
        out_img = QImage(w, new_h, QImage.Format_ARGB32)
        out_img.fill(Qt.transparent)
        
        painter = QPainter(out_img)
        
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

    def _extract_patch_unified(self, z, bbox, win_lo, win_hi, scale, offset):
        """
        Uses LazyTiffStack.get_poly_crop to extract straightened patch.
        Returns QImage (Grayscale8).
        """
        if not hasattr(self.data_stack, 'get_poly_crop'):
             return None
             
        roi_raw = self.data_stack.get_poly_crop(z, bbox)
        if roi_raw is None:
             return None
        
        # Process / Normalize
        roi_f32 = roi_raw.astype(np.float32)
        
        from .core_data import ImageNormalizer
        roi_u8 = ImageNormalizer.process(roi_f32, win_lo, win_hi, scale, offset)
        
        h_roi, w_roi = roi_u8.shape
        # Create QImage copy (Safe)
        q_src = QImage(roi_u8.data, w_roi, h_roi, w_roi, QImage.Format_Grayscale8).copy()
        
        return q_src

    def _draw_voids(self, target_img: QImage, current_layer, cx, cy, w, h, angle, mode='overlay'):
        """
        Draws voids relevant to this chip onto the target QImage.
        mode: 'overlay' (draw normally), 'mask' (hollow outlines, specific colors)
        """
        painter = QPainter(target_img)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pcx = target_img.width() / 2
        pcy = target_img.height() / 2
        
        # Calculate Chip Center in Deskewed Space using Main Window GL Widget
        gx_chip, gy_chip = self.main_window.glw.raw_to_deskew(cx, cy)
        
        # Decide which layers to draw
        layers_to_draw = []
        if mode == 'overlay':
            layers_to_draw = self.void_manager.voids.keys()
        else:
            layers_to_draw = [current_layer]
            
        for layer_key in layers_to_draw:
            voids = self.void_manager.voids.get(layer_key, [])
            is_active_layer = (layer_key == current_layer)
            
            for v in voids:
                vx, vy = v["globalCX"], v["globalCY"]
                
                # Transform Void to Deskewed
                gx_void, gy_void = self.main_window.glw.raw_to_deskew(vx, vy)
                
                # Relative Position
                dx = gx_void - gx_chip
                dy = gy_void - gy_chip
                
                px = pcx + dx
                py = pcy + dy
                
                # Bounds check
                if not (-w < (px-pcx) < w and -h < (py-pcy) < h):
                    continue
                
                tid = v.get("type_id", 0)
                type_data = self.void_manager.types.get(tid, {"color": (255, 0, 0)})
                col_rgb = type_data["color"]
                color = QColor(col_rgb[0], col_rgb[1], col_rgb[2])
                
                pen = QPen(color)
                pen.setWidth(2)
                
                rx = v["radiusX"]
                ry = v["radiusY"]
                
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
                self.void_manager.draw_void_shape(painter, shape, px, py, rx, ry)
            
        painter.end()

    def _export_json(self, output_dir):
        path = os.path.join(output_dir, "voids.json")
        # Use main_window properties
        self.void_manager.save_to_file(path, self.main_window.glw.grid_cfg, self.main_window.glw.bonding_map)

    def _export_csv(self, output_dir):
        path = os.path.join(output_dir, "voids.csv")
        grid_cfg = self.main_window.glw.grid_cfg
        bonding_map = self.main_window.glw.bonding_map
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["Layer", "CellRow", "CellCol", "Label", "VoidType", "GlobalX", "GlobalY", "RelativeX", "RelativeY", "RadiusX", "RadiusY"])
            
            # Iterate
            for layer, v_list in self.void_manager.voids.items():
                for v in v_list:
                    vx, vy = v["globalCX"], v["globalCY"]
                    
                    # Determine which cell it belongs to
                    gx = (vx - grid_cfg.start_x) / grid_cfg.pitch_x
                    gy = (vy - grid_cfg.start_y) / grid_cfg.pitch_y
                    c = int(np.floor(gx))
                    r = int(np.floor(gy))
                    
                    chip_x0 = grid_cfg.start_x + c * grid_cfg.pitch_x
                    chip_y0 = grid_cfg.start_y + r * grid_cfg.pitch_y
                    rel_x = vx - chip_x0
                    rel_y = vy - chip_y0
                    
                    label = ""
                    if bonding_map:
                        label = bonding_map.get_key(r, c)
                        
                    tid = v.get("type_id", 0)
                    tname = self.void_manager.types.get(tid, {}).get("name", "Unknown")
                    
                    writer.writerow([layer, r, c, label, tname, vx, vy, rel_x, rel_y, v["radiusX"], v["radiusY"]])
