
import os
import csv
import json
import time
import numpy as np
import math
from datetime import datetime

from PySide6.QtCore import QObject, Signal, Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QTransform, QFont

from .core_data import ImageNormalizer

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

    def request_stop(self):
        self.stop_requested = True

    def run_export(self, output_dir, options: dict, win_lo=0, win_hi=65535):
        """
        Main export loop.
        options = {'raw': bool, 'overlay': bool, 'mask': bool, 'merged': bool, 
                   'json': bool, 'csv': bool}
        win_lo, win_hi: Normalization range (from UI)
        - raw:     pure patch per chip and layer
        - overlay: patch with the void markings drawn on it
        - mask:    per chip, markings of all exported layers
        - merged:  per bonding-map value, markings of all chips with that value composited
        """
        self.stop_requested = False
        
        try:
            # 1. Create subdirectories
            os.makedirs(output_dir, exist_ok=True)
            subdirs = {}
            for k in ('raw', 'overlay', 'mask', 'merged'):
                if options.get(k):
                    subdirs[k] = os.path.join(output_dir, k)
                    os.makedirs(subdirs[k], exist_ok=True)

            # Metadata Exports
            if options.get('json'):
                self._export_json(output_dir)
            if options.get('csv'):
                self._export_csv(output_dir)
            self._export_bonding_map(output_dir)

            # 2. Layers, Configs & ROIs
            num_layers = len(self.data_stack)
            layer_numbers = [self.main_window.abs_layer(i) for i in range(num_layers)]
            cfg = self.main_window.glw.grid_cfg
            # Only cells with a LEG (bonding map value; rejected values count as none). The ROIs are
            # rebuilt so a map pasted or changed after they were computed is honoured.
            bmap = self.main_window.glw.bonding_map
            self.main_window.update_rois()
            rois = [item for item in self.main_window.rois
                    if bmap is not None and bmap.get_key(item['y'], item['x'])
                    and item.get('bbox') and self._roi_in_image(item['bbox'])]
            
            # Same rounding as the extracted patch size (get_poly_crop)
            w_pitch = int(round(cfg.pitch_x))
            h_pitch = int(round(cfg.pitch_y))

            # Voids indexed by grid cell once (instead of scanning all voids for every chip)
            void_index = self._index_voids(cfg)
            self._refs = self.void_manager.reference_bboxes(cfg, self.main_window.glw.coord_transform)
            
            want_patch = bool(options.get('raw') or options.get('overlay'))
            want_marks = bool(options.get('mask') or options.get('merged'))
            total = (len(rois) * num_layers if want_patch else 0) + (len(rois) if want_marks else 0)
            self._progress_init(total)
            
            patch_metadata_list = []

            # 3. Patches (raw / overlay): need image data -> layer by layer
            if want_patch:
                for layer_idx, abs_layer in enumerate(layer_numbers):
                    if self.stop_requested: break
                    
                    layer_name = f"Layer_{abs_layer:02d}"
                    layer_subdirs = {}
                    for k in ('raw', 'overlay'):
                        if k in subdirs:
                            layer_subdirs[k] = os.path.join(subdirs[k], layer_name)
                            os.makedirs(layer_subdirs[k], exist_ok=True)
                    
                    # Calibration Map for this layer (keyed by loaded layer index)
                    calib_map = self.main_window.layer_calib_data.get(layer_idx) if self.main_window.layer_calib_data else None
                    
                    for item in rois:
                        if self.stop_requested: break
                        r, c = item['y'], item['x']
                        safe_label = self._safe(item.get('label', f"{r}_{c}"))
                        
                        scale, offset = 1.0, 0.0
                        if calib_map is not None and r < calib_map.shape[0] and c < calib_map.shape[1]:
                            scale = float(calib_map[r, c, 0])
                            offset = float(calib_map[r, c, 1])
                        
                        patch_qimg = self._extract_patch_unified(layer_idx, item['bbox'], win_lo, win_hi, scale, offset)
                        self._progress_step(f"Layer {abs_layer}: chip {r}, {c}...")
                        if patch_qimg is None: continue
                        
                        title = f"X{c:02d}_Y{r:02d}_L{abs_layer:02d}_LEG:{safe_label}"
                        basename = f"X{c:02d}_Y{r:02d}_L{abs_layer:02d}_LEG_{safe_label}"
                        saved_rel_path = "" # Metadata prefers raw, then overlay

                        # A. Raw Export (grayscale PNG)
                        if 'raw' in layer_subdirs:
                            fname = f"{basename}.png"
                            path = os.path.join(layer_subdirs['raw'], fname)
                            final_img = self._add_title_bar(patch_qimg, title).convertToFormat(QImage.Format_Grayscale8)
                            if final_img.save(path):
                                saved_rel_path = f"raw/{layer_name}/{fname}"
                            else:
                                print(f"Error saving {path}")
                            
                        # B. Overlay (Burn-in)
                        if 'overlay' in layer_subdirs:
                            ov_img = patch_qimg.convertToFormat(QImage.Format_ARGB32)
                            self._draw_chip_voids(ov_img, void_index, cfg, r, c, mode='overlay', current_layer=abs_layer)
                            fname = f"{basename}_overlay.png"
                            path = os.path.join(layer_subdirs['overlay'], fname)
                            if self._add_title_bar(ov_img, title).save(path):
                                saved_rel_path = saved_rel_path or f"overlay/{layer_name}/{fname}"
                            else:
                                print(f"Error saving {path}")

                        # Collect Metadata (Deskewed BBox for visual alignment/mosaic)
                        if saved_rel_path:
                            x0, y0 = cfg.cell_origin(r, c)
                            x1, y1 = x0 + cfg.pitch_x, y0 + cfg.pitch_y
                            patch_metadata_list.append({
                                "layer": abs_layer,
                                "x": c,
                                "y": r,
                                "label": safe_label,
                                "bbox_raw": item['bbox'], # List of (x,y) tuples
                                "bbox_deskewed": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                                "filename": saved_rel_path
                            })

                # Release the frame decoded for patch extraction
                self.data_stack.release_cache()

            # 4. Mask / Merged: markings only, no image data -> once per chip over all exported layers
            if want_marks and not self.stop_requested:
                layers = set(layer_numbers)
                merged_accumulators = {}
                for item in rois:
                    if self.stop_requested: break
                    r, c = item['y'], item['x']
                    safe_label = self._safe(item.get('label', f"{r}_{c}"))
                    self._progress_step(f"Masks: chip {r}, {c}...")

                    if 'mask' in subdirs:
                        img = QImage(w_pitch, h_pitch, QImage.Format_ARGB32)
                        img.fill(Qt.transparent)
                        self._draw_chip_voids(img, void_index, cfg, r, c, mode='mask', layers=layers)
                        path = os.path.join(subdirs['mask'], f"X{c:02d}_Y{r:02d}_LEG_{safe_label}.png")
                        if not self._add_title_bar(img, f"X{c:02d}_Y{r:02d}_LEG:{safe_label}").save(path):
                            print(f"Error saving mask {path}")

                    if 'merged' in subdirs:
                        img = merged_accumulators.get(safe_label)
                        if img is None:
                            img = QImage(w_pitch, h_pitch, QImage.Format_ARGB32)
                            img.fill(Qt.transparent)
                            merged_accumulators[safe_label] = img
                        self._draw_chip_voids(img, void_index, cfg, r, c, mode='mask', layers=layers)

                if not self.stop_requested:
                    for label, img in merged_accumulators.items():
                        path = os.path.join(subdirs['merged'], f"Merged_Label_{label}.png")
                        if not self._add_title_bar(img, f"LEG:{label}_MERGED").save(path):
                            print(f"Error saving merged {path}")

            if self.stop_requested:
                print("Export canceled.")
                return
            
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

    @staticmethod
    def _safe(label):
        """Sanitized label for file names."""
        return "".join(ch if ch.isalnum() or ch in (' ', '_', '-') else '_' for ch in label)

    def _roi_in_image(self, bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (max(xs) > 0 and max(ys) > 0 and
                min(xs) < self.data_stack.width and min(ys) < self.data_stack.height)

    def _progress_init(self, total):
        self._progress_total = max(1, total)
        self._progress_done = 0
        self._progress_last = 0.0

    def _progress_step(self, msg):
        # Throttled (~20 Hz): every emit runs processEvents for the dialog / Cancel
        self._progress_done += 1
        now = time.monotonic()
        if now - self._progress_last >= 0.05 or self._progress_done >= self._progress_total:
            self._progress_last = now
            self.progress_update.emit(int(self._progress_done * 100 / self._progress_total), msg)

    def _index_voids(self, cfg):
        """{(row, col): [(layer, void, gx, gy)]} with Deskewed centers, computed once per export."""
        glw = self.main_window.glw
        index = {}
        for layer, v_list in self.void_manager.voids.items():
            for v in v_list:
                gx, gy = glw.raw_to_deskew(v["globalCX"], v["globalCY"])
                index.setdefault(cfg.cell_at(gx, gy), []).append((layer, v, gx, gy))
        return index

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
        Uses ImageStack.get_poly_crop to extract straightened patch.
        Returns QImage (Grayscale8).
        """
        roi_raw = self.data_stack.get_poly_crop(z, bbox)
        if roi_raw is None:
             return None
        
        # Normalize via a 16-bit lookup table (same result as ImageNormalizer.process per pixel)
        roi_u8 = np.ascontiguousarray(self._lut(win_lo, win_hi, scale, offset)[roi_raw])
        h_roi, w_roi = roi_u8.shape
        # Create QImage copy (Safe)
        return QImage(roi_u8.data, w_roi, h_roi, w_roi, QImage.Format_Grayscale8).copy()

    def _lut(self, win_lo, win_hi, scale, offset):
        key = (win_lo, win_hi, scale, offset)
        cache = self.__dict__.setdefault('_lut_cache', {})
        lut = cache.get(key)
        if lut is None:
            if len(cache) > 256:
                cache.clear()
            lut = ImageNormalizer.process(np.arange(65536, dtype=np.float32), win_lo, win_hi, scale, offset)
            cache[key] = lut
        return lut

    def _draw_chip_voids(self, target_img: QImage, void_index, cfg, r, c, mode='overlay', current_layer=None, layers=None):
        """
        Draws the voids around chip (r, c) onto an image centered on the chip.
        mode 'overlay': all layers; current_layer solid, other layers as ghosts.
        mode 'mask':    only `layers`, solid, without antialiasing (pure type colors).
        """
        w, h = target_img.width(), target_img.height()
        pcx, pcy = w / 2, h / 2
        gx_chip = cfg.start_x + (c + 0.5) * cfg.pitch_x
        gy_chip = cfg.start_y + (r + 0.5) * cfg.pitch_y

        painter = QPainter(target_img)
        painter.setRenderHint(QPainter.Antialiasing, mode != 'mask')
        ghosts, actives = [], []
        # Voids whose center is within one pitch of the chip center live in the 3x3 neighbourhood
        for rr in (r - 1, r, r + 1):
            for cc in (c - 1, c, c + 1):
                for layer, v, gx, gy in void_index.get((rr, cc), ()):
                    dx, dy = gx - gx_chip, gy - gy_chip
                    if not (-w < dx < w and -h < dy < h):
                        continue
                    entry = (v, dx, dy, self.void_manager.quadrant_of(v, gx, gy, self._refs, cfg))
                    if mode == 'mask':
                        if layer in layers:
                            actives.append(entry)
                    elif layer == current_layer:
                        actives.append(entry)
                    else:
                        ghosts.append(entry)
        for v, dx, dy, q in ghosts:
            self.void_manager.paint_void(painter, v, pcx + dx, pcy + dy, style="ghost", width=2, quadrant=q)
        for v, dx, dy, q in actives:
            self.void_manager.paint_void(painter, v, pcx + dx, pcy + dy, style="normal", width=2, quadrant=q)
        painter.end()

    def _export_json(self, output_dir):
        path = os.path.join(output_dir, "voids.json")
        # Same as "Save JSON" (the transform is needed for rotated grids)
        glw = self.main_window.glw
        self.void_manager.save_to_file(path, glw.grid_cfg, glw.bonding_map, glw.coord_transform)

    def _export_bonding_map(self, output_dir):
        """bonding_map.txt: the pasted map as is (can be pasted again); reject values alongside."""
        bmap = self.main_window.glw.bonding_map
        text = getattr(bmap, "source_text", None) if bmap is not None else None
        if not text:
            return
        with open(os.path.join(output_dir, "bonding_map.txt"), 'w', encoding='utf-8', newline='') as f:
            f.write(text)
        if bmap.reject_values:
            with open(os.path.join(output_dir, "bonding_map_reject.txt"), 'w', encoding='utf-8') as f:
                f.write(", ".join(bmap.reject_values) + "\n")

    def _export_csv(self, output_dir):
        path = os.path.join(output_dir, "voids.csv")
        glw = self.main_window.glw
        columns = [(key, name) for key, name in self.void_manager.csv_columns.items() if name]
        records = self.void_manager.void_records(glw.grid_cfg, glw.bonding_map, glw.coord_transform)
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name for _, name in columns])
            for rec in records:
                writer.writerow([rec[key] for key, _ in columns])
