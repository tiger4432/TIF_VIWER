from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QWidget, QSizePolicy, QSpacerItem
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QKeyEvent, QPen
from PySide6.QtCore import Qt, QSize

import numpy as np
from .core_data import ImageNormalizer, to_gray2d_uint16

class PatchViewer(QDialog):
    def __init__(self, parent=None, rois=None, data_source=None, grid_cfg=None, gl_widget=None, calib_data=None, win_lo=0, win_hi=65535):
        super().__init__(parent)
        self.setWindowTitle("Patch Viewer")
        self.resize(600, 700)
        
        # Data
        self.rois = rois if rois else []
        self.data_source = data_source
        self.grid_cfg = grid_cfg
        self.glw = gl_widget # Need for deskew_to_raw
        self.calib_data = calib_data
        self.win_lo = win_lo
        self.win_hi = win_hi
        
        self.current_idx = 0
        
        # UI
        self.layout = QVBoxLayout(self)
        
        # 1. Info Header
        self.lbl_info = QLabel("Chip: -")
        self.lbl_info.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(self.lbl_info)
        
        # 2. Image Display
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #self.lbl_img.setMinimumSize(400, 400)
        self.lbl_img.setStyleSheet("background-color: #202020; border: 1px solid #444;")
        self.layout.addWidget(self.lbl_img)
        
        # 3. Controls
        h_ctrl = QHBoxLayout()
        self.btn_prev = QPushButton("<< Prev")
        self.btn_prev.clicked.connect(self.prev_item)
        self.btn_next = QPushButton("Next >>")
        self.btn_next.clicked.connect(self.next_item)
        
        h_ctrl.addWidget(self.btn_prev)
        h_ctrl.addWidget(self.btn_next)
        self.layout.addLayout(h_ctrl)
        
        self.lbl_status = QLabel("Ready")
        self.layout.addWidget(self.lbl_status)
        
        # Initial Load
        if self.rois:
            self.show_item(0)
        else:
            self.lbl_info.setText("No ROIs found.")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Left:
            self.prev_item()
        elif event.key() == Qt.Key_Right:
            self.next_item()
        else:
            super().keyPressEvent(event)

    def prev_item(self):
        if not self.rois: return
        self.current_idx = (self.current_idx - 1) % len(self.rois)
        self.show_item(self.current_idx)

    def next_item(self):
        if not self.rois: return
        self.current_idx = (self.current_idx + 1) % len(self.rois)
        self.show_item(self.current_idx)

    def show_item(self, idx):
        if idx < 0 or idx >= len(self.rois): return
        
        roi_data = self.rois[idx]
        label = roi_data['label']
        c = roi_data['x']
        r = roi_data['y']
        
        self.lbl_info.setText(f"Index: {idx+1}/{len(self.rois)} | Label: {label} (R{r}, C{c})")
        
        # Extraction Logic
        if self.data_source is None or self.glw is None:
            self.lbl_status.setText("Error: No Data Source")
            return
            
        try:
            # 1. Get Image Layer (Assuming Layer 0 or current? Let's use current from glw?)
            # Wait, patch viewer usually wants to see what's on screen.
            # Let's assume current layer of GLWidget.
            z_idx = self.glw.current_layer
            # 1. Get Image Info (Dimensions)
            z_idx = self.glw.current_layer
            w, h = 0, 0
            if hasattr(self.data_source, 'width'):
                 w, h = self.data_source.width, self.data_source.height
            elif hasattr(self.data_source, 'shape'):
                 if self.data_source.ndim == 3:
                     h, w = self.data_source.shape[1:]
                 else:
                     h, w = self.data_source.shape[:2]
            
            # 2. Get Geometry
            bbox = roi_data.get('bbox', [])
            if not bbox:
                self.lbl_status.setText("Error: No BBox")
                return
            
            # 3. Fetch Crop (Efficiently using get_poly_crop)
            if hasattr(self.data_source, 'get_poly_crop'):
                 roi_img = self.data_source.get_poly_crop(z_idx, bbox)
            else:
                 self.lbl_status.setText("Error: DataSource missing get_poly_crop")
                 return
                 
            if roi_img is None:
                self.lbl_status.setText("Error: Crop failed (Out of bounds?)")
                return

            roi_img = roi_img.astype(np.float32)

            # 4. Calibration
            scale, offset = 1.0, 0.0
            if self.calib_data:
                # Per-layer calib data
                layer_map = self.calib_data.get(z_idx, None)
                if layer_map is not None:
                     if r < layer_map.shape[0] and c < layer_map.shape[1]:
                         scale = layer_map[r, c, 0]
                         offset = layer_map[r, c, 1]
            
            # 5. Process
            roi_u8 = ImageNormalizer.process(roi_img, self.win_lo, self.win_hi, scale, offset)
            
            # 6. Display (Directly)
            h_roi, w_roi = roi_u8.shape
            dst_w, dst_h = w_roi, h_roi
            
            q_dst = QImage(roi_u8.data, w_roi, h_roi, w_roi, QImage.Format_Grayscale8)
            # Convert to RGB for Overlay support
            q_dst = q_dst.convertToFormat(QImage.Format_ARGB32_Premultiplied)
            
            p = QPainter(q_dst)
            # Draw Crosshair (Debug)
            pen = QPen(Qt.red)
            pen.setWidth(3)
            p.setPen(pen)
            cx = int(dst_w / 2)
            cy = int(dst_h / 2)
            p.drawLine(cx - 20, cy, cx + 20, cy)
            p.drawLine(cx, cy - 20, cx, cy + 20)
            
            # Draw Border (Debug)
            pen_border = QPen(Qt.green)
            pen_border.setWidth(4)
            p.setPen(pen_border)
            p.setBrush(Qt.NoBrush)
            p.drawRect(0, 0, dst_w, dst_h)
            
            p.end()
            
            # Display
            pix = QPixmap.fromImage(q_dst)
            self.lbl_img.setPixmap(pix.scaled(self.lbl_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            status_msg = f"Range: [{roi_u8.min()}, {roi_u8.max()}] | " \
                         f"Win: {self.win_lo}-{self.win_hi} | " \
                         f"Calib: {scale:.3f}/{offset:.1f} | Size: {w_roi}x{h_roi}"
            self.lbl_status.setText(status_msg)
            
        except Exception as e:
            self.lbl_status.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()
            

