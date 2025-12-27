import sys
import os
import numpy as np
import tifffile
import time
import json # Used in VoidManager serialization, but here maybe not needed directly?
# MainWindow uses 'to_gray2d_uint16' from core_data.
# MainWindow calls void_manager.load_from_file which uses json.
# MainWindow logic seems free of direct json usage, but explicit imports are safer if I missed something.

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSpinBox, QSlider, QLabel, QToolBar, QDockWidget, QGroupBox, 
    QFormLayout, QPushButton, QDoubleSpinBox, QPlainTextEdit, 
    QFileDialog, QProgressDialog, QCheckBox, QRadioButton, 
    QButtonGroup, QComboBox
)
from PySide6.QtGui import QSurfaceFormat, QAction, QImage, QPainter
from PySide6.QtCore import Qt, QTimer

# Imports from sat_widgets
from sat_widgets.core_data import LazyTiffStack, GridConfig, BondingMap, to_gray2d_uint16
from sat_widgets.void_manager import VoidManager, VoidTypeDialog
from sat_widgets.gl_widget import GLImageWidget, TiledImage

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
        # v_ext.addWidget(btn_extract) # Duplicate in original?
        dock_layout.addWidget(gb_extract)
        
        # 3.2 Coordinate Display & Navigation
        gb_coord = QGroupBox("Coordinates & Navigation")
        form_coord = QFormLayout(gb_coord)
        
        self.lbl_cursor_pos = QLabel("Hover: -")
        form_coord.addRow(self.lbl_cursor_pos)
        
        h_jump = QHBoxLayout()
        self.sb_jump_x = QSpinBox(); self.sb_jump_x.setRange(0, 9999); self.sb_jump_x.setPrefix("X:")
        self.sb_jump_y = QSpinBox(); self.sb_jump_y.setRange(0, 9999); self.sb_jump_y.setPrefix("Y:")
        btn_jump = QPushButton("Go")
        btn_jump.clicked.connect(self.go_to_cell)
        
        h_jump.addWidget(self.sb_jump_x)
        h_jump.addWidget(self.sb_jump_y)
        h_jump.addWidget(btn_jump)
        
        form_coord.addRow("Jump to Cell:", h_jump)
        dock_layout.addWidget(gb_coord)

        # Connect Cursor Signal
        self.glw.view_center_changed.connect(self.on_view_center_changed)
        self.glw.navigation_requested.connect(self.on_navigation_requested)
        
        # 3.5 Void Marking Control
        self.void_manager = VoidManager()
        self.glw.void_manager = self.void_manager
        
        self.gb_void = QGroupBox("Void Marking")
        v_layout_void = QVBoxLayout(self.gb_void)
        
        # Toggle Mode (Hint Only)
        self.lbl_void_hint = QLabel("Hold SHIFT to Activate")
        self.lbl_void_hint.setStyleSheet("font-weight: bold; color: gray;")
        
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
        
        self.btn_clear_chip = QPushButton("Clear Chip")
        self.btn_clear_chip.clicked.connect(self.clear_chip_voids)
        
        h_layout_void_btns.addWidget(self.btn_load_voids)
        h_layout_void_btns.addWidget(self.btn_save_voids)
        h_layout_void_btns.addWidget(self.btn_clear_voids)
        h_layout_void_btns.addWidget(self.btn_clear_chip)
        
        v_layout_void.addWidget(self.lbl_void_hint)
        v_layout_void.addLayout(h_modes)
        v_layout_void.addLayout(h_type)
        v_layout_void.addWidget(self.lbl_void_count)
        v_layout_void.addLayout(h_layout_void_btns)
        v_layout_void.addWidget(self.btn_clear_voids)
        
        dock_layout.addWidget(self.gb_void)
        
        # Init Types
        self.populate_void_types()
        
        # Shortcuts for Void Mode
        # Removed Toggle ('q') as Shift is now used
    
    
        self.action_void_draw = QAction("Void Draw Mode", self)
        self.action_void_draw.setShortcut("w")
        self.action_void_draw.triggered.connect(self.set_void_mode_draw)
        self.addAction(self.action_void_draw)

        self.action_void_edit = QAction("Void Edit Mode", self)
        self.action_void_edit.setShortcut("e")
        self.action_void_edit.triggered.connect(self.set_void_mode_edit)
        self.addAction(self.action_void_edit)
        
        self.action_void_erase = QAction("Void Erase Mode", self)
        self.action_void_erase.setShortcut("r")
        self.action_void_erase.triggered.connect(self.set_void_mode_erase)
        self.addAction(self.action_void_erase)

        # Removed Esc (Exit Void Mode) as Shift release exits

        
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
            

    def on_view_center_changed(self, c, r):
        self.lbl_cursor_pos.setText(f"Center: Col {c}, Row {r}")
        
        # Also sync inputs if not focused?
        # self.spin_nav_x.setValue(c)
        # self.spin_nav_y.setValue(r)
        
    def on_navigation_requested(self, c, r):
        # Update inputs first
        self.sb_jump_x.blockSignals(True)
        self.sb_jump_y.blockSignals(True)
        self.sb_jump_x.setValue(c)
        self.sb_jump_y.setValue(r)
        self.sb_jump_x.blockSignals(False)
        self.sb_jump_y.blockSignals(False)
        
        # Trigger move
        self.go_to_cell()
        
    def go_to_cell(self):
        c = self.sb_jump_x.value()
        r = self.sb_jump_y.value()
        self.glw.fit_to_cell(c, r)
            
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

    def set_void_mode_draw(self):
        self.rb_draw.setChecked(True)
        self.on_void_mode_changed()

    def set_void_mode_edit(self):
        self.rb_edit.setChecked(True)
        self.on_void_mode_changed()

    def set_void_mode_erase(self):
        self.rb_erase.setChecked(True)
        self.on_void_mode_changed()

    def on_void_mode_changed(self):
        # Called when Radio Button changes
        if self.rb_draw.isChecked():
            self.glw.set_void_tool("DRAW")
            print("Void Tool: DRAW")
        elif self.rb_edit.isChecked():
            self.glw.set_void_tool("EDIT")
            print("Void Tool: EDIT")
        elif self.rb_erase.isChecked():
            self.glw.set_void_tool("ERASE")
            print("Void Tool: ERASE")
            
        self.glw.setFocus()
        self.glw.update()
        
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
            self.void_manager.save_to_file(path, self.glw.grid_cfg, self.glw.bonding_map)
            
    def clear_voids(self):
        # Clear CURRENT LAYER Only
        self.void_manager.clear_layer(self.current_z)
        self.glw.active_void = None
        self.glw.setFocus()
        self.glw.update()
        
    def clear_chip_voids(self):
        self.glw.clear_current_chip_voids()
        self.glw.setFocus()
        
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
        self.glw.current_layer = val # Sync with GL Widget for Void Manager
        
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
