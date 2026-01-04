
import sys
import os
import numpy as np
import time
from datetime import datetime
import json 

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSpinBox, QSlider, QLabel, QToolBar, QDockWidget, QGroupBox, 
    QFormLayout, QPushButton, QDoubleSpinBox, QPlainTextEdit, 
    QFileDialog, QProgressDialog, QCheckBox, QRadioButton, 
    QButtonGroup, QComboBox, QDialog, QScrollArea 
)
from PySide6.QtGui import QSurfaceFormat, QAction, QImage, QPainter
from PySide6.QtCore import Qt, QTimer

# Imports from sat_widgets
from sat_widgets.core_data import LazyTiffStack, MosaicTiffStack, GridConfig, BondingMap, to_gray2d_uint16, ImageNormalizer, CoordinateTransform
from sat_widgets.void_manager import VoidManager, VoidTypeDialog
from sat_widgets.gl_widget import GLImageWidget, TiledImage
from sat_widgets.exporter import ExportManager
from sat_widgets.patch_viewer import PatchViewer
from sat_widgets.roi_inspector import ROIInspector

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

        self.current_z = 0
        self.layer_offset = 0 # Absolute offset of loaded layers
        self.layer_cache = {} # Cache for TiledImage objects
        self.layer_calib_data = {} # Cache for calibration data {z: (rows, cols, 2)}
        self.rois = [] # Pre-calculated ROIs [{'label':.., 'x':.., 'y':.., 'bbox':[(x,y)*4]}, ...]
        
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

        a_proj = QAction("Open Project", self)
        a_proj.triggered.connect(self.open_project_folder)
        tb.addAction(a_proj)

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
        
        self.sb_grid_x = QDoubleSpinBox(); self.sb_grid_x.setRange(-10000, 50000); self.sb_grid_x.setValue(5000)
        self.sb_grid_y = QDoubleSpinBox(); self.sb_grid_y.setRange(-10000, 50000); self.sb_grid_y.setValue(5000)
        form.addRow("Origin X:", self.sb_grid_x)
        form.addRow("Origin Y:", self.sb_grid_y)
        
        self.sb_pitch_x = QDoubleSpinBox(); self.sb_pitch_x.setRange(1, 10000); self.sb_pitch_x.setValue(500)
        self.sb_pitch_y = QDoubleSpinBox(); self.sb_pitch_y.setRange(1, 10000); self.sb_pitch_y.setValue(500)
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
        self.sb_end_layer = QSpinBox(); self.sb_end_layer.setRange(0, 100); self.sb_end_layer.setValue(1)
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

        
        btn_adv_export = QPushButton("Advanced Export...")
        btn_adv_export.clicked.connect(self.open_advanced_export)
        v_ext.addWidget(btn_adv_export)
        
        btn_patch_view = QPushButton("Open Patch Viewer")
        btn_patch_view.clicked.connect(self.open_patch_viewer)
        v_ext.addWidget(btn_patch_view)
        
        btn_inspect_rois = QPushButton("Inspect ROIs (Debug)")
        btn_inspect_rois.clicked.connect(self.open_roi_inspector)
        v_ext.addWidget(btn_inspect_rois)
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
        # Connect Cursor Signal
        self.glw.view_center_changed.connect(self.on_view_center_changed)
        self.glw.navigation_requested.connect(self.on_navigation_requested)
        self.glw.cursor_moved.connect(self.on_cursor_moved)
        
        # 3.5 Void Marking Control
        self.void_manager = VoidManager()
        self.glw.void_manager = self.void_manager
        self.glw.void_type_shortcut_triggered.connect(self.on_void_type_shortcut)
        
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
        

        dock_layout.addWidget(gb_calib)
        
        dock_layout.addStretch()
        

        # Scroll Area Wrapper
        scroll = QScrollArea()
        scroll.setWidget(dock_content)
        scroll.setWidgetResizable(True)
        # scroll.setMaximumHeight(1000) # Removed fixed limit
        
        from PySide6.QtWidgets import QSizePolicy
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        dock.setWidget(scroll)

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
        
        # CRITICAL: Notify GLWidget to rebuild CoordinateTransform
        self.glw.update_grid_params()
        
        # Update ROIs
        self.update_rois()

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
        
        self.update_rois() # Update ROIs on manual move too

        if self.chk_use_calib.isChecked():
             # Disable auto-recalc on move (User Request)
             # User must click 'Recalculate'
             pass
             # self.calib_timer.start(200) # 200ms debounce

    def update_rois(self):
        """
        Pre-calculate ROIs for all (bonded) chips in Raw Image Coordinates.
        Stored in self.rois
        """
        self.rois = []
        cfg = self.glw.grid_cfg
        
        # If bonding map exists, use it. Else use all grid cells? 
        # User said "bonding map에 따라서" (according to bonding map).
        # If no map, maybe empty? Or all? Let's assume all if no map, or just logic handle both.
        # "bonding map... bonding 좌표들의 roi" implies only bonded ones.
        
        rows = cfg.rows
        cols = cfg.cols
        
        for r in range(rows):
            for c in range(cols):
                label = f"{r}_{c}"
                is_bonded = True
                
                if self.glw.bonding_map:
                    key = self.glw.bonding_map.get_key(r, c)
                    if not key:
                        is_bonded = False
                    else:
                        label = key
                
                if not is_bonded:
                    continue
                    
                # Calculate 4 corners in Deskewed Space
                # Top-Left: (c*p_x, r*p_y) + start
                x0 = cfg.start_x + c * cfg.pitch_x
                y0 = cfg.start_y + r * cfg.pitch_y
                x1 = x0 + cfg.pitch_x
                y1 = y0 + cfg.pitch_y
                
                # 4 Points (TL, TR, BR, BL)
                corners_deskewed = [
                    (x0, y0), (x1, y0), (x1, y1), (x0, y1)
                ]
                
                bbox_raw = []
                for gx, gy in corners_deskewed:
                    rx, ry = self.glw.deskew_to_raw(gx, gy)
                    bbox_raw.append((rx, ry))
                    
                self.rois.append({
                    'label': label,
                    'x': c,
                    'y': r,
                    'bbox': bbox_raw
                })
        
        # Debug print
        # print(f"Updated ROIs: {len(self.rois)} items")
        
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
                
            print(f"Map Imported: {bmap.rows}x{bmap.cols}, Unique Keys: {len(bmap.data_map)}")
            
            # Update ROIs
            self.update_rois()
            

    def on_view_center_changed(self, c, r):
        # We only update the "Center" info in the label if user isn't hovering.
        # OR we splits the label?
        # Current label text "Hover: -" suggests Hover is primary.
        # "Center: ..." was setting it.
        # Let's append or overwrite?
        # Let's say: "Center: C, R | Hover: X, Y (Val)"
        pass
        # self.lbl_cursor_pos.setText(f"Center: Col {c}, Row {r}")
        
    def on_cursor_moved(self, gx, gy):
        # gx, gy are Deskewed Image Coordinates (ints)
        
        # 1. Convert to Raw (Pixel Access)
        rx, ry = self.glw.deskew_to_raw(gx, gy)
        
        # 2. Lookup Value
        val_str = "-"
        if self.full_data is not None:
             # Check bounds
             h, w = self.full_data.shape[-2:] # Last 2 dims
             irx = int(rx)
             iry = int(ry)
             
             if 0 <= irx < w and 0 <= iry < h:
                 # Load value
                 # LazyStack supports slice? Or scalar?
                 # LazyStack[z] returns array.
                 # Optimization: Don't load full layer for 1 pixel.
                 # But LazyStack wraps PIL. 
                 # We can use our cache? 
                 # Or just try/except.
                 try:
                     # Accessing single pixel from LazyTiffStack might be slow if we load full page.
                     # But we have `layer_cache`?
                     # `layer_cache` stores `TiledImage`. `TiledImage` doesn't keep full numpy array (it chops tiles).
                     # `TiledImage` assumes tile data is in RAM or GPU? 
                     # `Tile.data` is numpy.
                     
                     # Simple approach: If `LazyTiffStack`, use `img.getpixel`?
                     # LazyStack doesn't expose `getpixel`.
                     # Let's skip value lookup if it's too heavy, valid coordinates first.
                     
                     val_str = ""
                 except:
                     pass
        
        # 3. Calculate Grid Cell (Deskewed)
        cfg = self.glw.grid_cfg
        # Adjust for start offset before dividing by pitch
        c = int(np.floor((gx - cfg.start_x) / cfg.pitch_x))
        r = int(np.floor((gy - cfg.start_y) / cfg.pitch_y))
        
        label_str = "-"
        if self.glw.bonding_map:
             label_str = self.glw.bonding_map.get_key(r, c) or "-"
             
        # Update Label (User Request: Show Raw & Deskew)
        # Format: Raw: (X, Y) | Deskew: (X, Y) | Cell: R_C (Label)
        self.lbl_cursor_pos.setText(
            f"Raw: ({rx:.1f}, {ry:.1f}) | Deskew: ({gx:.1f}, {gy:.1f}) | Cell: {r}_{c} ({label_str})"
        )
        
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
        # Auto Level
        self.auto_level()

    def open_project_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder (with patches.json)")
        if not folder: return
        
        # Check for patches.json
        p_json = os.path.join(folder, "patches.json")
        if not os.path.exists(p_json):
            print("patches.json not found!")
            return
            
        # 1. Init Mosaic Stack
        try:
            self.full_data = MosaicTiffStack(p_json)
            self.layer_offset = self.full_data.layer_offset
            self.layer_cache.clear()
        except Exception as e:
            print(f"Failed to load project: {e}")
            return
            
        # 2. Setup UI
        frames = len(self.full_data)
        self.spin_layer.setRange(0, frames - 1)
        self.spin_layer.setValue(0)
        self.spin_layer.setEnabled(True)
        self.current_z = 0
        
        # 3. Apply Grid Config from Stack
        # MosaicStack .grid_cfg is a dict
        gcfg = self.full_data.grid_cfg
        
        self.sb_grid_x.setValue(gcfg.get("start_x", 0.0))
        self.sb_grid_y.setValue(gcfg.get("start_y", 0.0))
        self.sb_pitch_x.setValue(gcfg.get("pitch_x", 100.0))
        self.sb_pitch_y.setValue(gcfg.get("pitch_y", 100.0))
        self.sb_rows.setValue(gcfg.get("rows", 1))
        self.sb_cols.setValue(gcfg.get("cols", 1))
        self.sb_angle.setValue(gcfg.get("angle", 0.0))
        
        # Update GL Widget Grid Config
        self.glw.grid_cfg.start_x = self.sb_grid_x.value()
        self.glw.grid_cfg.start_y = self.sb_grid_y.value()
        self.glw.grid_cfg.pitch_x = self.sb_pitch_x.value()
        self.glw.grid_cfg.pitch_y = self.sb_pitch_y.value()
        self.glw.grid_cfg.rows = self.sb_rows.value()
        self.glw.grid_cfg.cols = self.sb_cols.value()
        self.glw.grid_cfg.angle = self.sb_angle.value()
        
        # self.glw.update_grid_buffer() # Method does not exist, update() is sufficient
        self.glw.update()

        # 4. Load Voids (Raw)
        v_json = os.path.join(folder, "voids.json")
        if os.path.exists(v_json):
            # Virtual Raw Strategy: Mosaic is the Raw Image.
            # Voids are in Raw Coords.
            # Load directly without transformation.
            self.void_manager.load_from_file(v_json, self.glw.grid_cfg)

        
        # 5. Load Initial Layer
        self.current_z = 0
        self.load_layer(0)
        
        # Auto Level (Optional, or standard default)
        self.glw.update()

    def auto_level(self):
        c = self.sb_jump_x.value()
        r = self.sb_jump_y.value()
        self.glw.fit_to_cell(c, r)
            


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
            self.void_manager.load_from_file(path, self.glw.grid_cfg, self.glw.coord_transform)
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
        
        # Update cursor style if void mode is active
        if self.glw.void_mode:
            self.glw.update_cursor_style()

    def on_void_type_shortcut(self, type_id):
        """
        Called when GLWidget detects a shortcut key (1-9, 0).
        Sync the ComboBox.
        """
        # Find index for this type_id
        idx = self.cb_void_type.findData(type_id)
        if idx >= 0:
            self.cb_void_type.setCurrentIndex(idx)
        
    def open_type_manager(self):
        dlg = VoidTypeDialog(self.void_manager, self)
        dlg.exec()
        self.populate_void_types()
        self.glw.update()

    def save_voids(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Voids JSON", "voids.json", "JSON (*.json)")
        if path:
            self.void_manager.save_to_file(path, self.glw.grid_cfg, self.glw.bonding_map, self.glw.coord_transform)
            
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
            if hasattr(self.full_data, 'close'):
                self.full_data.close()
            self.full_data = None
            
            # 2. Create Lazy Stack (PIL Path)
            lazy_stack = LazyTiffStack(path)
            
            # Apply Limits
            # req_start is 0-indexed index. req_end is inclusive? 
            # UI Default: Start 0, End 10.
            # Range: [Start, End]. Count = End - Start + 1?
            # Or is End exclusive? "End Limit". Usually inclusive in user speak?
            # Let's assume inclusive for now or match previous logical intent.
            # If default is 0 to 10.
            if req_start <= req_end:
                 count = req_end - req_start + 1
                 lazy_stack.set_range(req_start, count)
                 print(f"Applied Layer Limit: Start {req_start}, Count {count}")
            
            print(f"Opened with Lazy Loading (Instant). Shape: {lazy_stack.shape}")
            
            self.full_data = lazy_stack
            self.layer_offset = req_start # Store simple offset for reference if needed
            
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
            # Handle MosaicTiffStack specialized loading
            img_2d = to_gray2d_uint16(self.full_data, z_index)
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

    def open_patch_viewer(self):
        if self.full_data is None:
            print("No image loaded.")
            return

        # Ensure we have ROIs (from grid update or import map)
        if not self.rois:
             print("No ROIs found. Attempting to update from grid...")
             self.update_rois()
             
        dlg = PatchViewer(
            self, 
            rois=self.rois, 
            data_source=self.full_data, 
            grid_cfg=self.glw.grid_cfg,
            gl_widget=self.glw,
            calib_data=self.layer_calib_data if self.chk_use_calib.isChecked() else None,
            win_lo=self.glw.win_lo,
            win_hi=self.glw.win_hi
        )
        dlg.exec()

    def open_roi_inspector(self):
        if not self.rois:
             print("No ROIs found. Updating...")
             self.update_rois()
             
        dlg = ROIInspector(self.rois, self)
        dlg.exec()

    def open_advanced_export(self):
        if not self.full_data and not self.glw.tiled_image:
            print("No image loaded.")
            return

        # 1. Generate Auto Path
        # save/{tif_name}_{timestamp}
        tif_basename = "unknown"
        if hasattr(self.full_data, 'path') and self.full_data.path:
             # Extract filename without extension
             fname = os.path.basename(self.full_data.path)
             tif_basename = os.path.splitext(fname)[0]
             
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = os.path.join(os.getcwd(), "save", f"{tif_basename}_{timestamp}")
        
        dlg = ExportDialog(self, default_dir=default_dir)
        if dlg.exec():
            # Run Export
            out_dir = dlg.output_dir
            opts = dlg.get_options()
            
            if not out_dir:
                print("No output directory selected.")
                return
                
            print(f"Starting Export to {out_dir} with opts: {opts}")
            
            # Setup Progress Dialog
            self.pd = QProgressDialog("Exporting...", "Cancel", 0, 100, self)
            self.pd.setWindowModality(Qt.WindowModal)
            self.pd.show()
            
            # Ensure ROIs are up to date
            if not self.rois:
                 print("Updating ROIs for export...")
                 self.update_rois()
            
            # Create Manager (Pass MainWindow)
            exporter = ExportManager(self)
            
            # Connect
            exporter.progress_update.connect(lambda p, msg: (self.pd.setValue(p), self.pd.setLabelText(msg), QApplication.processEvents()))
            
            # Run
            try:
                # Pass Window Levels for WYSIWYG
                win_lo = self.glw.win_lo
                win_hi = self.glw.win_hi
                exporter.run_export(out_dir, opts, win_lo, win_hi)
            except Exception as e:
                print(f"Export Error: {e}")
                
            self.pd.close()
            print("Export Finished.")

class ExportDialog(QDialog):
    def __init__(self, parent=None, default_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Advanced Export")
        self.resize(400, 300)
        
        self.layout = QVBoxLayout(self)
        
        # 1. Output Options
        gb_opt = QGroupBox("Export Artifacts")
        v_opt = QVBoxLayout(gb_opt)
        
        self.chk_raw = QCheckBox("Raw Patches (/raw)")
        self.chk_overlay = QCheckBox("Overlay Patches (/overlay)")
        self.chk_mask = QCheckBox("Mask Patches (/mask)")
        self.chk_merged = QCheckBox("Merged Patches (/merged)")
        self.chk_json = QCheckBox("Metadata JSON (voids.json)")
        self.chk_csv = QCheckBox("Metadata CSV (voids.csv)")
        
        # Defaults
        self.chk_raw.setChecked(True)
        self.chk_overlay.setChecked(True)
        self.chk_mask.setChecked(True)
        self.chk_merged.setChecked(True)
        self.chk_json.setChecked(True)
        self.chk_csv.setChecked(True)
        
        v_opt.addWidget(self.chk_raw)
        v_opt.addWidget(self.chk_overlay)
        v_opt.addWidget(self.chk_mask)
        v_opt.addWidget(self.chk_merged)
        v_opt.addWidget(self.chk_json)
        v_opt.addWidget(self.chk_csv)
        self.layout.addWidget(gb_opt)
        
        # 2. Directory
        h_dir = QHBoxLayout()
        self.output_dir = default_dir
        self.lbl_dir = QLabel(f"Out Dir: {default_dir}" if default_dir else "Out Dir: -")        
        btn_dir = QPushButton("Browse...")
        btn_dir.clicked.connect(self.browse_dir)
        h_dir.addWidget(self.lbl_dir, 1)
        h_dir.addWidget(btn_dir)
        self.layout.addLayout(h_dir)
        
        # 3. Actions
        h_btns = QHBoxLayout()
        self.btn_export = QPushButton("Start Export")
        self.btn_export.clicked.connect(self.accept) # We'll handle logic in Main
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        
        h_btns.addWidget(self.btn_export)
        h_btns.addWidget(self.btn_cancel)
        self.layout.addLayout(h_btns)
        
    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.output_dir = d
            self.lbl_dir.setText(f"Out: {d}")

    def get_options(self):
        return {
            'raw': self.chk_raw.isChecked(),
            'overlay': self.chk_overlay.isChecked(),
            'mask': self.chk_mask.isChecked(),
            'merged': self.chk_merged.isChecked(),
            'json': self.chk_json.isChecked(),
            'csv': self.chk_csv.isChecked()
        }

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
