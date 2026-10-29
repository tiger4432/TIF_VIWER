
import sys
import os
import numpy as np
import time
import traceback
from collections import OrderedDict
from datetime import datetime
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSpinBox, QSlider, QLabel, QToolBar, QDockWidget, QGroupBox,
    QFormLayout, QPushButton, QDoubleSpinBox, QPlainTextEdit,
    QFileDialog, QProgressDialog, QCheckBox, QRadioButton,
    QButtonGroup, QComboBox, QDialog, QScrollArea, QMessageBox, QLineEdit,
    QTextEdit, QAbstractSpinBox, QSizePolicy
)
from PySide6.QtGui import QSurfaceFormat, QAction, QImage, QPainter, QKeySequence
from PySide6.QtCore import Qt, QTimer, QEvent, QPointF

# Imports from sat_widgets
from sat_widgets.core_data import TiffStack, MosaicTiffStack, GridConfig, BondingMap, to_uint16, ImageNormalizer, CoordinateTransform
from sat_widgets.void_manager import VoidManager, VoidTypeDialog
from sat_widgets.gl_widget import GLImageWidget, TiledImage
from sat_widgets.layer_loader import LayerLoader, STATS_STEP
from sat_widgets.exporter import ExportManager
from sat_widgets.patch_viewer import PatchViewer
from sat_widgets.roi_inspector import ROIInspector
from sat_widgets.angle_preview import AnglePreview
from sat_widgets.auto_bbox import place_bboxes
from sat_widgets import void_detect
from sat_widgets import void_eval
from sat_widgets import results_api
from sat_widgets.products import load_products, GRID_KEYS
from sat_widgets.app_paths import declaration_path, sync_declarations, default_update_manifest
from sat_widgets.version import APP_VERSION
from sat_widgets.updater import UpdateChecker, show_update_dialog

# ==================================================================================
# 4. Main Window
# ==================================================================================
def calibrate_layer(stack, z, p, preview=None):
    """
    (rows, cols, 2) scale / offset per cell of layer z: the cell's brightness (mean / std, or
    median / IQR when robust) is mapped to the target. Cells without a bonding map value (and
    outside the image) get scale 1, offset 0. Thread-safe given its own stack reader.
    preview: (array, step), every step-th pixel of the layer kept by the background loader: the
    statistics are taken from it (no decoding), else from the stack.
    """
    h, w = stack.height, stack.width
    rows, cols = p["rows"], p["cols"]
    pitch_x, pitch_y = p["pitch_x"], p["pitch_y"]
    win_lo = p["win_lo"]
    win_range = max(1, p["win_hi"] - win_lo)
    bonded = p["bonded"]
    calib = np.zeros((rows, cols, 2), dtype=np.float32)
    calib[:, :, 0] = 1.0
    for r in range(rows):
        for c in range(cols):
            if bonded is not None and (r, c) not in bonded:
                continue # Not bonded -> no calibration
            # Cell Center: Deskewed (grid) -> Raw (same transform as the display)
            cx, cy = p["to_raw"](p["start_x"] + (c + 0.5) * pitch_x, p["start_y"] + (r + 0.5) * pitch_y)
            x0 = max(0, min(w, int(cx - pitch_x / 2)))
            x1 = max(0, min(w, int(cx + pitch_x / 2)))
            y0 = max(0, min(h, int(cy - pitch_y / 2)))
            y1 = max(0, min(h, int(cy + pitch_y / 2)))
            if x1 <= x0 or y1 <= y0:
                continue
            if preview is not None:
                arr, s = preview
                # Preview pixel k is source pixel k * s: keep those inside [x0, x1) x [y0, y1)
                sub = arr[-(-y0 // s):-(-y1 // s), -(-x0 // s):-(-x1 // s)]
                step = int(np.sqrt(sub.size / 10000)) if sub.size > 10000 else 1
                roi = to_uint16(sub[::max(1, step), ::max(1, step)])
            else:
                # Subsample to ~10k pixels while reading
                roi_size = (x1 - x0) * (y1 - y0)
                step = int(np.sqrt(roi_size / 10000)) if roi_size > 10000 else 1
                roi = to_uint16(stack.read_region(z, x0, y0, x1 - x0, y1 - y0, max(1, step)))
            if p["robust"]:
                mean = np.median(roi)
                q75, q25 = np.percentile(roi, [75, 25])
                std = (q75 - q25) / 1.35
            else:
                mean = np.mean(roi)
                std = np.std(roi)
            std = max(std, 1.0)
            scale = p["target_std"] / std
            calib[r, c, 0] = scale
            calib[r, c, 1] = (p["target_mean"] - win_lo - scale * (mean - win_lo)) / win_range
    return calib


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Tiled TIF Viewer (OpenGl Mosaic) v{APP_VERSION}")
        
        # Declarations (config.json, products.json): central shared folder -> local cache, see app_paths
        self.declaration_status = sync_declarations()
        self.resize(1280, 720)
        
        self.calib_timer = QTimer()
        self.calib_timer.setSingleShot(True)
        self.calib_timer.timeout.connect(self.calculate_calibration)
        
        self.full_data = None # ImageStack (TiffStack / MosaicTiffStack)

        self.current_z = 0 # Loaded layer index (0..n-1). Void data uses absolute layer numbers.
        self.layer_offset = 0 # Absolute number of the first loaded layer
        self.layer_cache = OrderedDict() # GPU TiledImage per layer index, LRU order
        # Background loading (see sat_widgets/layer_loader.py)
        self.layer_loader = None
        self.layer_proxies = {} # z -> uploaded low-resolution preview (TiledImage)
        self.layer_proxy_arrays = {} # z -> (every STATS_STEP-th pixel, step): per-cell statistics (calibration)
        self._ram_layers = {} # z -> decoded tiles of a neighbour layer that did not fit the GPU budget
        self._upload_queue = [] # TiledImages with tiles waiting for upload (current layer first)
        self._upload_timer = QTimer(self)
        self._upload_timer.setSingleShot(True)
        self._upload_timer.setInterval(0)
        self._upload_timer.timeout.connect(self._upload_step)
        self._layer_dir = 1 # Last browsing direction (+1 / -1): prefetched first
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
        
        a_reload = QAction("Reload Declarations", self)
        a_reload.setToolTip("Fetch config.json / products.json again (shared folder) and apply them")
        a_reload.triggered.connect(self.reload_declarations)
        tb.addAction(a_reload)

        # ---------------------------------------------------------
        # Dock Widget for Grid/Map Controls
        # ---------------------------------------------------------
        dock = QDockWidget("Grid & Map Tools", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        
        # 0. WF Product (products.json): grid size, bbox size and bbox layout
        gb_product = QGroupBox("WF Product")
        form_product = QFormLayout(gb_product)
        self.cb_product = QComboBox()
        form_product.addRow("Product:", self.cb_product)
        # Opened file and its wafer ID (parsed from the file name, editable)
        self.lbl_source = QLabel("-")
        self.lbl_source.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_source.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form_product.addRow("File:", self.lbl_source)
        h_wafer = QHBoxLayout()
        self.le_wafer_id = QLineEdit()
        self.le_wafer_id.setPlaceholderText("parsed from the file name")
        self.le_wafer_id.setToolTip('Wafer ID: parsed from the file name (config.json "wafer_id": {"pattern": ...}),\n'
                                    "editable. Used by the Results API plugin (payload, bonding map lookup).")
        btn_fetch_map = QPushButton("Fetch Map")
        btn_fetch_map.setToolTip("Get the bonding map of this wafer ID from the server\n"
                                 "(fetch_bonding_map of plugins/results_api.py)")
        btn_fetch_map.clicked.connect(self.fetch_bonding_map)
        self.btn_fetch_map = btn_fetch_map
        h_wafer.addWidget(self.le_wafer_id, 1)
        h_wafer.addWidget(btn_fetch_map)
        form_product.addRow("Wafer ID:", h_wafer)
        dock_layout.addWidget(gb_product)

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

        # Pixel size: chip / layout declarations in um -> pixels, um columns of the exports
        self.sb_um_per_px = QDoubleSpinBox(); self.sb_um_per_px.setRange(0, 100000); self.sb_um_per_px.setDecimals(4)
        self.sb_um_per_px.setSingleStep(0.1); self.sb_um_per_px.setSuffix(" um/px")
        self.sb_um_per_px.setSpecialValueText("not set") # 0
        self.sb_um_per_px.setToolTip("How many um one image pixel is. Converts the product's chip size (um) to the\n"
                                     "bbox size in pixels and fills the um columns of voids.json / voids.csv.")
        self.sb_um_per_px.valueChanged.connect(self.on_scale_changed)
        form.addRow("Scale:", self.sb_um_per_px)
        
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(50)
        form.addRow("Opacity:", self.slider_opacity)
        
        # Angle alignment preview (cells spread over the wafer)
        self.angle_preview = None # Small, always visible at the bottom of this group (created at the end of __init__)
        self.form_grid = form
        
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

        # GPU memory for cached layers (least recently used layers are dropped beyond this)
        self.sb_gpu_cache = QSpinBox(); self.sb_gpu_cache.setRange(256, 65536); self.sb_gpu_cache.setSingleStep(256)
        self.sb_gpu_cache.setValue(2048); self.sb_gpu_cache.setSuffix(" MB")
        form_load.addRow("GPU Cache:", self.sb_gpu_cache)
        
        dock_layout.addWidget(gb_load)
        
        # 2. Bonding Map Paste
        gb_map = QGroupBox("Bonding Map (Paste Excel)")
        v_map = QVBoxLayout(gb_map)
        
        # Map values to exclude (treated as not bonded), comma separated
        h_reject = QHBoxLayout()
        h_reject.addWidget(QLabel("Reject Values:"))
        self.le_reject = QLineEdit()
        self.le_reject.setPlaceholderText("e.g. X, NG, 0")
        self.le_reject.setToolTip("Bonding map values to exclude, comma separated. Press Enter to apply.")
        self.le_reject.editingFinished.connect(self.import_map_data)
        h_reject.addWidget(self.le_reject, 1)
        v_map.addLayout(h_reject)
        
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

        # Results API: send the markings to / load them from another system (plugins/results_api.py)
        h_api = QHBoxLayout()
        h_api.addWidget(QLabel("Results API:"))
        btn_api_preview = QPushButton("Preview")
        btn_api_preview.setToolTip("Show the payload that Send would send (build_payload of plugins/results_api.py)")
        btn_api_preview.clicked.connect(self.preview_results_payload)
        self.btn_api_send = QPushButton("Send")
        self.btn_api_send.setToolTip("Send the markings (send of plugins/results_api.py, config.json \"results_api\")")
        self.btn_api_send.clicked.connect(self.send_results)
        self.btn_api_load = QPushButton("Load")
        self.btn_api_load.setToolTip("Load markings from the other system (fetch + to_voids of plugins/results_api.py); Ctrl+Z undoes")
        self.btn_api_load.clicked.connect(self.load_results_from_api)
        for b in (btn_api_preview, self.btn_api_send, self.btn_api_load):
            h_api.addWidget(b)
        v_ext.addLayout(h_api)
        self.last_export_dir = None
        # (added at the bottom of the dock, below Void Marking)
        
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
        self.glw.cell_step_requested.connect(self.step_bonded_cell)
        self.glw.cursor_moved.connect(self.on_cursor_moved)
        
        # 3.5 Void Marking Control
        self.void_manager = VoidManager(declaration_path("config.json"))
        self.glw.void_manager = self.void_manager
        self.glw.voids_changed.connect(self.update_void_ui)
        self.glw.void_type_shortcut_triggered.connect(self.on_void_type_shortcut)
        
        self.gb_void = QGroupBox("Void Marking")
        v_layout_void = QVBoxLayout(self.gb_void)
        
        # Toggle Mode (Hint Only)
        self.lbl_void_hint = QLabel("Hold W: Draw  E: Edit  R: Erase (release: View)\n"
                                    "1-9, 10..: type of the hovered void (else of new voids)\n"
                                    "Q: move hovered void to this layer  |  Pan: right/middle drag")
        self.lbl_void_hint.setStyleSheet("font-weight: bold; color: gray;")
        
        # Mode Selection (Draw/Edit/Erase)
        h_modes = QHBoxLayout()
        self.rb_view = QRadioButton("View")
        self.rb_draw = QRadioButton("Draw")
        self.rb_edit = QRadioButton("Edit")
        self.rb_erase = QRadioButton("Erase")
        self.rb_view.setChecked(True)
        
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_view)
        self.mode_group.addButton(self.rb_draw)
        self.mode_group.addButton(self.rb_edit)
        self.mode_group.addButton(self.rb_erase)
        
        # Indicators only: a mode is active while its key is held (see eventFilter)
        for rb in (self.rb_view, self.rb_draw, self.rb_edit, self.rb_erase):
            rb.setAttribute(Qt.WA_TransparentForMouseEvents)
            rb.setFocusPolicy(Qt.NoFocus)
        
        h_modes.addWidget(self.rb_view)
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
        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.setToolTip("Removes every marking on all layers, bboxes included (Ctrl+Z restores)")
        self.btn_clear_all.clicked.connect(self.clear_all_voids)

        h_layout_void_btns.addWidget(self.btn_load_voids)
        h_layout_void_btns.addWidget(self.btn_save_voids)
        h_clear_btns = QHBoxLayout()
        h_clear_btns.addWidget(self.btn_clear_voids)
        h_clear_btns.addWidget(self.btn_clear_chip)
        h_clear_btns.addWidget(self.btn_clear_all)

        v_layout_void.addWidget(self.lbl_void_hint)
        v_layout_void.addLayout(h_modes)
        v_layout_void.addLayout(h_type)
        v_layout_void.addWidget(self.lbl_void_count)
        v_layout_void.addLayout(h_layout_void_btns)
        v_layout_void.addLayout(h_clear_btns)

        # Rough automatic detection (die-to-die comparison), marked with the selected type
        h_detect = QHBoxLayout()
        btn_detect = QPushButton("Detect Voids")
        btn_detect.setToolTip("Compare every chip with the median chip of its layer (layout-independent), from the top\n"
                              "layer down; each void is marked once, on the layer where it first appears, with the\n"
                              "selected type. Brighter and darker spots both count. Results are dashed until edited.")
        btn_detect.clicked.connect(self.detect_voids)
        self.sb_detect_thr = QDoubleSpinBox(); self.sb_detect_thr.setRange(2.0, 50.0); self.sb_detect_thr.setSingleStep(0.5)
        self.sb_detect_thr.setValue(float(self._config().get("void_detect", {}).get("threshold", void_detect.DEFAULTS["threshold"])))
        self.sb_detect_thr.setPrefix("Thr ")
        self.sb_detect_thr.setToolTip("Deviation from the median chip, in units of its chip-to-chip spread. Lower = more sensitive.")
        btn_next_det = QPushButton("Next Detected")
        btn_next_det.setToolTip("Go to the next detected void that was not reviewed (edited) yet, layer by layer")
        btn_next_det.clicked.connect(self.next_detected_void)
        # Scorer: die-to-die (default) or the optional PatchCore (onnxruntime + models/*.onnx)
        self.cb_detect_method = QComboBox()
        self.cb_detect_method.addItem("Die-to-die", ("die_to_die", None))
        self.cb_detect_method.addItem("PatchCore MobileNetV2", ("patchcore", "models/patchcore_mobilenetv2.onnx"))
        self.cb_detect_method.addItem("PatchCore ResNet50", ("patchcore", "models/patchcore_resnet50.onnx"))
        self.cb_detect_method.setToolTip("Die-to-die: pixel comparison with the median chip (fast, exact outlines).\n"
                                         "PatchCore: CNN features (optional, slower, coarse outlines), see docs/PATCHCORE.md.")
        self._detect_thr = {}
        self.cb_detect_method.currentIndexChanged.connect(self._on_detect_method)
        h_detect.addWidget(btn_detect)
        h_detect.addWidget(self.cb_detect_method)
        h_detect.addWidget(self.sb_detect_thr)
        h_detect.addWidget(btn_next_det)
        btn_eval = QPushButton("Evaluate")
        btn_eval.setToolTip("Run the detection without marking and compare it with the voids marked by hand on the loaded\n"
                            "layers: misses and false detections with their reasons, as a short code to type and send,\n"
                            "plus a report folder with chip images (see docs/VOID_EVAL.md).")
        btn_eval.clicked.connect(self.evaluate_detection)
        h_detect.addWidget(btn_eval)
        v_layout_void.addLayout(h_detect)
        # Layer of a spot found on several layers
        h_rule = QHBoxLayout()
        h_rule.addWidget(QLabel("Mark on layer:"))
        self.cb_layer_rule = QComboBox()
        for label, key in (("First (from top)", "first"), ("Strongest", "strongest"), ("Last", "last")):
            self.cb_layer_rule.addItem(label, key)
        self.cb_layer_rule.setToolTip("A spot found on several layers is marked once, on:\n"
                                      "First: the first layer where it appears (from the top)\n"
                                      "Strongest: the layer where it deviates most\n"
                                      "Last: the deepest layer where it appears")
        rule = str(self._config().get("void_detect", {}).get("layer_rule", "first")).lower()
        self.cb_layer_rule.setCurrentIndex(max(0, self.cb_layer_rule.findData(rule)))
        h_rule.addWidget(self.cb_layer_rule, 1)
        h_rule.addWidget(QLabel("Skip layers:"))
        self.le_skip_layers = QLineEdit()
        self.le_skip_layers.setToolTip("Absolute layer numbers Detect Voids does not search, e.g. 0 or 0, 1\n"
                                       "(the top layer is usually noisy: surface echo)")
        skip = self._config().get("void_detect", {}).get("skip_layers", void_detect.DEFAULTS["skip_layers"])
        self.le_skip_layers.setText(", ".join(map(str, skip)) if isinstance(skip, list) else str(skip))
        self.le_skip_layers.setMaximumWidth(80)
        h_rule.addWidget(self.le_skip_layers)
        v_layout_void.addLayout(h_rule)
        self._init_detect_method()
        
        dock_layout.addWidget(self.gb_void)
        
        # Init Types
        self.populate_void_types()
        
        # Shortcuts for Void Mode
    
    
        self._held_modes = [] # Mode keys held down, in press order
        QApplication.instance().installEventFilter(self)
        QApplication.instance().applicationStateChanged.connect(self._on_app_state)

        self.action_void_view = QAction("View Mode", self)
        self.action_void_view.setShortcut("Esc")
        self.action_void_view.triggered.connect(self.set_void_mode_view)
        self.addAction(self.action_void_view)

        # Q (handled in eventFilter, like the mode keys: works whatever widget has the focus)
        self.action_void_to_layer = QAction("Move Hovered Void To Current Layer", self)
        self.action_void_to_layer.triggered.connect(self.glw.move_hovered_void_to_current_layer)


        
        # Undo of void marking changes (add / move / resize / erase / clear / load / p)
        self.action_undo = QAction("Undo Void Change", self)
        self.action_undo.setShortcut(QKeySequence.Undo) # Text boxes with focus keep their own Ctrl+Z
        self.action_undo.triggered.connect(self.undo_void_change)
        self.addAction(self.action_undo)



        
        # 4. Auto Calibration
        gb_calib = QGroupBox("Auto Calibration")
        form_calib = QFormLayout(gb_calib)
        
        self.chk_use_calib = QPushButton("Enable Auto-Calib: OFF")
        self.chk_use_calib.setCheckable(True)
        self.chk_use_calib.toggled.connect(self.toggle_calib)
        form_calib.addRow(self.chk_use_calib)
        
        self.chk_calib_robust = QPushButton("Robust Mode (Ignore Voids): ON")
        self.chk_calib_robust.setCheckable(True)
        self.chk_calib_robust.setChecked(True)
        self.chk_calib_robust.clicked.connect(lambda c: self.chk_calib_robust.setText(f"Robust Mode (Ignore Voids): {'ON' if c else 'OFF'}"))
        self.chk_calib_robust.clicked.connect(self.calculate_calibration)
        form_calib.addRow(self.chk_calib_robust)
        
        self.chk_calib_all = QPushButton("Apply to All Layers: ON")
        self.chk_calib_all.setCheckable(True)
        self.chk_calib_all.setChecked(True)
        self.chk_calib_all.clicked.connect(lambda c: self.chk_calib_all.setText(f"Apply to All Layers: {'ON' if c else 'OFF'}"))
        form_calib.addRow(self.chk_calib_all)
        
        self.sb_target_mean = QDoubleSpinBox(); self.sb_target_mean.setRange(0, 65535); self.sb_target_mean.setValue(30000)
        self.sb_target_std = QDoubleSpinBox(); self.sb_target_std.setRange(0, 65535); self.sb_target_std.setValue(10000)
        form_calib.addRow("Target Mean:", self.sb_target_mean)
        form_calib.addRow("Target Std:", self.sb_target_std)
        
        btn_recalc = QPushButton("Recalculate")
        btn_recalc.clicked.connect(self.calculate_calibration)
        form_calib.addRow(btn_recalc)

        # Reference bboxes (topmost layer) and review of the estimated ones
        h_auto = QHBoxLayout()
        btn_auto_bbox = QPushButton("Detect BBOX")
        btn_auto_bbox.setToolTip("Detect the package edges of every (bonded) cell on the topmost layer and draw the bbox (declared size, detected center)")
        btn_auto_bbox.clicked.connect(self.auto_bbox)
        btn_next_est = QPushButton("Next Estimated")
        btn_next_est.setToolTip("Go to the next bbox that was estimated because detection failed (drawn dashed)")
        btn_next_est.clicked.connect(self.next_estimated_bbox)
        h_auto.addWidget(btn_auto_bbox)
        h_auto.addWidget(btn_next_est)
        form_calib.addRow(h_auto)

        # Panel order: ... Bonding Map, Auto Calibration, Coordinates, Void Marking, Extraction
        dock_layout.insertWidget(dock_layout.indexOf(gb_map) + 1, gb_calib)
        dock_layout.addWidget(gb_extract)
        dock_layout.addStretch()
        

        # Scroll Area Wrapper
        scroll = QScrollArea()
        scroll.setWidget(dock_content)
        scroll.setWidgetResizable(True)
        # scroll.setMaximumHeight(1000) # Removed fixed limit
        
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        dock.setWidget(scroll)

        # Sync initial state
        self.on_window_changed()
        self.update_grid_params()
        
        # Default values of the inputs (config.json "ui_defaults")
        self._apply_ui_defaults()

        # WF products (declared in products.json next to the program)
        self.load_product_list()
        self.cb_product.currentTextChanged.connect(self.apply_product)
        if self.cb_product.count() > 0:
            self.apply_product(self.cb_product.currentText())
        
        self._create_angle_preview()

        # Newer build available? (location declared in config.json "update.manifest")
        manifest = self._config().get("update", {}).get("manifest") or default_update_manifest()
        self.update_checker = UpdateChecker(APP_VERSION, manifest, self)
        self.update_checker.update_available.connect(lambda m: show_update_dialog(self, APP_VERSION, m))
        QTimer.singleShot(1500, self.update_checker.start)
        self.statusBar().showMessage(f"Declarations: {self.declaration_status}")

    def reload_declarations(self):
        """Re-fetches the declarations and applies void types, output columns and products."""
        self.declaration_status = sync_declarations()
        self.void_manager.config_path = declaration_path("config.json")
        self.void_manager.load_config()
        self.populate_void_types()
        self._apply_ui_defaults()
        current = self.cb_product.currentText()
        self.load_product_list()
        if current in self.products:
            self.cb_product.setCurrentText(current)
        if self.cb_product.count() > 0:
            self.apply_product(self.cb_product.currentText())
        self.glw.update()
        self.statusBar().showMessage(f"Declarations reloaded: {self.declaration_status}")

    def _ui_default_setters(self):
        """config.json "ui_defaults" key -> setter of the input (see README "UI 기본값")."""
        def toggle(btn, label):
            def f(v):
                btn.setChecked(bool(v))
                btn.setText(f"{label}: {'ON' if v else 'OFF'}")
            return f

        def void_type(v):
            i = self.cb_void_type.findData(self.void_manager.resolve_type(v))
            if i < 0:
                raise ValueError(f"unknown void type {v!r}")
            self.cb_void_type.setCurrentIndex(i)

        def layer_rule(v):
            i = self.cb_layer_rule.findData(str(v).lower())
            if i < 0:
                raise ValueError(f"expected first / strongest / last, got {v!r}")
            self.cb_layer_rule.setCurrentIndex(i)

        return {
            "reject_values": lambda v: self.le_reject.setText(", ".join(map(str, v)) if isinstance(v, list) else str(v)),
            "gpu_cache_mb": lambda v: self.sb_gpu_cache.setValue(int(v)),
            "start_layer": lambda v: self.sb_start_layer.setValue(int(v)),
            "end_layer": lambda v: self.sb_end_layer.setValue(int(v)),
            "show_grid": lambda v: self.chk_grid_show.setChecked(bool(v)),
            "grid_opacity": lambda v: self.slider_opacity.setValue(int(v)),
            "um_per_px": lambda v: self.sb_um_per_px.setValue(float(v)),
            "calib_enabled": lambda v: self.chk_use_calib.setChecked(bool(v)),
            "calib_robust": toggle(self.chk_calib_robust, "Robust Mode (Ignore Voids)"),
            "calib_all_layers": toggle(self.chk_calib_all, "Apply to All Layers"),
            "target_mean": lambda v: self.sb_target_mean.setValue(float(v)),
            "target_std": lambda v: self.sb_target_std.setValue(float(v)),
            "void_type": void_type,
            "layer_rule": layer_rule,
            "skip_layers": lambda v: self.le_skip_layers.setText(", ".join(map(str, v)) if isinstance(v, list) else str(v)),
            "export": lambda v: None, # Read by the export dialog when it opens
        }

    def _apply_ui_defaults(self):
        """Sets the inputs from config.json "ui_defaults" (the program never writes config.json)."""
        setters = self._ui_default_setters()
        for key, value in (self._config().get("ui_defaults") or {}).items():
            if key not in setters:
                print(f"[ui_defaults] unknown key '{key}' ignored. Valid keys: {', '.join(setters)}")
                continue
            try:
                setters[key](value)
            except Exception as e:
                print(f"[ui_defaults] {key} = {value!r}: {e}")

    def _config(self):
        try:
            with open(declaration_path("config.json"), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # -----------------------------------------------
    # WF Products
    # -----------------------------------------------
    def load_product_list(self):
        self.products, default = load_products(declaration_path("products.json"))
        self.cb_product.blockSignals(True)
        self.cb_product.clear()
        self.cb_product.addItems(list(self.products))
        if default:
            self.cb_product.setCurrentText(default)
        self.cb_product.blockSignals(False)
        self.cb_product.setEnabled(bool(self.products))
        if not self.products:
            print(f"No products defined ({declaration_path('products.json')})")

    def apply_product(self, name):
        """Applies the declared grid values, scale, bbox default size and layout of a product."""
        prod = self.products.get(name)
        if prod is None:
            return
        self.current_product = prod
        
        # Grid: only declared keys
        grid = prod.get("grid", {})
        spin = {"start_x": self.sb_grid_x, "start_y": self.sb_grid_y, "pitch_x": self.sb_pitch_x,
                "pitch_y": self.sb_pitch_y, "rows": self.sb_rows, "cols": self.sb_cols, "angle": self.sb_angle}
        for key in GRID_KEYS:
            if key in grid:
                spin[key].blockSignals(True)
                spin[key].setValue(grid[key])
                spin[key].blockSignals(False)
        
        if "um_per_px" in prod:
            self.sb_um_per_px.blockSignals(True)
            self.sb_um_per_px.setValue(float(prod["um_per_px"]))
            self.sb_um_per_px.blockSignals(False)
        self.void_manager.um_per_px = self.sb_um_per_px.value()
        self._apply_bbox_size()

        # Layout inside each bbox (also provides snap offsets such as SL). Declared in chip
        # units (um when chip_size_um is given) and scaled to each drawn bbox.
        layout = prod.get("layout")
        chip = prod.get("chip_size_um")
        if layout and chip and "design_size" not in layout:
            layout = dict(layout, design_size=list(chip))
        self.glw.layout = layout
        self.void_manager.layout = layout
        
        self.update_grid_params()
        print(f"Product: {name}")

    def _apply_bbox_size(self):
        """
        Default size of the bbox type for the selected product (session only, not written to
        config.json): chip_size_um / scale, or the legacy "bbox": {"w", "h"} in pixels.
        No scale -> 0 (drag to size; Detect BBOX uses the detected size). A product declaring
        neither gets the size declared in config.json (not the previous product's).
        """
        name = self._bbox_type_name()
        if name is None:
            return
        prod = getattr(self, "current_product", None) or {}
        chip = prod.get("chip_size_um")
        if chip:
            s = self.sb_um_per_px.value()
            w, h = (chip[0] / s, chip[1] / s) if s > 0 else (0.0, 0.0)
        elif prod.get("bbox"):
            w, h = prod["bbox"].get("w", 0.0), prod["bbox"].get("h", 0.0)
        else:
            declared = VoidManager.normalize_types(self._config().get("void_types", {})).get(name, {})
            w, h = declared.get("def_w", 0.0), declared.get("def_h", 0.0)
        t = self.void_manager.types[name]
        t["def_w"], t["def_h"] = float(w), float(h)
        # Existing bboxes follow the product size (centers kept); corner voids follow the bboxes
        if self.void_manager.apply_bbox_size():
            self.void_manager.resnap(self.glw.grid_cfg, self.glw.coord_transform)
            self.glw.update()

    def on_scale_changed(self, value):
        self.void_manager.um_per_px = value
        self._apply_bbox_size()
        if self.glw.void_mode:
            self.glw.update_cursor_style()
        self.glw.update()

    def undo_void_change(self):
        if self.void_manager.undo():
            self.glw.active_void = None
            self.glw.refresh_hover() # Undo restores copies of the voids
            self.update_void_ui()

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

    def _create_angle_preview(self):
        """Angle preview: small and always visible at the bottom of Grid Configuration."""
        self.angle_preview = AnglePreview(self)
        lbl = QLabel("Angle Preview")
        lbl.setStyleSheet("font-weight: bold;")
        self.form_grid.addRow(lbl)
        self.form_grid.addRow(self.angle_preview)

    def _refresh_angle_preview(self):
        if getattr(self, 'angle_preview', None) is not None:
            self.angle_preview.schedule_update()

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

        self._refresh_angle_preview()

    def on_layer_wheel_scroll(self, delta):
        # delta is +1 or -1
        if not self.spin_layer.isEnabled():
            return
            
        current = self.spin_layer.value()
        new_val = current + delta
        # Spinbox handles clamping but let's be safe
        new_val = max(self.spin_layer.minimum(), min(self.spin_layer.maximum(), new_val))
        self.spin_layer.setValue(new_val)

    def reject_values(self):
        return [v.strip() for v in self.le_reject.text().split(",") if v.strip()]

    def import_map_data(self):
        text = self.txt_map.toPlainText()
        if not text.strip():
            return
            
        bmap = BondingMap()
        bmap.parse_text(text, self.reject_values())
        
        if bmap.rows > 0 and bmap.cols > 0:
            self.glw.bonding_map = bmap
            self.glw.update_map_texture()
            
            # Auto-update grid dimensions to match map
            self.sb_rows.setValue(bmap.rows)
            self.sb_cols.setValue(bmap.cols)
            
            # Enable grid if not
            if not self.chk_grid_show.isChecked():
                self.chk_grid_show.setChecked(True)
                
            print(f"Map Imported: {bmap.rows}x{bmap.cols}, Values: {len(bmap.unique_keys)}, Rejected: {self.reject_values()}")
            
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
            
        if not self._confirm_discard_voids():
            return

        # 1. Init Mosaic Stack
        try:
            stack = MosaicTiffStack(p_json)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Open Project", f"Failed to load project:\n{e}")
            return

        self._close_data()
        self.full_data = stack
        self.layer_offset = stack.layer_number(0)
        self.glw.bbox_layer = self.layer_offset
        self.void_manager.replace_all()

        # 2. Setup UI
        self._setup_layer_controls()

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
        
        # 3.5 Bonding map saved with the export (bonding_map.txt; reject values alongside)
        bm_txt = os.path.join(folder, "bonding_map.txt")
        if os.path.exists(bm_txt):
            try:
                with open(bm_txt, "r", encoding="utf-8") as f:
                    text = f.read()
                rej = os.path.join(folder, "bonding_map_reject.txt")
                reject = ""
                if os.path.exists(rej):
                    with open(rej, "r", encoding="utf-8") as f:
                        reject = f.read().strip()
                self.le_reject.setText(reject) # The project's own values (none saved: none)
                self.txt_map.setPlainText(text)
                self.import_map_data()
            except Exception as e:
                print(f"Bonding map not loaded ({bm_txt}): {e}")

        # 4. Load Initial Layer (also sets up the coordinate transform for the void loader)
        self.load_layer(0)

        # 5. Load Voids (Raw)
        v_json = os.path.join(folder, "voids.json")
        if os.path.exists(v_json):
            # Virtual Raw Strategy: Mosaic is the Raw Image. Voids are in Raw Coords.
            self.void_manager.load_from_file(v_json, self.glw.grid_cfg, self.glw.coord_transform, self.layer_offset)
            self._scale_from_file()
            self.void_manager.reset_history()
            self.populate_void_types()
        self.update_void_ui()

        self._start_layer_loader()
        try:
            with open(p_json, "r", encoding="utf-8") as f:
                saved_wafer = json.load(f).get("wafer_id")
        except Exception:
            saved_wafer = None
        # The project's own bonding map wins over a server lookup
        self._set_source(folder, saved_wafer, allow_fetch=not os.path.exists(os.path.join(folder, "bonding_map.txt")))
        self.glw.update()

    # -----------------------------------------------
    # Source file / wafer ID / bonding map lookup
    # -----------------------------------------------
    def _set_source(self, path, wafer_id=None, allow_fetch=True):
        """Shows the opened file and fills the wafer ID (saved one, else parsed from the name)."""
        name = os.path.basename(path.rstrip("/\\"))
        self.lbl_source.setText(name)
        self.lbl_source.setToolTip(path)
        self.le_wafer_id.setText(wafer_id or self.parse_wafer_id(name))
        cfg = (self._results_settings().get("bonding_map") or {})
        if allow_fetch and cfg.get("auto_fetch") and self.le_wafer_id.text().strip():
            self.fetch_bonding_map()

    def parse_wafer_id(self, file_name):
        """
        Wafer ID from a file / folder name: plugin parse_wafer_id(file_name, settings) if it has
        one, else config.json "wafer_id": {"pattern": regex, "group": 1 or name}, else the name
        without extension.
        """
        stem = os.path.splitext(file_name)[0]
        settings = self._results_settings()
        try:
            plugin = results_api.load_plugin(settings)
            if callable(getattr(plugin, "parse_wafer_id", None)):
                wid = plugin.parse_wafer_id(file_name, settings)
                if wid:
                    return str(wid)
        except Exception as e:
            print(f"[wafer_id] plugin parse_wafer_id: {e}")
        rule = self._config().get("wafer_id") or {}
        pattern = rule.get("pattern")
        if pattern:
            try:
                m = re.search(pattern, file_name)
                if m:
                    return m.group(rule.get("group", 1 if m.groups() else 0))
                print(f"[wafer_id] pattern {pattern!r} does not match {file_name!r}")
            except (re.error, IndexError) as e:
                print(f"[wafer_id] pattern {pattern!r}: {e}")
        return stem

    def fetch_bonding_map(self):
        """Bonding map of the wafer ID from another server (plugin fetch_bonding_map, background)."""
        wafer_id = self.le_wafer_id.text().strip()
        if not wafer_id:
            QMessageBox.information(self, "Fetch Map", "Enter a wafer ID first.")
            return
        plugin, settings = self._results_plugin()
        if plugin is None:
            return
        if not callable(getattr(plugin, "fetch_bonding_map", None)):
            QMessageBox.warning(self, "Fetch Map", f"{results_api.plugin_path(settings)} has no fetch_bonding_map().")
            return
        ctx = results_api.build_context(self, settings)
        self.btn_fetch_map.setEnabled(False)
        self.statusBar().showMessage(f"Bonding map: fetching {wafer_id}...")
        job = results_api.BackgroundCall(lambda: plugin.fetch_bonding_map(ctx, settings), self)
        job.done.connect(lambda result, error: self._on_bonding_map_fetched(job, wafer_id, result, error))
        self._map_job = job

    def _on_bonding_map_fetched(self, job, wafer_id, result, error):
        self.btn_fetch_map.setEnabled(True)
        job.deleteLater()
        if error:
            self.statusBar().showMessage("Bonding map: fetch failed")
            self._results_error("fetch_bonding_map() failed.", error)
            return
        try:
            text, reject = results_api.normalize_bonding_map(result)
        except Exception:
            self._results_error("fetch_bonding_map() returned an unknown format.", traceback.format_exc())
            return
        if reject is not None:
            self.le_reject.setText(", ".join(reject))
        self.txt_map.setPlainText(text)
        self.import_map_data()
        bmap = self.glw.bonding_map
        size = f"{bmap.rows}x{bmap.cols}" if bmap is not None else "not parsed"
        self.statusBar().showMessage(f"Bonding map of {wafer_id}: {size}")

    def step_bonded_cell(self, direction=1):
        """
        Right click: the next cell with a bonding map value (left to right, then the next row;
        Shift + right click: previous), wrapping around. All grid cells without a bonding map.
        """
        cfg = self.glw.grid_cfg
        bmap = self.glw.bonding_map
        cells = [(r, c) for r in range(cfg.rows) for c in range(cfg.cols) if bmap is None or bmap.get_key(r, c)]
        if not cells:
            self.statusBar().showMessage("No cell with a bonding map value.")
            return
        here = tuple(self.glw.center_cell())
        if direction > 0:
            nxt = next((rc for rc in cells if rc > here), cells[0])
        else:
            nxt = next((rc for rc in reversed(cells) if rc < here), cells[-1])
        r, c = nxt
        self.glw.fit_to_cell(c, r)
        leg = bmap.get_key(r, c) if bmap is not None else "-"
        self.statusBar().showMessage(f"Cell {cells.index(nxt) + 1}/{len(cells)}: X{c:02d} Y{r:02d}  LEG {leg}")

    def auto_level(self):
        c = self.sb_jump_x.value()
        r = self.sb_jump_y.value()
        self.glw.fit_to_cell(c, r)
            


    MODE_KEYS = {Qt.Key_W: "DRAW", Qt.Key_E: "EDIT", Qt.Key_R: "ERASE"}

    TEXT_INPUTS = (QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox)

    def _typing(self, obj):
        """True when the key goes to a text input (it keeps its keys)."""
        return isinstance(obj, self.TEXT_INPUTS) or isinstance(QApplication.focusWidget(), self.TEXT_INPUTS)

    def eventFilter(self, obj, ev):
        # Application-wide, so the keys work whatever widget of this window has the focus.
        # W / E / R: the mode is active only while its key is held (last held key wins).
        # Digits: type number (gl_widget.type_number_key). Q: move the hovered void here.
        et = ev.type()
        if et in (QEvent.KeyPress, QEvent.KeyRelease) and isinstance(obj, QWidget) and obj.window() is self:
            key = ev.key()
            plain = not (ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier))
            if key in self.MODE_KEYS and plain and not self._typing(obj):
                if not ev.isAutoRepeat():
                    tool = self.MODE_KEYS[key]
                    if tool in self._held_modes:
                        self._held_modes.remove(tool)
                    if et == QEvent.KeyPress:
                        self._held_modes.append(tool)
                    self._apply_held_mode()
                return True
            if et == QEvent.KeyPress and Qt.Key_0 <= key <= Qt.Key_9 and plain and not self._typing(obj):
                if not ev.isAutoRepeat():
                    self.glw.type_number_key(key - Qt.Key_0)
                return True
            if et == QEvent.KeyPress and key == Qt.Key_Q and plain and not self._typing(obj):
                if not ev.isAutoRepeat():
                    self.action_void_to_layer.trigger()
                return True
        return super().eventFilter(obj, ev)

    def _on_app_state(self, state):
        # Key releases are lost while another application is active
        if state != Qt.ApplicationActive and self._held_modes:
            self._held_modes.clear()
            self._apply_held_mode()

    def _apply_held_mode(self):
        tool = self._held_modes[-1] if self._held_modes else None
        rb = {None: self.rb_view, "DRAW": self.rb_draw, "EDIT": self.rb_edit, "ERASE": self.rb_erase}[tool]
        rb.setChecked(True)
        self.glw.set_void_tool(tool)

    def set_void_mode_view(self):
        self._held_modes.clear()
        self._apply_held_mode()
        
    def load_voids(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Voids JSON", "", "JSON (*.json)")
        if path:
            self.void_manager.load_from_file(path, self.glw.grid_cfg, self.glw.coord_transform, self.layer_offset)
            self._scale_from_file()
            self.glw.refresh_hover()
            self.populate_void_types() # Refresh types
            self.update_void_ui()
            self.glw.update()
            
    def _scale_from_file(self):
        """A voids.json saved with a scale sets the Scale field."""
        s = self.void_manager.file_um_per_px
        if s and s > 0 and abs(s - self.sb_um_per_px.value()) > 1e-9:
            self.sb_um_per_px.setValue(s)
            print(f"Scale from file: {s} um/px")

    def populate_void_types(self):
        self.cb_void_type.blockSignals(True)
        self.cb_void_type.clear()
        
        # Declaration order (number keys 1..9, 10.. follow it)
        for i, name in enumerate(self.void_manager.types, 1):
            self.cb_void_type.addItem(f"{i}: {name}", name)
            
        self.cb_void_type.blockSignals(False)
        
        # Set first as default if available
        if self.cb_void_type.count() > 0:
            self.cb_void_type.setCurrentIndex(0)
            self.on_void_type_changed(0)
            
    def on_void_type_changed(self, idx):
        if idx < 0: return
        name = self.cb_void_type.currentData()
        self.glw.active_type = name
        print(f"Active Void Type: {name}")
        
        # Update cursor style if void mode is active
        if self.glw.void_mode:
            self.glw.update_cursor_style()

    def on_void_type_shortcut(self, type_name):
        """Called when a number key selects the drawing type: sync the ComboBox."""
        idx = self.cb_void_type.findData(type_name)
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
        self.void_manager.clear_layer(self.glw.current_layer)
        self.update_void_ui()
        self.glw.active_void = None
        self.glw.setFocus()
        self.glw.update()
        
    def clear_chip_voids(self):
        self.glw.clear_current_chip_voids()
        self.update_void_ui()
        self.glw.setFocus()

    def clear_all_voids(self):
        """Removes every marking on all layers (bboxes too) after a confirmation. One undo step."""
        n = sum(len(vs) for vs in self.void_manager.voids.values())
        if n == 0:
            return
        if QMessageBox.question(self, "Clear All",
                                f"Remove all {n} markings on all layers (bboxes included)?\nCtrl+Z restores them.",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.void_manager.remove_voids_where(lambda layer, v: True)
        self.glw.active_void = None
        self.update_void_ui()
        self.glw.refresh_hover()
        self.glw.setFocus()

    def _bbox_type_name(self):
        for name in self.void_manager.types:
            if name.strip().lower() == self.void_manager.BBOX_TYPE_NAME:
                return name
        return None

    def auto_bbox(self):
        """Detects and draws the reference bbox of every (bonded) cell on the topmost layer."""
        if self.full_data is None or self.glw.tiled_image is None:
            return
        tid = self._bbox_type_name()
        if tid is None:
            QMessageBox.warning(self, "Detect BBOX", "No void type named 'bbox' is declared in config.json.")
            return
        t = self.void_manager.types[tid]
        size = (t["def_w"], t["def_h"]) if t.get("def_w", 0) > 0 and t.get("def_h", 0) > 0 else None
        cfg = self.glw.grid_cfg
        bmap = self.glw.bonding_map
        cells = [(r, c) for r in range(cfg.rows) for c in range(cfg.cols) if bmap is None or bmap.get_key(r, c)]

        progress = QProgressDialog("Detecting bboxes...", "Cancel", 0, len(cells), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def step(i):
            if i % 10 == 0:
                progress.setValue(i)
                QApplication.processEvents()
            return not progress.wasCanceled()

        top_layer = self.abs_layer(0)
        vm = self.void_manager
        vm.begin_change() # One undo step with the corner voids following the new bboxes
        try:
            result = place_bboxes(self.full_data, 0, top_layer, cfg, self.glw.coord_transform, vm,
                                  tid, size, cells, self._config().get("auto_bbox"), step)
            if result is not None:
                vm.resnap(cfg, self.glw.coord_transform)
        finally:
            vm.end_change()
            progress.close()
        if result is None:
            return
        self.glw.update()
        self.update_void_ui()
        msg = (f"Detect BBOX on layer {top_layer}: {result['detected']} detected, "
               f"{len(result['estimated'])} estimated (dashed, check with 'Next Estimated'), "
               f"{len(result['failed'])} failed")
        self.statusBar().showMessage(msg)
        print(msg)
        if result["failed"] and not result["detected"]:
            QMessageBox.warning(self, "Detect BBOX", "No box edge could be detected. Check the grid origin/pitch "
                                "and the bbox size, or adjust auto_bbox in config.json.")

    def next_estimated_bbox(self):
        """Navigates through the bboxes estimated by Detect BBOX (row-major order)."""
        cfg = self.glw.grid_cfg
        cells = sorted({self.glw.raw_to_cell(v["globalCX"], v["globalCY"])
                        for vs in self.void_manager.voids.values() for v in vs if v.get("auto") == "estimated"})
        if not cells:
            self.statusBar().showMessage("No estimated bboxes left.")
            return
        here = self.glw.center_cell()
        nxt = next((rc for rc in cells if rc > here), cells[0])
        self.glw.fit_to_cell(nxt[1], nxt[0])
        self.statusBar().showMessage(f"Estimated bbox {cells.index(nxt) + 1}/{len(cells)}: X{nxt[1]:02d} Y{nxt[0]:02d}")
        
    def _init_detect_method(self):
        """PatchCore items are enabled when onnxruntime and their model file are there."""
        from sat_widgets import patchcore
        cfg = self._config().get("void_detect", {})
        custom = cfg.get("pc_model")
        if custom and all(self.cb_detect_method.itemData(i)[1] != custom for i in range(self.cb_detect_method.count())):
            self.cb_detect_method.addItem("PatchCore (config.json)", ("patchcore", custom))
        start = 0
        for i in range(1, self.cb_detect_method.count()):
            model = self.cb_detect_method.itemData(i)[1]
            reason = patchcore.available({"pc_model": model})
            item = self.cb_detect_method.model().item(i)
            item.setEnabled(not reason)
            item.setToolTip(reason or "CNN features, position-aware nearest neighbour (CPU), see docs/PATCHCORE.md")
            if not reason and cfg.get("method") == "patchcore" and model == (custom or patchcore.DEFAULTS["pc_model"]):
                start = i
        self.cb_detect_method.setCurrentIndex(start)
        self._on_detect_method()

    def _on_detect_method(self, *_):
        """Each method keeps its own threshold (config.json "void_detect": threshold / pc_threshold)."""
        from sat_widgets import patchcore
        cfg = self._config().get("void_detect", {})
        idx = self.cb_detect_method.currentIndex()
        method = self.cb_detect_method.currentData()[0]
        last = getattr(self, "_detect_last", None)
        if last is not None:
            self._detect_thr[last] = self.sb_detect_thr.value()
        default = cfg.get("pc_threshold", patchcore.DEFAULTS["pc_threshold"]) if method == "patchcore" else \
            cfg.get("threshold", void_detect.DEFAULTS["threshold"])
        self.sb_detect_thr.setValue(float(self._detect_thr.get(idx, default)))
        self._detect_last = idx

    # -----------------------------------------------
    # Results API (plugins/results_api.py)
    # -----------------------------------------------
    def _results_settings(self):
        return self._config().get("results_api") or {}

    def _results_error(self, title, text):
        print(text)
        box = QMessageBox(QMessageBox.Critical, "Results API", title, parent=self)
        box.setDetailedText(text) # Traceback for the plugin developer
        box.exec()

    def _results_plugin(self):
        settings = self._results_settings()
        try:
            return results_api.load_plugin(settings), settings
        except Exception:
            self._results_error("The plugin could not be loaded.", traceback.format_exc())
            return None, settings

    def _results_payload(self):
        if self.full_data is None:
            QMessageBox.information(self, "Results API", "Open an image or a project first.")
            return None, None, None
        plugin, settings = self._results_plugin()
        if plugin is None:
            return None, None, None
        try:
            return plugin, settings, plugin.build_payload(results_api.build_context(self, settings))
        except Exception:
            self._results_error("build_payload() failed.", traceback.format_exc())
            return None, None, None

    def preview_results_payload(self):
        plugin, settings, payload = self._results_payload()
        if plugin is None:
            return
        text = results_api.to_json(payload)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Results API payload ({len(text.encode('utf-8')) / 1024:.1f} KB)")
        dlg.resize(700, 600)
        lay = QVBoxLayout(dlg)
        edit = QPlainTextEdit(text)
        edit.setReadOnly(True)
        lay.addWidget(edit)
        row = QHBoxLayout()
        btn_save = QPushButton("Save...")
        btn_copy = QPushButton("Copy")
        btn_close = QPushButton("Close")
        row.addWidget(btn_save); row.addWidget(btn_copy); row.addStretch(); row.addWidget(btn_close)
        lay.addLayout(row)

        def save():
            path, _ = QFileDialog.getSaveFileName(dlg, "Save Payload", "payload.json", "JSON (*.json)")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
        btn_save.clicked.connect(save)
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        btn_close.clicked.connect(dlg.accept)
        dlg.exec()

    def send_results(self):
        plugin, settings, payload = self._results_payload()
        if plugin is None:
            return
        self.btn_api_send.setEnabled(False)
        self.statusBar().showMessage("Results API: sending...")
        job = results_api.BackgroundCall(lambda: plugin.send(payload, settings), self)
        job.done.connect(lambda result, error: self._on_results_sent(job, result, error))
        self._results_job = job

    def _on_results_sent(self, job, result, error):
        self.btn_api_send.setEnabled(True)
        job.deleteLater()
        if error:
            self.statusBar().showMessage("Results API: send failed")
            self._results_error("send() failed.", error)
            return
        ok, message = result if isinstance(result, tuple) else (bool(result), str(result))
        self.statusBar().showMessage(f"Results API: {'sent' if ok else 'not sent'} - {message}")
        (QMessageBox.information if ok else QMessageBox.warning)(self, "Results API", message)

    def load_results_from_api(self):
        if self.full_data is None or self.glw.tiled_image is None:
            QMessageBox.information(self, "Results API", "Open an image or a project first.")
            return
        plugin, settings = self._results_plugin()
        if plugin is None:
            return
        replace = (settings.get("receive") or {}).get("replace", True)
        if replace and not self._confirm_discard_voids():
            return
        ctx = results_api.build_context(self, settings)
        self.btn_api_load.setEnabled(False)
        self.statusBar().showMessage("Results API: loading...")
        job = results_api.BackgroundCall(lambda: plugin.fetch(ctx, settings), self)
        job.done.connect(lambda data, error: self._on_results_fetched(job, plugin, ctx, replace, data, error))
        self._results_job = job

    def _on_results_fetched(self, job, plugin, ctx, replace, data, error):
        self.btn_api_load.setEnabled(True)
        job.deleteLater()
        if error:
            self.statusBar().showMessage("Results API: load failed")
            self._results_error("fetch() failed.", error)
            return
        try:
            voids = plugin.to_voids(data, ctx)
            n = self.void_manager.load_from_data(voids, self.glw.grid_cfg, self.glw.coord_transform,
                                                 self.layer_offset, replace=replace)
        except Exception:
            self.statusBar().showMessage("Results API: load failed")
            self._results_error("to_voids() / loading the voids failed.", traceback.format_exc())
            return
        self._scale_from_file()
        self.populate_void_types()
        self.glw.active_void = None
        self.glw.refresh_hover()
        self.update_void_ui()
        self.statusBar().showMessage(f"Results API: {n} voids loaded ({'replaced' if replace else 'added'}; Ctrl+Z undoes)")

    def _detect_setup(self, title):
        """Inputs of Detect Voids / Evaluate: (type, cells, refs, params, layers), or None."""
        if self.full_data is None or self.glw.tiled_image is None:
            return None
        vm = self.void_manager
        type_name = self.glw.active_type
        if not type_name or type_name not in vm.types or vm.is_bbox({"type": type_name}):
            QMessageBox.warning(self, title, "Select the void type to mark with (not the bbox type).")
            return None
        cfg = self.glw.grid_cfg
        bmap = self.glw.bonding_map
        cells = [(r, c) for r in range(cfg.rows) for c in range(cfg.cols) if bmap is None or bmap.get_key(r, c)]
        refs = vm.reference_bboxes(cfg, self.glw.coord_transform)
        params = dict(self._config().get("void_detect", {}))
        rules = (getattr(self, "current_product", None) or {}).get("type_rules")
        if rules is not None: # Per product (products.json) wins over config.json
            params["type_rules"] = rules
        params["method"], model = self.cb_detect_method.currentData()
        if model:
            params["pc_model"] = model
        params["pc_threshold" if params["method"] == "patchcore" else "threshold"] = self.sb_detect_thr.value()
        params["layer_rule"] = self.cb_layer_rule.currentData()
        try:
            params["skip_layers"] = [int(x) for x in re.split(r"[,\s]+", self.le_skip_layers.text().strip()) if x]
        except ValueError:
            QMessageBox.warning(self, title, "Skip layers: layer numbers separated by commas, e.g. 0, 1")
            return None
        layers = [self.abs_layer(z) for z in range(len(self.full_data))]
        return type_name, cells, refs, params, layers

    def _detect_progress(self, text):
        progress = QProgressDialog(text, "Cancel", 0, 1000, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def step(i, n, label):
            if i % 20 == 0:
                progress.setValue(int(1000 * i / max(1, n)))
                progress.setLabelText(label)
                QApplication.processEvents()
            return not progress.wasCanceled()
        return progress, step

    def detect_voids(self):
        """Rough automatic void detection on all loaded layers (see sat_widgets/void_detect.py)."""
        setup = self._detect_setup("Detect Voids")
        if setup is None:
            return
        type_name, cells, refs, params, layers = setup
        vm, cfg = self.void_manager, self.glw.grid_cfg
        progress, step = self._detect_progress("Detecting voids...")
        result = void_detect.detect_voids(self.full_data, cfg, self.glw.coord_transform, refs, cells,
                                          layers, vm, type_name, params, step)
        progress.close()
        if result is None:
            return
        self._detect_done(result, refs)

    def evaluate_detection(self):
        """
        Detection without marking, compared with the hand markings (sat_widgets/void_eval.py):
        shows the short code and writes the report folder.
        """
        setup = self._detect_setup("Evaluate")
        if setup is None:
            return
        type_name, cells, refs, params, layers = setup
        vm, cfg, ct = self.void_manager, self.glw.grid_cfg, self.glw.coord_transform
        to_deskew = ct.raw_to_deskew if ct else (lambda x, y: (x, y))
        truths = []
        for layer, vs in vm.voids.items():
            if layer not in layers:
                continue
            for v in vs:
                if vm.is_bbox(v) or v.get("auto"):
                    continue # Hand markings only (reviewed detections count: editing clears "auto")
                gx, gy = to_deskew(v["globalCX"], v["globalCY"])
                rx, ry = v["radiusX"], v["radiusY"]
                q = vm.quadrant_of(v, gx, gy, refs, cfg)
                if q: # Drawn quadrant: corner -> corner + radius
                    xs, ys = sorted((gx, gx + q[0] * rx)), sorted((gy, gy + q[1] * ry))
                    box = (xs[0], ys[0], xs[1], ys[1])
                else:
                    box = (gx - rx, gy - ry, gx + rx, gy + ry)
                truths.append({"layer": layer, "cell": cfg.cell_at(gx, gy), "box": box, "type": v.get("type")})
        ev = void_eval.Evaluation(truths, layers, self.cb_detect_method.currentText())
        ev.refs_pct = round(100 * sum(1 for c in cells if c in refs) / max(1, len(cells)))
        progress, step = self._detect_progress("Evaluating detection...")
        try:
            result = void_detect.detect_voids(self.full_data, cfg, ct, refs, cells, layers, vm, type_name,
                                              params, step, diag=ev, apply=False)
        finally:
            progress.close()
        if result is None:
            return
        summary = ev.finish(cells)
        code = ev.code(summary)
        src = self.lbl_source.toolTip()
        stem = os.path.splitext(os.path.basename(src))[0] or "eval"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(os.path.expanduser("~"), "TIF_Viewer_eval", f"{stem}_{stamp}")
        try:
            report = ev.write_report(folder, summary, code, {"source": src, "time": stamp, "method": ev.method,
                                                            "wafer_id": self.le_wafer_id.text()})
        except OSError as e:
            report = None
            print(f"Evaluation report not written: {e}")
        print("Detection evaluation:\n" + code + (f"\nReport: {report}" if report else ""))
        EvalDialog(self, code, folder if report else None, report, summary["h"]).exec()

    def _detect_done(self, result, refs):
        self.glw.refresh_hover()
        self.update_void_ui()
        per = ", ".join(f"L{l}: {n}" for l, n in result["per_layer"].items())
        kinds = ", ".join(f"{t}: {n}" for t, n in result.get("types", {}).items())
        msg = (f"Detect Voids: {result['added']} new voids ({per}; {kinds}). "
               "Dashed until reviewed; 'Next Detected' walks through them.")
        if not refs:
            msg += " No reference bboxes: chips were aligned on the grid cells."
        self.statusBar().showMessage(msg)
        print(msg)

    def next_detected_void(self):
        """Navigates through the detected voids not reviewed yet (layer, row, col order)."""
        items = []
        for layer, vs in self.void_manager.voids.items():
            for v in vs:
                if v.get("auto") == "detected":
                    r, c = self.glw.raw_to_cell(v["globalCX"], v["globalCY"])
                    items.append((layer, r, c))
        if not items:
            self.statusBar().showMessage("No detected voids left to review.")
            return
        items = sorted(set(items))
        here = (self.glw.current_layer,) + tuple(self.glw.center_cell())
        nxt = next((it for it in items if it > here), items[0])
        layer, r, c = nxt
        z = next((z for z in range(len(self.full_data)) if self.abs_layer(z) == layer), None)
        if z is not None and z != self.current_z:
            self.spin_layer.setValue(z)
        self.glw.fit_to_cell(c, r)
        self.statusBar().showMessage(f"Detected void {items.index(nxt) + 1}/{len(items)}: layer {layer}, X{c:02d} Y{r:02d}")

    def update_void_ui(self):
        # Update label count (voids are keyed by absolute layer number)
        count = len(self.void_manager.voids.get(self.glw.current_layer, []))
        self.lbl_void_count.setText(f"Voids: {count}")

    # -----------------------------------------------
    # Data / Layer Management
    # -----------------------------------------------
    def abs_layer(self, z):
        """Absolute layer number of loaded layer index z (void data and file names use this)."""
        if self.full_data is not None:
            return self.full_data.layer_number(z)
        return self.layer_offset + z

    def _confirm_discard_voids(self):
        """Asks before replacing existing void markings. Returns False to abort."""
        if not any(self.void_manager.voids.values()):
            return True
        ret = QMessageBox.question(
            self, "Void Markings",
            "Current void markings will be cleared (save them first if needed).\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return ret == QMessageBox.Yes

    def _clear_gpu_cache(self):
        if self.layer_cache:
            self.glw.makeCurrent()
            for t in self.layer_cache.values():
                t.cleanup()
            self.glw.doneCurrent()
        self.layer_cache.clear()

    def _close_data(self):
        """Releases the current stack, its GPU textures and per-layer data."""
        self._stop_layer_loader()
        self._clear_gpu_cache()
        if self.full_data is not None:
            self.full_data.close()
        self.full_data = None
        self.layer_calib_data.clear()

    def _setup_layer_controls(self):
        n = len(self.full_data)
        self.spin_layer.blockSignals(True)
        self.spin_layer.setRange(0, max(0, n - 1))
        self.spin_layer.setValue(0)
        self.spin_layer.setEnabled(n > 1)
        self.spin_layer.blockSignals(False)

        # valueChanged is blocked above -> sync layer state explicitly
        self.current_z = 0
        self.glw.current_layer = self.abs_layer(0)
        self.glw.active_void = None
        self.spin_layer.setSuffix(f" (Abs: {self.glw.current_layer})")
        self.update_void_ui()

    def _layer_gpu_bytes(self):
        # Gray uint8 sources (incl. 8-bit RGB) are uploaded as GL_R8, everything else as GL_R16
        bpp = 1 if self.full_data.dtype == np.uint8 else 2
        return self.full_data.width * self.full_data.height * bpp

    def _evict_gpu_cache(self, keep, incoming=0):
        """Drops least recently used layers until the cache fits the budget. GL context must be current."""
        budget = self.sb_gpu_cache.value() * 1024 * 1024
        layer_bytes = self._layer_gpu_bytes() # Layers still uploading count at their full size
        total = sum(max(t.nbytes, layer_bytes) for t in self.layer_cache.values()) + incoming
        for z in list(self.layer_cache.keys()):
            if total <= budget:
                break
            if z in keep:
                continue
            t = self.layer_cache.pop(z)
            total -= max(t.nbytes, layer_bytes)
            t.cleanup()

    def _auto_level(self, z):
        """Sets the window to the min/max of a subsample of layer z."""
        try:
            sample = self.full_data.get_sample(z)
            vmin, vmax = int(sample.min()), int(sample.max())

            self.slider_lo.blockSignals(True)
            self.slider_hi.blockSignals(True)
            self.slider_lo.setValue(vmin)
            self.slider_hi.setValue(vmax)
            self.slider_lo.blockSignals(False)
            self.slider_hi.blockSignals(False)

            self.on_window_changed()
            print(f"Auto-Leveled to [{vmin}, {vmax}]")
        except Exception as e:
            print(f"Auto-Level failed: {e}")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "TIFF (*.tif *.tiff)")
        if not path:
            return
        if not self._confirm_discard_voids():
            return

        # Get Load Limits (inclusive range of absolute layer numbers)
        req_start = self.sb_start_layer.value()
        req_end = self.sb_end_layer.value()

        print(f"Loading {path}...")
        print(f"Requested Range: {req_start} to {req_end}")

        try:
            # 1. Release the previous file
            self._close_data()

            # 2. Open (only headers are read here; pixels are read per tile on demand)
            if req_start <= req_end:
                stack = TiffStack(path, req_start, req_end - req_start + 1)
            else:
                stack = TiffStack(path)
            print(f"Opened ({stack.backend}): layers {stack.layer_number(0)}..{stack.layer_number(len(stack) - 1)}, "
                  f"{stack.width}x{stack.height}, {stack.dtype}")

            self.full_data = stack
            self.layer_offset = stack.layer_number(0)
            self.glw.bbox_layer = self.layer_offset
            self.void_manager.replace_all()

            # 3. UI / State
            self._setup_layer_controls()

            # 4. Auto-Leveling (subsample of the first layer)
            self._auto_level(0)

            # 5. Show first layer, then load the others in the background (GPU cache budget)
            self.load_layer(0)
            self._start_layer_loader()

            # 6. File name / wafer ID (may fetch the bonding map, see results_api "bonding_map")
            self._set_source(path)

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Open TIF", f"Error loading file:\n{e}")

    def preload_gpu_cache(self):
        """Uploads further layers while they fit into the GPU cache budget (instant layer switching)."""
        if self.full_data is None:
            return

        n_layers = len(self.full_data)
        budget = self.sb_gpu_cache.value() * 1024 * 1024
        n_fit = min(n_layers, max(1, budget // max(1, self._layer_gpu_bytes())))
        todo = [z for z in range(n_fit) if z not in self.layer_cache]
        if not todo:
            return
        if n_fit < n_layers:
            print(f"GPU cache budget fits {n_fit}/{n_layers} layers; others are loaded on demand.")

        print(f"Preloading {len(todo)} layers to VRAM...")

        progress = QProgressDialog("Preloading layers to GPU...", "Cancel", 0, len(todo), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.resize(400, 100)
        progress.show()

        try:
            for i, z in enumerate(todo):
                if progress.wasCanceled():
                    print("Preloading canceled.")
                    break

                progress.setValue(i)
                QApplication.processEvents()

                # Uploading needs a current GL context (processEvents may have painted in between)
                tiled = TiledImage.from_stack(self.full_data, z)
                self.glw.makeCurrent()
                try:
                    tiled.upload_all()
                finally:
                    self.glw.doneCurrent()
                self.layer_cache[z] = tiled

            progress.setValue(len(todo))
        finally:
            progress.close()
            self.full_data.release_cache()
            # Keep the displayed layer most recently used
            if self.current_z in self.layer_cache:
                self.layer_cache.move_to_end(self.current_z)

    # ---------------- Layer loading: background decode, previews, time-sliced uploads ----------------
    def _start_layer_loader(self):
        """Background loading for the open stack; stacks that cannot be reopened preload instead."""
        self._stop_layer_loader()
        if self.full_data is None or not self.full_data.supports_reopen:
            self.preload_gpu_cache()
            return
        loader = LayerLoader(self.full_data, TiledImage.TILE_SIZE, parent=self)
        loader.loaded.connect(lambda z, tiles, proxy, stats, L=loader: self._on_layer_decoded(L, z, tiles, proxy, stats))
        self.layer_loader = loader
        self._schedule_layer_loads()

    def _stop_layer_loader(self):
        if self.layer_loader is not None:
            self.layer_loader.stop()
            self.layer_loader = None
        self._ram_layers.clear()
        self._upload_queue.clear()
        self.glw.proxy_image = None
        if self.layer_proxies:
            self.glw.makeCurrent()
            for p in self.layer_proxies.values():
                p.cleanup()
            self.glw.doneCurrent()
            self.layer_proxies.clear()
        self.layer_proxy_arrays.clear()

    def layers_loading(self):
        """True while layers are decoded or uploaded in the background."""
        return bool(self._upload_queue) or (self.layer_loader is not None and self.layer_loader.busy())

    def _has_layer_data(self, z):
        if z in self._ram_layers:
            return True
        t = self.layer_cache.get(z)
        return t is not None and all(tile.is_uploaded or tile.loader is not None for tile in t.tiles)

    def _schedule_layer_loads(self):
        """
        Decode jobs by distance from the current layer (browsing direction first): full tiles for
        the layers the GPU budget holds and for the direct neighbours, a preview for the others.
        """
        L = self.layer_loader
        if L is None or self.full_data is None:
            return
        n, cur, d = len(self.full_data), self.current_z, self._layer_dir
        order = [cur]
        for k in range(1, n):
            order += [z for z in (cur + d * k, cur - d * k) if 0 <= z < n]
        fit = max(1, (self.sb_gpu_cache.value() * 1024 * 1024) // max(1, self._layer_gpu_bytes()))
        full = set(order[:fit]) | {z for z in (cur - 1, cur, cur + 1) if 0 <= z < n}
        for z in list(self._ram_layers): # RAM only holds the direct neighbours
            if abs(z - cur) > 1:
                del self._ram_layers[z]
        jobs = []
        for z in order:
            if z in full and not self._has_layer_data(z):
                jobs.append((z, True))
            elif z not in self.layer_proxy_arrays: # Preview + statistics sample (calibration), also for GPU layers
                jobs.append((z, False))
        L.request(jobs)

    def _on_layer_decoded(self, loader, z, tiles, proxy, stats=None):
        loader.ack()
        if loader is not self.layer_loader or self.full_data is None:
            return # Result of a closed file
        w, h = self.full_data.width, self.full_data.height
        if z not in self.layer_proxies:
            p = TiledImage.proxy(w, h, proxy)
            self.glw.makeCurrent()
            p.upload_all()
            self.glw.doneCurrent()
            self.layer_proxies[z] = p
            if z == self.current_z and self.glw.tiled_image is not None and not self.glw.tiled_image.fully_uploaded:
                self.glw.proxy_image = p
                self.glw.update()
        if stats is not None:
            self.layer_proxy_arrays[z] = (stats, STATS_STEP)
        if tiles is None:
            return
        t = self.layer_cache.get(z)
        if t is not None: # Waiting in the GPU cache (the current layer, or reserved earlier)
            t.set_tile_data(tiles)
            self._queue_upload(t, front=(z == self.current_z))
        elif (len(self.layer_cache) + 1) * self._layer_gpu_bytes() <= self.sb_gpu_cache.value() * 1024 * 1024:
            t = TiledImage.pending(w, h)
            t.set_tile_data(tiles)
            self.layer_cache[z] = t
            self.layer_cache.move_to_end(z, last=False) # Prefetched: not "recently used" yet
            self._queue_upload(t)
        elif abs(z - self.current_z) <= 1:
            self._ram_layers[z] = tiles # Instant switch to the neighbour (upload only)

    def _queue_upload(self, tiled, front=False):
        if tiled in self._upload_queue:
            self._upload_queue.remove(tiled)
        if front:
            self._upload_queue.insert(0, tiled)
        else:
            self._upload_queue.append(tiled)
        self._upload_timer.start()

    def _next_tile(self, tiled):
        todo = [t for t in tiled.tiles if not t.is_uploaded and t.loader is not None]
        if not todo:
            return None
        if tiled is self.glw.tiled_image and len(todo) > 1:
            # What is on screen first
            cx, cy = self.glw.deskew_to_raw(*self.glw.get_image_coords(QPointF(self.glw.width() / 2, self.glw.height() / 2)))
            todo.sort(key=lambda t: (t.x + t.w / 2 - cx) ** 2 + (t.y + t.h / 2 - cy) ** 2)
        return todo[0]

    def _upload_step(self):
        """Uploads tiles for ~15 ms, then lets the UI paint and handle input before the next ones."""
        deadline = time.perf_counter() + 0.015
        uploaded = False
        self.glw.makeCurrent()
        try:
            while self._upload_queue and time.perf_counter() < deadline:
                tiled = self._upload_queue[0]
                tile = self._next_tile(tiled) if any(tiled is t for t in self.layer_cache.values()) else None
                if tile is None: # Done, waiting for data, or evicted
                    self._upload_queue.pop(0)
                    continue
                tile.upload()
                uploaded = True
        finally:
            self.glw.doneCurrent()
        if uploaded:
            self.glw.update()
        if self._upload_queue:
            self._upload_timer.start()

    def load_layer(self, z_index):
        if self.full_data is None:
            return

        tiled = self.layer_cache.get(z_index)
        background = self.layer_loader is not None
        if tiled is not None:
            self.layer_cache.move_to_end(z_index)
        elif background:
            # Shown at once: preview (if decoded already) under the tiles as they arrive
            self.glw.makeCurrent()
            self._evict_gpu_cache(keep={z_index}, incoming=self._layer_gpu_bytes())
            self.glw.doneCurrent()
            tiled = TiledImage.pending(self.full_data.width, self.full_data.height)
            ram = self._ram_layers.pop(z_index, None)
            if ram is not None:
                tiled.set_tile_data(ram)
            self.layer_cache[z_index] = tiled
        else:
            print(f"Loading Layer {z_index} (Abs {self.abs_layer(z_index)})...")
            # Make room first, then stream the tiles to the GPU (set_tiled_image uploads)
            self.glw.makeCurrent()
            self._evict_gpu_cache(keep={z_index}, incoming=self._layer_gpu_bytes())
            self.glw.doneCurrent()
            tiled = TiledImage.from_stack(self.full_data, z_index)
            self.layer_cache[z_index] = tiled

        if background:
            self.glw.proxy_image = self.layer_proxies.get(z_index)
            self.glw.set_tiled_image(tiled, cleanup_old=False, upload=False)
            if not tiled.fully_uploaded:
                self._queue_upload(tiled, front=True)
            self._schedule_layer_loads()
        else:
            # Set Image (uploads if needed)
            self.glw.set_tiled_image(tiled, cleanup_old=False)
            self.full_data.release_cache()
        
        # Keep the cache within the budget (it may have been lowered meanwhile)
        self.glw.makeCurrent()
        self._evict_gpu_cache(keep={z_index})
        self.glw.doneCurrent()

        # Update Calibration Texture for this layer
        calib = self.layer_calib_data.get(z_index, None)
        self.glw.update_calib_texture(calib)

        self.update()

    def on_layer_changed(self, val):
        if val != self.current_z:
            self._layer_dir = 1 if val > self.current_z else -1
        self.current_z = val
        abs_layer = self.abs_layer(val)
        self.glw.current_layer = abs_layer # Void data is keyed by absolute layer
        self.glw.active_void = None

        self.spin_layer.setSuffix(f" (Abs: {abs_layer})")

        self.load_layer(val)
        self.glw.refresh_hover()
        self.update_void_ui()
        self._refresh_angle_preview()

    def on_window_changed(self):
        self.glw.win_lo = self.slider_lo.value()
        self.glw.win_hi = self.slider_hi.value()
        self.glw.update()
        self._refresh_angle_preview()
        
    def toggle_calib(self, checked):
        self.chk_use_calib.setText(f"Enable Auto-Calib: {'ON' if checked else 'OFF'}")
        self.glw.grid_cfg.use_calib = checked
        if checked:
            self.calculate_calibration()
        else:
            self.glw.update()

    def calculate_calibration(self):
        """
        Per-cell brightness calibration (scale, offset) of the current layer, or of all loaded
        layers ("Apply to All Layers"). Only cells with a bonding map value are computed
        (others keep scale 1, offset 0). Layers are decoded and computed in parallel, each worker
        with its own reader of the file; the GUI stays responsive (progress, Cancel).
        """
        if self.full_data is None: return
        n_layers = len(self.full_data)
        layers_to_calc = list(range(n_layers)) if self.chk_calib_all.isChecked() else [self.current_z]

        cfg = self.glw.grid_cfg
        bmap = self.glw.bonding_map
        params = {
            "start_x": cfg.start_x, "start_y": cfg.start_y, "pitch_x": cfg.pitch_x, "pitch_y": cfg.pitch_y,
            "rows": cfg.rows, "cols": cfg.cols,
            "to_raw": self.glw.deskew_to_raw,
            "bonded": None if bmap is None else {(r, c) for r in range(cfg.rows) for c in range(cfg.cols) if bmap.get_key(r, c)},
            "robust": self.chk_calib_robust.isChecked(),
            "target_mean": self.sb_target_mean.value(), "target_std": self.sb_target_std.value(),
            "win_lo": self.glw.win_lo, "win_hi": self.glw.win_hi,
        }
        progress = None
        if len(layers_to_calc) > 1:
            progress = QProgressDialog("Calibrating layers...", "Cancel", 0, len(layers_to_calc), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

        stack = self.full_data
        # Layers whose preview is in memory (background loader): computed at once, no decoding
        for z in [z for z in layers_to_calc if z in self.layer_proxy_arrays]:
            self.layer_calib_data[z] = calibrate_layer(stack, z, params, self.layer_proxy_arrays[z])
        done_now = len(layers_to_calc)
        layers_to_calc = [z for z in layers_to_calc if z not in self.layer_proxy_arrays]
        done_now -= len(layers_to_calc)
        if progress:
            progress.setMaximum(done_now + len(layers_to_calc))
            progress.setValue(done_now)
        workers = min(len(layers_to_calc), os.cpu_count() or 2, 4) if stack.supports_reopen else 1
        if workers <= 1:
            for i, z in enumerate(layers_to_calc):
                if progress:
                    if progress.wasCanceled(): break
                    progress.setValue(done_now + i)
                    QApplication.processEvents()
                self.layer_calib_data[z] = calibrate_layer(stack, z, params)
            stack.release_cache()
        else:
            # One reader per worker thread (decoding releases the GIL)
            local = threading.local()
            readers = []
            canceled = threading.Event() # Set from the GUI thread (no widget access in workers)

            def job(z):
                if canceled.is_set():
                    return z, None
                rd = getattr(local, "reader", None)
                if rd is None:
                    rd = local.reader = stack.reopen()
                    readers.append(rd)
                out = calibrate_layer(rd, z, params)
                rd.release_cache()
                return z, out

            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(job, z) for z in layers_to_calc]
                done = 0
                while done < len(futures):
                    done = sum(f.done() for f in futures)
                    if progress:
                        progress.setValue(done_now + done)
                        if progress.wasCanceled():
                            canceled.set()
                    QApplication.processEvents()
                    time.sleep(0.02)
                for f in futures:
                    z, out = f.result()
                    if out is not None:
                        self.layer_calib_data[z] = out
            for rd in readers:
                rd.close()

        if progress:
            progress.setValue(progress.maximum())
            progress.close()

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
            win_hi=self.glw.win_hi,
            layer_index=self.current_z
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
                
            if self.glw.bonding_map is None and any(opts.get(k) for k in ('raw', 'overlay', 'mask', 'merged')):
                QMessageBox.warning(self, "Export", "Images are exported only for cells with a LEG (bonding map value).\n"
                                                    "Paste a bonding map first.")
                return

            print(f"Starting Export to {out_dir} with opts: {opts}")
            
            # Setup Progress Dialog
            self.pd = QProgressDialog("Exporting...", "Cancel", 0, 100, self)
            self.pd.setWindowModality(Qt.WindowModal)
            self.pd.show()
            
            # Create Manager (Pass MainWindow)
            exporter = ExportManager(self)
            errors = []

            # Connect
            exporter.progress_update.connect(lambda p, msg: (self.pd.setValue(p), self.pd.setLabelText(msg), QApplication.processEvents()))
            exporter.error_occurred.connect(errors.append)
            self.pd.canceled.connect(exporter.request_stop)

            # Run
            try:
                # Pass Window Levels for WYSIWYG
                win_lo = self.glw.win_lo
                win_hi = self.glw.win_hi
                exporter.run_export(out_dir, opts, win_lo, win_hi)
            except Exception as e:
                traceback.print_exc()
                errors.append(str(e))

            # close() emits canceled: disconnect first, or a finished export reads as canceled
            self.pd.canceled.disconnect(exporter.request_stop)
            self.pd.close()
            if errors:
                QMessageBox.critical(self, "Export", f"Export failed:\n{errors[0]}")
            elif exporter.stop_requested:
                QMessageBox.information(self, "Export", f"Export canceled.\nPartial output: {out_dir}")
            else:
                print("Export Finished.")
                self.last_export_dir = out_dir
                if (self._results_settings().get("send") or {}).get("after_export"):
                    self.send_results() # config.json "results_api": {"send": {"after_export": true}}

class EvalDialog(QDialog):
    """Result of Evaluate: the short code (to type or scan on a phone) and the report folder."""

    def __init__(self, parent, code, folder, report, n_hand):
        super().__init__(parent)
        self.setWindowTitle("Detection Evaluation")
        self.code, self.folder, self.report = code, folder, report
        lay = QVBoxLayout(self)
        hint = "Type this code on the phone and send it (letters, digits, dots and spaces only)."
        if n_hand == 0:
            hint = ("No hand markings on the loaded layers: only detection and chip statistics are shown.\n"
                    "Mark voids by hand (or load a reviewed voids.json) and evaluate again to compare.\n") + hint
        lay.addWidget(QLabel(hint))
        txt = QPlainTextEdit(code)
        txt.setReadOnly(True)
        f = txt.font()
        f.setFamily("Consolas")
        f.setStyleHint(f.StyleHint.Monospace)
        f.setPointSize(16)
        txt.setFont(f)
        txt.setLineWrapMode(QPlainTextEdit.NoWrap)
        txt.setMinimumSize(720, 380)
        lay.addWidget(txt, 1)
        btns = QHBoxLayout()
        b_copy = QPushButton("Copy")
        b_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.code))
        btns.addWidget(b_copy)
        b_rep = QPushButton("Open Report")
        b_rep.setEnabled(bool(report))
        b_rep.clicked.connect(lambda: self._open(self.report))
        btns.addWidget(b_rep)
        b_dir = QPushButton("Open Folder")
        b_dir.setEnabled(bool(folder))
        b_dir.clicked.connect(lambda: self._open(self.folder))
        btns.addWidget(b_dir)
        btns.addStretch(1)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        btns.addWidget(b_close)
        lay.addLayout(btns)
        if folder:
            lay.addWidget(QLabel(f"Report: {folder}"))

    @staticmethod
    def _open(path):
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


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
        # config.json "ui_defaults": {"export": {"raw": false, ...}}
        defaults = (parent._config().get("ui_defaults") or {}).get("export", {}) if hasattr(parent, "_config") else {}
        for key, chk in (("raw", self.chk_raw), ("overlay", self.chk_overlay), ("mask", self.chk_mask),
                         ("merged", self.chk_merged), ("json", self.chk_json), ("csv", self.chk_csv)):
            if key in defaults:
                chk.setChecked(bool(defaults[key]))
        
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
