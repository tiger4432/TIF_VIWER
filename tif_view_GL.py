
import sys
import os
import numpy as np
import time
import traceback
from collections import OrderedDict
from datetime import datetime
import json
import re
import copy
import threading
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSpinBox, QSlider, QLabel, QToolBar, QDockWidget, QGroupBox,
    QFormLayout, QPushButton, QDoubleSpinBox, QPlainTextEdit,
    QFileDialog, QProgressDialog, QCheckBox, QRadioButton,
    QButtonGroup, QComboBox, QDialog, QScrollArea, QMessageBox, QLineEdit,
    QTextEdit, QAbstractSpinBox, QSizePolicy, QProgressBar, QToolButton
)
from PySide6.QtGui import QSurfaceFormat, QAction, QImage, QPainter, QKeySequence
from PySide6.QtCore import Qt, QTimer, QEvent, QPointF, QSettings

# Imports from sat_widgets
from sat_widgets.core_data import TiffStack, MosaicTiffStack, GridConfig, BondingMap, to_uint16, ImageNormalizer, CoordinateTransform
from sat_widgets.void_manager import VoidManager, VoidTypeDialog
from sat_widgets.gl_widget import GLImageWidget, TiledImage
from sat_widgets.layer_loader import LayerLoader, STATS_STEP
from sat_widgets.exporter import ExportManager
from sat_widgets.patch_viewer import PatchViewer
from sat_widgets.roi_inspector import ROIInspector
from sat_widgets.angle_preview import AnglePreview
from sat_widgets.auto_bbox import place_bboxes, refine_bboxes, detect_layout, fit_grid, fit_edges
from sat_widgets import void_detect
from sat_widgets.detect_job import DetectJob, read_only_copy
from sat_widgets import void_layer_change
from sat_widgets import void_eval
from sat_widgets import void_learn
from sat_widgets import void_seg, seg_data
from sat_widgets import results_api
from sat_widgets.auto_run import AutoRunController
from sat_widgets.products import load_products, grid_values, pitch_um
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
        dock = QDockWidget("Grid && Map Tools", self)
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
        self.sb_angle.setDecimals(3) # Fit Grid gives the angle to 0.001 deg
        # Arrow keys / wheel: 0.001 deg (config ui_defaults "angle_step"); PgUp / PgDn: x10
        self.sb_angle.setSingleStep(float((self._config().get("ui_defaults") or {}).get("angle_step", 0.001)))
        form.addRow("Angle:", self.sb_angle)

        # Pixel size: chip / layout declarations in um -> pixels, um columns of the exports
        self.sb_um_per_px = QDoubleSpinBox(); self.sb_um_per_px.setRange(0, 100000); self.sb_um_per_px.setDecimals(4)
        self.sb_um_per_px.setSingleStep(0.1); self.sb_um_per_px.setSuffix(" um/px")
        self.sb_um_per_px.setSpecialValueText("not set") # 0
        self.sb_um_per_px.setToolTip("How many um one image pixel is. Converts the product's chip size (um) to the\n"
                                     "bbox size in pixels and fills the um columns of voids.json / voids.csv.")
        self.sb_um_per_px.valueChanged.connect(self.on_scale_changed)
        btn_measure = QPushButton("Measure")
        btn_measure.setToolTip("Scale from a known length: click its two ends on the image, then type the real\n"
                               "length in um (Esc cancels). Scale = length / pixel distance.")
        btn_measure.clicked.connect(self.measure_scale)
        h_scale = QHBoxLayout()
        h_scale.addWidget(self.sb_um_per_px, 1)
        h_scale.addWidget(btn_measure)
        form.addRow("Scale:", h_scale)
        
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
        gb_coord = QGroupBox("Coordinates && Navigation")
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
        # Ctrl+C on the image: the chip under the mouse (else at the view centre) with its markings
        act_copy = QAction("Copy Chip Image", self.glw)
        act_copy.setShortcut(QKeySequence.Copy)
        act_copy.setShortcutContext(Qt.WidgetShortcut)
        act_copy.triggered.connect(self.copy_chip_image)
        self.glw.addAction(act_copy)
        # A on the image: the detected (dashed) markings of the chip under the mouse are reviewed
        act_review = QAction("Review Chip", self.glw)
        act_review.setShortcut(QKeySequence(Qt.Key_A))
        act_review.setShortcutContext(Qt.WidgetShortcut)
        act_review.triggered.connect(self.review_chip)
        self.glw.addAction(act_review)
        
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
        self.btn_clear_keep_bbox = QPushButton("Clear Voids")
        self.btn_clear_keep_bbox.setToolTip("Removes every void on all layers and keeps the bboxes (Ctrl+Z restores)")
        self.btn_clear_keep_bbox.clicked.connect(self.clear_voids_keep_bbox)

        h_layout_void_btns.addWidget(self.btn_load_voids)
        h_layout_void_btns.addWidget(self.btn_save_voids)
        h_clear_btns = QHBoxLayout()
        h_clear_btns.addWidget(self.btn_clear_voids)
        h_clear_btns.addWidget(self.btn_clear_chip)
        h_clear_btns.addWidget(self.btn_clear_keep_bbox)
        h_clear_btns.addWidget(self.btn_clear_all)

        v_layout_void.addWidget(self.lbl_void_hint)
        v_layout_void.addLayout(h_modes)
        v_layout_void.addLayout(h_type)
        v_layout_void.addWidget(self.lbl_void_count)
        v_layout_void.addLayout(h_layout_void_btns)
        v_layout_void.addLayout(h_clear_btns)

        # Rough automatic detection (die-to-die comparison), marked with the selected type
        h_detect = QHBoxLayout()
        self.btn_detect = btn_detect = QPushButton("Detect Voids")
        btn_detect.setToolTip("Compare every chip with the median chip of its layer (layout-independent), from the top\n"
                              "layer down; each void is marked once, on the layer where it first appears, with the\n"
                              "selected type. Brighter and darker spots both count. Results are dashed until edited.\n"
                              "Runs in the background: the voids appear as they are found; viewing works meanwhile,\n"
                              "editing the markings waits until it ends (Cancel in the status bar keeps what was found).")
        btn_detect.clicked.connect(self.detect_voids)
        self.sb_detect_thr = QDoubleSpinBox(); self.sb_detect_thr.setRange(2.0, 50.0); self.sb_detect_thr.setSingleStep(0.5)
        self.sb_detect_thr.setValue(float(self._config().get("void_detect", {}).get("threshold", void_detect.DEFAULTS["threshold"])))
        self.sb_detect_thr.setPrefix("Thr ")
        self.sb_detect_thr.setToolTip("Deviation from the median chip, in units of its chip-to-chip spread. Lower = more sensitive.")
        btn_next_det = QPushButton("Next Detected")
        btn_next_det.setToolTip("Go to the next detected void that was not reviewed (edited) yet, layer by layer")
        btn_next_det.clicked.connect(self.next_detected_void)
        btn_next_l0 = QPushButton("Next Layer-0")
        btn_next_l0.setToolTip("Go to the next void flagged as already visible on layer 0 (badge \"0\", bubble?), layer by layer.\n"
                               "B over a void sets / clears the flag (Layer change sets it, see README)")
        btn_next_l0.clicked.connect(self.next_layer0_void)
        # Scorer: die-to-die (default) or the optional PatchCore (onnxruntime + models/*.onnx)
        self.cb_detect_method = QComboBox()
        self.cb_detect_method.addItem("Die-to-die", ("die_to_die", None))
        self.cb_detect_method.addItem("Layer change", ("layer_change", None))
        self.cb_detect_method.addItem("SSIM die-to-die", ("ssim", None))
        self.cb_detect_method.addItem("PatchCore MobileNetV2", ("patchcore", "models/patchcore_mobilenetv2.onnx"))
        self.cb_detect_method.addItem("PatchCore ResNet50", ("patchcore", "models/patchcore_resnet50.onnx"))
        self.cb_detect_method.addItem("Segmentation U-Net", ("segmentation", "models/void_unet.onnx"))
        self.cb_detect_method.addItem("Reconstruction U-Net", ("segmentation", "models/void_recon.onnx"))
        self.cb_detect_method.setToolTip("Die-to-die: pixel comparison with the median chip (fast, exact outlines).\n"
                                         "Layer change: every loaded layer at once; each layer explained from the layers\n"
                                         "above it and the median chips; a void is what is new on its first layer.\n"
                                         "PatchCore: CNN features (optional, slower, coarse outlines), see docs/PATCHCORE.md.")
        self._detect_thr = {}
        self.cb_detect_method.currentIndexChanged.connect(self._on_detect_method)
        btn_next_det.setText("Next Detected \u25B6")
        btn_prev_det = QPushButton("\u25C0")
        btn_prev_det.setToolTip("Go to the previous detected void that was not reviewed yet")
        btn_prev_det.setMaximumWidth(36)
        btn_prev_det.clicked.connect(self.prev_detected_void)
        self.lbl_review = QLabel("")
        self.lbl_review.setToolTip("Places (layer + cell) with detected voids not reviewed (edited) yet;\n"
                                   "after Next / Prev: position among them")
        self.btn_next_l0 = btn_next_l0
        btn_next_l0.setText("Layer-0")

        # Void Detection: run + review on top; method, layer options and training under "Advanced"
        self.gb_detect = QGroupBox("Void Detection")
        v_det = QVBoxLayout(self.gb_detect)
        h_detect.addWidget(btn_detect, 1)
        h_detect.addWidget(self.sb_detect_thr)
        v_det.addLayout(h_detect)
        h_review = QHBoxLayout()
        h_review.addWidget(btn_prev_det)
        h_review.addWidget(btn_next_det, 1)
        h_review.addWidget(self.lbl_review)
        h_review.addWidget(btn_next_l0)
        v_det.addLayout(h_review)
        # SNR filter: detections below it are hidden, not edited / counted, and not saved (hand markings: 100)
        h_snr = QHBoxLayout()
        h_snr.addWidget(QLabel("SNR \u2265"))
        self.slider_snr = QSlider(Qt.Horizontal)
        self.slider_snr.setRange(0, 200) # 0.5 steps up to 100
        self.sb_snr = QDoubleSpinBox()
        self.sb_snr.setRange(0.0, 100.0); self.sb_snr.setSingleStep(0.5); self.sb_snr.setDecimals(1)
        tip = ("SNR filter: detected voids whose SNR (signal on their layer / background noise) is below this are\n"
               "hidden, skipped by Next / editing / counts, and left out when saving or exporting (the value is\n"
               "written to meta.json as snr_min). Hand markings have SNR 100; bboxes are never filtered. 0 = off.")
        for w_ in (self.slider_snr, self.sb_snr):
            w_.setToolTip(tip)
        self.slider_snr.valueChanged.connect(lambda v: self.sb_snr.setValue(v / 2.0))
        self.sb_snr.valueChanged.connect(self._set_snr_min)
        h_snr.addWidget(self.slider_snr, 1)
        h_snr.addWidget(self.sb_snr)
        v_det.addLayout(h_snr)

        self.btn_detect_adv = QToolButton()
        self.btn_detect_adv.setText("Advanced")
        self.btn_detect_adv.setCheckable(True)
        self.btn_detect_adv.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_detect_adv.setAutoRaise(True)
        self.btn_detect_adv.setToolTip("Method, layer options, learned filter training and evaluation")
        v_det.addWidget(self.btn_detect_adv)
        self.detect_adv = QWidget()
        v_adv = QVBoxLayout(self.detect_adv)
        v_adv.setContentsMargins(12, 0, 0, 0)
        h_method = QHBoxLayout()
        h_method.addWidget(QLabel("Method:"))
        h_method.addWidget(self.cb_detect_method, 1)
        v_adv.addLayout(h_method)
        self.btn_eval = btn_eval = QPushButton("Evaluate")
        btn_eval.setToolTip("Run the detection without marking and compare it with the voids marked by hand on the loaded\n"
                            "layers: misses and false detections with their reasons, as a short code to type and send,\n"
                            "plus a report folder with chip images (see docs/VOID_EVAL.md).")
        btn_eval.clicked.connect(self.evaluate_detection)
        # Layer of a spot found on several layers
        h_rule = QHBoxLayout()
        h_rule.addWidget(QLabel("Mark on layer:"))
        self.cb_layer_rule = QComboBox()
        for label, key in (("First (from top)", "first"), ("First strong", "first_strong"), ("Strongest", "strongest"),
                           ("Last", "last")):
            self.cb_layer_rule.addItem(label, key)
        self.cb_layer_rule.setToolTip("A spot found on several layers is marked once, on:\n"
                                      "First: the first layer where it appears (from the top)\n"
                                      "First strong (Layer change): the first layer where it is clearly there (evidence at\n"
                                      "least half its peak, lc_first_frac), not where it only starts to show through\n"
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
        v_adv.addLayout(h_rule)
        h_learn = QHBoxLayout()
        self.chk_learned = QCheckBox("Learned filter")
        self.chk_learned.setToolTip("Die-to-die / Layer change: candidates scored by a model trained from hand markings\n"
                                    "(Train; one model per method, Layer change one per product) instead of the Thr\n"
                                    "threshold (sat_widgets/void_learn.py). Turned on when the model exists.")
        self.btn_train = btn_train = QPushButton("Train")
        btn_train.setToolTip("Add this file's hand markings to the training set and retrain the learned filter\n"
                             "(one entry per file; training the same file again replaces it)")
        btn_train.clicked.connect(self.train_detector)
        btn_reset_learn = QPushButton("Reset")
        btn_reset_learn.setToolTip("Starts the learned filter of this method (Layer change: of this product) over: its model and\n"
                                   "training set are renamed to *.bak (kept, can be restored). A shared model declared in\n"
                                   "products.json / config.json is not touched.")
        btn_reset_learn.clicked.connect(self.reset_learned)
        h_learn.addWidget(self.chk_learned)
        h_learn.addWidget(btn_train)
        h_learn.addWidget(btn_reset_learn)
        h_learn.addWidget(btn_eval)
        btn_seg = QPushButton("Seg Data")
        btn_seg.setToolTip("Build and review the U-Net training data from marked files (docs/SEGMENTATION.md)")
        btn_seg.clicked.connect(self.open_seg_data)
        h_learn.addWidget(btn_seg)
        v_adv.addLayout(h_learn)
        self.chk_detect_state = QCheckBox("Cell status")
        self.chk_detect_state.setChecked(True)
        self.chk_detect_state.setToolTip("Detect Voids progress per cell: a circle at the top-right corner of each inspected cell\n"
                                         "(empty: pending, filling: in progress, check: done); shown with the grid")
        self.chk_detect_state.toggled.connect(self._toggle_detect_state)
        self.chk_outlines = QCheckBox("Shapes (debug)")
        self.chk_outlines.setToolTip("Draw the shape Layer change found for each void (magenta): the region of strong\n"
                                     "evidence its box comes from. For checking the detection; saved in voids.json\n"
                                     "(outline), follows the void when it is moved or resized.")
        self.chk_outlines.toggled.connect(self._toggle_outlines)
        self.chk_outlines.setChecked(self._ui_state("show_outlines", False))
        h_dbg = QHBoxLayout()
        h_dbg.addWidget(self.chk_detect_state)
        h_dbg.addWidget(self.chk_outlines)
        h_dbg.addStretch(1)
        v_adv.addLayout(h_dbg)
        v_det.addWidget(self.detect_adv)
        self.btn_detect_adv.toggled.connect(self._show_detect_advanced)
        self.btn_detect_adv.setChecked(self._ui_state("detect_advanced", False))
        self._show_detect_advanced(self.btn_detect_adv.isChecked())
        self._init_detect_method()
        self._detect_job = None # Detect Voids running in the background (sat_widgets/detect_job.py)
        self._detect_added = 0
        self._review_n, self._review_pos = 0, None # Void Detection review counter
        
        dock_layout.addWidget(self.gb_void)
        dock_layout.addWidget(self.gb_detect)
        self._build_edge_stain_group(dock_layout)
        
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
        # B (eventFilter too): set / clear the "visible on layer 0" flag of the hovered void
        self.action_toggle_layer0 = QAction("Toggle Layer-0 Flag Of Hovered Void", self)
        self.action_toggle_layer0.triggered.connect(self.toggle_layer0_flag)


        
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
        btn_auto_bbox.setToolTip("Place the bbox of every (bonded) cell (declared size): aligned on the chip layout of the\n"
                                 "loaded layers, no package edge needed (config auto_bbox.method \"edge\": the package edges)")
        btn_auto_bbox.clicked.connect(self.auto_bbox)
        btn_refine_bbox = QPushButton("Refine BBOX")
        btn_refine_bbox.setToolTip("Align the bboxes on the layout again. Bboxes placed or moved by hand stay\n"
                                   "and set the absolute place for the others")
        btn_refine_bbox.clicked.connect(self.refine_bbox)
        btn_edge_bbox = QPushButton("Edge BBOX")
        btn_edge_bbox.setToolTip("Move every bbox onto the chip outline: the steepest brightness step near each of its four\n"
                                 "edges (profile through the chip middle, raw image, sub px). Size: the declared one.\n"
                                 "A chip with one side off (stain, chipped edge) keeps the steeper side. Hand-placed stay.")
        btn_edge_bbox.clicked.connect(self.edge_bbox)
        self.chk_bbox_from_grid = QCheckBox("From grid")
        self.chk_bbox_from_grid.setToolTip("Detect / Refine BBOX keep the grid's place: the bboxes are aligned with each other\n"
                                           "around the cell centres, not moved as a whole by the package edges.\n"
                                           "For a grid set well by hand, or narrow streets between the chips.")
        self.chk_bbox_from_grid.setChecked(self._ui_state("bbox_from_grid", (self._config().get("auto_bbox") or {}).get("absolute") == "grid"))
        self.chk_bbox_from_grid.toggled.connect(lambda on: self._set_ui_state("bbox_from_grid", on))
        btn_fit_grid = QPushButton("Fit Grid")
        btn_fit_grid.setToolTip("Fit the grid origin, pitch and angle to the bbox centers (asks before applying)")
        btn_fit_grid.clicked.connect(self.fit_grid_to_bboxes)
        self.chk_lock_bbox = QCheckBox("Lock BBOX")
        self.chk_lock_bbox.setToolTip("The bboxes are shown but not hovered, selected, moved, erased or retyped:\n"
                                      "the markings inside them are edited on the topmost layer without catching the bbox")
        self.chk_lock_bbox.toggled.connect(self._set_bbox_lock)
        self.chk_lock_bbox.setChecked(self._ui_state("bbox_locked", True)) # Remembered per PC; on by default
        btn_grid_bbox = QPushButton("Grid BBOX")
        btn_grid_bbox.setToolTip("Put the bbox (declared size) of every bonded cell exactly at the grid cell centre:\n"
                                 "no image analysis. For a grid fitted well by hand (Ctrl+Z undoes).")
        btn_grid_bbox.clicked.connect(self.grid_bbox)
        btn_next_est = QPushButton("Next Estimated")
        btn_next_est.setToolTip("Go to the next bbox that was estimated because detection failed (drawn dashed)")
        btn_next_est.clicked.connect(self.next_estimated_bbox)
        btn_bbox_grid = QPushButton("Auto BBOX + Grid")
        btn_bbox_grid.setToolTip("Detect BBOX, Fit Grid (applied without asking) and Detect BBOX again on the fitted grid,\n"
                                 "as the Auto Run does. Warns when the fit looks wrong (config auto_run qc limits).")
        btn_bbox_grid.clicked.connect(self.auto_bbox_grid)
        form_calib.addRow(btn_bbox_grid)
        h_auto.addWidget(btn_auto_bbox)
        h_auto.addWidget(btn_refine_bbox)
        h_auto.addWidget(btn_edge_bbox)
        h_auto.addWidget(self.chk_bbox_from_grid)
        form_calib.addRow(h_auto)
        h_auto2 = QHBoxLayout()
        h_auto2.addWidget(btn_grid_bbox)
        h_auto2.addWidget(self.chk_lock_bbox)
        h_auto2.addWidget(btn_fit_grid)
        h_edge = QHBoxLayout()
        h_edge.addWidget(QLabel("Edge pick"))
        self.cb_edge_pick = QComboBox()
        for key, label in (("steepest", "steepest"), ("outer", "outer"), ("inner", "inner")):
            self.cb_edge_pick.addItem(label, key)
        self.cb_edge_pick.setCurrentIndex(max(0, self.cb_edge_pick.findData(
            ((self._config().get("auto_bbox") or {}).get("edges") or {}).get("edge_pick", "steepest"))))
        self.cb_edge_pick.setToolTip("Edge BBOX: which brightness step near each bbox edge is the chip edge: the steepest,\n"
                                     "or the outermost / innermost of the steps at least half as steep (a bright rim, a halo\n"
                                     "or a scan tail next to the outline)")
        h_edge.addWidget(self.cb_edge_pick)
        btn_edge_dbg = QPushButton("Edge debug")
        btn_edge_dbg.setToolTip("Edge BBOX on the chip at the view centre without changing anything: the window with the\n"
                                "bbox now (yellow) and fitted (green), and the brightness profiles across the chip with the\n"
                                "search ranges and the steps taken")
        btn_edge_dbg.clicked.connect(self.edge_bbox_debug)
        h_edge.addWidget(btn_edge_dbg)
        h_edge.addStretch(1)
        h_auto2.addWidget(btn_next_est)
        form_calib.addRow(h_auto2)
        form_calib.addRow(h_edge)

        # Panel order: ... Bonding Map, Auto Calibration, Coordinates, Void Marking, Extraction
        dock_layout.insertWidget(dock_layout.indexOf(gb_map) + 1, gb_calib)
        dock_layout.addWidget(gb_extract)
        # Auto Run: watched measured files -> automatic marking (sat_widgets/auto_run.py)
        self.auto_run = AutoRunController(self)
        dock_layout.addWidget(self.auto_run.panel)
        dock_layout.addStretch()
        

        # Scroll Area Wrapper
        scroll = QScrollArea()
        scroll.setWidget(dock_content)
        scroll.setWidgetResizable(True)
        # scroll.setMaximumHeight(1000) # Removed fixed limit
        
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # No horizontal scrolling: the dock is never narrower than its content (+ vertical scroll bar)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(dock_content.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
                               + 2 * scroll.frameWidth())

        dock.setWidget(scroll)
        self.resizeDocks([dock], [540], Qt.Horizontal) # Initial width; the splitter can still move it

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
                raise ValueError(f"expected first / first_strong / strongest / last, got {v!r}")
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
            "void_line_width": lambda v: setattr(self.glw, "void_line_width", max(0.5, float(v))),
            "void_min_px": lambda v: setattr(self.glw, "void_min_px", max(0.0, float(v))),
            "angle_step": lambda v: self.sb_angle.setSingleStep(max(1e-3, float(v))),
            "snr_min": lambda v: self.sb_snr.setValue(float(v)),
            "origin_x": lambda v: self.sb_grid_x.setValue(float(v)),
            "origin_y": lambda v: self.sb_grid_y.setValue(float(v)),
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
        self.current_product_name = name
        if prod.get("detect_method") and hasattr(self, "cb_detect_method"):
            self._select_detect_method(prod["detect_method"])
        if hasattr(self, "chk_learned"):
            self._update_learned_state() # Layer change: one learned model per product
        
        if "um_per_px" in prod:
            self.sb_um_per_px.blockSignals(True)
            self.sb_um_per_px.setValue(float(prod["um_per_px"]))
            self.sb_um_per_px.blockSignals(False)
        self.void_manager.um_per_px = self.sb_um_per_px.value()

        # Grid: only declared keys (a pitch in um through the Scale)
        self._set_grid_spins(grid_values(prod, self.sb_um_per_px.value()))
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

    def _set_grid_spins(self, values):
        spin = {"start_x": self.sb_grid_x, "start_y": self.sb_grid_y, "pitch_x": self.sb_pitch_x,
                "pitch_y": self.sb_pitch_y, "rows": self.sb_rows, "cols": self.sb_cols, "angle": self.sb_angle}
        for key, value in values.items():
            spin[key].blockSignals(True)
            spin[key].setValue(value)
            spin[key].blockSignals(False)

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

    def copy_chip_image(self):
        """
        Ctrl+C on the image: the chip under the mouse (else at the view centre) as the export's overlay
        image (current layer, its markings solid, other layers' as ghosts, product layout, title bar)
        into the clipboard.
        """
        from PySide6.QtGui import QCursor
        glw = self.glw
        if self.full_data is None or glw.tiled_image is None:
            return
        cfg, vm = glw.grid_cfg, self.void_manager
        pos = glw.mapFromGlobal(QCursor.pos())
        at = QPointF(pos) if glw.rect().contains(pos) else QPointF(glw.width() / 2, glw.height() / 2)
        r, c = cfg.cell_at(*glw.get_image_coords(at))
        if not cfg.contains(r, c):
            self.statusBar().showMessage("Copy: no grid cell here")
            return
        x0, y0 = cfg.cell_origin(r, c)
        quad = [glw.deskew_to_raw(x, y) for x, y in ((x0, y0), (x0 + cfg.pitch_x, y0),
                                                       (x0 + cfg.pitch_x, y0 + cfg.pitch_y), (x0, y0 + cfg.pitch_y))]
        z = self.current_z
        layer = self.abs_layer(z)
        scale, offset = 1.0, 0.0
        calib = self.layer_calib_data.get(z) if self.layer_calib_data else None
        if calib is not None and r < calib.shape[0] and c < calib.shape[1]:
            scale, offset = float(calib[r, c, 0]), float(calib[r, c, 1])
        ex = ExportManager(self) # The export's drawing: same picture as overlay/
        ex._refs = vm.reference_bboxes(cfg, glw.coord_transform)
        patch = ex._extract_patch_unified(z, quad, glw.win_lo, glw.win_hi, scale, offset)
        self.full_data.release_cache()
        if patch is None:
            self.statusBar().showMessage("Copy: the chip is outside the image")
            return
        img = patch.convertToFormat(QImage.Format_RGB32)
        ex._draw_chip_voids(img, ex._index_voids(cfg), cfg, r, c, mode='overlay', current_layer=layer)
        leg = (glw.bonding_map.get_key(r, c) if glw.bonding_map else "") or ""
        title = f"X{c:02d}_Y{r:02d}_L{layer:02d}" + (f"_LEG:{ex._safe(leg)}" if leg else "")
        img = ex._add_title_bar(img, title)
        QApplication.clipboard().setImage(img)
        self.statusBar().showMessage(f"Copied {title} ({img.width()} x {img.height()}) to the clipboard")

    def measure_scale(self):
        """Measure: two clicks on the image (GLImageWidget.start_measure), then the real length -> Scale."""
        if self.full_data is None:
            return
        if not getattr(self, "_measure_connected", False):
            self.glw.measure_done.connect(self._on_measured)
            self._measure_connected = True
        self.glw.start_measure()
        self.statusBar().showMessage("Measure: click the two ends of a known length on the image (Esc cancels)")

    def _on_measured(self, px):
        from PySide6.QtWidgets import QInputDialog
        if px < 1.0:
            self.statusBar().showMessage("Measure: the two points are the same, measure again")
            return
        um, ok = QInputDialog.getDouble(self, "Measure Scale", f"Distance on the image: {px:.2f} px\n"
                                        "Real length (um):", getattr(self, "_measure_last_um", 1000.0), 0.0001, 1e9, 3)
        if not ok or um <= 0:
            self.statusBar().showMessage("Measure canceled")
            return
        self._measure_last_um = um
        self.sb_um_per_px.setValue(um / px) # Updates the bbox size and a pitch declared in um
        msg = f"Scale = {um:g} um / {px:.2f} px = {um / px:.4f} um/px"
        self.statusBar().showMessage(msg)
        print(msg)

    def on_scale_changed(self, value):
        self.void_manager.um_per_px = value
        self._apply_bbox_size()
        prod = getattr(self, "current_product", None)
        if value > 0 and pitch_um(prod): # A pitch declared in um follows the Scale
            self._set_grid_spins({k: v for k, v in grid_values(prod, value).items() if k in ("pitch_x", "pitch_y")})
            self.update_grid_params()
        if self.glw.void_mode:
            self.glw.update_cursor_style()
        self.glw.update()

    def undo_void_change(self):
        if self._detect_busy():
            return
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
                if isinstance(wid, dict): # {"wafer_id", "lot"}
                    wid = wid.get("wafer_id")
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
            text, reject = results_api.normalize_bonding_map(result, self._results_settings().get("bonding_map"))
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
        # B: set / clear the layer-0 flag of the hovered void.
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
            if et == QEvent.KeyPress and key == Qt.Key_B and plain and not self._typing(obj):
                if not ev.isAutoRepeat():
                    self.action_toggle_layer0.trigger()
                return True
        return super().eventFilter(obj, ev)

    def _on_app_state(self, state):
        # Key releases are lost while another application is active
        if state != Qt.ApplicationActive and self._held_modes:
            self._held_modes.clear()
            self._apply_held_mode()

    def _apply_held_mode(self):
        tool = self._held_modes[-1] if self._held_modes and not self.glw.edit_locked else None
        rb = {None: self.rb_view, "DRAW": self.rb_draw, "EDIT": self.rb_edit, "ERASE": self.rb_erase}[tool]
        rb.setChecked(True)
        self.glw.set_void_tool(tool)

    def set_void_mode_view(self):
        self._held_modes.clear()
        self._apply_held_mode()
        
    def load_voids(self):
        if self._detect_busy():
            return
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
        if self._detect_busy():
            return
        dlg = VoidTypeDialog(self.void_manager, self)
        dlg.exec()
        self.populate_void_types()
        self.glw.update()

    def save_voids(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Voids JSON", "voids.json", "JSON (*.json)")
        if path:
            self.void_manager.save_to_file(path, self.glw.grid_cfg, self.glw.bonding_map, self.glw.coord_transform,
                                           self.void_meta())
            vm = self.void_manager
            hidden = sum(1 for vs in vm.voids.values() for v in vs if not vm.passes(v))
            if hidden:
                self.statusBar().showMessage(f"Saved without {hidden} voids below SNR {vm.snr_min:.1f} (snr_min in meta.json)")

    def void_meta(self):
        """Source information written to meta.json next to voids.json."""
        prod = getattr(self, "current_product", None)
        return {"app_version": APP_VERSION, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source_file": self.lbl_source.toolTip(), "wafer_id": self.le_wafer_id.text(),
                "product": self.cb_product.currentText() if prod is not None else "",
                "layers": [self.abs_layer(z) for z in range(len(self.full_data))] if self.full_data is not None else []}
            
    def clear_voids(self):
        if self._detect_busy():
            return
        # Clear CURRENT LAYER Only
        self.void_manager.clear_layer(self.glw.current_layer)
        self.update_void_ui()
        self.glw.active_void = None
        self.glw.setFocus()
        self.glw.update()
        
    def clear_chip_voids(self):
        if self._detect_busy():
            return
        self.glw.clear_current_chip_voids()
        self.update_void_ui()
        self.glw.setFocus()

    def clear_all_voids(self):
        """Removes every marking on all layers (bboxes too) after a confirmation. One undo step."""
        if self._detect_busy():
            return
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

    def clear_voids_keep_bbox(self):
        """Removes every void on all layers (SNR-hidden ones too) and keeps the bboxes. One undo step."""
        if self._detect_busy():
            return
        vm = self.void_manager
        n = sum(1 for vs in vm.voids.values() for v in vs if not vm.is_bbox(v))
        if n == 0:
            self.statusBar().showMessage("No voids to remove (bboxes kept).")
            return
        hidden = sum(1 for vs in vm.voids.values() for v in vs if not vm.is_bbox(v) and not vm.passes(v))
        if QMessageBox.question(self, "Clear Voids",
                                f"Remove all {n} voids on all layers" + (f" ({hidden} hidden by the SNR filter)" if hidden else "")
                                + "?\nThe bboxes stay. Ctrl+Z restores them.",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        vm.remove_voids_where(lambda layer, v: not vm.is_bbox(v))
        self.glw.active_void = None
        self.update_void_ui()
        self.glw.refresh_hover()
        self.glw.update()
        self.statusBar().showMessage(f"Removed {n} voids; bboxes kept (Ctrl+Z restores).")
        self.glw.setFocus()

    def _bbox_type_name(self):
        for name in self.void_manager.types:
            if name.strip().lower() == self.void_manager.BBOX_TYPE_NAME:
                return name
        return None

    def auto_bbox_grid(self):
        """One click: Detect BBOX -> Fit Grid (applied) -> Detect BBOX on the fitted grid (pipeline.step_bbox)."""
        if self._detect_busy() or self.full_data is None or self.glw.tiled_image is None:
            return
        cfg = self.glw.grid_cfg
        before = {k: getattr(cfg, k) for k in ("start_x", "start_y", "pitch_x", "pitch_y", "angle")}
        first = self.auto_bbox(report=False)
        if first is None:
            return
        new = self.fit_grid_to_bboxes(confirm=False)
        if new is None:
            self._report_layout_bbox("Auto BBOX + Grid (grid not fitted)", first, len(self._bbox_setup("Auto BBOX")[2]))
            return
        second = self.auto_bbox(report=False)
        if second is None:
            return
        n_cells = len(self._bbox_setup("Auto BBOX")[2])
        self._report_layout_bbox("Auto BBOX + Grid", second, n_cells)
        msg = (f"Auto BBOX + Grid: origin ({before['start_x']:.1f}, {before['start_y']:.1f}) -> ({new['Origin X']:.1f}, "
               f"{new['Origin Y']:.1f}), pitch ({new['Pitch X']:.2f}, {new['Pitch Y']:.2f}), angle {before['angle']:.3f} -> "
               f"{new['Angle']:.3f} deg, residual {new['rms']:.2f} px; {second['refined']} aligned, "
               f"{len(second['estimated'])} estimated")
        self.statusBar().showMessage(msg)
        print(msg)
        qc = dict(self._config().get("auto_run", {}).get("qc") or {})
        lim_rms, lim_ang = qc.get("max_fit_rms_px", 3.0), qc.get("max_angle_change", 1.0)
        bad = []
        if lim_rms is not None and new["rms"] > lim_rms:
            bad.append(f"residual {new['rms']:.2f} px > {lim_rms} px")
        if lim_ang is not None and abs(new["Angle"] - before["angle"]) > lim_ang:
            bad.append(f"angle changed by {abs(new['Angle'] - before['angle']):.2f} deg (> {lim_ang})")
        if bad:
            QMessageBox.warning(self, "Auto BBOX + Grid", "The fitted grid looks wrong: " + "; ".join(bad) +
                                ".\nCheck the origin / pitch (Ctrl+Z undoes the bboxes; set the grid back by hand).")

    def auto_bbox(self, report=True):
        """Detects and draws the reference bbox of every (bonded) cell on the topmost layer; returns the result."""
        if self._detect_busy():
            return None
        if self.full_data is None or self.glw.tiled_image is None:
            return None
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

        shown = [0.0]

        def step(i): # Repaint / Cancel at most every 0.1 s, whatever the step numbers
            now = time.monotonic()
            if now - shown[0] >= 0.1:
                shown[0] = now
                progress.setValue(min(int(i), progress.maximum()))
                QApplication.processEvents()
            return not progress.wasCanceled()

        top_layer = self.abs_layer(0)
        vm = self.void_manager
        params = dict(self._config().get("auto_bbox") or {})
        if self.chk_bbox_from_grid.isChecked():
            params["absolute"] = "grid"
        layout = params.get("method", "layout") == "layout"
        zs = list(range(len(self.full_data)))
        if layout:
            progress.setMaximum(max(1, len(cells) * min(len(zs), int(params.get("refine_layers", 4)))))
        vm.begin_change() # One undo step with the corner voids following the new bboxes
        try:
            if layout: # From the cell centres, aligned on the layout (blurred / noisy package edges)
                result = detect_layout(self.full_data, zs, top_layer, cfg, self.glw.coord_transform, vm, tid, size,
                                       cells, params, step)
            else:
                result = place_bboxes(self.full_data, 0, top_layer, cfg, self.glw.coord_transform, vm,
                                      tid, size, cells, params, step)
                if result is not None and params.get("layout_refine", True) and result["detected"] >= 3:
                    progress.setValue(0)
                    progress.setMaximum(max(1, len(cells) * min(len(zs), int(params.get("refine_layers", 4)))))
                    ref = refine_bboxes(self.full_data, zs, top_layer, cfg, self.glw.coord_transform, vm, tid, size,
                                        cells, params, step)
                    if ref is not None:
                        result["layout"] = ref
            if result is not None:
                vm.resnap(cfg, self.glw.coord_transform)
        finally:
            vm.end_change()
            progress.close()
        if result is None:
            if layout:
                QMessageBox.warning(self, "Detect BBOX", "Not enough chips to align (at least 3 bonded cells).")
            return None
        self.glw.update()
        self.update_void_ui()
        if layout:
            if report:
                self._report_layout_bbox("Detect BBOX", result, len(cells))
            return result
        msg = (f"Detect BBOX on layer {top_layer}: {result['detected']} detected, "
               f"{len(result['estimated'])} estimated (dashed, check with 'Next Estimated'), "
               f"{len(result['failed'])} failed")
        if "layout" in result:
            msg += f"; aligned on the layout: moved {result['layout']['moved_px']:.1f} px (median)"
        self.statusBar().showMessage(msg)
        print(msg)
        if result["failed"] and not result["detected"]:
            QMessageBox.warning(self, "Detect BBOX", "No box edge could be detected. Check the grid origin/pitch "
                                "and the bbox size, or adjust auto_bbox in config.json.")
        return result

    def _report_layout_bbox(self, title, result, n_cells):
        est = len(result["estimated"])
        loc = result.get("located")
        msg = (f"{title}: {result['refined']} aligned on the layout, {est} estimated (dashed, 'Next Estimated'), "
               f"{result['hand']} kept (by hand); place from {result['absolute']}; match {result['ncc']:.2f}; "
               f"moved {result['moved_px']:.1f} px (median)")
        if loc:
            msg += f"; chip outline over a whole pitch: shifted ({loc['dx']:.0f}, {loc['dy']:.0f}) px"
        self.statusBar().showMessage(msg)
        print(msg)
        if "located" in result and not loc and result["absolute"] == "mean place" and not result["hand"]:
            QMessageBox.warning(self, title, "The chip outline was not found: the bboxes are aligned with each other, "
                                "but all of them may be shifted by the same amount.\nIf so, move one bbox onto its chip "
                                "by hand and press Refine BBOX: the others follow it.")
        if n_cells and est > 0.3 * n_cells:
            QMessageBox.warning(self, title, f"{est} of {n_cells} bonded cells do not match the chip layout.\n"
                                "Check the grid origin (a shift by a cell puts the bonding map on other chips), "
                                "the pitch and the bbox size.")

    def _bbox_setup(self, title):
        if self.full_data is None or self.glw.tiled_image is None:
            return None
        tid = self._bbox_type_name()
        if tid is None:
            QMessageBox.warning(self, title, "No void type named 'bbox' is declared in config.json.")
            return None
        t = self.void_manager.types[tid]
        size = (t["def_w"], t["def_h"]) if t.get("def_w", 0) > 0 and t.get("def_h", 0) > 0 else None
        cfg = self.glw.grid_cfg
        bmap = self.glw.bonding_map
        cells = [(r, c) for r in range(cfg.rows) for c in range(cfg.cols) if bmap is None or bmap.get_key(r, c)]
        return tid, size, cells

    def edge_bbox(self):
        """Edge BBOX: every bbox moved onto the chip outline (auto_bbox.fit_edges), one undo step."""
        if self._detect_busy():
            return
        setup = self._bbox_setup("Edge BBOX")
        if setup is None:
            return
        tid, size, cells = setup
        cfg, vm = self.glw.grid_cfg, self.void_manager
        params = dict((self._config().get("auto_bbox") or {}).get("edges") or {})
        params["edge_pick"] = self.cb_edge_pick.currentData()
        progress = QProgressDialog("Fitting the bboxes to the chip edges...", "Cancel", 0, max(1, len(cells)), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        shown = [0.0]

        def step(i):
            now = time.monotonic()
            if now - shown[0] >= 0.1:
                shown[0] = now
                progress.setValue(min(int(i), progress.maximum()))
                QApplication.processEvents()
            return not progress.wasCanceled()
        z = self.glw.current_layer - self.layer_offset if 0 <= self.glw.current_layer - self.layer_offset < len(self.full_data) else 0
        vm.begin_change()
        try:
            result = fit_edges(self.full_data, z, cfg, self.glw.coord_transform, vm, tid, size, cells, params, step)
            if result is not None:
                vm.resnap(cfg, self.glw.coord_transform)
        finally:
            vm.end_change()
            progress.close()
        if result is None:
            if not progress.wasCanceled():
                QMessageBox.warning(self, "Edge BBOX", "Draw or detect the bboxes first.")
            return
        self.glw.update()
        self.update_void_ui()
        sz = result.get("size")
        size_txt = "" if not sz else (f", size {sz[0]:.1f} x {sz[1]:.1f} px" + ("" if size and abs(size[0] - sz[0]) < 1e-6 else
                                      f" (measured; declared {size[0]:.0f} x {size[1]:.0f} is off)" if size else " (measured)"))
        msg = (f"Edge BBOX (layer {self.abs_layer(z)}){size_txt}: {result['fitted']} fitted to the chip edges, moved "
               f"{result['moved_px']:.1f} px (median); {result['one_side']} sides off (kept the steeper side); "
               f"{len(result['failed'])} without a clear edge (unchanged); {result['hand']} hand-placed kept. Ctrl+Z undoes")
        self.statusBar().showMessage(msg)
        print(msg)

    def edge_bbox_debug(self):
        """Edge BBOX on the chip at the view centre, nothing changed: window + profiles with the steps taken."""
        from PIL import Image, ImageDraw
        setup = self._bbox_setup("Edge debug")
        if setup is None:
            return
        tid, size, cells = setup
        glw, vm = self.glw, self.void_manager
        cell = glw.grid_cfg.cell_at(*glw.get_image_coords(QPointF(glw.width() / 2, glw.height() / 2)))
        params = dict((self._config().get("auto_bbox") or {}).get("edges") or {})
        params.update(edge_pick=self.cb_edge_pick.currentData(), apply=False, trace={"cell": cell})
        z = glw.current_layer - self.layer_offset if 0 <= glw.current_layer - self.layer_offset < len(self.full_data) else 0
        res = fit_edges(self.full_data, z, glw.grid_cfg, glw.coord_transform, vm, tid, size, cells, params)
        tr = params["trace"]
        if res is None or "px" not in tr:
            QMessageBox.information(self, "Edge debug", f"Chip X{cell[1]:02d}_Y{cell[0]:02d}: no bbox to fit (or placed by hand).")
            return
        L, Rr, T, B = tr["steps"]
        X0, Y0 = tr["X0"], tr["Y0"]
        win = tr["window"]
        lo, hi = np.percentile(win, [1, 99])
        g = np.clip((win - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        k = max(1, int(900 // max(win.shape)))
        im = Image.fromarray(g).convert("RGB").resize((win.shape[1] * k, win.shape[0] * k), Image.NEAREST)
        d = ImageDraw.Draw(im)

        def box(b, color):
            if b is None:
                return
            cx, cy, w, h = b
            d.rectangle([(cx - w / 2 - X0) * k, (cy - h / 2 - Y0) * k, (cx + w / 2 - X0) * k, (cy + h / 2 - Y0) * k], outline=color, width=2)
        box(tr.get("old"), (255, 220, 0))
        box(tr.get("new"), (0, 255, 80))
        for st, axis in ((L, "x"), (Rr, "x"), (T, "y"), (B, "y")): # The steps taken (thin red)
            if st is None:
                continue
            if axis == "x":
                d.line([(st[0] * k, 0), (st[0] * k, im.size[1])], fill=(255, 40, 40), width=1)
            else:
                d.line([(0, st[0] * k), (im.size[0], st[0] * k)], fill=(255, 40, 40), width=1)

        def plot(prof, e, m, st, title):
            """The profile around one bbox edge (+- 2 x the search range), zoomed."""
            i0, i1 = max(0, int(e - 2 * m)), min(len(prof), int(e + 2 * m) + 2)
            seg = prof[i0:i1]
            n = len(seg)
            kx = max(1, int(560 // max(n, 1)))
            W, H = n * kx, 260
            pim = Image.new("RGB", (W, H), (20, 20, 20))
            pd_ = ImageDraw.Draw(pim)
            pd_.rectangle([(e - m - i0) * kx, 0, (e + m - i0) * kx, H], fill=(45, 45, 45)) # Search range
            gr = np.abs(np.diff(seg))
            for arr, color, top, x_off in ((seg, (230, 230, 230), 20, 0.5), (gr, (255, 150, 0), 140, 1.0)):
                if len(arr) < 2:
                    continue
                a0, a1 = float(arr.min()), float(arr.max())
                pts = [((j + x_off) * kx, top + 100 - (v - a0) / max(a1 - a0, 1e-6) * 100) for j, v in enumerate(arr)]
                pd_.line(pts, fill=color, width=2)
            pd_.line([((e - i0) * kx, 0), ((e - i0) * kx, H)], fill=(255, 220, 0), width=1)
            if st is not None:
                pd_.line([((st[0] - i0) * kx, 0), ((st[0] - i0) * kx, H)], fill=(0, 255, 80), width=2)
            pd_.text((4, 2), title, fill=(255, 255, 255))
            return np.asarray(pim)
        cap = "profile (white) and step size (orange): search range (gray), bbox now (yellow), step taken (green)"
        steps = [("1 window: bbox now (yellow), fitted (green), steps taken (red)", np.asarray(im), "rgb"),
                 ("2 left edge: " + cap, plot(tr["px"], tr["ex"][0], tr["mx"], L, "left"), "rgb"),
                 ("3 right edge", plot(tr["px"], tr["ex"][1], tr["mx"], Rr, "right"), "rgb"),
                 ("4 top edge", plot(tr["py"], tr["ey"][0], tr["my"], T, "top"), "rgb"),
                 ("5 bottom edge", plot(tr["py"], tr["ey"][1], tr["my"], B, "bottom"), "rgb")]
        old, new = tr.get("old"), tr.get("new")
        info = [f"Chip X{cell[1]:02d}_Y{cell[0]:02d}, layer {self.abs_layer(z)}, Edge pick {params['edge_pick']}"]
        if old and new:
            info.append(f"bbox now: centre ({old[0]:.2f}, {old[1]:.2f}), {old[2]:.1f} x {old[3]:.1f} px; fitted: centre "
                        f"({new[0]:.2f}, {new[1]:.2f}) = moved ({new[0] - old[0]:+.2f}, {new[1] - old[1]:+.2f}) px, {new[2]:.1f} x {new[3]:.1f}")
        names = ("left", "right", "top", "bottom")
        def step_txt(i, n, st):
            if st is None:
                return f"{n} -"
            pos = st[0] + (X0 if i < 2 else Y0)
            return f"{n} {pos:.2f} ({'rising' if st[2] > 0 else 'falling'}, {st[1]:.1f})"
        info.append("steps (raw px): " + ", ".join(step_txt(i, n, st) for i, (n, st) in enumerate(zip(names, (L, Rr, T, B)))))
        info.append(f"step direction of each side (majority of the chips): {tr['signs']}; measured size {res.get('measured')}")
        self._show_trace(f"Edge BBOX debug: X{cell[1]:02d}_Y{cell[0]:02d}", info, steps, {})

    def refine_bbox(self):
        """Aligns the bboxes on the layout again; hand-placed ones stay and set the place (auto_bbox.py)."""
        if self._detect_busy():
            return
        setup = self._bbox_setup("Refine BBOX")
        if setup is None:
            return
        tid, size, cells = setup
        cfg, vm = self.glw.grid_cfg, self.void_manager
        params = dict(self._config().get("auto_bbox") or {})
        if self.chk_bbox_from_grid.isChecked():
            params["absolute"] = "grid"
        zs = list(range(len(self.full_data)))
        progress = QProgressDialog("Aligning bboxes on the layout...", "Cancel", 0,
                                   max(1, len(cells) * min(len(zs), int(params.get("refine_layers", 4)))), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        shown = [0.0]

        def step(i): # Repaint / Cancel at most every 0.1 s, whatever the step numbers
            now = time.monotonic()
            if now - shown[0] >= 0.1:
                shown[0] = now
                progress.setValue(min(int(i), progress.maximum()))
                QApplication.processEvents()
            return not progress.wasCanceled()

        vm.begin_change()
        try:
            result = refine_bboxes(self.full_data, zs, self.abs_layer(0), cfg, self.glw.coord_transform, vm, tid, size,
                                   cells, params, step)
            if result is not None:
                vm.resnap(cfg, self.glw.coord_transform)
        finally:
            vm.end_change()
            progress.close()
        if result is None:
            if not progress.wasCanceled():
                QMessageBox.warning(self, "Refine BBOX", "Draw or detect the bboxes first (at least 3).")
            return
        self.glw.update()
        self.update_void_ui()
        self._report_layout_bbox("Refine BBOX", result, len(cells))

    def fit_grid_to_bboxes(self, confirm=True):
        """Grid origin / pitch / angle fitted to the bbox centers (auto_bbox.fit_grid), applied when confirmed
        (confirm=False: applied directly): the cells end up centred on the bboxes (config auto_bbox keep_offset:
        the bboxes keep their median offset from the cell centres instead). Returns the applied values
        (+ "rms", "n") or None."""
        if self._detect_busy():
            return None
        if self.full_data is None:
            return None
        cfg, ct, vm = self.glw.grid_cfg, self.glw.coord_transform, self.void_manager
        f = fit_grid(vm, cfg, ct, self.full_data.width, self.full_data.height)
        if f is None:
            QMessageBox.warning(self, "Fit Grid", "At least 4 bboxes over 2 rows and 2 columns are needed (Detect BBOX first).")
            return None
        to_deskew = ct.raw_to_deskew if ct else (lambda x, y: (x, y))
        offs = []
        for vs in vm.voids.values():
            for v in vs:
                if vm.is_bbox(v) and v.get("auto") != "estimated":
                    gx, gy = to_deskew(v["globalCX"], v["globalCY"])
                    k = cfg.cell_at(gx, gy)
                    if k is not None:
                        offs.append((gx - cfg.start_x - (k[1] + 0.5) * cfg.pitch_x, gy - cfg.start_y - (k[0] + 0.5) * cfg.pitch_y))
        # The origin is corrected too (cells centred on the bboxes), unless config auto_bbox keep_offset: then
        # the bboxes keep their median place in the cells (a declared off-centre origin rule)
        keep = bool((self._config().get("auto_bbox") or {}).get("keep_offset", False))
        off = np.median(offs, axis=0) if (offs and keep) else (0.0, 0.0)
        new = {"Origin X": f["start_x"] - off[0], "Origin Y": f["start_y"] - off[1], "Pitch X": f["pitch_x"],
               "Pitch Y": f["pitch_y"], "Angle": f["angle"]}
        old = {"Origin X": cfg.start_x, "Origin Y": cfg.start_y, "Pitch X": cfg.pitch_x, "Pitch Y": cfg.pitch_y, "Angle": cfg.angle}
        rows = "\n".join(f"{k}: {old[k]:.3f} -> {new[k]:.3f}" for k in new)
        if confirm and QMessageBox.question(
                self, "Fit Grid", f"Grid fitted to {f['n']} bboxes (residual {f['rms']:.2f} px):\n\n{rows}\n\n"
                "Apply? (products.json is not changed)", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return None
        for sb, k in ((self.sb_grid_x, "Origin X"), (self.sb_grid_y, "Origin Y"), (self.sb_pitch_x, "Pitch X"),
                      (self.sb_pitch_y, "Pitch Y"), (self.sb_angle, "Angle")):
            sb.setValue(new[k])
        cfg.start_x, cfg.start_y = new["Origin X"], new["Origin Y"] # Exact (the boxes round)
        cfg.pitch_x, cfg.pitch_y, cfg.angle = new["Pitch X"], new["Pitch Y"], new["Angle"]
        self.glw.update_grid_params()
        self.statusBar().showMessage(f"Fit Grid applied ({f['n']} bboxes, residual {f['rms']:.2f} px)")
        return dict(new, rms=f["rms"], n=f["n"])

    def _set_bbox_lock(self, on):
        self._set_ui_state("bbox_locked", on)
        if not hasattr(self, "glw"):
            return
        self.glw.bbox_locked = on
        if self.glw.active_void is not None and self.void_manager.is_bbox(self.glw.active_void) and on:
            self.glw.active_void = None
        self.glw.refresh_hover()
        self.glw.update()

    def grid_bbox(self):
        """Bbox of every bonded cell at the grid cell centre (declared size), no image analysis. One undo step."""
        if self._detect_busy():
            return
        setup = self._bbox_setup("Grid BBOX")
        if setup is None:
            return
        tid, size, cells = setup
        if size is None:
            QMessageBox.warning(self, "Grid BBOX", "The bbox size is not known: set the Scale (chip_size_um) or the "
                                "bbox size of the product.")
            return
        cfg, vm = self.glw.grid_cfg, self.void_manager
        to_raw = self.glw.deskew_to_raw
        layer = self.abs_layer(0)
        vm.begin_change()
        try:
            for l, vs in list(vm.voids.items()):
                for v in [v for v in vs if vm.is_bbox(v)]:
                    vm.remove_void(l, v)
            for r, c in cells:
                cx = cfg.start_x + (c + 0.5) * cfg.pitch_x
                cy = cfg.start_y + (r + 0.5) * cfg.pitch_y
                v = vm.add_void(layer, *to_raw(cx, cy), size[0] / 2.0, size[1] / 2.0, tid)
                v["placed"] = "grid"
            vm.resnap(cfg, self.glw.coord_transform)
        finally:
            vm.end_change()
        self.glw.update()
        self.update_void_ui()
        msg = f"Grid BBOX: {len(cells)} bboxes at the grid cell centres ({size[0]:.0f} x {size[1]:.0f} px)"
        self.statusBar().showMessage(msg)
        print(msg)

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
        want = cfg.get("method") or "layer_change"
        for i in range(1, self.cb_detect_method.count()):
            method, model = self.cb_detect_method.itemData(i)
            item = self.cb_detect_method.model().item(i)
            if method == "layer_change":
                item.setToolTip("All loaded layers at once (load the layers to compare). Each layer of a chip is explained\n"
                                "from the chip's layers above it and the median chips; what is left is new on that layer.\n"
                                "Marks the layer where the void first appears (Mark on layer). Skipped layers still serve\n"
                                "as the layers above. Thr: evidence from the void's first layer down (default 26;\n"
                                "lower finds more and marks more falsely).")
                if want == "layer_change":
                    start = i
                continue
            if method == "ssim":
                item.setToolTip("Experimental: die-to-die on the local structure (SSIM against the median chip), robust to\n"
                                "a few px of misalignment; also for single-layer images. Thr in chip-to-chip spreads.")
                if want == "ssim":
                    start = i
                continue
            if method == "segmentation":
                reason = void_seg.available(model)
                item.setEnabled(not reason)
                item.setToolTip(reason or "Small U-Net trained from the reviewed markings (Seg Data, tools/train_seg.py), "
                                          "see docs/SEGMENTATION.md. Thr 5 = 50 % (segmentation)")
                continue
            reason = patchcore.available({"pc_model": model})
            item.setEnabled(not reason)
            item.setToolTip(reason or "CNN features, position-aware nearest neighbour (CPU), see docs/PATCHCORE.md")
            if not reason and cfg.get("method") == "patchcore" and model == (custom or patchcore.DEFAULTS["pc_model"]):
                start = i
        self.cb_detect_method.setCurrentIndex(start)
        self._on_detect_method()
        prod = getattr(self, "current_product", None)
        if prod and prod.get("detect_method"):
            self._select_detect_method(prod["detect_method"])

    def _select_detect_method(self, method):
        """Selects the first enabled item of a method ("die_to_die", "layer_change", "patchcore", "segmentation")."""
        for i in range(self.cb_detect_method.count()):
            if self.cb_detect_method.itemData(i)[0] == method and self.cb_detect_method.model().item(i).isEnabled():
                self.cb_detect_method.setCurrentIndex(i)
                return True
        print(f"[products] detect_method {method!r} is not available here")
        return False

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
            cfg.get("seg_threshold", 5.0) if method == "segmentation" else \
            cfg.get("lc_threshold", void_layer_change.DEFAULTS["lc_threshold"]) if method == "layer_change" else \
            cfg.get("ss_threshold", cfg.get("threshold", void_detect.DEFAULTS["threshold"])) if method == "ssim" else \
            cfg.get("threshold", void_detect.DEFAULTS["threshold"])
        self.sb_detect_thr.setValue(float(self._detect_thr.get(idx, default)))
        self._detect_last = idx
        if hasattr(self, "chk_learned"):
            self._update_learned_state()

    def _product_key(self):
        """(name, products.json entry) of the selected product, or None."""
        name = getattr(self, "current_product_name", None)
        return (name, getattr(self, "current_product", None) or {}) if name else None

    def _update_learned_state(self):
        """Learned filter: the model of this method (die-to-die and layer change have their own; layer
        change one per product, sat_widgets/void_learn.model_path)."""
        method = self.cb_detect_method.currentData()[0]
        cfg = self._config().get("void_detect", {})
        ok = method in ("die_to_die", "layer_change") and os.path.exists(void_learn.model_path(cfg, method, self._product_key()))
        self.chk_learned.setEnabled(ok)
        self.chk_learned.setChecked(ok) # Used whenever there is a model (uncheck to compare with Thr)

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
        if self._detect_busy():
            return
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

    def _build_edge_stain_group(self, dock_layout):
        """Edge Stains group: the button and its settings (start values: config void_detect.edge_band)."""
        from sat_widgets import edge_band
        cfg = dict(edge_band.DEFAULTS)
        cfg.update((self._config().get("void_detect") or {}).get("edge_band") or {})
        gb = QGroupBox("Edge Stains")
        form = QFormLayout(gb)
        btn = QPushButton("Detect Edge Stains")
        btn.setToolTip("Contamination along the inside of the chip edges (a long, dark, specked line with the chip edge):\n"
                       "every chip compared with the median chip (cut as it lies in the image, whole px, no resampling),\n"
                       "the dark specks joined into a line along the edge, the stain = chip edge up to the line, marked\n"
                       "as a dashed, filled shape of the selected void type (sat_widgets/edge_band.py). Chips must look alike.")
        btn.clicked.connect(self.detect_edge_stains)
        form.addRow(btn)
        self._stain_inputs = {}

        def spin(key, label, lo, hi, step, decimals, tip):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(decimals)
            sb.setValue(float(cfg[key]) * (100.0 if key in ("depth_frac", "clean_ref") else 1.0))
            sb.setToolTip(tip)
            self._stain_inputs[key] = sb
            return sb

        def row(*items):
            h = QHBoxLayout()
            for it in items:
                h.addWidget(QLabel(it) if isinstance(it, str) else it)
            form.addRow(h)

        row("Length ≥", spin("min_length_px", "", 5, 100000, 10, 0, "Shortest stain along the edge (image px)"),
            "Merge r", spin("merge_radius_px", "", 0, 1000, 1, 0, "Dark specks this close (image px, across the edge) are one line"),
            "Along ×", spin("merge_along", "", 1, 20, 0.5, 1, "... and this many times farther along the edge (a sparse rim joins into one line)"),
            "Link gap", spin("link_gap_px", "", 0, 100000, 5, 0, "Pieces of a line this far apart along the edge (image px) are one stain"))
        row("Depth %", spin("depth_frac", "", 1, 50, 1, 0, "Edge zone searched: this share of the chip size from the edge inward"),
            "Dark", spin("dark_threshold", "", 0.5, 50, 0.5, 1, "A dark speck: darker than the median chip's local min by this many chip-to-chip spreads"),
            "Density", spin("speck_density", "", 0.0, 1.0, 0.01, 2, "Specks around a speck (merge radius box), at least this share:\n"
                                                                     "dense rim specks stay, scattered noise specks go (lower if the rim is sparse)"),
            "Aspect", spin("min_aspect", "", 0.5, 50, 0.1, 1, "Length along the edge / depth, at least"))
        self.cb_stain_compare = QComboBox()
        for key, label in (("both", "dark + SSIM"), ("intensity", "dark"), ("ssim", "SSIM")):
            self.cb_stain_compare.addItem(label, key)
        self.cb_stain_compare.setCurrentIndex(max(0, self.cb_stain_compare.findData(cfg.get("compare", "both"))))
        self.cb_stain_compare.setToolTip("What counts as a speck of the line: darker than the median chip (dark), the local\n"
                                         "structure / texture changed (SSIM: pads covered or smeared while the mean brightness\n"
                                         "hardly changes), or both")
        row("Compare", self.cb_stain_compare,
            "SSIM thr", spin("ssim_threshold", "", 0.5, 500, 1, 1, "SSIM: a speck where the structure differs from the median chip by this many chip-to-chip spreads"),
            "SSIM σ", spin("ssim_sigma_px", "", 0.5, 100, 0.5, 1, "SSIM window (Gaussian sigma, image px)"),
            "Blur σ", spin("blur_px", "", 0, 100, 0.5, 1, "Gaussian blur of every chip before the comparison (sigma, image px; 0 = off)"))
        row("Skip", spin("edge_skip_px", "", 0, 1000, 1, 0, "Outermost image px of the chip left out (bbox / package edge)"),
            "Min depth", spin("min_depth_px", "", 0, 10000, 1, 0, "Shallowest stain (image px from the edge)"),
            "Align tol", spin("edge_tol_px", "", 0, 10, 0.5, 1, "A chip may sit this far (image px) off the median chip (chips are cut on whole px):\n"
                                                                  "a pixel within the median chip's values moved by +- this is no difference\n"
                                                                  "(Shift tol > 0: the wider square range instead)"),
            "Along", spin("edge_along_px", "", 0, 1000, 1, 0, "Within this many image px of the chip edge the median chip's range is taken along the\n"
                                                              "edge only (the steep package edge does not hide a stain there; 0 = off)"),
            "Shift tol", spin("shift_tol_px", "", 0, 100, 1, 0, "A pixel inside the median chip's min..max over +- this (image px) is no difference"))
        self.chk_stain_shading = QCheckBox("Shading")
        self.chk_stain_shading.setChecked(bool(cfg.get("shading", True)))
        self.chk_stain_shading.setToolTip("Take a brightness gradient across each chip (illumination over the wafer) off\n"
                                          "before the comparison (plane of chip - median chip, fitted robustly)")
        row("Texture", spin("texture_px", "", 0, 50, 1, 1, "Pad grid pitch (image px): every chip is averaged over a box of it before the\n"
                                                           "comparison, so the pads do not read as a darker grid (pads every other px: 2; 0 = off)"),
            self.chk_stain_shading)
        h4b = QHBoxLayout()
        h4b.addWidget(QLabel("Edges"))
        self._stain_edges = {}
        for key, label in (("t", "T"), ("b", "B"), ("l", "L"), ("r", "R")):
            cb = QCheckBox(label)
            cb.setToolTip({"t": "Top edge", "b": "Bottom edge", "l": "Left edge", "r": "Right edge"}[key] + " searched")
            cb.setChecked(key in str(cfg.get("edges", "tblr")).lower())
            self._stain_edges[key] = cb
            h4b.addWidget(cb)
        h4b.addWidget(QLabel("Match"))
        self.cb_stain_match = QComboBox()
        for key, label in (("none", "none"), ("linear", "linear"), ("curve", "curve")):
            self.cb_stain_match.addItem(label, key)
        self.cb_stain_match.setCurrentIndex(max(0, self.cb_stain_match.findData(cfg.get("match", "none"))))
        self.cb_stain_match.setToolTip("Per-chip brightness matching onto the median chip, measured on the chip centre:\n"
                                       "none (default), linear (gain / offset), curve (contrast / gamma too)")
        h4b.addWidget(self.cb_stain_match)
        h4b.addWidget(QLabel("Clean %"))
        h4b.addWidget(spin("clean_ref", "", 5, 100, 5, 0, "Reference (median chip, spread) from this share of the chips that differ least\n"
                                                         "(at most 64; 100 = a sample of all): stains at the same place on many chips\n"
                                                         "(long ones, corners) do not get into the reference. Golden chips picked: their median"))
        form.addRow(h4b)
        h5 = QHBoxLayout()
        h5.addWidget(QLabel("Mark as"))
        self.cb_stain_type = QComboBox()
        self.cb_stain_type.setToolTip("Void type of the stain marks (selected: the type selected for marking)")
        self.cb_stain_type.addItem("(selected type)", None)
        for name in self.void_manager.types:
            if not self.void_manager.is_bbox({"type": name}):
                self.cb_stain_type.addItem(name, name)
        if cfg.get("type"):
            i = self.cb_stain_type.findData(cfg["type"])
            if i >= 0:
                self.cb_stain_type.setCurrentIndex(i)
        h5.addWidget(self.cb_stain_type, 1)
        self.chk_stain_mask = QCheckBox("Show mask")
        self.chk_stain_mask.setToolTip("Last run, every chip (first searched layer): dark specks (red), the specks joined into\n"
                                       "lines (yellow) and the stain areas found (cyan, = the marks' shapes)")
        self.chk_stain_mask.toggled.connect(lambda on: (setattr(self.glw, "show_stain_masks", on), self.glw.update()))
        h5.addWidget(self.chk_stain_mask)
        form.addRow(h5)
        h6 = QHBoxLayout()
        self.btn_stain_golden = QPushButton("Golden (0)")
        self.btn_stain_golden.setToolTip("Golden chips for the median chip: Ctrl+G on the image adds / removes the chip under\n"
                                         "the mouse (gold frame). None picked: the cleanest chips are taken automatically.\n"
                                         "Click: clear the picked golden chips")
        self.btn_stain_golden.clicked.connect(self.clear_golden_chips)
        h6.addWidget(self.btn_stain_golden)
        self.btn_stain_paint = QPushButton("Paint line")
        self.btn_stain_paint.setCheckable(True)
        self.btn_stain_paint.setToolTip("Mark a stain by hand: paint over its dark line on the image (left drag). The stain is the\n"
                                        "nearest chip edge up to the painted line; strokes on the same edge within Merge r are\n"
                                        "joined into one stain (the gap filled). Esc or this button ends painting. Ctrl+Z undoes")
        self.btn_stain_paint.toggled.connect(self.toggle_paint_line)
        h6.addWidget(self.btn_stain_paint)
        h6.addWidget(QLabel("Brush"))
        self.sb_paint_width = QDoubleSpinBox()
        self.sb_paint_width.setRange(1, 200); self.sb_paint_width.setDecimals(0); self.sb_paint_width.setValue(6)
        self.sb_paint_width.setToolTip("Brush width (image px)")
        h6.addWidget(self.sb_paint_width)
        btn_dbg = QPushButton("Debug chip")
        btn_dbg.setToolTip("Every processing step of one chip (Ctrl+Shift+D on the image: the chip under the mouse;\n"
                           "this button: the chip at the view centre): raw window, cut, reference, deviation, specks,\n"
                           "lines, areas, and why places were dropped. Runs the detection without marking")
        btn_dbg.clicked.connect(lambda: self.debug_stain_chip(at_centre=True))
        h6.addWidget(btn_dbg)
        form.addRow(h6)
        self.glw.golden_cells = set()
        self.glw.paint_done.connect(self.on_paint_stroke)
        for key, slot in ((QKeySequence("Ctrl+G"), self.toggle_golden_chip), (QKeySequence("Ctrl+Shift+D"), self.debug_stain_chip)):
            act = QAction(self.glw)
            act.setShortcut(key)
            act.setShortcutContext(Qt.WidgetShortcut)
            act.triggered.connect(slot)
            self.glw.addAction(act)
        dock_layout.addWidget(gb)

    def _edge_stain_params(self):
        out = {k: sb.value() for k, sb in getattr(self, "_stain_inputs", {}).items()}
        for k in ("depth_frac", "clean_ref"):
            if k in out:
                out[k] /= 100.0
        if getattr(self, "cb_stain_compare", None) is not None:
            out["compare"] = self.cb_stain_compare.currentData()
        if getattr(self, "chk_stain_shading", None) is not None:
            out["shading"] = self.chk_stain_shading.isChecked()
        if getattr(self.glw, "golden_cells", None):
            out["golden_cells"] = sorted(self.glw.golden_cells)
        if getattr(self, "cb_stain_match", None) is not None:
            out["match"] = self.cb_stain_match.currentData()
        if getattr(self, "_stain_edges", None):
            out["edges"] = "".join(k for k, cb in self._stain_edges.items() if cb.isChecked())
        if getattr(self, "cb_stain_type", None) is not None and self.cb_stain_type.currentData():
            out["type"] = self.cb_stain_type.currentData()
        return out

    def review_chip(self):
        """A on the image: every detected (dashed) marking of the chip under the mouse is reviewed (one undo step)."""
        from PySide6.QtGui import QCursor
        glw, vm = self.glw, self.void_manager
        if self.full_data is None:
            return
        cfg = glw.grid_cfg
        pos = glw.mapFromGlobal(QCursor.pos())
        at = QPointF(pos) if glw.rect().contains(pos) else QPointF(glw.width() / 2, glw.height() / 2)
        cell = cfg.cell_at(*glw.get_image_coords(at))
        todo = [v for vs in vm.voids.values() for v in vs
                if v.get("auto") == "detected" and cfg.cell_at(*glw.raw_to_deskew(v["globalCX"], v["globalCY"])) == cell]
        if not todo:
            self.statusBar().showMessage(f"A: no detected markings to review in chip X{cell[1]:02d}_Y{cell[0]:02d}")
            return
        vm.begin_change()
        try:
            for v in todo:
                v.pop("auto", None)
            vm.touch()
        finally:
            vm.end_change()
        glw.update()
        self.update_void_ui()
        self.statusBar().showMessage(f"A: {len(todo)} detected marking(s) of chip X{cell[1]:02d}_Y{cell[0]:02d} reviewed (Ctrl+Z undoes)")

    def _cell_under_mouse(self):
        """Grid cell under the mouse on the image (else at the view centre)."""
        from PySide6.QtGui import QCursor
        glw = self.glw
        pos = glw.mapFromGlobal(QCursor.pos())
        at = QPointF(pos) if glw.rect().contains(pos) else QPointF(glw.width() / 2, glw.height() / 2)
        return glw.grid_cfg.cell_at(*glw.get_image_coords(at))

    def _update_golden_label(self):
        if getattr(self, "btn_stain_golden", None) is not None:
            self.btn_stain_golden.setText(f"Golden ({len(self.glw.golden_cells)})")

    def toggle_golden_chip(self):
        """Ctrl+G on the image: the chip under the mouse becomes / stops being a golden chip (Edge Stains reference)."""
        if self.full_data is None:
            return
        cell = self._cell_under_mouse()
        if not self.glw.grid_cfg.contains(*cell):
            return
        g = self.glw.golden_cells
        g.symmetric_difference_update({cell})
        self._update_golden_label()
        self.glw.update()
        self.statusBar().showMessage(f"Golden chip X{cell[1]:02d}_Y{cell[0]:02d} {'added' if cell in g else 'removed'}: "
                                     f"{len(g)} golden chip(s) (Edge Stains median chip)")

    def clear_golden_chips(self):
        self.glw.golden_cells.clear()
        self._update_golden_label()
        self.glw.update()
        self.statusBar().showMessage("Golden chips cleared: Edge Stains takes the cleanest chips automatically (Ctrl+G picks one)")

    def toggle_paint_line(self, on):
        if on and self.full_data is None:
            self.btn_stain_paint.setChecked(False)
            return
        if on:
            self.glw.start_paint(self.sb_paint_width.value())
            self.statusBar().showMessage("Paint line: drag over the dark line of a stain (the chip edge up to it becomes the stain); Esc ends")
        elif getattr(self.glw, "paint_width", None) is not None:
            self.glw.stop_paint()

    def on_paint_stroke(self, stroke):
        """A painted stroke (Paint line): the stain from the nearest chip edge up to the line, joined with the painted
        stains on the same edge within Merge r."""
        from sat_widgets import edge_band
        if not stroke: # Esc: paint mode ended
            self.btn_stain_paint.setChecked(False)
            return
        glw, vm = self.glw, self.void_manager
        glw.paint_width = self.sb_paint_width.value()
        cfg = glw.grid_cfg
        mx = sum(q[0] for q in stroke) / len(stroke)
        my = sum(q[1] for q in stroke) / len(stroke)
        cell = cfg.cell_at(mx, my)
        if not cfg.contains(*cell):
            return
        refs = vm.reference_bboxes(cfg, glw.coord_transform)
        if cell in refs:
            rect = refs[cell]
        else:
            x0, y0 = cfg.cell_origin(*cell)
            rect = (x0, y0, cfg.pitch_x, cfg.pitch_y)
        width = self.sb_paint_width.value()
        res = edge_band.stain_from_strokes([stroke], rect, width)
        if res is None:
            return
        side, (t0, t1) = res[2], res[3]
        # Strokes on the same edge within Merge r (along the edge) are one stain, the gap between them filled
        gap = self._stain_inputs["merge_radius_px"].value() if "merge_radius_px" in self._stain_inputs else 8.0
        to_raw, to_deskew = glw.deskew_to_raw, glw.raw_to_deskew
        strokes = [stroke]
        joined = []
        for layer, vs in vm.voids.items():
            for v in vs:
                pl = v.get("paint_line")
                if pl and tuple(pl["cell"]) == tuple(cell) and pl["side"] == side and \
                        pl["t"][0] - gap <= t1 and t0 <= pl["t"][1] + gap:
                    joined.append((layer, v))
                    strokes += [[to_deskew(x, y) for x, y in st] for st in pl["strokes"]]
                    width = max(width, float(pl.get("width", width)))
        outline, (gx0, gy0, gx1, gy1), side, (t0, t1) = edge_band.stain_from_strokes(strokes, rect, width, side)
        type_name = (self.cb_stain_type.currentData() if getattr(self, "cb_stain_type", None) is not None else None) \
            or glw.active_type
        layer = joined[0][0] if joined else glw.current_layer
        vm.begin_change()
        try:
            for ly, v in joined:
                vm.remove_void(ly, v)
            v = vm.add_void(layer, *to_raw((gx0 + gx1) / 2, (gy0 + gy1) / 2), (gx1 - gx0) / 2, (gy1 - gy0) / 2, type_name)
            cx, cy = v["globalCX"], v["globalCY"]
            v["outline"] = [[round(px - cx, 1), round(py - cy, 1)] for px, py in (to_raw(x, y) for x, y in outline)]
            v["outline_r"] = [round(v["radiusX"], 2), round(v["radiusY"], 2)]
            v["seg_fill"] = True
            v["paint_line"] = {"cell": list(cell), "side": side, "t": [t0, t1], "width": width,
                               "strokes": [[list(to_raw(x, y)) for x, y in st] for st in strokes]}
            vm.touch()
        finally:
            vm.end_change()
        glw.update()
        self.update_void_ui()
        self.statusBar().showMessage(f"Painted stain on X{cell[1]:02d}_Y{cell[0]:02d} ({side} edge, {t1 - t0:.0f} px long"
                                     f"{', joined with %d' % len(joined) if joined else ''}); Ctrl+Z undoes")

    def debug_stain_chip(self, at_centre=False):
        """Edge Stains debug: every processing step of one chip (no marking)."""
        from sat_widgets import edge_band
        if self._detect_busy():
            return
        glw = self.glw
        cell = glw.grid_cfg.cell_at(*glw.get_image_coords(QPointF(glw.width() / 2, glw.height() / 2))) if at_centre \
            else self._cell_under_mouse()
        setup = self._detect_setup("Edge Stains")
        if setup is None:
            return
        type_name, cells, refs, params, layers = setup
        if cell not in cells:
            QMessageBox.information(self, "Edge Stains debug", f"Chip X{cell[1]:02d}_Y{cell[0]:02d} is not among the inspected chips.")
            return
        ep = dict(params.get("edge_band") or {})
        ep.update(self._edge_stain_params())
        skip = set(params.get("skip_layers") or []) if len(layers) > 1 else set()
        zs = [z for z, l in enumerate(layers) if l not in skip]
        z = glw.current_layer if glw.current_layer in zs else zs[0]
        trace = ep["trace"] = {"cell": cell}
        ep["debug"] = []
        pd = QProgressDialog("Edge stains debug...", "Cancel", 0, 2, self)
        pd.setWindowModality(Qt.WindowModal)
        pd.show()

        def progress(i, n, text):
            pd.setMaximum(n); pd.setValue(i); pd.setLabelText(text)
            QApplication.processEvents()
            return not pd.wasCanceled()
        try:
            found = edge_band.detect(self.full_data, [z], [layers[z]], glw.grid_cfg, glw.coord_transform, refs, cells, ep, progress)
        finally:
            pd.close()
        if found is None:
            return
        mine = [f for f in found if tuple(f[5]) == tuple(cell)]
        info = [f"Chip X{cell[1]:02d}_Y{cell[0]:02d}, layer {layers[z]}: {len(mine)} stain(s) found"] + (trace.get("info") or [])
        self._show_trace(f"Edge Stains debug: X{cell[1]:02d}_Y{cell[0]:02d}", info, trace.get("steps") or [], ep)

    def _show_trace(self, title, info, steps, ep):
        """Dialog with the images of the steps (gray / signed / mask), Save writes them as PNGs."""
        from PIL import Image
        from PySide6.QtWidgets import QDialog, QScrollArea, QGridLayout, QWidget, QVBoxLayout
        from PySide6.QtGui import QPixmap

        def to_rgb(a, kind):
            if kind == "rgb":
                return np.asarray(a, np.uint8)
            a = np.asarray(a, np.float32)
            if kind == "mask":
                g = (a > 0).astype(np.uint8) * 255
                return np.stack([g, g, g], -1)
            if kind == "signed": # Blue darker, red brighter (+- 2 x Dark)
                lim = 2.0 * float(ep.get("dark_threshold", 3.0))
                t = np.clip(a / lim, -1, 1)
                rgb = np.full(a.shape + (3,), 255.0, np.float32)
                rgb[..., 0] -= 255 * np.clip(-t, 0, 1); rgb[..., 1] -= 255 * np.abs(t); rgb[..., 2] -= 255 * np.clip(t, 0, 1)
                return rgb.astype(np.uint8)
            if kind == "pos":
                lim = 2.0 * float(ep.get("ssim_threshold", 25.0))
                t = np.clip(a / lim, 0, 1)
                return np.stack([255 * np.clip(2 * t, 0, 1), 255 * np.clip(2 * t - 1, 0, 1), 0 * t], -1).astype(np.uint8)
            lo, hi = np.percentile(a, [1, 99])
            g = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            return np.stack([g, g, g], -1)

        imgs = []
        for name, a, kind in steps:
            im = Image.fromarray(to_rgb(a, kind))
            if max(im.size) > 900:
                k = 900.0 / max(im.size)
                im = im.resize((max(1, int(im.size[0] * k)), max(1, int(im.size[1] * k))), Image.BILINEAR)
            imgs.append((name, im))
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        lay = QVBoxLayout(dlg)
        lab = QLabel("\n".join(info))
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(lab)
        area = QScrollArea()
        inner = QWidget()
        grid = QGridLayout(inner)
        kinds = [k for _, _, k in steps]
        for i, (name, im) in enumerate(imgs):
            side = 420 if kinds[i] == "rgb" else 300
            rgb = np.ascontiguousarray(np.asarray(im.convert("RGB")))
            q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(q).scaled(side, side, Qt.KeepAspectRatio, Qt.FastTransformation)
            cell = QWidget()
            v = QVBoxLayout(cell)
            pl = QLabel(); pl.setPixmap(pix)
            pl.setToolTip(name)
            cap = QLabel(name); cap.setWordWrap(True); cap.setMaximumWidth(side)
            v.addWidget(pl); v.addWidget(cap)
            grid.addWidget(cell, i // 4, i % 4)
        area.setWidget(inner)
        area.setWidgetResizable(True)
        lay.addWidget(area, 1)
        btns = QHBoxLayout()
        bsave = QPushButton("Save PNGs...")

        def save():
            folder = QFileDialog.getExistingDirectory(dlg, "Save debug images")
            if not folder:
                return
            for i, (name, im) in enumerate(imgs):
                safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:60]
                im.save(os.path.join(folder, f"{i:02d}_{safe}.png"))
            with open(os.path.join(folder, "info.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(info) + "\n")
            self.statusBar().showMessage(f"Debug images saved to {folder}")
        bsave.clicked.connect(save)
        bclose = QPushButton("Close")
        bclose.clicked.connect(dlg.close)
        btns.addStretch(1); btns.addWidget(bsave); btns.addWidget(bclose)
        lay.addLayout(btns)
        dlg.resize(1300, 900)
        dlg.show()
        self._trace_dialog = dlg

    def detect_edge_stains(self):
        """Edge Stains button: sat_widgets/edge_band.py on the loaded layers, dashed boxes of the selected type."""
        from sat_widgets import edge_band
        if self._detect_busy():
            return
        setup = self._detect_setup("Edge Stains")
        if setup is None:
            return
        type_name, cells, refs, params, layers = setup
        ep = dict(params.get("edge_band") or {})
        ep.update(self._edge_stain_params()) # The group's inputs
        type_name = ep.get("type") or type_name
        if type_name not in self.void_manager.types:
            QMessageBox.warning(self, "Edge Stains", f"Void type {type_name!r} (void_detect.edge_band.type) is not declared.")
            return
        skip = set(params.get("skip_layers") or []) if len(layers) > 1 else set() # A single-layer image: always searched
        zs = [z for z, l in enumerate(layers) if l not in skip]
        pd = QProgressDialog("Edge stains...", "Cancel", 0, max(1, 2 * len(zs)), self)
        pd.setWindowModality(Qt.WindowModal)
        pd.show()

        def progress(i, n, text):
            pd.setMaximum(n)
            pd.setValue(i)
            pd.setLabelText(text)
            QApplication.processEvents()
            return not pd.wasCanceled()

        masks = {}
        why = ep["debug"] = [] # Why found places were not kept (summary below)
        try:
            found = edge_band.detect(self.full_data, zs, [layers[z] for z in zs], self.glw.grid_cfg,
                                     self.glw.coord_transform, refs, cells, ep, progress, masks)
        finally:
            pd.close()
        self.glw.stain_masks = masks # Show mask (debug)
        self.glw.update()
        if found is None:
            self.statusBar().showMessage("Edge stains: canceled")
            return
        vm, to_raw, to_deskew = self.void_manager, self.glw.deskew_to_raw, self.glw.raw_to_deskew
        added = 0
        vm.begin_change()
        try:
            for layer, gx, gy, rx, ry, cell, score, outline in found:
                # Skip what is marked already (any layer): a rerun does not double the marks
                if any(not vm.is_bbox(v) and abs(to_deskew(v["globalCX"], v["globalCY"])[0] - gx) < rx + v["radiusX"]
                       and abs(to_deskew(v["globalCX"], v["globalCY"])[1] - gy) < ry + v["radiusY"]
                       for vs in vm.voids.values() for v in vs):
                    continue
                v = vm.add_void(layer, *to_raw(gx, gy), rx, ry, type_name)
                v["auto"] = "detected" # Dashed until reviewed (Next Detected)
                v["score"] = score
                if outline: # Segmented stain (Shapes, debug): raw px offsets from the centre
                    cx, cy = v["globalCX"], v["globalCY"]
                    v["outline"] = [[round(px - cx, 1), round(py - cy, 1)] for px, py in (to_raw(x, y) for x, y in outline)]
                    v["outline_r"] = [round(v["radiusX"], 2), round(v["radiusY"], 2)]
                    v["seg_fill"] = True # Drawn filled (the stain area)
                added += 1
        finally:
            vm.end_change()
        self.glw.update()
        self.update_void_ui()
        msg = f"Edge stains: {added} marked ({type_name}, dashed: review with Next Detected), {len(found) - added} already marked"
        # Where the others were lost: the lines dropped by a filter (the longest one of each: most likely a stain)
        drops = {}
        for w in why:
            drops.setdefault(w[1], []).append(w)
        hints = {"short": "Length ≥ / Link gap", "not elongated": "Aspect", "thin": "Min depth", "not bright inside": "bright_min"}
        parts = []
        for k, ws in drops.items():
            big = max(ws, key=lambda w: w[2])
            parts.append(f"{len(ws)} lines dropped as {k} (longest {big[2]} px long, {big[3]} px deep: {hints.get(k, k)})")
        parts = parts or ["no other dark lines"]
        msg += " | " + "; ".join(parts)
        self.statusBar().showMessage(msg)
        print(msg)

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
        if params["method"] == "segmentation":
            # U-Net map in place of the die-to-die deviation (sat_widgets/void_seg.py)
            try:
                params["seg_model"] = void_seg.Model(model)
            except Exception as e:
                QMessageBox.warning(self, title, f"Segmentation model not usable ({model}): {e}")
                return None
            params["method"] = "die_to_die"
            params["threshold"] = self.sb_detect_thr.value()
            model = None
        if model:
            params["pc_model"] = model
        params[{"patchcore": "pc_threshold", "layer_change": "lc_threshold"}.get(params["method"], "threshold")] = \
            self.sb_detect_thr.value()
        params["layer_rule"] = self.cb_layer_rule.currentData()
        if self.chk_learned.isChecked() and self.chk_learned.isEnabled() and params["method"] in ("die_to_die", "layer_change"):
            path = void_learn.model_path(params, params["method"], self._product_key())
            try:
                params["learned_model"] = void_learn.Model.load(
                    path, void_layer_change.FEATURES if params["method"] == "layer_change" else None)
            except Exception as e:
                QMessageBox.warning(self, title, f"Learned filter not usable ({path}): {e}")
                return None
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

        last = [0.0]

        def step(i, n, label):
            if i % 20 == 0 or time.monotonic() - last[0] > 0.25: # Slow steps (Layer change) refresh too
                last[0] = time.monotonic()
                progress.setValue(int(1000 * i / max(1, n)))
                progress.setLabelText(label)
                QApplication.processEvents()
            return not progress.wasCanceled()
        return progress, step

    def detect_voids(self):
        """
        Rough automatic void detection on all loaded layers (see sat_widgets/void_detect.py), in the
        background (sat_widgets/detect_job.py): the voids appear as they are found and the view stays
        usable; editing the markings waits until the run ends. The whole run is one undo step.
        """
        if self._detect_busy():
            return
        setup = self._detect_setup("Detect Voids")
        if setup is None:
            return
        type_name, cells, refs, params, layers = setup
        vm = self.void_manager
        # The grid as it is now, for the worker and the placement (the grid may be moved meanwhile)
        cfg, ct = copy.deepcopy(self.glw.grid_cfg), copy.deepcopy(self.glw.coord_transform)
        rects = void_detect.chip_rects(cfg, refs, cells)
        job = DetectJob(self.full_data.reopen, (cfg, ct, refs, cells, layers, read_only_copy(vm), type_name, params), self)
        job.found.connect(lambda batch: self._detect_found(job, batch, rects, cfg, ct))
        job.cells.connect(lambda cs, frac: self._detect_cells(job, cs, frac))
        job.progress.connect(lambda i, n, text: self._detect_step(job, i, n, text))
        job.finished.connect(lambda result, error: self._detect_finished(job, result, error, refs))
        self._detect_job, self._detect_added = job, 0
        vm.begin_change() # One undo step for the whole run (Ctrl+Z waits until it ends)
        self._lock_editing(True)
        self.glw.detect_cells = {c: 0.0 for c in cells}
        self._detect_widgets(True)
        self.statusBar().showMessage("Detect Voids: running in the background...")
        job.start()

    def _detect_busy(self):
        """True (with a note in the status bar) while Detect Voids runs: the markings are not editable."""
        if self._detect_job is None:
            return False
        self.statusBar().showMessage("Detect Voids is running: markings can be edited when it ends "
                                     "(Cancel in the status bar keeps what was found).", 5000)
        return True

    def _lock_editing(self, on):
        self.glw.edit_locked = on
        if on:
            self.glw.active_void = None
        self._apply_held_mode()
        for w in (self.btn_detect, self.btn_eval, self.btn_train, self.btn_load_voids, self.btn_clear_voids,
                  self.btn_clear_chip, self.btn_clear_all, self.btn_clear_keep_bbox, self.btn_edit_types):
            w.setEnabled(not on)

    def _detect_widgets(self, show):
        """Progress bar and Cancel button in the status bar (created on first use)."""
        if not hasattr(self, "detect_bar"):
            self.detect_label = QLabel()
            self.detect_bar = QProgressBar()
            self.detect_bar.setRange(0, 1000)
            self.detect_bar.setMaximumWidth(220)
            self.detect_cancel = QPushButton("Cancel")
            self.detect_cancel.setToolTip("Stops Detect Voids; the voids found so far stay (Ctrl+Z removes them)")
            self.detect_cancel.clicked.connect(self.cancel_detect)
            for w in (self.detect_label, self.detect_bar, self.detect_cancel):
                self.statusBar().addPermanentWidget(w)
        self.detect_label.setText("Detect Voids")
        self.detect_bar.setValue(0)
        self.detect_cancel.setEnabled(True)
        for w in (self.detect_label, self.detect_bar, self.detect_cancel):
            w.setVisible(show)

    def _detect_step(self, job, i, n, text):
        if job is self._detect_job:
            self.detect_bar.setValue(int(1000 * i / max(1, n)))
            self.detect_label.setText(f"{text}  ({self._detect_added} voids)")

    def _detect_found(self, job, batch, rects, cfg, ct):
        """Places voids found by the worker (GUI thread, inside the run's undo step)."""
        if job is not self._detect_job:
            return
        void_detect.place_found(self.void_manager, batch, rects, cfg, ct)
        self._detect_added += len(batch)
        self.glw.refresh_hover()
        self.update_void_ui()
        self.glw.update()

    def _detect_cells(self, job, cells, frac):
        state = self.glw.detect_cells
        if job is not self._detect_job or state is None:
            return
        for c in cells:
            if c in state:
                state[c] = max(state[c], frac)
        self.glw.update()

    def cancel_detect(self):
        """Stops Detect Voids; the voids placed so far stay (one undo step)."""
        if self._detect_job is not None:
            self._detect_job.cancel()
            self.detect_cancel.setEnabled(False)
            self.detect_label.setText("Cancelling...")

    def _end_detect(self):
        self._detect_job = None
        self.void_manager.end_change()
        self._lock_editing(False)
        self._detect_widgets(False)

    def _abort_detect(self):
        """Stops a running Detect Voids without waiting (file closed); its results are dropped."""
        if self._detect_job is not None:
            self._detect_job.cancel() # The worker reads with its own reader: it may finish later
            self._end_detect()
        self.glw.detect_cells = None

    def _detect_finished(self, job, result, error, refs):
        job.deleteLater() # Also for an aborted run (file closed): frees the job and its slots' captured data
        if job is not self._detect_job:
            return
        self._end_detect()
        if error:
            print(error)
            QMessageBox.warning(self, "Detect Voids", f"Detect Voids failed ({self._detect_added} voids kept):\n"
                                + error.strip().splitlines()[-1])
        if result is None:
            if not error:
                msg = f"Detect Voids cancelled: {self._detect_added} voids kept (Ctrl+Z removes them)."
                self.statusBar().showMessage(msg)
                print(msg)
            self.glw.update()
            return
        self.glw.detect_cells = dict.fromkeys(self.glw.detect_cells or (), 1.0)
        self._detect_done(result, refs)
        self.glw.update()

    def _toggle_outlines(self, on):
        self.glw.show_outlines = bool(on)
        self._set_ui_state("show_outlines", bool(on))
        self.glw.update()

    def _toggle_detect_state(self, on):
        self.glw.show_detect_state = on
        self.glw.update()

    def _hand_truths(self, layers, refs, unreviewed=False):
        """Hand markings (not bboxes, not unreviewed detections) on the loaded layers, as drawn boxes;
        unreviewed=True: the detections not reviewed yet instead (auto "detected", still dashed)."""
        vm, cfg, ct = self.void_manager, self.glw.grid_cfg, self.glw.coord_transform
        to_deskew = ct.raw_to_deskew if ct else (lambda x, y: (x, y))
        truths = []
        for layer, vs in vm.voids.items():
            if layer not in layers:
                continue
            for v in vs:
                if vm.is_bbox(v) or (v.get("auto") == "detected") != unreviewed or (not unreviewed and v.get("auto")):
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
        return truths

    def seg_params(self):
        """Detect Voids settings for extracting U-Net training data (die-to-die, current Skip layers)."""
        params = dict(void_detect.DEFAULTS)
        params.update(self._config().get("void_detect", {}))
        params["method"] = "die_to_die"
        try:
            params["skip_layers"] = [int(x) for x in re.split(r"[,\s]+", self.le_skip_layers.text().strip()) if x]
        except ValueError:
            pass
        return params

    def open_seg_data(self):
        from sat_widgets.seg_review import SegDataDialog
        SegDataDialog(self).exec()

    def add_seg_samples_from_open(self, progress_factory):
        """Training samples of the open file and its markings (Seg Data window). Returns the folder or None."""
        if self.full_data is None or self.glw.tiled_image is None:
            QMessageBox.information(self, "Seg Data", "Open a file and load its markings first.")
            return None
        vm, cfg, ct = self.void_manager, self.glw.grid_cfg, self.glw.coord_transform
        bmap = self.glw.bonding_map
        cells = [(r, c) for r in range(cfg.rows) for c in range(cfg.cols) if bmap is None or bmap.get_key(r, c)]
        refs = vm.reference_bboxes(cfg, ct)
        layers = [self.abs_layer(z) for z in range(len(self.full_data))]
        dlg, step = progress_factory("Extracting training samples...")
        try:
            out = seg_data.extract(self.full_data, cfg, ct, refs, cells, layers, vm, self.seg_params(), step)
        finally:
            dlg.close()
        if out is None:
            return None
        layer_refs, samples = out
        source = self.lbl_source.toolTip() or self.le_wafer_id.text() or "unnamed"
        return void_seg.write_source(source, self.void_meta(), layer_refs, samples)

    def train_detector(self):
        """
        Learned filter (sat_widgets/void_learn.py): adds the candidates of this file, labelled by its
        hand markings, to the training set and retrains the model on all of it.
        """
        if self._detect_busy():
            return
        setup = self._detect_setup("Train")
        if setup is None:
            return
        type_name, cells, refs, params, layers = setup
        vm, cfg, ct = self.void_manager, self.glw.grid_cfg, self.glw.coord_transform
        truths = self._hand_truths(layers, refs)
        pending = self._hand_truths(layers, refs, unreviewed=True)
        if not truths and QMessageBox.question(
                self, "Train", "No hand markings on the loaded layers: this file adds only non-voids.\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if pending and QMessageBox.question(
                self, "Train", f"{len(pending)} detected voids are not reviewed yet (dashed).\n"
                "Their places are left out of the training (neither void nor non-void).\n"
                "Review them first (E + click: correct, R + click: wrong) to use them. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        params.pop("learned_model", None)
        if params.get("method") == "patchcore":
            params["method"] = "die_to_die" # The learned filter works on the die-to-die deviation
        method = params["method"]
        lc = method == "layer_change" # Own candidates / features / model (sat_widgets/void_layer_change.py)
        cand_thr = float(params.get("lc_cand_threshold", void_layer_change.DEFAULTS["lc_cand_threshold"]) if lc
                         else params.get("cand_threshold", 1.0))
        collected = []
        params["learn"] = lambda layer, cell, cands: collected.append((layer, cell, cands))
        params["cand_threshold"] = cand_thr
        progress, step = self._detect_progress("Collecting training candidates...")
        try:
            result = void_detect.detect_voids(self.full_data, cfg, ct, refs, cells, layers, vm, type_name,
                                              params, step, apply=False)
        finally:
            progress.close()
        if result is None:
            return
        # Layer change: the filter decides the place only (the layer comes from the layer rule)
        rows, covered = void_learn.label_rows(collected, truths, any_layer=lc, unknown=pending)
        key = self.lbl_source.toolTip() or self.le_wafer_id.text() or "unnamed"
        product = self._product_key() if lc else None # Layer change: one model per product
        dataset = void_learn.dataset_path(method, product)
        void_learn.add_to_dataset(key, rows, dataset)
        path = void_learn.model_path(self._config().get("void_detect", {}), method, product)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Layer change: cut for a false-detection budget per wafer (config lc_fp_per_wafer, default 1)
            fp_target = float(self._config().get("void_detect", {}).get("lc_fp_per_wafer", 1.0)) if lc else None
            data = void_learn.train_model(cand_thr, path, rounds=400 if lc else 200, dataset=dataset,
                                          features=void_layer_change.FEATURES if lc else None, trees=lc,
                                          fp_per_file=fp_target)
        except ValueError as e:
            QMessageBox.warning(self, "Train", f"Added {len(rows)} candidates of this file. Not trained yet: {e}.")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.chk_learned.setEnabled(True)
        self.chk_learned.setChecked(True)
        pct = lambda v: int(round(100 * v))
        short = (f"L{'C' if lc else '1'} f{data['files']} v{data['voids']} n{data['others']} c{covered} h{len(truths)}\n"
                 f"R{pct(data['cv_recall'])} P{pct(data['cv_precision'])} F{data.get('cv_false_per_file', 0):.1f} "
                 + " ".join(data["top_features"][:3]))
        full = (f"This file: {len(truths)} hand markings, {covered} found among {len(rows)} candidates\n"
                f"Training set: {data['files']} files, {data['voids']} voids, {data['others']} other candidates\n"
                f"Cross-validated ({'by file' if lc and data['files'] >= 3 else 'by chip'}): recall {pct(data['cv_recall'])} %, "
                f"precision {pct(data['cv_precision'])} %, {data.get('cv_false_per_file', 0):.1f} false per file "
                f"(cut: {data.get('cut_rule', 'best F1')})\n"
                f"Most used features: {', '.join(data['top_features'])}\n"
                + (f"Product: {product[0]}\n" if product else "") + f"Model: {path}\nTraining set: {dataset}")
        print("Train:\n" + full)
        EvalDialog(self, full, os.path.dirname(path), None, len(truths), short).exec()

    def reset_learned(self):
        """Learned filter of the current method / product back to none: model and training set renamed to
        *.<time>.bak under the training folder; a declared (shared) model elsewhere stays."""
        if self._detect_busy():
            return
        method = self.cb_detect_method.currentData()[0]
        if method not in ("die_to_die", "layer_change"):
            QMessageBox.information(self, "Reset", "The learned filter is for Die-to-die and Layer change.")
            return
        product = self._product_key() if method == "layer_change" else None
        model = void_learn.model_path(self._config().get("void_detect", {}), method, product)
        dataset = void_learn.dataset_path(method, product)
        own = os.path.abspath(void_learn.TRAIN_DIR)
        local = lambda f: os.path.abspath(f).startswith(own + os.sep)
        files = [f for f in (model, dataset) if os.path.exists(f) and local(f)]
        shared = os.path.exists(model) and not local(model)
        what = f"{'Layer change' if method == 'layer_change' else 'Die-to-die'}" + (f", product {product[0]}" if product else "")
        if not files:
            QMessageBox.information(self, "Reset", f"No local learned filter for {what}." +
                                    (f"\nThe shared model {model} (declared) is not changed here." if shared else ""))
            return
        if QMessageBox.question(self, "Reset", f"Reset the learned filter for {what}?\n\n" + "\n".join(files) +
                                "\n\nThe files are renamed to *.bak (kept). Train again to build a new one." +
                                (f"\n\nThe shared model {model} is declared and stays in use." if shared else ""),
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        moved = []
        for f in files:
            dst = f"{f}.{stamp}.bak"
            os.replace(f, dst)
            moved.append(dst)
        self._update_learned_state()
        msg = f"Learned filter reset ({what}): " + ", ".join(os.path.basename(m) for m in moved)
        self.statusBar().showMessage(msg)
        print(msg)

    def evaluate_detection(self):
        """
        Detection without marking, compared with the hand markings (sat_widgets/void_eval.py):
        shows the short code and writes the report folder.
        """
        if self._detect_busy():
            return
        setup = self._detect_setup("Evaluate")
        if setup is None:
            return
        type_name, cells, refs, params, layers = setup
        vm, cfg, ct = self.void_manager, self.glw.grid_cfg, self.glw.coord_transform
        to_deskew = ct.raw_to_deskew if ct else (lambda x, y: (x, y))
        truths = self._hand_truths(layers, refs)
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
        short = ev.short_code(summary)
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
        print("Detection evaluation:\n" + short + "\n\n" + code + (f"\nReport: {report}" if report else ""))
        EvalDialog(self, code, folder if report else None, report, summary["h"], short).exec()

    def _detect_done(self, result, refs):
        self.glw.refresh_hover()
        self.update_void_ui()
        per = ", ".join(f"L{l}: {n}" for l, n in result["per_layer"].items())
        kinds = ", ".join(f"{t}: {n}" for t, n in result.get("types", {}).items())
        msg = (f"Detect Voids: {result['added']} new voids ({per}; {kinds}). "
               "Dashed until reviewed; 'Next Detected' walks through them.")
        n0 = sum(1 for vs in self.void_manager.voids.values() for v in vs if v.get("layer0") and v.get("auto") == "detected")
        if n0:
            msg += f" {n0} already visible on layer 0 (badge \"0\", bubble?): 'Next Layer-0', B clears the flag."
        if not refs:
            msg += " No reference bboxes: chips were aligned on the grid cells."
        self.statusBar().showMessage(msg)
        print(msg)

    def next_detected_void(self):
        """Navigates through the detected voids not reviewed yet (layer, row, col order)."""
        self._next_void(lambda v: v.get("auto") == "detected", "Detected void", "No detected voids left to review.")

    def prev_detected_void(self):
        self._next_void(lambda v: v.get("auto") == "detected", "Detected void", "No detected voids left to review.", step=-1)

    def next_layer0_void(self):
        """Navigates through the voids flagged as already visible on layer 0 (layer, row, col order)."""
        self._next_void(lambda v: v.get("layer0"), "Layer-0 void", "No voids flagged as visible on layer 0.")

    def toggle_layer0_flag(self):
        state = self.glw.toggle_hovered_layer0()
        if state is not None:
            n = sum(1 for vs in self.void_manager.voids.values() for v in vs if v.get("layer0"))
            self.statusBar().showMessage(f"Layer-0 flag {'set' if state else 'cleared'} ({n} flagged; Ctrl+Z undoes)")

    def _next_void(self, pick, what, none_msg, step=1):
        items = []
        for layer, vs in self.void_manager.voids.items():
            for v in vs:
                if pick(v) and self.void_manager.passes(v):
                    r, c = self.glw.raw_to_cell(v["globalCX"], v["globalCY"])
                    items.append((layer, r, c))
        if not items:
            self.statusBar().showMessage(none_msg)
            return
        items = sorted(set(items))
        here = (self.glw.current_layer,) + tuple(self.glw.center_cell())
        if step > 0:
            nxt = next((it for it in items if it > here), items[0])
        else:
            nxt = next((it for it in reversed(items) if it < here), items[-1])
        layer, r, c = nxt
        z = next((z for z in range(len(self.full_data)) if self.abs_layer(z) == layer), None)
        if z is not None and z != self.current_z:
            self.spin_layer.setValue(z)
        self.glw.fit_to_cell(c, r)
        self.statusBar().showMessage(f"{what} {items.index(nxt) + 1}/{len(items)}: layer {layer}, X{c:02d} Y{r:02d}")
        if what == "Detected void":
            self._review_pos = items.index(nxt) + 1
            self._update_review_ui()

    def update_void_ui(self):
        # Update label count (voids are keyed by absolute layer number)
        vm = self.void_manager
        here = vm.voids.get(self.glw.current_layer, [])
        shown = sum(1 for v in here if vm.passes(v))
        self.lbl_void_count.setText(f"Voids: {shown}" + (f"  ({len(here) - shown} hidden by SNR)" if shown < len(here) else ""))
        self._update_review_ui()

    def _update_review_ui(self):
        """Void Detection: detected voids left to review, layer-0 flags."""
        if not hasattr(self, "lbl_review"):
            return
        places, l0 = set(), 0 # Next / Prev stop once per layer and cell
        for layer, vs in self.void_manager.voids.items():
            for v in vs:
                if not self.void_manager.passes(v):
                    continue
                if v.get("auto") == "detected":
                    places.add((layer,) + tuple(self.glw.raw_to_cell(v["globalCX"], v["globalCY"])))
                l0 += bool(v.get("layer0"))
        todo = len(places)
        pos = getattr(self, "_review_pos", None)
        if todo == 0:
            self._review_pos = None
            self.lbl_review.setText("")
        elif pos is not None and todo == self._review_n:
            self.lbl_review.setText(f"{min(pos, todo)}/{todo}")
        else:
            self._review_pos = None
            self.lbl_review.setText(f"{todo} to review")
        self._review_n = todo
        self.btn_next_l0.setText(f"Layer-0 ({l0})" if l0 else "Layer-0")
        self.btn_next_l0.setEnabled(l0 > 0)

    def _set_snr_min(self, value):
        """SNR filter changed (slider / box): redraw, recount; hidden voids are not saved."""
        value = float(value)
        if int(round(value * 2)) != self.slider_snr.value():
            self.slider_snr.blockSignals(True)
            self.slider_snr.setValue(int(round(value * 2)))
            self.slider_snr.blockSignals(False)
        vm = self.void_manager
        vm.snr_min = value
        if self.glw.active_void is not None and not vm.passes(self.glw.active_void):
            self.glw.active_void = None
        self._review_pos = None
        self.glw.refresh_hover()
        self.update_void_ui()
        hidden = sum(1 for vs in vm.voids.values() for v in vs if not vm.passes(v))
        self.statusBar().showMessage(f"SNR filter {value:.1f}: {hidden} voids hidden (not saved)" if value > 0
                                     else "SNR filter off")
        self.glw.update()

    def _ui_state(self, key, default):
        """Per-user window state (QSettings), not a declaration: e.g. the Advanced section open."""
        try:
            v = QSettings("TIF_Viewer", "TIF_Viewer").value(f"ui/{key}", default)
            return v in (True, "true", "1", 1) if isinstance(default, bool) else v
        except Exception:
            return default

    def _set_ui_state(self, key, value):
        try:
            QSettings("TIF_Viewer", "TIF_Viewer").setValue(f"ui/{key}", value)
        except Exception:
            pass

    def _show_detect_advanced(self, on):
        self.detect_adv.setVisible(on)
        self.btn_detect_adv.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self._set_ui_state("detect_advanced", bool(on))

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
        self._abort_detect()
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

    def open_file(self, path=None, confirm=True):
        """Open TIF (dialog), or `path` (Auto Run: confirm=False, no question about unsaved markings)."""
        if not isinstance(path, str): # Menu action: `checked`
            path, _ = QFileDialog.getOpenFileName(
                self, "Open", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;TIFF (*.tif *.tiff);;All files (*)")
        if not path:
            return
        if confirm and not self._confirm_discard_voids():
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
                
            if self.glw.bonding_map is None and any(opts.get(k) for k in ('raw', 'overlay', 'mask', 'merged', 'for_copy')):
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
                stats_errors = getattr(exporter, "defect_stats_errors", None)
                if stats_errors:
                    QMessageBox.warning(self, "Export", f"Defect spec plugin ({len(stats_errors)} error(s)); those types count as "
                                        "spec out where they occur:\n" + "\n".join(stats_errors[:8]))
                if opts.get("for_copy"):
                    n = getattr(exporter, "for_copy_count", 0)
                    groups = getattr(exporter, "for_copy_groups", {})
                    per = ", ".join(f"{g} {k}" for g, k in sorted(groups.items()))
                    msg = f"Export finished: {n} defect image(s) in {os.path.join(out_dir, 'for_copy')}" + (f" ({per})" if per else "")
                    print(msg)
                    self.statusBar().showMessage(msg)
                self.last_export_dir = out_dir
                if (self._results_settings().get("send") or {}).get("after_export"):
                    self.send_results() # config.json "results_api": {"send": {"after_export": true}}

class EvalDialog(QDialog):
    """Result of Evaluate: the short code (to type or scan on a phone) and the report folder."""

    def __init__(self, parent, code, folder, report, n_hand, short=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Evaluation")
        self.full, self.short = code, short or code
        self.code, self.folder, self.report = self.short, folder, report
        lay = QVBoxLayout(self)
        hint = "Type this code on the phone and send it. Full code: more detail (per layer)."
        if n_hand == 0:
            hint = ("No hand markings on the loaded layers: only detection and chip statistics are shown.\n"
                    "Mark voids by hand (or load a reviewed voids.json) and evaluate again to compare.\n") + hint
        lay.addWidget(QLabel(hint))
        txt = QPlainTextEdit(self.code)
        self.txt = txt
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
        self.b_full = QPushButton("Full Code")
        self.b_full.setCheckable(True)
        self.b_full.toggled.connect(self._toggle_full)
        btns.addWidget(self.b_full)
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

    def _toggle_full(self, on):
        self.code = self.full if on else self.short
        self.txt.setPlainText(self.code)
        self.b_full.setText("Short Code" if on else "Full Code")

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
        self.chk_for_copy = QCheckBox("Defect images only, one folder (/for_copy)")
        self.chk_for_copy.setToolTip("The overlay images of the chips with a defect marked on that layer\n"
                                     "(not the bbox), all in one folder to copy elsewhere")
        self.chk_for_copy_all = QCheckBox("    ... all layers of a defective die")
        self.chk_for_copy_all.setToolTip("for_copy: every exported layer of a die with a defect on any layer\n"
                                         "(off: only the layers where a defect is marked)")
        self.chk_for_copy.toggled.connect(self.chk_for_copy_all.setEnabled)
        self.chk_json = QCheckBox("Voids JSON (voids.json + meta.json)")
        self.chk_csv = QCheckBox("Metadata CSV (voids.csv)")
        
        # Defaults
        self.chk_raw.setChecked(True)
        self.chk_overlay.setChecked(True)
        self.chk_mask.setChecked(True)
        self.chk_merged.setChecked(True)
        self.chk_for_copy.setChecked(True)
        self.chk_for_copy_all.setChecked(True)
        self.chk_json.setChecked(True)
        self.chk_csv.setChecked(True)
        # config.json "ui_defaults": {"export": {"raw": false, ...}}
        defaults = (parent._config().get("ui_defaults") or {}).get("export", {}) if hasattr(parent, "_config") else {}
        for key, chk in (("raw", self.chk_raw), ("overlay", self.chk_overlay), ("mask", self.chk_mask),
                         ("merged", self.chk_merged), ("for_copy", self.chk_for_copy),
                         ("for_copy_all_layers", self.chk_for_copy_all), ("json", self.chk_json),
                         ("csv", self.chk_csv)):
            if key in defaults:
                chk.setChecked(bool(defaults[key]))
        
        v_opt.addWidget(self.chk_raw)
        v_opt.addWidget(self.chk_overlay)
        v_opt.addWidget(self.chk_mask)
        v_opt.addWidget(self.chk_merged)
        v_opt.addWidget(self.chk_for_copy)
        v_opt.addWidget(self.chk_for_copy_all)
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
            'for_copy': self.chk_for_copy.isChecked(),
            'for_copy_all_layers': self.chk_for_copy_all.isChecked(),
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
