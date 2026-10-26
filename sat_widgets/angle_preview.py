"""
Angle alignment preview: deskewed patches of a few cells spread over the round wafer, with the
grid drawn on top. A wrong grid angle shows as chip edges drifting off the grid lines, more the
farther a cell is from the wafer center.

Preview points are polar positions on the wafer circle (center = grid center, radius = half the
smaller grid side), declared in config.json (defaults below):

    "angle_preview": {
        "points": [{"name": "Center", "angle": 0, "r": 0.0},
                   {"name": "Top 3/4", "angle": 90, "r": 0.75}, ...],
        "view_scale": 1.4
    }

angle: degrees counter-clockwise from +x (screen right), r: fraction of the wafer radius.
With a bonding map the nearest bonded cell is used, so edge cells without a chip are skipped.
"""
import math
import time

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QDialog, QGridLayout, QLabel, QVBoxLayout, QWidget, QSizePolicy

from .core_data import ImageNormalizer, to_uint16
from .products import layout_boxes

DEFAULT_POINTS = [
    {"name": "Center", "angle": 0, "r": 0.0},
    {"name": "Top 3/4", "angle": 90, "r": 0.75},
    {"name": "Right 3/4", "angle": 0, "r": 0.75},
    {"name": "Top-Right 1/2", "angle": 45, "r": 0.5},
]
DEFAULT_VIEW_SCALE = 1.4 # Patch size in cell pitches


def preview_cells(grid_cfg, points, bonding_map=None):
    """[(name, row, col)] for the declared polar points on the wafer circle."""
    w, h = grid_cfg.cols * grid_cfg.pitch_x, grid_cfg.rows * grid_cfg.pitch_y
    cx, cy = grid_cfg.start_x + w / 2.0, grid_cfg.start_y + h / 2.0
    radius = min(w, h) / 2.0

    bonded = None
    if bonding_map is not None:
        bonded = [(r, c) for r in range(grid_cfg.rows) for c in range(grid_cfg.cols) if bonding_map.get_key(r, c)]

    out = []
    for pt in points:
        a = math.radians(pt.get("angle", 0.0))
        px = cx + pt.get("r", 0.0) * radius * math.cos(a)
        py = cy - pt.get("r", 0.0) * radius * math.sin(a) # Screen y points down
        if bonded:
            def dist(rc):
                ccx = grid_cfg.start_x + (rc[1] + 0.5) * grid_cfg.pitch_x
                ccy = grid_cfg.start_y + (rc[0] + 0.5) * grid_cfg.pitch_y
                return (ccx - px) ** 2 + (ccy - py) ** 2
            row, col = min(bonded, key=dist)
        else:
            row, col = grid_cfg.cell_at(px, py)
            row = min(max(row, 0), grid_cfg.rows - 1)
            col = min(max(col, 0), grid_cfg.cols - 1)
        out.append((pt.get("name", ""), row, col))
    return out


class _Tile(QWidget):
    clicked = Signal(int, int) # col, row

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.title = QLabel("-")
        self.title.setStyleSheet("font-weight: bold;")
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(200, 200)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image.setStyleSheet("background-color: #202020;")
        lay.addWidget(self.title)
        lay.addWidget(self.image, 1)
        self.cell = None
        self.resized = None # Callback: re-render at the new size

    def mousePressEvent(self, e):
        if self.cell is not None:
            self.clicked.emit(self.cell[1], self.cell[0])

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.resized:
            self.resized()

    def side(self):
        """Square render size in pixels (rendered at display size, no extra scaling)."""
        return max(64, min(self.image.width(), self.image.height()) - 4)

    def set_content(self, title, cell, pixmap):
        self.title.setText(title)
        self.cell = cell
        self.image.setPixmap(pixmap)


class AnglePreview(QDialog):
    """
    Non-modal 2x2 preview; call schedule_update() whenever grid, layer or window changes.

    Fast path (target < 30 ms per update): the raw region around each preview cell is read once
    with a one-pitch margin and kept in memory; angle / origin / window changes only re-warp it at
    display size and apply the window level to that small output.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("Angle Preview")
        self.resize(760, 800)
        self.mw = main_window

        lay = QVBoxLayout(self)
        self.info = QLabel("Grid lines should run along the chip edges in every patch. Click a patch to go there.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.grid_layout = QGridLayout()
        lay.addLayout(self.grid_layout, 1)
        self.tiles = []
        for _ in range(4):
            self._add_tile()

        # 0 ms single shot: coalesces the several signals of one change into one refresh
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.refresh)

        self._regions = {} # (z, row, col) -> (x0, y0, float 'F' image of the raw region)
        self._settings_cache = None
        self._stack = None
        self._lut, self._lut_key = None, None

    def _add_tile(self):
        t = _Tile()
        t.clicked.connect(self._navigate)
        t.resized = self.schedule_update
        self.grid_layout.addWidget(t, len(self.tiles) // 2, len(self.tiles) % 2)
        self.tiles.append(t)

    def schedule_update(self):
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        self._settings_cache = None # Re-read config.json when reopened
        self.timer.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self._regions.clear()
        # The decoded frame was kept while open (cell changes re-read without decoding again)
        if self._stack is not None:
            self._stack.release_cache()

    def _navigate(self, col, row):
        self.mw.glw.fit_to_cell(col, row)

    def _settings(self):
        if self._settings_cache is None:
            cfg = self.mw._config().get("angle_preview", {})
            self._settings_cache = (cfg.get("points") or DEFAULT_POINTS, float(cfg.get("view_scale", DEFAULT_VIEW_SCALE)))
        return self._settings_cache

    def refresh(self):
        t0 = time.perf_counter()
        mw, glw = self.mw, self.mw.glw
        stack = mw.full_data
        if stack is None or glw.coord_transform is None:
            self.info.setText("No image loaded.")
            return
        if stack is not self._stack: # New image: drop cached regions
            self._stack = stack
            self._regions.clear()
        grid = glw.grid_cfg
        points, view_scale = self._settings()
        cells = preview_cells(grid, points, glw.bonding_map)
        while len(self.tiles) < len(cells):
            self._add_tile()

        refs = mw.void_manager.reference_bboxes(grid, glw.coord_transform)
        for tile, (name, row, col) in zip(self.tiles, cells):
            pix, _ = self._render(stack, mw.current_z, grid, row, col, view_scale, refs, tile.side())
            tile.set_content(f"{name}  (X{col:02d}, Y{row:02d})", (row, col), pix)
        ms = (time.perf_counter() - t0) * 1000.0
        self.last_refresh_ms = ms
        self.info.setText(f"Angle {grid.angle:.3f}°  |  {ms:.0f} ms  |  grid lines should run along the chip edges. Click a patch to go there.")

    def _region(self, stack, z, row, col, rx0, ry0, rx1, ry1, margin):
        """Cached uint16 raw region covering [rx0, rx1) x [ry0, ry1); read with a margin on a miss."""
        key = (z, row, col)
        hit = self._regions.get(key)
        if hit is not None:
            x0, y0, arr = hit
            if x0 <= max(0, rx0) and y0 <= max(0, ry0) and \
               x0 + arr.width >= min(stack.width, rx1) and y0 + arr.height >= min(stack.height, ry1):
                return hit, False
        x0 = max(0, int(math.floor(rx0 - margin)))
        y0 = max(0, int(math.floor(ry0 - margin)))
        x1 = min(stack.width, int(math.ceil(rx1 + margin)))
        y1 = min(stack.height, int(math.ceil(ry1 + margin)))
        arr = to_uint16(stack.read_region(z, x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
        # Float image: bilinear warp is correct (PIL's I;16 bilinear is not); the window level is
        # applied after warping, on the small output
        hit = (x0, y0, Image.fromarray(arr.astype(np.float32)))
        self._regions[key] = hit
        return hit, True

    def _render(self, stack, z, grid, row, col, view_scale, refs, size):
        glw = self.mw.glw
        half = 0.5 * view_scale * max(grid.pitch_x, grid.pitch_y)
        cx = grid.start_x + (col + 0.5) * grid.pitch_x
        cy = grid.start_y + (row + 0.5) * grid.pitch_y
        x0, y0 = cx - half, cy - half # Patch top-left (Deskewed)

        # Deskewed square -> raw quad (TL, TR, BR, BL), like the main view
        corners = [glw.deskew_to_raw(x, y) for x, y in
                   [(x0, y0), (cx + half, y0), (cx + half, cy + half), (x0, cy + half)]]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        (rx0, ry0, region), did_read = self._region(stack, z, row, col, min(xs), min(ys), max(xs), max(ys),
                                                    margin=max(grid.pitch_x, grid.pitch_y))

        # Warp at display size, then window level -> 8-bit on the small output only
        quad = []
        for px, py in (corners[0], corners[3], corners[2], corners[1]): # PIL QUAD: UL, LL, LR, UR
            quad += [px - rx0, py - ry0]
        warped = np.asarray(region.transform((size, size), Image.QUAD, tuple(quad), resample=Image.BILINEAR))
        lkey = (glw.win_lo, glw.win_hi)
        if self._lut_key != lkey:
            self._lut = ImageNormalizer.process(np.arange(65536, dtype=np.float32), glw.win_lo, glw.win_hi)
            self._lut_key = lkey
        u8 = np.ascontiguousarray(self._lut[np.clip(warped, 0, 65535).astype(np.uint16)])
        img = QImage(u8.data, size, size, size, QImage.Format_Grayscale8).convertToFormat(QImage.Format_ARGB32)
        sx = sy = size / (2 * half)

        # Everything drawn here is axis-aligned: no antialiasing (it dominated the draw time)
        p = QPainter(img)
        # Grid lines inside the grid extent
        pen = QPen(QColor(255, 255, 0, 220))
        pen.setWidth(1)
        p.setPen(pen)
        gx_end = grid.start_x + grid.cols * grid.pitch_x
        gy_end = grid.start_y + grid.rows * grid.pitch_y
        for k in range(col - 2, col + 4):
            x = grid.start_x + k * grid.pitch_x
            if 0 <= k <= grid.cols and x0 <= x <= x0 + 2 * half:
                p.drawLine(QPointF((x - x0) * sx, (max(y0, grid.start_y) - y0) * sy),
                           QPointF((x - x0) * sx, (min(y0 + 2 * half, gy_end) - y0) * sy))
        for k in range(row - 2, row + 4):
            y = grid.start_y + k * grid.pitch_y
            if 0 <= k <= grid.rows and y0 <= y <= y0 + 2 * half:
                p.drawLine(QPointF((max(x0, grid.start_x) - x0) * sx, (y - y0) * sy),
                           QPointF((min(x0 + 2 * half, gx_end) - x0) * sx, (y - y0) * sy))

        # Reference bbox and product layout of this cell
        bbox = refs.get((row, col))
        if bbox is not None:
            bx, by, bw, bh = bbox
            pen = QPen(QColor(255, 255, 255, 230))
            pen.setWidth(1)
            p.setPen(pen)
            p.drawRect(QRectF((bx - x0) * sx, (by - y0) * sy, bw * sx, bh * sy))
            layout = glw.layout
            if layout:
                dw, dh = layout.get("design_size", (bw, bh))
                pen = QPen(QColor(0, 200, 255, 200))
                pen.setDashPattern([4, 4])
                p.setPen(pen)
                for (lx, ly, lw, lh) in layout_boxes(layout):
                    p.drawRect(QRectF((bx + lx * bw / dw - x0) * sx, (by + ly * bh / dh - y0) * sy,
                                      lw * bw / dw * sx, lh * bh / dh * sy))

        # Cell center cross
        pen = QPen(QColor(255, 0, 0, 200))
        pen.setWidthF(1.0)
        p.setPen(pen)
        ccx, ccy = (cx - x0) * sx, (cy - y0) * sy
        p.drawLine(QPointF(ccx - 8, ccy), QPointF(ccx + 8, ccy))
        p.drawLine(QPointF(ccx, ccy - 8), QPointF(ccx, ccy + 8))
        p.end()
        return QPixmap.fromImage(img), did_read
