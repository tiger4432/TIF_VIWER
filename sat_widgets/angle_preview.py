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

import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QDialog, QGridLayout, QLabel, QVBoxLayout, QWidget, QSizePolicy

from .core_data import ImageNormalizer
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
        self.image.setMinimumSize(240, 240)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image.setStyleSheet("background-color: #202020;")
        lay.addWidget(self.title)
        lay.addWidget(self.image, 1)
        self.cell = None
        self.pixmap = None

    def mousePressEvent(self, e):
        if self.cell is not None:
            self.clicked.emit(self.cell[1], self.cell[0])

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def set_content(self, title, cell, pixmap):
        self.title.setText(title)
        self.cell = cell
        self.pixmap = pixmap
        self._fit()

    def _fit(self):
        if self.pixmap is not None:
            self.image.setPixmap(self.pixmap.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class AnglePreview(QDialog):
    """Non-modal 2x2 preview; call schedule_update() whenever grid, layer or window changes."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("Angle Preview")
        self.resize(760, 800)
        self.mw = main_window

        lay = QVBoxLayout(self)
        self.info = QLabel("Grid lines should run along the chip edges in every patch. Click a patch to go there.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        grid = QGridLayout()
        lay.addLayout(grid, 1)
        self.tiles = []
        for i in range(4):
            t = _Tile()
            t.clicked.connect(self._navigate)
            grid.addWidget(t, i // 2, i % 2)
            self.tiles.append(t)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(120) # Debounce spin box / drag changes
        self.timer.timeout.connect(self.refresh)

    def schedule_update(self):
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        self.timer.start()

    def _navigate(self, col, row):
        self.mw.glw.fit_to_cell(col, row)

    def _settings(self):
        cfg = self.mw._config().get("angle_preview", {})
        return cfg.get("points") or DEFAULT_POINTS, float(cfg.get("view_scale", DEFAULT_VIEW_SCALE))

    def refresh(self):
        mw, glw = self.mw, self.mw.glw
        stack = mw.full_data
        if stack is None:
            self.info.setText("No image loaded.")
            return
        grid = glw.grid_cfg
        points, view_scale = self._settings()
        cells = preview_cells(grid, points, glw.bonding_map)

        # More tiles if more points are declared
        while len(self.tiles) < len(cells):
            t = _Tile()
            t.clicked.connect(self._navigate)
            self.layout().itemAt(1).layout().addWidget(t, len(self.tiles) // 2, len(self.tiles) % 2)
            self.tiles.append(t)

        refs = mw.void_manager.reference_bboxes(grid, glw.coord_transform)
        for tile, (name, row, col) in zip(self.tiles, cells):
            pix = self._render(stack, mw.current_z, grid, row, col, view_scale, refs)
            tile.set_content(f"{name}  (X{col:02d}, Y{row:02d})", (row, col), pix)
        stack.release_cache()
        self.info.setText(f"Angle {grid.angle:.3f}°  |  grid lines should run along the chip edges. Click a patch to go there.")

    def _render(self, stack, z, grid, row, col, view_scale, refs):
        glw = self.mw.glw
        half = 0.5 * view_scale * max(grid.pitch_x, grid.pitch_y)
        cx = grid.start_x + (col + 0.5) * grid.pitch_x
        cy = grid.start_y + (row + 0.5) * grid.pitch_y
        x0, y0 = cx - half, cy - half # Patch top-left (Deskewed)

        # Deskewed square -> raw quad (TL, TR, BR, BL) -> straightened patch, like the main view
        corners = [(x0, y0), (cx + half, y0), (cx + half, cy + half), (x0, cy + half)]
        patch = stack.get_poly_crop(z, [glw.deskew_to_raw(x, y) for x, y in corners])
        size = int(round(2 * half))
        if patch is None:
            img = QImage(size, size, QImage.Format_ARGB32)
            img.fill(Qt.black)
        else:
            u8 = np.ascontiguousarray(ImageNormalizer.process(patch, glw.win_lo, glw.win_hi))
            h, w = u8.shape
            img = QImage(u8.data, w, h, w, QImage.Format_Grayscale8).convertToFormat(QImage.Format_ARGB32)
        sx = img.width() / (2 * half)
        sy = img.height() / (2 * half)

        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        # Grid lines inside the grid extent
        pen = QPen(QColor(255, 255, 0, 220))
        pen.setWidthF(1.5)
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
            pen.setWidthF(1.5)
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
        return QPixmap.fromImage(img)
