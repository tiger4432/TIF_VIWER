from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, 
    QPushButton, QHBoxLayout, QLabel
)
from PySide6.QtCore import Qt

class ROIInspector(QDialog):
    def __init__(self, rois, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Inspector")
        self.resize(800, 600)
        self.rois = rois
        
        layout = QVBoxLayout(self)
        
        # Header
        self.lbl_info = QLabel(f"Total ROIs: {len(self.rois)}")
        layout.addWidget(self.lbl_info)
        
        # Table
        self.table = QTableWidget()
        cols = ["Index", "Label", "Grid(R,C)", "TL (Raw)", "TR", "BR", "BL", "W x H", "Angle"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.table)
        
        # Buttons
        h_btn = QHBoxLayout()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.populate)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        h_btn.addWidget(btn_refresh)
        h_btn.addWidget(btn_close)
        layout.addLayout(h_btn)
        
        self.populate()
        
    def populate(self):
        self.table.setRowCount(0)
        if not self.rois:
            return
            
        self.table.setRowCount(len(self.rois))
        
        import numpy as np
        
        for i, roi in enumerate(self.rois):
            # Index
            self.table.setItem(i, 0, QTableWidgetItem(str(i)))
            
            # Label
            self.table.setItem(i, 1, QTableWidgetItem(roi.get('label', '-')))
            
            # Grid
            self.table.setItem(i, 2, QTableWidgetItem(f"{roi.get('y')},{roi.get('x')}"))
            
            # BBox
            bbox = roi.get('bbox', [])
            if len(bbox) == 4:
                # Format: (x,y)
                tl = f"{bbox[0][0]:.1f},{bbox[0][1]:.1f}"
                tr = f"{bbox[1][0]:.1f},{bbox[1][1]:.1f}"
                br = f"{bbox[2][0]:.1f},{bbox[2][1]:.1f}"
                bl = f"{bbox[3][0]:.1f},{bbox[3][1]:.1f}"
                
                self.table.setItem(i, 3, QTableWidgetItem(tl))
                self.table.setItem(i, 4, QTableWidgetItem(tr))
                self.table.setItem(i, 5, QTableWidgetItem(br))
                self.table.setItem(i, 6, QTableWidgetItem(bl))
                
                # Geometry calc (same as PatchViewer)
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                p0, p1, p2, p3 = bbox
                
                w1 = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
                h1 = np.hypot(p3[0]-p0[0], p3[1]-p0[1])
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                self.table.setItem(i, 7, QTableWidgetItem(f"{w1:.1f}x{h1:.1f}"))
                self.table.setItem(i, 8, QTableWidgetItem(f"{angle:.2f}"))
            else:
                self.table.setItem(i, 3, QTableWidgetItem("Invalid BBox"))
