"""
Seg Data window: builds and reviews the U-Net training data (sat_widgets/void_seg.py).

- Add Open File: samples of the open file and its markings.
- Add Folder...: samples of saved results under a folder (voids.json + meta.json naming the TIF
  and the grid; bonding_map.txt beside it is used when present), without opening them.
- Sources (left): one per file; the check box keeps or leaves out the whole file.
- Samples (right): chip thumbnails, void green, unknown yellow. Click a sample to keep / leave it
  out (left-out samples are crossed). Filter: with voids / without voids / all.
Choices are written to the source's index.json at once; tools/train_seg.py trains on the kept ones.
"""
import os

from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem,
                               QLabel, QComboBox, QSplitter, QWidget, QMessageBox, QFileDialog, QProgressDialog,
                               QApplication)

from . import void_seg, seg_data


class SegDataDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("Segmentation Training Data")
        self.resize(1200, 760)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        for text, slot, tip in (
                ("Add Open File", self.add_open, "Samples of the open file and its markings (hand markings = voids)"),
                ("Add Folder...", self.add_folder, "Saved results under a folder: voids.json + meta.json (TIF path and grid inside)"),
                ("Remove Source", self.remove_source, "Delete the selected file's samples"),
                ("Open Data Folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(void_seg.DATA_DIR)), void_seg.DATA_DIR)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            top.addWidget(b)
        top.addStretch(1)
        self.summary = QLabel()
        top.addWidget(self.summary)
        lay.addLayout(top)

        split = QSplitter()
        self.sources = QListWidget()
        self.sources.currentRowChanged.connect(self.show_source)
        self.sources.itemChanged.connect(self._source_toggled)
        split.addWidget(self.sources)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItems(["With voids", "Without voids", "All"])
        self.filter.currentIndexChanged.connect(lambda _: self.show_source(self.sources.currentRow()))
        bar.addWidget(QLabel("Show:"))
        bar.addWidget(self.filter)
        for text, value in (("Keep shown", True), ("Leave out shown", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, v=value: self.set_shown(v))
            bar.addWidget(b)
        bar.addStretch(1)
        bar.addWidget(QLabel("Click a chip: keep / leave out. Green = void, yellow = unknown (not trained)."))
        rl.addLayout(bar)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(128, 128))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setSpacing(4)
        self.grid.itemClicked.connect(self._sample_clicked)
        rl.addWidget(self.grid, 1)
        split.addWidget(right)
        split.setSizes([300, 900])
        lay.addWidget(split, 1)
        self.data = []
        self.reload()

    # ----- data -----
    def reload(self, select=None):
        self.data = seg_data_sources()
        self.sources.blockSignals(True)
        self.sources.clear()
        for folder, index in self.data:
            n = len(index["samples"])
            nv = sum(s["voids"] for s in index["samples"])
            kept = sum(1 for s in index["samples"] if s.get("include", True))
            it = QListWidgetItem(f"{os.path.basename(index.get('source', folder))}\n{n} chips, {nv} with voids, {kept} kept")
            it.setToolTip(index.get("source", folder))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if index.get("include", True) else Qt.Unchecked)
            self.sources.addItem(it)
        self.sources.blockSignals(False)
        self._update_summary()
        row = 0
        if select is not None:
            row = next((i for i, (f, _) in enumerate(self.data) if f == select), 0)
        if self.data:
            self.sources.setCurrentRow(row)
        else:
            self.grid.clear()

    def _update_summary(self):
        files = sum(1 for _, ix in self.data if ix.get("include", True))
        kept = [s for _, ix in self.data if ix.get("include", True) for s in ix["samples"] if s.get("include", True)]
        self.summary.setText(f"Training set: {files} files, {len(kept)} chips, {sum(s['voids'] for s in kept)} with voids")

    def show_source(self, row):
        self.grid.clear()
        if row < 0 or row >= len(self.data):
            return
        folder, index = self.data[row]
        mode = self.filter.currentIndex()
        for i, s in enumerate(index["samples"]):
            if (mode == 0 and not s["voids"]) or (mode == 1 and s["voids"]):
                continue
            it = QListWidgetItem(f"L{s['layer']} r{s['row']} c{s['col']}")
            it.setData(Qt.UserRole, i)
            self._set_icon(it, folder, s)
            self.grid.addItem(it)

    def _set_icon(self, item, folder, s):
        pm = QPixmap(os.path.join(folder, s["id"] + ".png"))
        if not s.get("include", True):
            p = QPainter(pm)
            p.fillRect(pm.rect(), QColor(0, 0, 0, 140))
            pen = QPen(QColor(255, 60, 60), 4)
            p.setPen(pen)
            p.drawLine(0, 0, pm.width(), pm.height())
            p.drawLine(pm.width(), 0, 0, pm.height())
            p.end()
        item.setIcon(QIcon(pm))

    def _sample_clicked(self, item):
        folder, index = self.data[self.sources.currentRow()]
        s = index["samples"][item.data(Qt.UserRole)]
        s["include"] = not s.get("include", True)
        void_seg.save_index(folder, index)
        self._set_icon(item, folder, s)
        self._refresh_source_text()

    def set_shown(self, value):
        row = self.sources.currentRow()
        if row < 0:
            return
        folder, index = self.data[row]
        for k in range(self.grid.count()):
            it = self.grid.item(k)
            s = index["samples"][it.data(Qt.UserRole)]
            s["include"] = value
            self._set_icon(it, folder, s)
        void_seg.save_index(folder, index)
        self._refresh_source_text()

    def _source_toggled(self, item):
        row = self.sources.row(item)
        folder, index = self.data[row]
        index["include"] = item.checkState() == Qt.Checked
        void_seg.save_index(folder, index)
        self._update_summary()

    def _refresh_source_text(self):
        row = self.sources.currentRow()
        folder, index = self.data[row]
        n = len(index["samples"])
        nv = sum(s["voids"] for s in index["samples"])
        kept = sum(1 for s in index["samples"] if s.get("include", True))
        self.sources.blockSignals(True)
        self.sources.item(row).setText(f"{os.path.basename(index.get('source', folder))}\n{n} chips, {nv} with voids, {kept} kept")
        self.sources.blockSignals(False)
        self._update_summary()

    # ----- adding -----
    def _progress(self, text):
        dlg = QProgressDialog(text, "Cancel", 0, 1000, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()

        def step(i, n, label):
            if i % 20 == 0:
                dlg.setValue(int(1000 * i / max(1, n)))
                dlg.setLabelText(label)
                QApplication.processEvents()
            return not dlg.wasCanceled()
        return dlg, step

    def add_open(self):
        folder = self.mw.add_seg_samples_from_open(self._progress)
        if folder:
            self.reload(select=folder)

    def add_folder(self):
        root = QFileDialog.getExistingDirectory(self, "Folder with saved results (voids.json + meta.json)")
        if not root:
            return
        found = seg_data.find_saved(root)
        if not found:
            QMessageBox.information(self, "Add Folder", "No voids.json with a meta.json beside it under this folder.")
            return
        params = self.mw.seg_params()
        done, failed, last = [], [], None
        for i, vp in enumerate(found):
            dlg, step = self._progress(f"{i + 1}/{len(found)}: {vp}")
            try:
                out = seg_data.extract_saved(vp, params, step)
                if out is None:
                    break # Cancelled
                source, meta, refs, samples = out
                last = void_seg.write_source(source, meta, refs, samples)
                done.append(f"{os.path.basename(source)}: {len(samples)} chips, {sum(int((s['y'] == 1).any()) for s in samples)} with voids")
            except Exception as e:
                failed.append(f"{vp}: {e}")
            finally:
                dlg.close()
        self.reload(select=last)
        QMessageBox.information(self, "Add Folder", f"Added {len(done)} of {len(found)}.\n\n" + "\n".join(done[:20])
                                + ("\n\nNot added:\n" + "\n".join(failed[:20]) if failed else ""))

    def remove_source(self):
        row = self.sources.currentRow()
        if row < 0:
            return
        folder, index = self.data[row]
        if QMessageBox.question(self, "Remove Source", f"Delete the samples of\n{index.get('source', folder)}?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        self.reload()


def seg_data_sources():
    return void_seg.list_sources()
