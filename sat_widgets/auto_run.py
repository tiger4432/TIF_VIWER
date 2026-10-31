"""
Auto Run in the viewer window: watches glob patterns of measured files (config.json
"auto_run": {"watch": [...]}, editable in the panel) and runs every new complete file through the
steps of sat_widgets/pipeline.py, one file at a time (files arriving meanwhile are queued).

The steps run on a worker thread with their own stack reader, grid and VoidManager (the window
stays responsive); after each step its result is shown in the window on the GUI thread: the file is
opened as with Open TIF, the product selected, the bonding map applied, the grid / bboxes and the
detected voids replace the window's. The saved results (voids.json, meta.json, summary.json) come
from the run itself. A failed QC gate or an error marks the wafer, notifies and goes on with the
next file. Stop cancels the current wafer (it is processed again on the next Start).
"""
import os
import threading
import time
import traceback

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout)

from . import pipeline
from .app_paths import declaration_path
from .products import GRID_KEYS


class AutoRunController(QObject):
    step_signal = Signal(object, str, str, object) # run, step, phase, snapshot (worker -> GUI thread)
    done_signal = Signal(object, object)            # run, summary
    scan_signal = Signal(object)                    # ready paths

    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.watcher = None
        self.settings = None
        self.queue = []
        self.current = None # (run, cancel event, started)
        self.last = None
        self._scanning = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.clock = QTimer(self) # Elapsed time of the running step in the status line
        self.clock.timeout.connect(self._show_status)
        self.step_signal.connect(self._on_step)
        self.done_signal.connect(self._on_done)
        self.scan_signal.connect(self._on_scanned)
        self._step = ("", 0.0)
        self.panel = self._build_panel()

    # ---------------------------------------------------------------- panel
    def _build_panel(self):
        gb = QGroupBox("Auto Run")
        lay = QVBoxLayout(gb)
        h = QHBoxLayout()
        self.le_watch = QLineEdit("; ".join(pipeline.watch_globs(self._auto_settings())))
        self.le_watch.setPlaceholderText("D:/SAT/raw/**/*.tif; \\\\server\\sat\\*\\*.tif")
        self.le_watch.setToolTip("Measured files to watch: glob patterns separated by ';' (** = any subfolders).\n"
                                 "Default: config.json \"auto_run\": {\"watch\": [...]}")
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(28)
        btn_browse.setToolTip("Watch a folder (its TIFs, subfolders included)")
        btn_browse.clicked.connect(self._browse)
        h.addWidget(self.le_watch)
        h.addWidget(btn_browse)
        lay.addLayout(h)
        self.btn_start = QPushButton("Start Auto Run")
        self.btn_start.setCheckable(True)
        self.btn_start.setToolTip("Each new complete file: open, wafer ID, product, bonding map, Detect BBOX + Fit Grid,\n"
                                  "Detect Voids, save (voids.json, meta.json, summary.json), Results API, review request.\n"
                                  "See docs/AUTO_RUN.md")
        self.btn_start.toggled.connect(self._toggle)
        lay.addWidget(self.btn_start)
        self.lbl_status = QLabel("Stopped")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        return gb

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self.mw, "Watch folder (measured TIFs)")
        if folder:
            self.le_watch.setText(f"{folder}/**/*.tif; {folder}/**/*.tiff")

    def _auto_settings(self):
        return dict(self.mw._config().get("auto_run") or {})

    def _toggle(self, on):
        self.start() if on else self.stop()

    # ---------------------------------------------------------------- watching
    def start(self):
        s = self._auto_settings()
        s["config_file"] = declaration_path("config.json")
        s["products_file"] = declaration_path("products.json")
        s["watch"] = [p.strip() for p in self.le_watch.text().split(";") if p.strip()]
        # One layer range for the run and the window (auto_run's, else the window's)
        for key, sb in (("start_layer", self.mw.sb_start_layer), ("end_layer", self.mw.sb_end_layer)):
            if s.get(key) is not None:
                sb.setValue(int(s[key]))
            s[key] = sb.value()
        if not s["watch"]:
            self.lbl_status.setText("Enter the files to watch (glob patterns).")
            self.btn_start.setChecked(False)
            return
        self.settings = pipeline.merged_settings(s)
        self.watcher = pipeline.FileWatcher.for_settings(self.settings)
        self.queue = []
        self.le_watch.setEnabled(False)
        self.btn_start.setText("Stop Auto Run")
        self.timer.start(int(1000 * float(self.settings["poll_seconds"])))
        print(f"[auto_run] watching {'; '.join(self.watcher.globs)}")
        self._poll()
        self._show_status()

    def stop(self):
        self.timer.stop()
        if self.current is not None:
            self.current[1].set() # The run ends at its next check; the file is offered again later
        for p in self.queue:
            self.watcher.forget(p)
        self.queue = []
        self.le_watch.setEnabled(True)
        self.btn_start.setText("Start Auto Run")
        self._show_status()

    def running(self):
        return self.timer.isActive()

    def _poll(self):
        if self._scanning or self.watcher is None:
            return
        self._scanning = True
        watcher = self.watcher

        def scan(): # Globs over a network share can be slow: not on the GUI thread
            try:
                ready = watcher.poll()
            except Exception:
                traceback.print_exc()
                ready = []
            self.scan_signal.emit((watcher, ready))
        threading.Thread(target=scan, daemon=True).start()

    def _on_scanned(self, result):
        watcher, ready = result
        self._scanning = False
        if watcher is not self.watcher:
            return
        if not self.running():
            for p in ready:
                watcher.forget(p)
            return
        for p in ready:
            print(f"[auto_run] new file: {p}")
            self.queue.append(p)
        self._next()
        self._show_status()

    # ---------------------------------------------------------------- one wafer
    def _next(self):
        if self.current is not None or not self.queue or not self.running():
            return
        path = self.queue.pop(0)
        cancel = threading.Event()

        def on_step(run, name, phase): # Worker thread: copy what the window shows
            snap = run.snapshot() if phase == "done" and name in ("open", "product", "bonding", "bbox", "voids") else None
            self.step_signal.emit(run, name, phase, snap)

        run = pipeline.WaferRun(path, self.settings, on_step=on_step, cancel=cancel)
        self.current = (run, cancel, time.monotonic())
        self._step = ("starting", time.monotonic())

        def work():
            try:
                summary = run.run()
            except Exception as e:
                traceback.print_exc()
                summary = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
            self.done_signal.emit(run, summary)
        threading.Thread(target=work, daemon=True).start()
        self.clock.start(1000)

    def _on_step(self, run, name, phase, snap):
        if self.current is None or self.current[0] is not run:
            return
        self._step = (name, time.monotonic()) if phase == "start" else self._step
        self._show_status()
        if snap is None or run.cancel.is_set():
            return
        try:
            getattr(self, "_show_" + name)(run, snap)
        except Exception:
            traceback.print_exc() # The window is only a view: the run goes on
        self.mw.glw.update()

    def _show_open(self, run, snap):
        self.mw.open_file(run.path, confirm=False)

    def _show_product(self, run, snap):
        mw = self.mw
        mw.le_wafer_id.setText(snap["wafer_id"])
        if mw.cb_product.currentText() != snap["product"]:
            mw.cb_product.setCurrentText(snap["product"]) # Applies it
        else:
            mw.apply_product(snap["product"])

    def _show_bonding(self, run, snap):
        b = snap["bonding"]
        if b is None:
            return
        self.mw.le_reject.setText(", ".join(b["reject_values"]))
        self.mw.txt_map.setPlainText(b["text"])
        self.mw.import_map_data()

    def _show_grid(self, grid):
        mw = self.mw
        spins = {"start_x": mw.sb_grid_x, "start_y": mw.sb_grid_y, "pitch_x": mw.sb_pitch_x, "pitch_y": mw.sb_pitch_y,
                 "rows": mw.sb_rows, "cols": mw.sb_cols, "angle": mw.sb_angle}
        for k in GRID_KEYS:
            spins[k].blockSignals(True)
            spins[k].setValue(grid[k])
            spins[k].blockSignals(False)
        mw.update_grid_params()
        cfg = mw.glw.grid_cfg
        for k in GRID_KEYS: # Exact values (the spin boxes round)
            setattr(cfg, k, grid[k])
        mw.glw.update_grid_params()

    def _show_bbox(self, run, snap):
        self._show_grid(snap["grid"])
        self._show_voids(run, snap)

    def _show_voids(self, run, snap):
        mw = self.mw
        mw.void_manager.replace_all(snap["voids"])
        mw.glw.active_void = None
        mw.glw.refresh_hover()
        mw.update_void_ui()

    def _on_done(self, run, summary):
        if self.current is not None and self.current[0] is run:
            self.current = None
        self.clock.stop()
        if self.watcher is not None:
            self.watcher.finished(run.path, summary)
        self.last = (run.wafer_id or os.path.basename(run.path), summary.get("status"), summary.get("reason", ""))
        msg = f"Auto Run: {self.last[0]} {self.last[1]}" + (f" ({self.last[2]})" if self.last[2] else "")
        self.mw.statusBar().showMessage(msg)
        self._show_status()
        self._next()

    def _show_status(self):
        lines = []
        if self.running():
            lines.append(f"Watching: {'; '.join(self.watcher.globs)}")
        else:
            lines.append("Stopped" + (" (finishing the current wafer...)" if self.current else ""))
        if self.current is not None:
            run = self.current[0]
            name, t0 = self._step
            lines.append(f"Now: {run.wafer_id or os.path.basename(run.path)} - {name or 'starting'} "
                         f"({time.monotonic() - t0:.0f} s)")
        lines.append(f"Queue: {len(self.queue)}")
        if self.last is not None:
            lines.append(f"Last: {self.last[0]} {self.last[1]}" + (f" - {self.last[2]}" if self.last[2] else ""))
        self.lbl_status.setText("\n".join(lines))
