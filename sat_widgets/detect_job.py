"""
Detect Voids in the background (see MainWindow.detect_voids).

The worker thread reads the image with its own reader (stack.reopen(), like layer_loader.py) and a
read-only copy of the void data; it never touches the VoidManager. Found voids, per-cell progress
and the result arrive on the GUI thread through signals (queued, in emission order), where they
are placed (void_detect.place_found) inside one undo step.
"""
import threading
import time
import traceback

from PySide6.QtCore import QObject, Signal

from . import void_detect


def read_only_copy(vm):
    """Copy of a VoidManager for the worker: its own void dicts and type table, no undo history."""
    import copy
    ro = copy.copy(vm)
    ro.voids = vm._snapshot()
    ro.types = {k: dict(v) for k, v in vm.types.items()}
    ro._undo = []
    return ro


class DetectJob(QObject):
    found = Signal(object) # [found tuples] to place (void_detect.place_found)
    cells = Signal(object, float) # cells, share of their work done (1.0 = done)
    progress = Signal(int, int, str) # i, n, text (throttled)
    finished = Signal(object, object) # result dict (None: cancelled or failed), error text or None

    def __init__(self, open_reader, args, parent=None):
        """args: detect_voids arguments after the stack (grid_cfg .. params), see void_detect.detect_voids."""
        super().__init__(parent)
        self._open_reader = open_reader
        self._args = args
        self._cancel = threading.Event()
        self._last = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def cancel(self):
        self._cancel.set()

    def cancelled(self):
        return self._cancel.is_set()

    def wait(self, timeout=None):
        self._thread.join(timeout)

    def _step(self, i, n, text):
        now = time.monotonic()
        if now - self._last > 0.1:
            self._last = now
            self.progress.emit(i, n, text)
        return not self._cancel.is_set()

    def _run(self):
        result, error, reader = None, None, None
        try:
            reader = self._open_reader()
            result = void_detect.detect_voids(reader, *self._args, progress=self._step,
                                              on_found=self.found.emit, on_cells=self.cells.emit)
        except Exception:
            error = traceback.format_exc()
        finally:
            if reader is not None:
                reader.close()
            self._args = None # The void snapshot etc. are not needed any more
        self.finished.emit(None if self._cancel.is_set() else result, error)
