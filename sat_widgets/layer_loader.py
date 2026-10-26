"""
Background layer decoding, for smooth layer browsing.

Worker threads decode layers with their own reader (stack.reopen(), created in the thread), so
the UI never waits for a decode and a layer skipped while scrolling is never decoded:
- the layer being shown, if it is not on the GPU yet (urgent),
- its neighbours (next in the browsing direction first), so stepping one layer is instant,
- a low-resolution preview of every layer, shown at once on a switch while the full
  resolution tiles are uploaded.

Results are delivered on the GUI thread through the `loaded` signal; GPU uploads stay on the GUI
thread (see MainWindow._upload_step).
"""
import threading

import numpy as np
from PySide6.QtCore import QObject, Signal

from .core_data import to_gray

PROXY_MAX = 2048 # Longest side of a preview texture


def proxy_step(width, height):
    return max(1, int(np.ceil(max(width, height) / PROXY_MAX)))


class LayerLoader(QObject):
    # z, tiles (list of (h, w) arrays in TiledImage tile order) or None, preview array
    loaded = Signal(int, object, object)

    def __init__(self, stack, tile_size, workers=2, parent=None):
        super().__init__(parent)
        self._open_reader = stack.reopen
        self.width, self.height = stack.width, stack.height
        self.tile_size = tile_size
        self._cv = threading.Condition()
        self._queue = [] # [(z, want_tiles)] in priority order
        self._running = set() # Layers being decoded
        self._undelivered = 0 # Results emitted but not handled on the GUI thread yet (see ack)
        self._stop = False
        self._threads = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for t in self._threads:
            t.start()

    def request(self, jobs):
        """Replaces the pending jobs: [(z, want_tiles)], most urgent first."""
        with self._cv:
            self._queue = [j for j in jobs if j[0] not in self._running]
            self._cv.notify_all()

    def busy(self):
        with self._cv:
            return bool(self._queue or self._running or self._undelivered)

    def ack(self):
        """Called by the receiver of `loaded` once a result is handled."""
        with self._cv:
            self._undelivered -= 1

    def stop(self):
        with self._cv:
            self._stop = True
            self._queue = []
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=5)

    def _run(self):
        reader = None
        try:
            while True:
                with self._cv:
                    while not self._queue and not self._stop:
                        self._cv.wait()
                    if self._stop:
                        return
                    z, want_tiles = self._queue.pop(0)
                    self._running.add(z)
                try:
                    if reader is None:
                        reader = self._open_reader()
                    tiles, proxy = self._decode(reader, z, want_tiles)
                except Exception as e:
                    print(f"[LayerLoader] layer {z}: {e}")
                    tiles = proxy = None
                with self._cv:
                    self._running.discard(z)
                    if self._stop:
                        return
                    if proxy is not None:
                        self._undelivered += 1
                if proxy is not None:
                    self.loaded.emit(z, tiles, proxy)
        finally:
            if reader is not None:
                reader.close()

    def _decode(self, reader, z, want_tiles):
        w, h = self.width, self.height
        arr = to_gray(reader.read_region(z, 0, 0, w, h))
        s = proxy_step(w, h)
        proxy = np.ascontiguousarray(arr[::s, ::s])
        tiles = None
        if want_tiles:
            # Views into one decoded array (no copies); each tile is made contiguous at upload.
            # A decoded frame stays alive through the views after release_cache(); a memory
            # map is read here, not on the GUI thread at upload time.
            T = self.tile_size
            if isinstance(arr, np.memmap):
                arr = np.array(arr)
            tiles = [arr[y:y + T, x:x + T] for y in range(0, h, T) for x in range(0, w, T)] # TiledImage order
        reader.release_cache()
        return tiles, proxy
