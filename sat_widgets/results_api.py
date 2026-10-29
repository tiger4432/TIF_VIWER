"""
Host of the Results API plugin (plugins/results_api.py, written by the site developer):
loads it from file every time it is used, builds the context, runs the requests on a background
thread and reports errors (with the traceback, for the developer).
"""
import datetime
import getpass
import importlib.util
import json
import os
import socket
import threading
import traceback

from PySide6.QtCore import QObject, Signal

from .app_paths import app_file
from .version import APP_VERSION

DEFAULT_PLUGIN = "plugins/results_api.py"


def plugin_path(settings):
    path = (settings or {}).get("plugin") or DEFAULT_PLUGIN
    return path if os.path.isabs(path) else app_file(path)


def load_plugin(settings):
    """Fresh import of the plugin file (edits apply without restarting)."""
    path = plugin_path(settings)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results API plugin not found: {path}")
    spec = importlib.util.spec_from_file_location("results_api_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("build_payload", "send", "fetch", "to_voids"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"{path}: function {name}() is missing")
    return module


def build_context(mw, settings):
    """Read-only snapshot of the current state for the plugin (see plugins/results_api.py)."""
    glw, vm = mw.glw, mw.void_manager
    g = glw.grid_cfg
    stack = mw.full_data
    source = getattr(stack, "path", None) or ""
    if source.endswith("patches.json"):
        source = os.path.dirname(source) # Project folder
    bmap = glw.bonding_map
    bonding = None
    if bmap is not None:
        bonding = {"text": getattr(bmap, "source_text", ""),
                   "reject_values": list(getattr(bmap, "reject_values", [])),
                   "rows": [list(r) for r in bmap.data_map]}
    layers = [mw.abs_layer(z) for z in range(len(stack))] if stack is not None else []
    return {
        "app_version": APP_VERSION,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "product": mw.cb_product.currentText(),
        "source_file": source,
        "source_name": os.path.splitext(os.path.basename(source.rstrip("/\\")))[0],
        "wafer_id": mw.le_wafer_id.text().strip(),
        "layers": layers,
        "current_layer": glw.current_layer,
        "grid": {"start_x": g.start_x, "start_y": g.start_y, "pitch_x": g.pitch_x, "pitch_y": g.pitch_y,
                 "rows": g.rows, "cols": g.cols, "angle": g.angle},
        "um_per_px": vm.um_per_px or 0.0,
        "bonding_map": bonding,
        "voids_json": vm.to_data(g, bmap, glw.coord_transform),
        "records": vm.void_records(g, bmap, glw.coord_transform),
        "types": {k: dict(v) for k, v in vm.types.items()},
        "last_export_dir": getattr(mw, "last_export_dir", None),
        "settings": settings,
    }


def normalize_bonding_map(result):
    """
    fetch_bonding_map() result -> (map text as pasted from Excel, reject values or None).
    Accepted: text (tab / comma separated rows), a list of rows, or a dict with "text" or "rows"
    and optionally "reject_values" (list or comma separated text).
    """
    reject = None
    if isinstance(result, dict):
        reject = result.get("reject_values")
        if isinstance(reject, str):
            reject = [v.strip() for v in reject.split(",") if v.strip()]
        elif reject is not None:
            reject = [str(v) for v in reject]
        result = result.get("text", result.get("rows"))
    if isinstance(result, (list, tuple)):
        result = "\n".join("\t".join("" if v is None else str(v) for v in row) for row in result)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"no bonding map in the result ({type(result).__name__})")
    if "\t" not in result and "," in result: # CSV -> tab separated
        result = "\n".join("\t".join(c.strip() for c in line.split(",")) for line in result.splitlines())
    return result, reject


def to_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


class BackgroundCall(QObject):
    """Runs fn() on a thread; `done(result, error_text)` arrives on the GUI thread."""
    done = Signal(object, object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            result, error = self._fn(), None
        except Exception:
            result, error = None, traceback.format_exc()
        self.done.emit(result, error)
