"""
Automatic run of one measured wafer: what an operator does in the viewer, step by step, each step
with a QC gate. The steps work on their own objects (stack reader, grid, VoidManager), no window:
the viewer's Auto Run (sat_widgets/auto_run.py) runs them on a worker thread and shows the result
of every step in the window; tools/auto_run.py runs them without a window.

    summary = WaferRun(path, settings, on_step=None, cancel=None).run()

Steps (summary["steps"]):
    wafer_id  patches.json "wafer_id" (projects) > plugin parse_wafer_id(file_name, settings)
              (str, or {"wafer_id", "lot"}) > config.json "wafer_id" pattern ("group", "lot_group")
              > file name
    open      TiffStack (a project folder / patches.json: MosaicTiffStack), layer range
    product   plugin product_for(wafer_id, file_name, settings) > first auto_run "products" rule
              whose given patterns ("lot", "wafer_id", "file") all match; none -> needs_attention.
              Applied like the viewer's WF Product (grid keys, um_per_px, bbox size, layout, type_rules)
    bonding   plugin fetch_bonding_map(ctx, settings) -> BondingMap; grid rows / cols from the map
    bbox      detect_layout -> fit_grid -> grid moved (bboxes keep their place in the cells) ->
              detect_layout again on the fitted grid
    voids     void_detect.detect_voids (method from settings, default layer change; learned filter
              of the product when its model file exists)
    save      voids.json + meta.json, summary.json, pipeline.log in the wafer's output folder
    send      optional: plugin send(build_payload(ctx), settings) (auto_run "send_results")
    notify    plugin notify_review(title, body, recipients, ctx, settings) -> (ok, message);
              without the plugin function: printed and appended to <output_root>/review_queue.jsonl

A failed gate stops the wafer with status "needs_attention" and the reason; an exception gives
"error"; cancel (threading.Event) gives "cancelled". Except when cancelled, what exists is saved and
the reviewer is notified. Every result is marked "review": "pending" (summary.json and meta.json).

on_step(run, name, phase): called on the running thread before ("start") and after ("done") every
step; at "done" the step's result is in the run (grid, bmap, vm, ...).

Settings: config.json "auto_run" (read only), see docs/AUTO_RUN.md. Plugin functions get
config.json "results_api" as settings (as in the viewer), product_for gets "auto_run".
"""
import copy
import datetime
import getpass
import glob
import io
import json
import os
import platform
import re
import socket
import sys
import threading
import time
import traceback

import numpy as np

from . import results_api, void_detect, void_layer_change, void_learn
from .app_paths import declaration_path
from .auto_bbox import detect_layout, fit_grid
from .core_data import BondingMap, CoordinateTransform, GridConfig, MosaicTiffStack, TiffStack
from .products import GRID_KEYS, load_products
from .version import APP_VERSION
from .void_manager import VoidManager

DEFAULTS = {
    "output_root": "auto_results",
    "output_dir": "{out}/{product}/{wafer_id}", # {out}, {product}, {wafer_id}, {lot}, {source_name}, {date}
    "watch": [],              # Glob patterns of the measured files ("**" recursive)
    "stable_seconds": 10.0,   # Size and mtime unchanged this long: the file is complete
    "poll_seconds": 5.0,
    "state_file": None,       # Default: <output_root>/auto_run_state.json
    "method": "layer_change",
    "use_learned": True,
    "require_bonding_map": True,
    "fit_grid": True,
    "send_results": False,
    "qc": {
        "max_estimated_frac": 0.3, # Bboxes estimated from the lattice / bonded cells
        "max_fit_rms_px": 3.0,     # Fit Grid residual
        "max_angle_change": 1.0,   # Fitted angle vs the declared one (deg)
        "require_package_box": True, # The chip package edge found in the golden chip (else the grid
                                     # may sit a fraction of a cell off: bboxes on the streets)
        "max_voids_per_chip": None, # e.g. 5: more on average -> needs_attention
    },
}

STEPS = ("wafer_id", "open", "product", "bonding", "bbox", "voids")


class GateFailed(Exception):
    """A QC gate stopped the wafer (status needs_attention)."""


class Cancelled(Exception):
    """The run was stopped (Auto Run: Stop)."""


class _Tee(io.TextIOBase):
    """Console output also written to the wafer's log (the modules report with print)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def merged_settings(settings):
    s = dict(DEFAULTS)
    s.update(settings or {})
    s["qc"] = dict(DEFAULTS["qc"], **(settings or {}).get("qc", {}))
    return s


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _safe(name):
    return re.sub(r'[<>:"/\\|?*\s]+', "_", str(name)).strip("._") or "unknown"


class WaferRun:
    """State of one wafer through the steps (the headless counterpart of MainWindow)."""

    def __init__(self, path, settings, on_step=None, cancel=None):
        self.path = os.path.abspath(path)
        self.s = merged_settings(settings)
        self.on_step = on_step
        self.cancel = cancel or threading.Event()
        self.config_path = self.s.get("config_file") or declaration_path("config.json")
        self.products_path = self.s.get("products_file") or declaration_path("products.json")
        self.config = read_json(self.config_path, {}) or {}
        self.results_settings = dict(self.config.get("results_api") or {})
        if self.s.get("plugin"):
            self.results_settings["plugin"] = self.s["plugin"]
        self.plugin = None
        self.plugin_error = None
        self.stack = None
        self.grid = GridConfig()
        self.ct = None
        self.bmap = None
        self.vm = None
        self.product_name, self.product = "", None
        self.wafer_id, self.lot = "", ""
        self.cells = []
        self.out_dir = None
        self.summary = {
            "status": "running", "reason": "", "source_file": self.path,
            "source_name": self.source_name(), "wafer_id": "", "lot": "", "product": "",
            "started_at": _now(), "steps": [], "qc": {}, "warnings": [], "models": {},
            "review": "pending",
            "versions": {"app": APP_VERSION, "python": platform.python_version(), "numpy": np.__version__},
        }

    # ---------------------------------------------------------------- helpers
    def source_name(self):
        return os.path.splitext(self.file_name())[0]

    def file_name(self):
        p = self.path
        if os.path.basename(p) == "patches.json":
            p = os.path.dirname(p)
        return os.path.basename(p.rstrip("/\\"))

    def warn(self, text):
        print(f"[auto_run] warning: {text}")
        self.summary["warnings"].append(text)

    def load_plugin(self):
        """The Results API plugin, or None (not configured / broken: the fallbacks are used)."""
        if self.plugin is None and self.plugin_error is None:
            try:
                self.plugin = results_api.load_plugin(self.results_settings)
            except Exception as e:
                self.plugin_error = str(e)
                self.warn(f"plugin not loaded: {e}")
        return self.plugin

    def plugin_fn(self, name):
        plugin = self.load_plugin()
        fn = getattr(plugin, name, None) if plugin is not None else None
        return fn if callable(fn) else None

    def abs_layer(self, z):
        return self.stack.layer_number(z)

    def layers(self):
        return [self.abs_layer(z) for z in range(len(self.stack))] if self.stack is not None else []

    def bbox_type(self):
        for name in self.vm.types:
            if name.strip().lower() == VoidManager.BBOX_TYPE_NAME:
                return name
        return None

    def update_transform(self):
        self.ct = CoordinateTransform(self.grid.angle, self.stack.width, self.stack.height)

    def bonded_cells(self):
        g, b = self.grid, self.bmap
        return [(r, c) for r in range(g.rows) for c in range(g.cols) if b is None or b.get_key(r, c)]

    def check_cancel(self):
        if self.cancel.is_set():
            raise Cancelled("stopped")

    def _progress1(self, *_):
        return not self.cancel.is_set()

    def snapshot(self):
        """Copy of what the window shows (taken on the running thread, applied on the GUI thread)."""
        return {"wafer_id": self.wafer_id, "lot": self.lot, "product": self.product_name,
                "grid": {k: getattr(self.grid, k) for k in GRID_KEYS},
                "bonding": None if self.bmap is None else
                {"text": getattr(self.bmap, "source_text", ""), "reject_values": list(self.bmap.reject_values)},
                "voids": copy.deepcopy(self.vm.voids) if self.vm is not None else None,
                "layers": self.layers()}

    # ---------------------------------------------------------------- steps
    def step_wafer_id(self):
        name = self.file_name()
        wid, lot, how = None, None, None
        pj = self.path if self.path.endswith("patches.json") else os.path.join(self.path, "patches.json")
        if os.path.isfile(pj): # Saved with the export (as when the viewer opens it)
            saved = read_json(pj, {}) or {}
            wid, lot = saved.get("wafer_id"), saved.get("lot")
            how = "patches.json" if wid else None
        fn = self.plugin_fn("parse_wafer_id")
        if not wid and fn is not None:
            try:
                res = fn(name, self.results_settings)
                if isinstance(res, dict):
                    wid, lot = res.get("wafer_id"), res.get("lot")
                else:
                    wid = res
                how = "plugin"
            except Exception as e:
                self.warn(f"plugin parse_wafer_id: {e}")
        rule = self.config.get("wafer_id") or {}
        if (not wid or not lot) and rule.get("pattern"):
            try:
                m = re.search(rule["pattern"], name)
                if m:
                    if not wid:
                        wid, how = m.group(rule.get("group", 1 if m.groups() else 0)), "config pattern"
                    if not lot and rule.get("lot_group") is not None:
                        lot = m.group(rule["lot_group"])
                elif not wid:
                    self.warn(f"wafer_id pattern {rule['pattern']!r} does not match {name!r}")
            except (re.error, IndexError) as e:
                self.warn(f"wafer_id pattern: {e}")
        if not wid:
            wid, how = os.path.splitext(name)[0], "file name"
        self.wafer_id, self.lot = str(wid).strip(), str(lot or "").strip()
        self.summary.update(wafer_id=self.wafer_id, lot=self.lot)
        return {"wafer_id": self.wafer_id, "lot": self.lot, "from": how}

    def step_open(self):
        p = self.path
        if os.path.isdir(p):
            p = os.path.join(p, "patches.json")
        if os.path.basename(p) == "patches.json":
            self.stack = MosaicTiffStack(p)
        else:
            start, end = self.s.get("start_layer"), self.s.get("end_layer")
            ui = self.config.get("ui_defaults") or {}
            start = ui.get("start_layer", 0) if start is None else start
            end = ui.get("end_layer", -1) if end is None else end
            if int(start) <= int(end):
                self.stack = TiffStack(p, int(start), int(end) - int(start) + 1)
            else:
                self.stack = TiffStack(p)
        layers = self.layers()
        if len(layers) < 2:
            raise GateFailed(f"only {len(layers)} layer(s) loaded")
        return {"layers": [layers[0], layers[-1]], "size": [self.stack.width, self.stack.height],
                "backend": getattr(self.stack, "backend", "mosaic")}

    def match_product(self):
        """(product name, how) from the plugin or the first matching rule, else (None, None)."""
        fn = self.plugin_fn("product_for")
        if fn is not None:
            try:
                name = fn(self.wafer_id, self.file_name(), self.s)
                if name:
                    return name, "plugin"
            except Exception as e:
                self.warn(f"plugin product_for: {e}")
        values = {"lot": self.lot, "wafer_id": self.wafer_id, "file": self.file_name()}
        for i, rule in enumerate(self.s.get("products") or []):
            given = [k for k in ("lot", "wafer_id", "file") if rule.get(k)]
            if not given or not rule.get("product"):
                self.warn(f"products rule {i + 1}: needs \"product\" and at least one of lot / wafer_id / file")
                continue
            try:
                if all(re.search(rule[k], values[k]) for k in given):
                    return rule["product"], f"rule {i + 1} ({', '.join(f'{k}={rule[k]!r}' for k in given)})"
            except re.error as e:
                self.warn(f"products rule {i + 1}: {e}")
        return None, None

    def step_product(self):
        products, _ = load_products(self.products_path)
        name, how = self.match_product()
        if not name:
            raise GateFailed(f"no product for lot {self.lot!r} / wafer ID {self.wafer_id!r} "
                             '(config.json "auto_run": "products" rules)')
        if name not in products:
            raise GateFailed(f"product {name!r} not declared in {self.products_path}")
        self.apply_product(name, products[name])
        info = {"product": name, "from": how, "um_per_px": self.vm.um_per_px}
        t = self.vm.types.get(self.bbox_type() or "", {})
        info["bbox_size"] = [t.get("def_w", 0.0), t.get("def_h", 0.0)]
        return info

    def apply_product(self, name, prod):
        """As MainWindow.apply_product: declared grid keys, scale, bbox default size, layout."""
        self.product_name, self.product = name, prod
        self.summary["product"] = name
        for key in GRID_KEYS:
            if key in prod.get("grid", {}):
                setattr(self.grid, key, prod["grid"][key])
        if isinstance(self.stack, MosaicTiffStack): # The project's own (exported) grid wins
            for key in GRID_KEYS:
                if key in self.stack.grid_cfg:
                    setattr(self.grid, key, self.stack.grid_cfg[key])
        self.vm = VoidManager(self.config_path) # Void types, output columns from config.json
        ui = self.config.get("ui_defaults") or {}
        self.vm.um_per_px = float(prod.get("um_per_px", ui.get("um_per_px", 0.0)) or 0.0)
        self.apply_bbox_size()
        layout = prod.get("layout")
        chip = prod.get("chip_size_um")
        if layout and chip and "design_size" not in layout:
            layout = dict(layout, design_size=list(chip))
        self.vm.layout = layout
        self.update_transform()

    def apply_bbox_size(self):
        """As MainWindow._apply_bbox_size: chip_size_um / um_per_px, else legacy "bbox" w / h."""
        name = self.bbox_type()
        if name is None:
            return
        prod = self.product or {}
        chip = prod.get("chip_size_um")
        if chip:
            s = self.vm.um_per_px
            w, h = (chip[0] / s, chip[1] / s) if s > 0 else (0.0, 0.0)
        elif prod.get("bbox"):
            w, h = prod["bbox"].get("w", 0.0), prod["bbox"].get("h", 0.0)
        else:
            declared = VoidManager.normalize_types(self.config.get("void_types", {})).get(name, {})
            w, h = declared.get("def_w", 0.0), declared.get("def_h", 0.0)
        t = self.vm.types[name]
        t["def_w"], t["def_h"] = float(w), float(h)

    def step_bonding(self):
        info = {}
        fn = self.plugin_fn("fetch_bonding_map")
        result = None
        if isinstance(self.stack, MosaicTiffStack): # Saved with the export
            txt = os.path.join(self.stack.base_dir, "bonding_map.txt")
            if os.path.exists(txt):
                with open(txt, "r", encoding="utf-8") as f:
                    result = {"text": f.read()}
                rej = os.path.join(self.stack.base_dir, "bonding_map_reject.txt")
                if os.path.exists(rej):
                    with open(rej, "r", encoding="utf-8") as f:
                        result["reject_values"] = f.read().strip()
                info["from"] = "bonding_map.txt"
        if result is None and fn is not None:
            result = fn(build_context(self), self.results_settings) # Errors -> status error, with traceback
            info["from"] = "plugin"
        if result is None:
            if self.s["require_bonding_map"]:
                raise GateFailed("no bonding map (plugin fetch_bonding_map missing" +
                                 (f": {self.plugin_error})" if self.plugin_error else ")"))
            self.warn("no bonding map: every grid cell is inspected")
            info["from"] = None
        else:
            text, reject = results_api.normalize_bonding_map(result)
            if reject is None:
                ui = (self.config.get("ui_defaults") or {}).get("reject_values") or []
                reject = [v.strip() for v in ui.split(",")] if isinstance(ui, str) else [str(v) for v in ui]
            bmap = BondingMap()
            bmap.parse_text(text, reject)
            if bmap.rows <= 0 or bmap.cols <= 0:
                raise GateFailed("bonding map could not be parsed")
            declared = (self.product or {}).get("grid", {})
            if ("rows" in declared and declared["rows"] != bmap.rows) or ("cols" in declared and declared["cols"] != bmap.cols):
                self.warn(f"bonding map {bmap.rows}x{bmap.cols} differs from the product grid "
                          f"{declared.get('rows')}x{declared.get('cols')} (map used)")
            self.bmap = bmap
            self.grid.rows, self.grid.cols = bmap.rows, bmap.cols # As import_map_data
            info.update(size=[bmap.rows, bmap.cols], reject_values=list(bmap.reject_values))
        self.cells = self.bonded_cells()
        info["bonded_cells"] = len(self.cells)
        if not self.cells:
            raise GateFailed("no bonded cell in the bonding map")
        return info

    def _layout(self, tid, size, params, zs):
        res = detect_layout(self.stack, zs, self.abs_layer(0), self.grid, self.ct, self.vm, tid, size,
                            self.cells, params, self._progress1)
        self.check_cancel()
        if res is None:
            raise GateFailed("bbox: not enough chips to align (at least 3 bonded cells)")
        self.vm.resnap(self.grid, self.ct)
        return res

    def step_bbox(self):
        tid = self.bbox_type()
        if tid is None:
            raise GateFailed("no void type named 'bbox' declared in config.json")
        t = self.vm.types[tid]
        size = (t["def_w"], t["def_h"]) if t.get("def_w", 0) > 0 and t.get("def_h", 0) > 0 else None
        params = dict(self.config.get("auto_bbox") or {})
        params.update(self.s.get("auto_bbox") or {})
        zs = list(range(len(self.stack)))
        qc = self.s["qc"]
        declared = {k: getattr(self.grid, k) for k in GRID_KEYS}
        res = self._layout(tid, size, params, zs)
        info = {"first_pass": _layout_info(res, len(self.cells)), "grid_declared": declared}
        if self.s.get("fit_grid", True):
            f = fit_grid(self.vm, self.grid, self.ct, self.stack.width, self.stack.height)
            if f is None:
                raise GateFailed("Fit Grid: at least 4 bboxes over 2 rows and 2 columns are needed")
            off = self._bbox_offset()
            info["fit"] = {k: f[k] for k in ("start_x", "start_y", "pitch_x", "pitch_y", "angle", "rms", "n")}
            self.summary["qc"]["fit_rms_px"] = f["rms"]
            self.summary["qc"]["angle_change"] = f["angle"] - declared["angle"]
            if qc.get("max_fit_rms_px") is not None and f["rms"] > qc["max_fit_rms_px"]:
                raise GateFailed(f"Fit Grid residual {f['rms']:.2f} px > {qc['max_fit_rms_px']} px")
            if qc.get("max_angle_change") is not None and abs(f["angle"] - declared["angle"]) > qc["max_angle_change"]:
                raise GateFailed(f"fitted angle {f['angle']:.3f} deg is {abs(f['angle'] - declared['angle']):.2f} deg "
                                 "off the declared one")
            # As MainWindow.fit_grid_to_bboxes: the bboxes keep their median offset in the cells
            self.grid.start_x, self.grid.start_y = f["start_x"] - off[0], f["start_y"] - off[1]
            self.grid.pitch_x, self.grid.pitch_y, self.grid.angle = f["pitch_x"], f["pitch_y"], f["angle"]
            self.update_transform()
            res = self._layout(tid, size, params, zs)
            info["second_pass"] = _layout_info(res, len(self.cells))
        info["grid"] = {k: getattr(self.grid, k) for k in GRID_KEYS}
        est = len(res["estimated"]) / max(1, len(self.cells))
        self.summary["qc"].update(estimated_frac=est, layout_ncc=res["ncc"], bbox_place=res["absolute"])
        n_bbox = sum(1 for vs in self.vm.voids.values() for v in vs if self.vm.is_bbox(v))
        info["bboxes"] = n_bbox
        if qc.get("max_estimated_frac") is not None and est > qc["max_estimated_frac"]:
            raise GateFailed(f"grid / bonding map mismatch: {len(res['estimated'])} of {len(self.cells)} bonded "
                             "cells do not match the chip layout (estimated)")
        if qc.get("require_package_box") and res["absolute"] == "mean place":
            raise GateFailed("the chip package edge is not found in the bboxes: the grid origin may be off by a "
                             "fraction of a cell (bboxes on the streets); check the product grid")
        return info

    def _bbox_offset(self):
        to_deskew = self.ct.raw_to_deskew
        g, offs = self.grid, []
        for vs in self.vm.voids.values():
            for v in vs:
                if self.vm.is_bbox(v) and v.get("auto") != "estimated":
                    gx, gy = to_deskew(v["globalCX"], v["globalCY"])
                    k = g.cell_at(gx, gy)
                    offs.append((gx - g.start_x - (k[1] + 0.5) * g.pitch_x, gy - g.start_y - (k[0] + 0.5) * g.pitch_y))
        return np.median(offs, axis=0) if offs else (0.0, 0.0)

    def void_type(self):
        """Type of the automatic markings: auto_run "void_type", ui_defaults "void_type", else the first non-bbox type."""
        want = self.s.get("void_type") or (self.config.get("ui_defaults") or {}).get("void_type")
        if want:
            name = self.vm.resolve_type(want)
            if name in self.vm.types and not self.vm.is_bbox({"type": name}):
                return name
            self.warn(f"void type {want!r} not declared; using the first type")
        return next((n for n in self.vm.types if not self.vm.is_bbox({"type": n})), None)

    def detect_params(self):
        """As MainWindow._detect_setup, with the inputs taken from the settings."""
        params = dict(self.config.get("void_detect", {}))
        params.update(self.s.get("void_detect") or {})
        rules = (self.product or {}).get("type_rules")
        if rules is not None:
            params["type_rules"] = rules
        method = self.s.get("method") or params.get("method") or "layer_change"
        params["method"] = method
        if method == "segmentation":
            from . import void_seg
            model = self.s.get("seg_model") or "models/void_unet.onnx"
            params["seg_model"] = void_seg.Model(model)
            params["method"] = "die_to_die"
            self.summary["models"]["segmentation"] = model
        if self.s.get("threshold") is not None:
            key = {"patchcore": "pc_threshold", "layer_change": "lc_threshold"}.get(params["method"], "threshold")
            params[key] = float(self.s["threshold"])
        for key in ("layer_rule", "skip_layers"):
            if self.s.get(key) is not None:
                params[key] = self.s[key]
            elif (self.config.get("ui_defaults") or {}).get(key) is not None:
                params[key] = self.config["ui_defaults"][key]
        if isinstance(params.get("skip_layers"), str):
            params["skip_layers"] = [int(x) for x in re.split(r"[,\s]+", params["skip_layers"].strip()) if x]
        if self.s.get("use_learned", True) and params["method"] in ("die_to_die", "layer_change"):
            product = (self.product_name, self.product or {}) if self.product_name else None
            path = void_learn.model_path(params, params["method"], product)
            if os.path.exists(path):
                params["learned_model"] = void_learn.Model.load(
                    path, void_layer_change.FEATURES if params["method"] == "layer_change" else None)
                self.summary["models"]["learned_filter"] = path
            else:
                self.summary["models"]["learned_filter"] = None
        if params["method"] == "patchcore":
            self.summary["models"]["patchcore"] = params.get("pc_model")
        return params

    @staticmethod
    def threshold(params):
        """Threshold the method runs with (given, else the method's default)."""
        from . import patchcore
        key = {"patchcore": "pc_threshold", "layer_change": "lc_threshold"}.get(params["method"], "threshold")
        defaults = {"pc_threshold": patchcore.DEFAULTS.get("pc_threshold"),
                    "lc_threshold": void_layer_change.DEFAULTS["lc_threshold"], "threshold": void_detect.DEFAULTS["threshold"]}
        return params.get(key, defaults[key])

    def step_voids(self):
        type_name = self.void_type()
        if type_name is None:
            raise GateFailed("no void type to mark with (only the bbox type is declared)")
        params = self.detect_params()
        refs = self.vm.reference_bboxes(self.grid, self.ct)
        res = void_detect.detect_voids(self.stack, self.grid, self.ct, refs, self.cells, self.layers(), self.vm,
                                       type_name, params, self._progress1, apply=True)
        self.stack.release_cache()
        self.check_cancel()
        if res is None:
            raise GateFailed("void detection stopped")
        info = {"method": params["method"], "type": type_name, "added": res["added"],
                "per_layer": {str(k): v for k, v in res["per_layer"].items()},
                "per_type": dict(res.get("types", {})),
                "threshold": self.threshold(params),
                "skip_layers": params.get("skip_layers"), "layer_rule": params.get("layer_rule")}
        per_chip = res["added"] / max(1, len(self.cells))
        self.summary["qc"]["voids_per_chip"] = per_chip
        lim = self.s["qc"].get("max_voids_per_chip")
        if lim is not None and per_chip > lim:
            raise GateFailed(f"{res['added']} voids ({per_chip:.1f} per chip > {lim}): check the wafer / grid")
        return info

    # ---------------------------------------------------------------- results
    def output_root(self):
        return self.s.get("output_root") or DEFAULTS["output_root"]

    def output_dir(self):
        pattern = self.s.get("output_dir") or DEFAULTS["output_dir"]
        path = pattern.format(out=self.output_root(), product=_safe(self.product_name or "no_product"),
                              wafer_id=_safe(self.wafer_id or self.source_name()), lot=_safe(self.lot or "no_lot"),
                              source_name=_safe(self.source_name()), date=datetime.date.today().isoformat())
        os.makedirs(path, exist_ok=True)
        return path

    def counts(self):
        per_layer, per_type = {}, {}
        n_bbox = 0
        for layer, vs in self.vm.voids.items() if self.vm is not None else ():
            for v in vs:
                if self.vm.is_bbox(v):
                    n_bbox += 1
                    continue
                per_layer[str(layer)] = per_layer.get(str(layer), 0) + 1
                per_type[v.get("type", "")] = per_type.get(v.get("type", ""), 0) + 1
        per_layer = dict(sorted(per_layer.items(), key=lambda kv: int(kv[0])))
        return {"bboxes": n_bbox, "voids": sum(per_layer.values()), "per_layer": per_layer, "per_type": per_type,
                "bonded_cells": len(self.cells)}

    def save(self):
        """voids.json + meta.json (when markings exist) and summary.json; returns the folder."""
        self.out_dir = self.output_dir()
        self.summary["output_dir"] = os.path.abspath(self.out_dir)
        self.summary["counts"] = self.counts()
        if self.vm is not None and self.stack is not None:
            if self.ct is None:
                self.update_transform()
            extra = {"app_version": APP_VERSION, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "source_file": self.path, "wafer_id": self.wafer_id, "lot": self.lot,
                     "product": self.product_name, "layers": self.layers(),
                     "auto_run": {"status": self.summary["status"], "reason": self.summary["reason"]},
                     "review": "pending"}
            self.vm.save_to_file(os.path.join(self.out_dir, "voids.json"), self.grid, self.bmap, self.ct, extra)
            if self.bmap is not None:
                with open(os.path.join(self.out_dir, "bonding_map.txt"), "w", encoding="utf-8") as f:
                    f.write(getattr(self.bmap, "source_text", ""))
        self.write_summary()
        return self.out_dir

    def write_summary(self):
        with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False, default=_jsonable)

    def send(self):
        if not self.s.get("send_results"):
            return {"skipped": True}
        if self.plugin_fn("send") is None or self.vm is None:
            raise RuntimeError("send_results: plugin not loaded")
        ctx = build_context(self)
        ok, msg = self.plugin.send(self.plugin.build_payload(ctx), self.results_settings)
        if not ok:
            self.warn(f"results not sent: {msg}")
        return {"ok": bool(ok), "message": str(msg)}

    def review_text(self):
        """(title, body) of the review request."""
        s, c = self.summary, self.summary.get("counts") or {}
        who = f"{self.product_name or '-'} {self.wafer_id or self.source_name()}"
        if s["status"] == "ok":
            title = f"[TIF Viewer] {who}: {c.get('voids', 0)} voids, review"
        else:
            title = f"[TIF Viewer] {who} {s['status'].replace('_', ' ')}: {s['reason']}"
        fmt = lambda d: ", ".join(f"{k}: {v}" for k, v in (d or {}).items()) or "-"
        qc = ", ".join(f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}" for k, v in s["qc"].items()) or "-"
        lines = [f"Product: {self.product_name or '-'}", f"Lot: {self.lot or '-'}", f"Wafer ID: {self.wafer_id or '-'}",
                 f"File: {self.path}", f"Status: {s['status']}" + (f" ({s['reason']})" if s["reason"] else ""),
                 f"Voids: {c.get('voids', 0)}  (per layer: {fmt(c.get('per_layer'))}; per type: {fmt(c.get('per_type'))})",
                 f"BBoxes: {c.get('bboxes', 0)} of {c.get('bonded_cells', 0)} bonded cells",
                 f"QC: {qc}", f"Results: {s.get('output_dir', '-')}"]
        if s["warnings"]:
            lines.append("Warnings: " + "; ".join(s["warnings"]))
        return title, "\n".join(lines)

    def review_queue_path(self):
        return os.path.join(self.output_root(), "review_queue.jsonl")

    def notify(self):
        title, body = self.review_text()
        recipients = list((self.results_settings.get("notify") or {}).get("recipients") or [])
        self.summary["review_request"] = {"title": title, "recipients": recipients}
        print(f"[auto_run] {self.summary['status'].upper()}: {title}", flush=True)
        fn = self.plugin_fn("notify_review")
        if fn is None:
            append_review_queue(self.review_queue_path(), title, body, recipients, self.summary)
            return {"plugin": False, "ok": True, "message": "written to review_queue.jsonl"}
        ctx = dict(build_context(self), summary=self.summary, review_queue=self.review_queue_path())
        try:
            ok, msg = fn(title, body, recipients, ctx, self.results_settings)
        except Exception as e:
            print(traceback.format_exc())
            ok, msg = False, f"{type(e).__name__}: {e}"
        if not ok: # Never lost: kept in review_queue.jsonl
            self.warn(f"review request not sent ({msg}); written to review_queue.jsonl")
            append_review_queue(self.review_queue_path(), title, body, recipients, self.summary)
        return {"plugin": True, "ok": bool(ok), "message": str(msg)}

    # ---------------------------------------------------------------- run
    def run_step(self, name, fn):
        t0 = time.monotonic()
        entry = {"name": name, "ok": False}
        self.summary["steps"].append(entry)
        print(f"[auto_run] {name}...", flush=True)
        if self.on_step is not None:
            self.on_step(self, name, "start")
        try:
            entry["info"] = fn()
            entry["ok"] = True
        finally:
            entry["seconds"] = round(time.monotonic() - t0, 2)
        if self.on_step is not None:
            self.on_step(self, name, "done")

    def run(self):
        log = io.StringIO()
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
        t0 = time.monotonic()
        try:
            try:
                for name in STEPS:
                    self.check_cancel()
                    self.run_step(name, getattr(self, "step_" + name))
                self.summary["status"] = "ok"
            except Cancelled:
                self.summary.update(status="cancelled", reason="stopped by the user")
            except GateFailed as e:
                self.summary.update(status="needs_attention", reason=str(e))
                self.summary["steps"][-1]["error"] = str(e)
                print(f"[auto_run] needs attention: {e}")
            except Exception as e:
                tb = traceback.format_exc()
                self.summary.update(status="error", reason=f"{type(e).__name__}: {e}", traceback=tb)
                if self.summary["steps"]:
                    self.summary["steps"][-1]["error"] = str(e)
                print(tb)
            self.summary["finished_at"] = _now()
            self.summary["seconds"] = round(time.monotonic() - t0, 2)
            if self.summary["status"] == "cancelled": # Processed again later: nothing saved or sent
                return self.summary
            for name, fn in (("save", self.save), ("send", self.send), ("notify", self.notify)):
                try:
                    self.run_step(name, fn)
                except Exception as e:
                    print(traceback.format_exc())
                    self.warn(f"{name} failed: {e}")
                    if name == "save" and self.out_dir is None: # Somewhere to report it
                        self.out_dir = os.path.join(self.output_root(), "_failed", _safe(self.source_name()))
                        os.makedirs(self.out_dir, exist_ok=True)
                        self.summary["output_dir"] = os.path.abspath(self.out_dir)
        finally:
            sys.stdout, sys.stderr = out, err
            if self.stack is not None:
                try:
                    self.stack.close()
                except Exception:
                    pass
        if self.out_dir:
            self.write_summary() # With the timings of save / send / notify
            with open(os.path.join(self.out_dir, "pipeline.log"), "w", encoding="utf-8") as f:
                f.write(log.getvalue())
        return self.summary


def append_review_queue(path, title, body, recipients, summary=None):
    """Default review request: printed and appended to review_queue.jsonl (one JSON object per line)."""
    print(f"[review] {title}\n{body}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    entry = {"title": title, "body": body, "recipients": list(recipients or []), "queued_at": _now()}
    for k in ("status", "reason", "wafer_id", "lot", "product", "source_file", "output_dir", "review"):
        if summary and k in summary:
            entry[k] = summary[k]
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=_jsonable) + "\n")


def _layout_info(res, n_cells):
    return {"refined": res["refined"], "estimated": len(res["estimated"]), "hand": res["hand"],
            "ncc": res["ncc"], "moved_px": res["moved_px"], "absolute": res["absolute"], "cells": n_cells}


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_context(run):
    """Headless results_api.build_context: the same keys, from the pipeline state."""
    vm, g = run.vm, run.grid
    bmap = run.bmap
    bonding = None
    if bmap is not None:
        bonding = {"text": getattr(bmap, "source_text", ""),
                   "reject_values": list(getattr(bmap, "reject_values", [])),
                   "rows": [list(r) for r in bmap.data_map]}
    source = run.path
    if source.endswith("patches.json"):
        source = os.path.dirname(source)
    layers = run.layers()
    return {
        "app_version": APP_VERSION,
        "timestamp": _now(),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "product": run.product_name,
        "source_file": source,
        "source_name": run.source_name(),
        "wafer_id": run.wafer_id,
        "lot": run.lot,
        "layers": layers,
        "current_layer": layers[0] if layers else 0,
        "grid": {k: getattr(g, k) for k in ("start_x", "start_y", "pitch_x", "pitch_y", "rows", "cols", "angle")},
        "um_per_px": (vm.um_per_px if vm is not None else 0.0) or 0.0,
        "bonding_map": bonding,
        "voids_json": vm.to_data(g, bmap, run.ct) if vm is not None else {"voids": []},
        "records": vm.void_records(g, bmap, run.ct) if vm is not None else [],
        "types": {k: dict(v) for k, v in vm.types.items()} if vm is not None else {},
        "last_export_dir": run.out_dir,
        "settings": run.results_settings,
        "auto_run": True, # Called by the automatic run
    }


def run_wafer(path, settings=None):
    """Runs every step on one measured file / project folder; returns the summary (also in summary.json)."""
    return WaferRun(path, settings).run()


# -------------------------------------------------------------------- watching
def watch_globs(settings):
    """auto_run "watch": one glob pattern or a list ("**" recursive)."""
    w = settings.get("watch") or []
    return [w] if isinstance(w, str) else [str(p) for p in w]


class State:
    """Finished files: path -> {size, mtime, status, output_dir, finished_at}. Written atomically."""

    def __init__(self, path):
        self.path = path
        self.data = read_json(path, {}) or {}

    def done(self, path, size, mtime):
        e = self.data.get(path)
        return e is not None and e.get("size") == size and e.get("mtime") == mtime

    def record(self, path, size, mtime, summary):
        self.data[path] = {"size": size, "mtime": mtime, "status": summary.get("status"),
                           "reason": summary.get("reason", ""), "output_dir": summary.get("output_dir"),
                           "finished_at": summary.get("finished_at")}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


class FileWatcher:
    """
    Polls glob patterns; a matching file is ready when its size and mtime did not change for
    stable_seconds (copy finished) and the state has no record of it with that size / mtime.
    poll() returns each ready file once (until forget(path)). Safe to call from any one thread.
    """

    def __init__(self, globs, stable_seconds, state):
        self.globs = list(globs)
        self.stable = float(stable_seconds)
        self.state = state
        self.seen = {} # path -> (size, mtime, unchanged since)
        self.handed = {} # path -> (size, mtime) returned by poll, not recorded yet

    @staticmethod
    def for_settings(settings):
        s = merged_settings(settings)
        state = State(s.get("state_file") or os.path.join(s["output_root"], "auto_run_state.json"))
        return FileWatcher(watch_globs(s), s["stable_seconds"], state)

    def files(self):
        out = set()
        for pattern in self.globs:
            try:
                for p in glob.glob(os.path.expanduser(pattern), recursive=True):
                    if os.path.isfile(p):
                        out.add(os.path.abspath(p))
            except OSError as e: # Network share away for a moment
                print(f"[auto_run] {pattern}: {e}")
        return sorted(out)

    def poll(self):
        ready, now = [], time.time()
        for p in self.files():
            try:
                st = os.stat(p)
            except OSError:
                continue # Removed / renamed meanwhile
            key = (st.st_size, st.st_mtime)
            if self.state.done(p, *key) or self.handed.get(p) == key:
                continue
            prev = self.seen.get(p)
            if prev is None or prev[:2] != key:
                self.seen[p] = key + (now,)
                continue
            if now - prev[2] < self.stable or st.st_size == 0:
                continue
            self.handed[p] = key
            self.seen.pop(p, None)
            ready.append(p)
        return ready

    def finished(self, path, summary):
        """Records a processed file (not a cancelled one: it is offered again)."""
        key = self.handed.pop(path, None)
        if key is not None and summary.get("status") != "cancelled":
            self.state.record(path, key[0], key[1], summary)

    def forget(self, path):
        self.handed.pop(path, None)
