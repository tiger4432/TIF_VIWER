"""
GUI test of Detect Voids in the background (sat_widgets/detect_job.py) on a synthetic wafer.

    SEED=204 python tools/synth/make_wafer.py /tmp/data ml_204
    xvfb-run -a -s "-screen 0 1920x1080x24" python tools/test_background_detect.py /tmp/data/ml_204.tif

Checks, for Die-to-die and Layer change: the event loop keeps running during the run (view changes
work, editing is refused), Die-to-die marks appear before the run ends, every cell ends checked,
the voids equal a synchronous void_detect.detect_voids on the same data, Ctrl+Z removes the whole
run in one step; then Cancel keeps the voids found so far as one undo step.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PySide6.QtWidgets import QApplication, QFileDialog # noqa: E402
from PySide6.QtGui import QSurfaceFormat # noqa: E402
from PySide6.QtCore import QTimer, QThread # noqa: E402

import tif_view_GL # noqa: E402
from sat_widgets import void_detect # noqa: E402
from sat_widgets.core_data import TiffStack # noqa: E402
from sat_widgets.detect_job import read_only_copy # noqa: E402
from tools.synth.save_truth import ORG, P, B, ANG, R # noqa: E402

TYPE = "Signal"
fails = []


def check(ok, text):
    print(("PASS " if ok else "FAIL ") + text)
    if not ok:
        fails.append(text)


def auto_voids(vm):
    return sorted((layer, round(v["globalCX"], 3), round(v["globalCY"], 3), round(v["radiusX"], 3), round(v["radiusY"], 3),
                   v["type"], v.get("score")) for layer, vs in vm.voids.items() for v in vs if v.get("auto") == "detected")


def shot(app, w, name):
    """Screen grab (the view's QPainter overlay is not in QWidget.grab of a QOpenGLWidget)."""
    path = os.path.join(os.environ.get("SHOT_DIR", "/tmp"), f"bg_detect_{name}.png")
    app.primaryScreen().grabWindow(w.winId()).save(path)


def wait(app, cond, timeout=600):
    t0 = time.monotonic()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.monotonic() - t0 > timeout:
            raise RuntimeError("timeout")


def run(app, w, path, method_label, cancel_after=None):
    """One background run with a 20 ms timer watching it. Returns the samples and the elapsed time."""
    vm = w.void_manager
    w.cb_detect_method.setCurrentIndex(w.cb_detect_method.findText(method_label))
    # The synchronous reference on the same state (copy of the void data, own reader)
    type_name, cells, refs, params, layers = w._detect_setup("test")
    base = read_only_copy(vm)
    st = TiffStack(path, 0, 8)
    t0 = time.monotonic()
    ref = void_detect.detect_voids(st, w.glw.grid_cfg, w.glw.coord_transform, refs, cells, layers, base, type_name, params)
    t_sync = time.monotonic() - t0
    st.close()
    before = sum(len(vs) for vs in vm.voids.values())
    undo_before = len(vm._undo)

    shots = []
    samples = [] # (time, auto voids, cells done, sum of cell shares, job active)
    threads = set()
    orig_found = w._detect_found

    def found(job, batch, *a):
        threads.add(QThread.currentThread() is app.thread())
        orig_found(job, batch, *a)
    w._detect_found = found

    def tick():
        st_ = w.glw.detect_cells or {}
        if not shots and w._detect_job is not None and sum(st_.values()) > 0.4 * len(st_):
            shots.append(1)
            shot(app, w, method_label.split()[0].lower() + ("_cancel" if cancel_after else "") + "_running")
        samples.append((time.monotonic(), len(auto_voids(vm)), sum(1 for f in st_.values() if f >= 1.0),
                        sum(st_.values()), w._detect_job is not None))
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(20)
    t0 = time.monotonic()
    w.detect_voids()
    check(w._detect_job is not None, f"{method_label}: runs in the background (detect_voids returned at once)")
    # Viewing while it runs: zoom, pan, layer change; editing refused
    w.glw.zoom *= 1.5
    w.glw.pan_x += 50
    w.spin_layer.setValue(3)
    w.glw.update()
    app.processEvents()
    w.glw.fit_to_cell(3, 3)
    w.glw.zoom /= 3
    w.glw.set_void_tool("DRAW")
    check(w.glw.void_mode is False, f"{method_label}: draw tool refused while running")
    n0 = sum(len(vs) for vs in vm.voids.values())
    w.undo_void_change()
    w.clear_voids()
    check(sum(len(vs) for vs in vm.voids.values()) >= n0, f"{method_label}: undo / clear refused while running")
    w.detect_voids() # Second run refused
    if cancel_after is not None:
        wait(app, lambda: len(auto_voids(vm)) >= cancel_after or w._detect_job is None)
        w.cancel_detect()
    wait(app, lambda: w._detect_job is None)
    elapsed = time.monotonic() - t0
    for _ in range(10):
        app.processEvents()
    timer.stop()
    w._detect_found = orig_found
    check(threads <= {True}, f"{method_label}: voids placed on the GUI thread")
    active = [s for s in samples if s[4]]
    gaps = [b[0] - a[0] for a, b in zip(active, active[1:])]
    print(f"  {method_label}: background {elapsed:.1f}s (sync {t_sync:.1f}s), {len(active)} ticks while running, "
          f"max gap {max(gaps) if gaps else 0:.2f}s")
    check(len(active) > 10 and max(gaps) < 1.0, f"{method_label}: event loop kept running (max gap < 1 s)")
    return ref, base, before, undo_before, samples


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/data/ml_204.tif"
    app = QApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    w = tif_view_GL.MainWindow()
    w.resize(1600, 1000)
    w.show()
    app.processEvents()

    w.sb_start_layer.setValue(0)
    w.sb_end_layer.setValue(7)
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (path, ""))
    w.open_file()
    wait(app, lambda: w.full_data is not None and w.glw.tiled_image is not None)
    for sb, v in ((w.sb_grid_x, ORG), (w.sb_grid_y, ORG), (w.sb_pitch_x, P), (w.sb_pitch_y, P),
                  (w.sb_rows, R), (w.sb_cols, R), (w.sb_angle, ANG)):
        sb.setValue(v)
    w.chk_grid_show.setChecked(True)
    app.processEvents()
    vm, ct = w.void_manager, w.glw.coord_transform
    bbox = w._bbox_type_name()
    vm.types[bbox]["def_w"] = vm.types[bbox]["def_h"] = float(B) # Session only (undo resizes bboxes to it)
    for r in range(R):
        for c in range(R):
            vm.add_void(w.glw.bbox_layer, *ct.deskew_to_raw(ORG + c * P + P / 2, ORG + r * P + P / 2), B / 2, B / 2, bbox)
    vm.reset_history()
    w.glw.active_type = TYPE
    w.le_skip_layers.setText("")
    w.glw.fit_to_cell(3, 3)
    w.glw.zoom /= 3 # A few cells on screen
    app.processEvents()
    n_cells = R * R

    for label in ("Die-to-die", "Layer change"):
        ref, base, before, undo_before, samples = run(app, w, path, label)
        got = auto_voids(vm)
        exp = auto_voids(base)
        check(got == exp and len(got) == ref["added"],
              f"{label}: {len(got)} voids, same as the synchronous run ({ref['added']}, per layer {ref['per_layer']})")
        msg = w.statusBar().currentMessage()
        check(msg.startswith(f"Detect Voids: {ref['added']} new voids (L"), f"{label}: status '{msg[:60]}...'")
        final = samples[-1][1]
        early = [s for s in samples if s[4] and 0 < s[1] < final]
        partial = [s for s in samples if s[4] and 0 < s[3] < n_cells]
        if label == "Die-to-die":
            check(bool(early), f"{label}: marks appeared before the run ended ({len(early)} ticks with some marks)")
        else:
            check(not [s for s in samples if s[4] and s[1] > 0 and s[3] < n_cells], f"{label}: marks added at the end")
        check(bool(partial), f"{label}: cell circles progressed during the run ({len(partial)} ticks)")
        check(w.glw.detect_cells is not None and len(w.glw.detect_cells) == n_cells
              and all(f >= 1.0 for f in w.glw.detect_cells.values()), f"{label}: all {n_cells} cells checked")
        check(len(vm._undo) == undo_before + 1, f"{label}: one undo step")
        shot(app, w, label.split()[0].lower() + "_done")
        w.action_undo.trigger()
        check(not auto_voids(vm) and sum(len(vs) for vs in vm.voids.values()) == before,
              f"{label}: Ctrl+Z removed the whole run")

    # Cancel: the voids found so far stay, one undo step
    ref, base, before, undo_before, samples = run(app, w, path, "Die-to-die", cancel_after=1)
    got = auto_voids(vm)
    msg = w.statusBar().currentMessage()
    check(0 < len(got) < ref["added"] and "cancelled" in msg, f"Cancel: {len(got)} of {ref['added']} voids kept ('{msg}')")
    check(set(got) <= set(auto_voids(base)), "Cancel: kept voids are voids of the full run")
    check(not w.glw.edit_locked and w.btn_detect.isEnabled(), "Cancel: editing enabled again")
    check(len(vm._undo) == undo_before + 1, "Cancel: one undo step")
    w.action_undo.trigger()
    check(not auto_voids(vm), "Cancel: Ctrl+Z removed the kept voids")

    print("FAILED: " + "; ".join(fails) if fails else "ALL PASSED")
    w.close()
    os._exit(1 if fails else 0)


if __name__ == "__main__":
    main()
