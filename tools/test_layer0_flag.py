"""
GUI test of the "visible on layer 0" review flag (Layer change, v["layer0"]) on a synthetic wafer.

    SEED=204 python tools/synth/make_wafer.py /tmp/data ml_204
    xvfb-run -a -s "-screen 0 1920x1080x24" python tools/test_layer0_flag.py /tmp/data/ml_204.tif

Detect Voids (Layer change, Skip layers 0) on layers 0..7, then checks: some voids are flagged and
drawn with the magenta "0" badge (screen grab), Next Layer-0 walks through every flagged void,
B over a void clears / sets the flag and Ctrl+Z undoes it, editing (E) keeps the flag, the flag and
the review state ("auto") survive Save JSON -> Load, files without the columns still load, CSV
carries Layer0 / AutoState.
"""
import csv
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PySide6.QtWidgets import QApplication, QFileDialog # noqa: E402
from PySide6.QtGui import QSurfaceFormat, QCursor, QImage # noqa: E402
from PySide6.QtCore import Qt, QPoint # noqa: E402
from PySide6.QtTest import QTest # noqa: E402

import tif_view_GL # noqa: E402
from sat_widgets.exporter import ExportManager # noqa: E402
from tools.synth.save_truth import ORG, P, B, ANG, R # noqa: E402

TYPE = "Signal"
fails = []


def check(ok, text):
    print(("PASS " if ok else "FAIL ") + text)
    if not ok:
        fails.append(text)


def wait(app, cond, timeout=900):
    t0 = time.monotonic()
    while not cond():
        app.processEvents()
        time.sleep(0.005)
        if time.monotonic() - t0 > timeout:
            raise RuntimeError("timeout")


def settle(app, n=10):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def magenta(app, w, name=None):
    """Magenta (badge) pixels in a screen grab of the window (the QPainter overlay is not in QWidget.grab)."""
    settle(app)
    img = app.primaryScreen().grabWindow(w.winId()).toImage()
    if name:
        img.save(os.path.join(os.environ.get("SHOT_DIR", "/tmp"), f"layer0_{name}.png"))
    img = img.convertToFormat(QImage.Format_RGB32)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    b, g, r = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return int(((r > g + 70) & (b > g + 70) & (abs(r - b) < 60)).sum()) # Also the translucent ghost badge


def flagged(vm):
    return [(layer, v) for layer, vs in vm.voids.items() for v in vs if v.get("layer0")]


def key_of(layer, v):
    return (layer, round(v["globalCX"], 2), round(v["globalCY"], 2))


def hover(app, w, v):
    """Puts the mouse on the void's center and lets the view pick it as the hovered void."""
    glw = w.glw
    sx, sy = glw.deskew_to_screen(*glw.raw_to_deskew(v["globalCX"], v["globalCY"]))
    QCursor.setPos(glw.mapToGlobal(QPoint(int(sx), int(sy))))
    glw.refresh_hover()
    if glw.hover_void is not v: # No pointer under some X servers: pick it directly
        glw.hover_void = v
    settle(app)
    return QPoint(int(sx), int(sy))


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
    vm.types[bbox]["def_w"] = vm.types[bbox]["def_h"] = float(B)
    for r in range(R):
        for c in range(R):
            vm.add_void(w.glw.bbox_layer, *ct.deskew_to_raw(ORG + c * P + P / 2, ORG + r * P + P / 2), B / 2, B / 2, bbox)
    vm.reset_history()
    w.glw.active_type = TYPE
    w.le_skip_layers.setText("0")
    w.cb_detect_method.setCurrentIndex(w.cb_detect_method.findText("Layer change"))
    w.glw.fit_to_cell(3, 3)
    app.processEvents()

    check(magenta(app, w) == 0, "no magenta before detection")
    w.detect_voids()
    wait(app, lambda: w._detect_job is None)
    settle(app)
    fl = flagged(vm)
    n_all = sum(1 for vs in vm.voids.values() for v in vs if v.get("auto") == "detected")
    print(f"  {n_all} detected, {len(fl)} flagged as visible on layer 0: {sorted(k[0] for k in fl)}")
    check(len(fl) > 0, f"Layer change flagged {len(fl)} of {n_all} detected voids")
    check(all(v.get("auto") == "detected" for _, v in fl), "flagged voids are detected (dashed) voids")
    msg = w.statusBar().currentMessage()
    check("already visible on layer 0" in msg and str(len(fl)) in msg, f"status mentions them: '{msg[-110:]}'")

    # Next Layer-0 walks through every flagged (layer, row, col)
    expected = {(layer,) + tuple(w.glw.raw_to_cell(v["globalCX"], v["globalCY"])) for layer, v in fl}
    seen = set()
    for _ in range(len(expected) + 2):
        w.next_layer0_void()
        settle(app, 2)
        seen.add((w.glw.current_layer,) + tuple(w.glw.center_cell()))
    check(seen == expected, f"Next Layer-0 visited all {len(expected)} flagged cells ({len(seen)} seen)")
    check(w.statusBar().currentMessage().startswith("Layer-0 void "), f"status '{w.statusBar().currentMessage()}'")

    # Drawn distinctly: badge on the current layer and on ghosts
    layer, v = fl[0]
    z = next(z for z in range(len(w.full_data)) if w.abs_layer(z) == layer)
    w.spin_layer.setValue(z)
    w.glw.fit_to_cell(*reversed(w.glw.raw_to_cell(v["globalCX"], v["globalCY"])))
    on_layer = magenta(app, w, "on_layer")
    check(on_layer > 30, f"badge drawn on its layer ({on_layer} magenta px)")
    other = next(zz for zz in range(len(w.full_data)) if w.abs_layer(zz) != layer)
    w.spin_layer.setValue(other)
    ghost = magenta(app, w, "ghost")
    check(ghost > 10, f"badge drawn on the ghost (layer {w.abs_layer(other)}, {ghost} magenta px)")
    w.spin_layer.setValue(z)
    settle(app)

    # B clears the flag of the hovered void, Ctrl+Z restores it, B again sets it
    others_here = [u for l2, u in fl if u is not v and w.glw.raw_to_cell(u["globalCX"], u["globalCY"])
                   == w.glw.raw_to_cell(v["globalCX"], v["globalCY"])]
    undo0 = len(vm._undo)
    hover(app, w, v)
    QTest.keyClick(w.glw, Qt.Key_B)
    settle(app)
    check(not v.get("layer0") and len(vm._undo) == undo0 + 1, "B cleared the flag (one undo step)")
    after = magenta(app, w, "cleared")
    check(after < on_layer or others_here, f"badge gone ({on_layer} -> {after} magenta px)")
    w.action_undo.trigger()
    settle(app)
    v = next(u for l2, vs in vm.voids.items() for u in vs if l2 == layer and key_of(l2, u) == key_of(layer, v))
    check(bool(v.get("layer0")), "Ctrl+Z restored the flag")
    hover(app, w, v)
    QTest.keyClick(w.glw, Qt.Key_B)
    QTest.keyClick(w.glw, Qt.Key_B)
    settle(app)
    check(bool(v.get("layer0")), "B twice: flag set again")
    # B on a void without the flag sets it
    plain = next(((l2, u) for l2, vs in vm.voids.items() for u in vs if not u.get("layer0") and not vm.is_bbox(u)), None)
    if plain:
        w.spin_layer.setValue(next(zz for zz in range(len(w.full_data)) if w.abs_layer(zz) == plain[0]))
        w.glw.fit_to_cell(*reversed(w.glw.raw_to_cell(plain[1]["globalCX"], plain[1]["globalCY"])))
        hover(app, w, plain[1])
        QTest.keyClick(w.glw, Qt.Key_B)
        check(plain[1].get("layer0") is True, "B on an unflagged void sets the flag")
        w.action_undo.trigger()
        check(not any(key_of(*plain) == key_of(l2, u) and u.get("layer0") for l2, u in flagged(vm)), "Ctrl+Z cleared it")
        w.spin_layer.setValue(z)
        w.glw.fit_to_cell(*reversed(w.glw.raw_to_cell(v["globalCX"], v["globalCY"])))
        settle(app)

    # Editing (E: press / release on the void) clears "auto" but keeps the flag
    v = next(u for l2, vs in vm.voids.items() for u in vs if l2 == layer and key_of(l2, u) == key_of(layer, v))
    pos = hover(app, w, v)
    w.glw.set_void_tool("EDIT")
    QTest.mousePress(w.glw, Qt.LeftButton, Qt.NoModifier, pos)
    QTest.mouseMove(w.glw, pos + QPoint(2, 0))
    QTest.mouseRelease(w.glw, Qt.LeftButton, Qt.NoModifier, pos + QPoint(2, 0))
    w.glw.set_void_tool(None)
    settle(app)
    check(v.get("auto") is None and v.get("layer0") is True, f"edited: auto={v.get('auto')}, layer0={v.get('layer0')}")

    # Save -> load keeps the flag and the review state
    state = sorted((key_of(l2, u), bool(u.get("layer0")), u.get("auto")) for l2, vs in vm.voids.items() for u in vs)
    tmp = tempfile.mkdtemp()
    jpath = os.path.join(tmp, "voids.json")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (jpath, ""))
    w.save_voids()
    data = json.load(open(jpath))
    meta = json.load(open(os.path.join(tmp, "meta.json")))
    n_flag = sum(1 for d in data if d.get("layer0") is True)
    check(n_flag == len(flagged(vm)) and all("layer0" in d and "autoState" in d for d in data),
          f"voids.json: {n_flag} with layer0 true, every void has layer0 / autoState")
    check(meta.get("layer0_voids") == n_flag and meta.get("unreviewed_voids") == n_all - 1,
          f"meta.json layer0_voids {meta.get('layer0_voids')}, unreviewed_voids {meta.get('unreviewed_voids')}")
    vm.replace_all()
    check(not flagged(vm), "cleared")
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (jpath, ""))
    w.load_voids()
    state2 = sorted((key_of(l2, u), bool(u.get("layer0")), u.get("auto")) for l2, vs in vm.voids.items() for u in vs)
    check(state2 == state, f"load restored {len(state2)} voids with the same layer0 / auto state")
    settle(app)
    check(magenta(app, w) > 30, "badge drawn after load")

    # Older file without the columns: loads, no flags, not "detected"
    old = [{k: val for k, val in d.items() if k not in ("layer0", "autoState")} for d in data]
    opath = os.path.join(tmp, "old", "voids.json")
    os.makedirs(os.path.dirname(opath))
    json.dump(old, open(opath, "w"))
    json.dump(meta, open(os.path.join(tmp, "old", "meta.json"), "w"))
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (opath, ""))
    w.load_voids()
    n_old = sum(len(vs) for vs in vm.voids.values())
    check(n_old == len(data) and not flagged(vm) and not any(u.get("auto") for vs in vm.voids.values() for u in vs),
          f"file without the columns: {n_old} voids, no flags")

    # CSV
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (jpath, ""))
    w.load_voids()
    ExportManager(w)._export_csv(tmp)
    rows = list(csv.DictReader(open(os.path.join(tmp, "voids.csv"))))
    check(sum(1 for r in rows if r.get("Layer0") == "1") == n_flag and {r.get("Layer0") for r in rows} <= {"0", "1"}
          and sum(1 for r in rows if r.get("AutoState") == "detected") == n_all - 1,
          f"voids.csv: Layer0 1 x {n_flag}, AutoState detected x {n_all - 1}")

    print("FAILED: " + "; ".join(fails) if fails else "ALL PASSED")
    w.close()
    os._exit(1 if fails else 0)


if __name__ == "__main__":
    main()
