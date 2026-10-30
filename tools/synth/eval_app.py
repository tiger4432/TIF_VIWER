"""
Runs the viewer's Detect Voids on a synthetic wafer and scores it with score_detections.py.

    python tools/synth/eval_app.py OUTDIR NAME [--mark first] [--param key=value ...] [--seg-model PATH]

Examples:  --param method=die_to_die --param threshold=6 --param layer_rule=first --param "skip_layers=[]"
           --seg-model models/void_unet.onnx --param threshold=5
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from sat_widgets.core_data import TiffStack, CoordinateTransform # noqa: E402
from sat_widgets.void_manager import VoidManager # noqa: E402
from sat_widgets import void_detect # noqa: E402
from tools.synth.save_truth import grid, ORG, P, B, M, R # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("name")
    ap.add_argument("--mark", choices=["first", "last"], default="first")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--seg-model")
    a = ap.parse_args()
    p = dict(void_detect.DEFAULTS)
    p.update({"method": "die_to_die", "skip_layers": [], "layer_rule": a.mark})
    for kv in a.param:
        k, v = kv.split("=", 1)
        try:
            p[k] = json.loads(v)
        except ValueError:
            p[k] = v
    if a.seg_model:
        from sat_widgets import void_seg
        p["seg_model"] = void_seg.Model(a.seg_model)
    st = TiffStack(os.path.join(a.outdir, a.name + ".tif"))
    g = grid()
    ct = CoordinateTransform(g.angle, st.width, st.height)
    vm = VoidManager(config_path=os.path.join(a.outdir, "_none_.json"))
    vm.types = {"bbox": {"name": "bbox", "color": (255, 255, 0), "shape": "rectangle"},
                "Void": {"name": "Void", "color": (255, 0, 0), "shape": "ellipse"}}
    for r in range(R):
        for c in range(R):
            vm.add_void(0, *ct.deskew_to_raw(ORG + c * P + P / 2, ORG + r * P + P / 2), B / 2, B / 2, "bbox")
    refs = vm.reference_bboxes(g, ct)
    cells = [(r, c) for r in range(R) for c in range(R)]
    res = void_detect.detect_voids(st, g, ct, refs, cells, list(range(len(st))), vm, "Void", p, apply=False)
    dets = []
    for layer, gx, gy, rx, ry, score, vtype, cell, ext, _ in res["found"]:
        r, c = cell
        ox, oy = ORG + c * P + M, ORG + r * P + M
        dets.append({"layer": layer, "row": r, "col": c, "x0": ext[0] - ox, "y0": ext[1] - oy, "x1": ext[2] - ox, "y1": ext[3] - oy})
    path = os.path.join(tempfile.gettempdir(), f"dets_{a.name}_{os.getpid()}.json")
    json.dump(dets, open(path, "w"))
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "synth", "score_detections.py"), a.outdir, a.name, path, "--mark", a.mark])


if __name__ == "__main__":
    main()
