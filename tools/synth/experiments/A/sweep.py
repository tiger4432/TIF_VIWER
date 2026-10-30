"""
Experiment A: eval_app.py for several thresholds at once (the U-Net output and the neighbour layers
are computed once and reused; Detect Voids itself runs unchanged per threshold).

    python tools/synth/experiments/A/sweep.py OUTDIR NAME MODEL.onnx 3 4 5 6 [--param key=value ...]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)
from sat_widgets.core_data import TiffStack, CoordinateTransform # noqa: E402
from sat_widgets.void_manager import VoidManager # noqa: E402
from sat_widgets import void_detect, void_seg # noqa: E402
from tools.synth.save_truth import grid, ORG, P, B, M, R # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("name"); ap.add_argument("model")
    ap.add_argument("thresholds", nargs="+", type=float)
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--variant", action="append", help="extra params per run, ';'-separated (runs every threshold x variant)")
    a = ap.parse_args()
    model = void_seg.Model(a.model)
    cache = {}
    predict = model.predict

    def cached(x):
        k = hashlib.md5(x.tobytes()).hexdigest()
        if k not in cache:
            cache[k] = predict(x)
        return cache[k]
    model.predict = cached
    lc = void_detect._layer_chips
    lc_cache = []

    def layer_chips(*args):
        if not lc_cache:
            lc_cache.append(lc(*args))
        return lc_cache[0]
    void_detect._layer_chips = layer_chips
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
    variants = [v.split(";") for v in (a.variant or [""])]
    for thr, var in [(t, v) for t in a.thresholds for v in variants]:
        p = dict(void_detect.DEFAULTS)
        p.update({"method": "die_to_die", "skip_layers": [], "layer_rule": "first", "seg_model": model, "threshold": thr})
        for kv in a.param + [x for x in var if x]:
            k, v = kv.split("=", 1)
            try:
                p[k] = json.loads(v)
            except ValueError:
                p[k] = v
        res = void_detect.detect_voids(st, g, ct, refs, cells, list(range(len(st))), vm, "Void", p, apply=False)
        dets = []
        for layer, gx, gy, rx, ry, score, vtype, cell, ext, _ in res["found"]:
            r, c = cell
            ox, oy = ORG + c * P + M, ORG + r * P + M
            dets.append({"layer": layer, "row": r, "col": c, "x0": ext[0] - ox, "y0": ext[1] - oy, "x1": ext[2] - ox, "y1": ext[3] - oy})
        path = os.path.join(tempfile.gettempdir(), f"dets_{a.name}_{os.getpid()}.json")
        json.dump(dets, open(path, "w"))
        out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "synth", "score_detections.py"), a.outdir, a.name, path,
                              "--mark", "first"], capture_output=True, text=True).stdout.strip()
        json.dump(dets, open(os.path.join(tempfile.gettempdir(), f"dets_{a.name}_{thr}_{'_'.join(var)}.json".replace("=", "-").replace(";", "_")), "w"))
        print(f"threshold {thr} {' '.join(a.param + var)}: {out}", flush=True)


if __name__ == "__main__":
    main()
