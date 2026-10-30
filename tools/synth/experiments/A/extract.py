"""
Experiment A: training data for the viewer's U-Net from synthetic wafers, through the app pipeline
(voids.json + meta.json like save_truth.py -> seg_data.extract_saved -> void_seg.write_source).

    python tools/synth/experiments/A/extract.py DATA_DIR RESULTS_DIR SEG_DIR --label first|all ml_201 ml_202 ml_203

--label first: each void marked on the layer where it first appears (save_truth.py --mark first).
--label all:   marked on every layer it is visible on, with that layer's size (NAME_layers.json from
               make_wafer_layers.py); Detect Voids' layer rule then picks the first.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)
from sat_widgets import seg_data, void_detect, void_seg # noqa: E402
from sat_widgets.core_data import TiffStack, CoordinateTransform # noqa: E402
from sat_widgets.void_manager import VoidManager # noqa: E402
from tools.synth.save_truth import grid, ORG, P, B, M, R # noqa: E402


def save_marks(outdir, name, result_dir, label):
    tif = os.path.join(outdir, name + ".tif")
    st = TiffStack(tif)
    g = grid()
    ct = CoordinateTransform(g.angle, st.width, st.height)
    vm = VoidManager(config_path=os.path.join(result_dir, "_none_.json"))
    vm.types = {"bbox": {"name": "bbox", "color": (255, 255, 0), "shape": "rectangle"},
                "Void": {"name": "Void", "color": (255, 0, 0), "shape": "ellipse"}}
    for r in range(R):
        for c in range(R):
            vm.add_void(0, *ct.deskew_to_raw(ORG + c * P + P / 2, ORG + r * P + P / 2), B / 2, B / 2, "bbox")
    truth = json.load(open(os.path.join(outdir, name + "_truth.json")))
    layers = json.load(open(os.path.join(outdir, name + "_layers.json"))) if label == "all" else {}
    for k, (cx, cy, rx, ry, l0, l1, _) in truth.items():
        r, c = map(int, k.split(","))
        pos = ct.deskew_to_raw(ORG + c * P + M + cx, ORG + r * P + M + cy)
        for layer, sk in ([(l0, 1.0)] if label == "first" else [(lk, s) for lk, s, _ in layers[k]]):
            vm.add_void(int(layer), *pos, rx * sk, ry * sk, "Void")
    os.makedirs(result_dir, exist_ok=True)
    vm.save_to_file(os.path.join(result_dir, "voids.json"), g, None, ct,
                    {"source_file": os.path.abspath(tif), "layers": list(range(len(st)))})
    st.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("results"); ap.add_argument("seg_dir")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--label", choices=["first", "all"], default="first")
    a = ap.parse_args()
    params = dict(void_detect.DEFAULTS)
    params.update({"method": "die_to_die", "skip_layers": []})
    for name in a.names:
        rdir = os.path.join(a.results, f"{name}_{a.label}")
        save_marks(a.outdir, name, rdir, a.label)
        for vp in seg_data.find_saved(rdir):
            source, meta, refs, samples = seg_data.extract_saved(vp, params)
            folder = void_seg.write_source(source, meta, refs, samples, a.seg_dir)
            print(name, len(samples), "samples,", sum(int((s["y"] == 1).any()) for s in samples), "with voids ->", folder, flush=True)


if __name__ == "__main__":
    main()
