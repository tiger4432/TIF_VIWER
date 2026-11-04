"""
Saves the truth of a synthetic wafer as hand markings (voids.json + meta.json), the format the
viewer saves, so Seg Data / Add Folder and Evaluate can use it.

    python tools/synth/save_truth.py OUTDIR NAME RESULT_DIR [--mark first|last]

--mark first (default): each void is marked on the layer where it first appears.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sat_widgets.core_data import TiffStack, GridConfig, CoordinateTransform # noqa: E402
from sat_widgets.void_manager import VoidManager # noqa: E402

_B = int(os.environ.get("CHIP", 280)) # Chip (bbox) size in px; the layout pattern scales with it
ORG, P, B, ANG, R = 100, int(os.environ.get("PITCH", _B + 20)), _B, 0.4, int(os.environ.get("NCHIP", 14))
M = (P - B) / 2


def grid():
    g = GridConfig()
    g.start_x = g.start_y = ORG
    g.pitch_x = g.pitch_y = P
    g.rows = g.cols = R
    g.angle = ANG
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("name"); ap.add_argument("result_dir")
    ap.add_argument("--mark", choices=["first", "last"], default="first")
    a = ap.parse_args()
    tif = os.path.join(a.outdir, a.name + ".tif")
    st = TiffStack(tif)
    g = grid()
    ct = CoordinateTransform(g.angle, st.width, st.height)
    vm = VoidManager(config_path=os.path.join(a.result_dir, "_none_.json"))
    vm.types = {"bbox": {"name": "bbox", "color": (255, 255, 0), "shape": "rectangle"},
                "Void": {"name": "Void", "color": (255, 0, 0), "shape": "ellipse"}}
    for r in range(R):
        for c in range(R):
            vm.add_void(0, *ct.deskew_to_raw(ORG + c * P + P / 2, ORG + r * P + P / 2), B / 2, B / 2, "bbox")
    truth = json.load(open(os.path.join(a.outdir, a.name + "_truth.json")))
    for k, (cx, cy, rx, ry, l0, l1, _) in truth.items():
        r, c = map(int, k.split(","))
        vm.add_void(l0 if a.mark == "first" else l1, *ct.deskew_to_raw(ORG + c * P + M + cx, ORG + r * P + M + cy), rx, ry, "Void")
    os.makedirs(a.result_dir, exist_ok=True)
    vm.save_to_file(os.path.join(a.result_dir, "voids.json"), g, None, ct,
                    {"source_file": os.path.abspath(tif), "layers": list(range(len(st)))})
    st.close()


if __name__ == "__main__":
    main()
