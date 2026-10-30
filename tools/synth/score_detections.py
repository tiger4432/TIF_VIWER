"""
Scores detections on a synthetic wafer against its truth, the same way for every method.

    python tools/synth/score_detections.py OUTDIR NAME detections.json [--mark first|last]

detections.json: a list of {"layer": int, "row": int, "col": int, "x0", "y0", "x1", "y1"} with the
box in chip pixels (origin = the chip's (bbox) top-left corner, 280 x 280, axis-aligned to the grid,
i.e. after undoing the 0.4 deg rotation).

A detection matches a void when they are in the same chip and their boxes meet (2 px tolerance):
- ok: same layer as the marking (--mark first: the layer where the void first appears)
- ly: the void's chip and place but another layer (found, wrong layer)
- fp: no void there; ms: void not found.
Prints one JSON line: {"voids", "detections", "ok", "ly", "ms", "fp", "recall", "precision", "f1"}
(recall = ok / voids, precision = ok / detections; also "recall_any_layer" = (ok + ly) / voids).
"""
import argparse
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("name"); ap.add_argument("detections")
    ap.add_argument("--mark", choices=["first", "last"], default="first")
    a = ap.parse_args()
    truth = json.load(open(os.path.join(a.outdir, a.name + "_truth.json")))
    dets = json.load(open(a.detections))
    T = []
    for k, (cx, cy, rx, ry, l0, l1, _) in truth.items():
        r, c = map(int, k.split(","))
        T.append({"cell": (r, c), "layer": l0 if a.mark == "first" else l1, "box": (cx - rx, cy - ry, cx + rx, cy + ry)})
    tol = 2.0
    meet = lambda p, q: p[0] - tol <= q[2] and q[0] - tol <= p[2] and p[1] - tol <= q[3] and q[1] - tol <= p[3]
    pairs = []
    for i, t in enumerate(T):
        for j, d in enumerate(dets):
            db = (d["x0"], d["y0"], d["x1"], d["y1"])
            if (d["row"], d["col"]) == t["cell"] and meet(t["box"], db):
                dist = math.hypot((t["box"][0] + t["box"][2] - db[0] - db[2]) / 2, (t["box"][1] + t["box"][3] - db[1] - db[3]) / 2)
                pairs.append((d["layer"] != t["layer"], dist, i, j))
    pairs.sort()
    tm, dm = {}, {}
    for other, _, i, j in pairs:
        if i not in tm and j not in dm:
            tm[i] = dm[j] = "ly" if other else "ok"
    ok = sum(1 for v in tm.values() if v == "ok")
    ly = sum(1 for v in tm.values() if v == "ly")
    fp = len(dets) - len(dm)
    rec = ok / max(len(T), 1)
    prec = ok / max(len(dets), 1)
    print(json.dumps({"voids": len(T), "detections": len(dets), "ok": ok, "ly": ly, "ms": len(T) - ok - ly, "fp": fp,
                      "recall": round(rec, 3), "precision": round(prec, 3),
                      "f1": round(2 * rec * prec / max(rec + prec, 1e-9), 3),
                      "recall_any_layer": round((ok + ly) / max(len(T), 1), 3)}))


if __name__ == "__main__":
    main()
