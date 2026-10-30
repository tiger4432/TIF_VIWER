"""Tuning helpers on the training wafers (ml_201..203): same scoring rule as score_detections.py."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def score(truth, dets, mark="first"):
    T = []
    for k, (cx, cy, rx, ry, l0, l1, _) in truth.items():
        r, c = map(int, k.split(","))
        T.append({"cell": (r, c), "layer": l0 if mark == "first" else l1, "box": (cx - rx, cy - ry, cx + rx, cy + ry)})
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
    return dict(voids=len(T), det=len(dets), ok=ok, ly=ly, fp=len(dets) - len(dm))


def total(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    rec = s["ok"] / max(s["voids"], 1); prec = s["ok"] / max(s["det"], 1)
    s["f1"] = round(2 * rec * prec / max(rec + prec, 1e-9), 3)
    return s


def truth_of(outdir, name):
    return json.load(open(os.path.join(outdir, name + "_truth.json")))
