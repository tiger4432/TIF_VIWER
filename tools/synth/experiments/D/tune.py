"""Grid-search the box post-processing on saved probabilities of a TRAINING wafer.

    python tools/synth/experiments/D/tune.py data ml_203 probs_203.npy
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from draem import to_boxes, to_boxes_present  # noqa: E402

outdir, name, probs = sys.argv[1:4]
P = np.load(probs).astype(np.float32)
tmp = probs + ".det.json"
score = os.path.join(HERE, "..", "..", "score_detections.py")
res = []
for dec in ("present", "first"):
    for thr in (0.3, 0.5):
        for peak in (0.6, 0.8, 0.9, 0.95):
            for ma in (6, 15):
                for frac in ((0.3, 0.5, 0.7) if dec == "present" else (0,)):
                    for mpc in (None, 1):
                        if dec == "present":
                            d = to_boxes_present(P, thr, peak, ma, frac, max_per_chip=mpc)
                        else:
                            d = to_boxes(P, thr, peak, ma, max_per_chip=mpc)
                        json.dump(d, open(tmp, "w"))
                        s = json.loads(subprocess.check_output([sys.executable, score, outdir, name, tmp, "--mark", "first"]))
                        res.append((s["f1"], dec, thr, peak, ma, frac, mpc, s))
res.sort(key=lambda r: -r[0])
for r in res[:15]:
    print(r[1:7], json.dumps(r[7]))
