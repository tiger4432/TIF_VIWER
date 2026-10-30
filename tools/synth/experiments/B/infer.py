"""Detect voids on a wafer with a trained model; optionally sweep thresholds and score.

    python infer.py --data data --name ml_204 --model model.pt --out dets.json [--thr 0.5] [--sweep]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NL, S, PAD, UNet, features, wafer_pcs, n_in, pad, to_boxes, load_wafer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE = os.path.join(HERE, "..", "..", "score_detections.py")


def predict(model, X, K=40, flip=False):
    R = X.shape[1]
    pcs = wafer_pcs(X, K)
    probs = np.zeros((R, R, NL + 1, S, S), np.float32)
    with torch.no_grad():
        for r in range(R):
            xb = torch.from_numpy(np.stack([pad(features(X[:, r, c], pcs)) for c in range(R)]))
            if flip:   # only for models trained with left-right flips
                p = (model(xb).softmax(1) + model(xb.flip(3)).softmax(1).flip(3)) / 2
            else:
                p = model(xb).softmax(1)
            probs[r] = p[:, :, PAD:-PAD, PAD:-PAD].numpy()
    return probs


def detect(probs, thr, min_area, peak):
    R = probs.shape[0]
    return [d for r in range(R) for c in range(R) for d in to_boxes(probs[r, c], r, c, thr, min_area, peak)]


def score(data, name, dets, path):
    json.dump(dets, open(path, "w"))
    out = subprocess.run([sys.executable, SCORE, data, name, path, "--mark", "first"], capture_output=True, text=True).stdout
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--name", default="ml_204")
    ap.add_argument("--model", default="model.pt"); ap.add_argument("--out", default="dets.json")
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--min-area", type=int, default=6)
    ap.add_argument("--peak", type=float, default=None)
    ap.add_argument("--K", type=int, default=40); ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.model)
    model = UNet(n_in(), ck["base"]); model.load_state_dict(ck["state"]); model.eval()
    X, _ = load_wafer(a.data, a.name)
    t0 = time.time()
    probs = predict(model, X, a.K)
    print(f"inference {time.time() - t0:.1f}s", file=sys.stderr)
    if a.sweep:
        np.save(a.out + ".probs.npy", probs)
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for peak in [None, 0.8, 0.9]:
                for ma in [6, 15]:
                    s = score(a.data, a.name, detect(probs, thr, ma, peak), a.out)
                    print(f"thr {thr} peak {peak} min_area {ma}: {json.dumps(s)}")
    dets = detect(probs, a.thr, a.min_area, a.peak)
    print(json.dumps(score(a.data, a.name, dets, a.out)))


if __name__ == "__main__":
    main()
