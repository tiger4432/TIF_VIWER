"""Pre-compute a training pool of feature crops (with pasted voids) in parallel.

    python make_pool.py --data data --train ml_201,ml_202 --out pool.npz [--variants 8] [--crop 80]
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NL, S, FACTOR, features, wafer_pcs, paste_void, label_for, load_wafer  # noqa: E402

G = {}


def work(job):
    i, seed = job
    rng = np.random.default_rng(seed)
    stack, vs = G["items"][i]
    stack, vs = stack.copy(), list(vs)
    if rng.random() < G["p_paste"]:
        for _ in range(rng.integers(1, 3)):
            rx, ry = rng.uniform(6, 25, 2) / FACTOR
            cx, cy = rng.uniform(35, 280 - 35, 2) / FACTOR
            if any(abs(cx - v[0]) < 2.5 * (rx + v[2]) and abs(cy - v[1]) < 2.5 * (ry + v[3]) for v in vs):
                continue
            l0 = int(rng.integers(0, NL))
            paste_void(stack, rng, cx, cy, rx, ry, l0, rng.uniform(0.06, 0.14))
            vs.append((cx, cy, rx, ry, l0))
    x = features(stack, G["pcs"][i])
    y = label_for(np.zeros((S, S), np.int64), vs)
    C = G["crop"]
    xs, ys = [], []
    for j in range(G["crops"]):
        if vs and j % 2 == 0:   # half the crops hold a void
            v = vs[rng.integers(len(vs))]
            x0 = int(np.clip(v[0] - rng.uniform(10, C - 10), 0, S - C))
            y0 = int(np.clip(v[1] - rng.uniform(10, C - 10), 0, S - C))
        else:
            x0, y0 = rng.integers(0, S - C + 1, 2)
        xs.append(x[:, y0:y0 + C, x0:x0 + C].astype(np.float16))
        ys.append(y[y0:y0 + C, x0:x0 + C].astype(np.int8))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--train", default="ml_201,ml_202")
    ap.add_argument("--out", default="pool.npz"); ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--crops", type=int, default=2); ap.add_argument("--crop", type=int, default=80)
    ap.add_argument("--p-paste", type=float, default=0.75); ap.add_argument("--no-resid", action="store_true")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--K", type=int, default=40)
    a = ap.parse_args()
    items, pcs = [], []
    for n in a.train.split(","):
        X, truth = load_wafer(a.data, n)
        P = wafer_pcs(X, a.K)
        for r in range(X.shape[1]):
            for c in range(X.shape[2]):
                t = truth.get((r, c))
                pcs.append(P)
                items.append((X[:, r, c].copy(), [] if t is None else
                              [(t[0] / FACTOR, t[1] / FACTOR, t[2] / FACTOR, t[3] / FACTOR, int(t[4]))]))
    G.update(items=items, pcs=pcs, p_paste=a.p_paste, resid=not a.no_resid, crop=a.crop, crops=a.crops)
    jobs = [(i, a.seed * 1000003 + v * 10007 + i) for v in range(a.variants) for i in range(len(items))]
    with Pool(4) as p:
        res = p.map(work, jobs, chunksize=16)
    X = np.stack([x for xs, _ in res for x in xs]); Y = np.stack([y for _, ys in res for y in ys])
    np.savez(a.out, X=X, Y=Y, resid=not a.no_resid)
    print(a.out, X.shape, "void px frac", float((Y > 0).mean()))


if __name__ == "__main__":
    main()
