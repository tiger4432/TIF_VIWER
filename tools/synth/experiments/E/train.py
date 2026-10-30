"""
Method E: gradient boosting on candidates.

Two LightGBM models:
  void   - candidate is a void (any layer) vs not, location-level features;
  layer  - for a void candidate, which layer it first appears on: binary over (candidate, l) rows
           with relative features, argmax over l.
Cut on the void probability chosen on out-of-fold predictions (grouped by chip, 5 folds, over the
three training wafers) to maximise F1 of the scorer (ok on the first layer, after per-chip NMS).

    python train.py                 # CV on 201..203, fit, write detections for 204 + score
Export: LightGBM -> ONNX via onnxmltools (convert_lightgbm) or its dumped trees evaluated in numpy.
"""
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import lightgbm as lgb

import label

C = "../../../../cache"
D = "../../../../data"
TRAIN = (201, 202, 203)
TEST = 204
L = 8
HALF = float(os.environ.get("BOXHALF", 4))  # detection box half size in 140 px; x2 for 280 px
P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1)
PL = dict(P, n_estimators=200)


def load(s):
    cands, rows, lrows = pickle.load(open(f"{C}/G_{s}.pkl", "rb"))
    keys = list(rows[0])
    Xv = np.array([[r[k] for k in keys] for r in rows], np.float32)
    lkeys = list(lrows[0])
    Xl = np.array([[r[k] for k in lkeys] for r in lrows], np.float32).reshape(len(rows), L, -1)
    Xl = np.concatenate([Xl, rel_profile(rows)], axis=2)
    lkeys = lkeys + [f"rel_{m}_s{si}_d{d}" for m in ("loo", "fwd", "bwd") for si in range(3) for d in OFFS] \
        + [f"rel_pw_d{d}" for d in OFFS] + [f"cum_{m}_{a}" for m in ("loo", "fwd", "bwd") for a in ("before", "after", "diff")]
    if os.environ.get("NOPW"):  # ablation: without the pairwise change maps
        kv = [n for n, k in enumerate(keys) if not k.startswith("pw")]
        kl = [n for n, k in enumerate(lkeys) if not k.startswith(("pw", "rel_pw"))]
        Xv, keys, Xl, lkeys = Xv[:, kv], [keys[n] for n in kv], Xl[:, :, kl], [lkeys[n] for n in kl]
    y, l0, vid, tv = label.label([c[:4] for c in cands], f"{D}/ml_{s}_truth.json")
    chip = np.array([c[0] for c in cands]) + s * 1000
    return dict(cands=cands, Xv=Xv, Xl=Xl, y=y, l0=l0, vid=vid + np.where(vid >= 0, s * 1000, 0), chip=chip,
                keys=keys, lkeys=lkeys, tv=tv, s=s)


OFFS = (-3, -2, -1, 0, 1, 2, 3)


def rel_profile(rows):
    """Per (candidate, l): |response| at layers l + d (d in OFFS, NaN outside) for each map / scale,
    and the mean |response| (max over scales) before / from l."""
    n = len(rows)
    out = np.full((n, L, 9 * len(OFFS) + len(OFFS) + 9), np.nan, np.float32)
    for c, r in enumerate(rows):
        V = np.abs(np.array([[[r[f"{m}_s{si}_k{k}"] for k in range(L)] for si in range(3)]
                             for m in ("loo", "fwd", "bwd")], np.float32))  # (3, 3, L)
        V[1, :, 0] = np.nan
        V[2, :, -1] = np.nan
        pw = np.array([r.get(f"pw_pres_k{k}", np.nan) for k in range(L)], np.float32)
        B = np.nanmax(V, axis=1)  # (3, L)
        for l in range(L):
            col = 0
            for m in range(3):
                for si in range(3):
                    for d in OFFS:
                        if 0 <= l + d < L:
                            out[c, l, col] = V[m, si, l + d]
                        col += 1
            for d in OFFS:
                if 0 <= l + d < L:
                    out[c, l, col] = pw[l + d]
                col += 1
            for m in range(3):
                b = np.nanmean(B[m, :l]) if l and np.isfinite(B[m, :l]).any() else 0.0
                a = np.nanmean(B[m, l:]) if np.isfinite(B[m, l:]).any() else 0.0
                out[c, l, col:col + 3] = (b, a, a - b)
                col += 3
    return out


def cat(ws, k):
    return np.concatenate([w[k] for w in ws])


def fit_void(Xv, y):
    return lgb.LGBMClassifier(**P).fit(Xv, y)


RANK = int(os.environ.get("RANK", 0))


def fit_layer(Xl, y, l0):
    pos = y == 1
    Xs = Xl[pos].reshape(-1, Xl.shape[-1])
    ys = (np.arange(L)[None, :] == l0[pos][:, None]).astype(int).ravel()
    if RANK:
        return lgb.LGBMRanker(**dict(PL, objective="lambdarank")).fit(Xs, ys, group=[L] * int(pos.sum()))
    return lgb.LGBMClassifier(**PL).fit(Xs, ys)


def pred_layer(m, Xl):
    Xs = Xl.reshape(-1, Xl.shape[-1])
    p = (m.predict(Xs) if RANK else m.predict_proba(Xs)[:, 1]).reshape(len(Xl), L)
    return p.argmax(axis=1), p


def evaluate(pv, layer, y, l0, vid, chip, nvoids, cut):
    """Scorer-like counts: one detection per kept candidate; a void counts once (best match)."""
    keep = pv >= cut
    ok_v, ly_v = set(), set()
    fp = 0
    for n in np.nonzero(keep)[0]:
        if y[n] and vid[n] not in ok_v:
            if layer[n] == l0[n]:
                ok_v.add(vid[n]); ly_v.discard(vid[n])
            elif vid[n] not in ly_v:
                ly_v.add(vid[n])
            else:
                fp += 1
        elif y[n]:
            fp += 1
        else:
            fp += 1
    det = keep.sum()
    ok = len(ok_v)
    rec, prec = ok / nvoids, ok / max(det, 1)
    return dict(ok=ok, ly=len(ly_v), fp=fp, det=int(det), f1=2 * rec * prec / max(rec + prec, 1e-9))


def dedupe(pv, cands, y=None, r=8):
    """Keep only the best candidate within r px in the same chip (the classifier may fire twice)."""
    order = np.argsort(-pv)
    kept = np.zeros(len(pv), bool)
    seen = {}
    for n in order:
        i, yy, xx = cands[n][:3]
        if any(abs(yy - a) <= r and abs(xx - b) <= r for a, b in seen.get(i, [])):
            continue
        seen.setdefault(i, []).append((yy, xx))
        kept[n] = True
    return np.where(kept, pv, 0.0)


def main():
    t0 = time.time()
    ws = [load(s) for s in TRAIN]
    Xv, Xl, y, l0, vid, chip = (cat(ws, k) for k in ("Xv", "Xl", "y", "l0", "vid", "chip"))
    cands = sum([w["cands"] for w in ws], [])
    nvoids = sum(len(w["tv"]) for w in ws)
    print("train cands", len(y), "pos", y.sum(), "voids with cand", len(set(vid[vid >= 0])), "/", nvoids)
    # grouped CV by chip
    rng = np.random.default_rng(0)
    uchips = np.unique(chip)
    fold_of = dict(zip(uchips, rng.integers(0, 5, len(uchips))))
    fold = np.array([fold_of[c] for c in chip])
    pv = np.zeros(len(y)); lay = np.zeros(len(y), int)
    for f in range(5):
        tr, te = fold != f, fold == f
        mv = fit_void(Xv[tr], y[tr])
        pv[te] = mv.predict_proba(Xv[te])[:, 1]
        ml = fit_layer(Xl[tr], y[tr], l0[tr])
        lay[te] = pred_layer(ml, Xl[te])[0]
    pvd = dedupe(pv, cands)
    print("CV layer acc on positives", np.mean(lay[y == 1] == l0[y == 1]).round(3))
    cm = np.zeros((L, L), int)
    for a, b in zip(l0[y == 1], lay[y == 1]):
        cm[a, b] += 1
    print("confusion (true rows, pred cols)\n", cm)
    best = None
    for cut in np.arange(0.05, 0.95, 0.025):
        e = evaluate(pvd, lay, y, l0, vid, chip, nvoids, cut)
        if best is None or e["f1"] > best[1]["f1"]:
            best = (cut, e)
    print("CV best cut", round(best[0], 3), best[1])
    for cut in (0.2, 0.3, 0.4, 0.5, 0.6):
        print("  cut", cut, evaluate(pvd, lay, y, l0, vid, chip, nvoids, cut))
    cut = float(os.environ.get("CUT", best[0]))
    mv = fit_void(Xv, y)
    ml = fit_layer(Xl, y, l0)
    ttrain = time.time() - t0
    imp = sorted(zip(mv.feature_importances_, ws[0]["keys"]), reverse=True)[:15]
    print("void top features", [k for _, k in imp])
    impl = sorted(zip(ml.feature_importances_, ws[0]["lkeys"]), reverse=True)[:10]
    print("layer top features", [k for _, k in impl])
    pickle.dump((mv, ml, cut), open(f"{C}/model_E.pkl", "wb"))
    # test
    t1 = time.time()
    w = load(TEST)
    p = dedupe(mv.predict_proba(w["Xv"])[:, 1], w["cands"])
    lt = pred_layer(ml, w["Xl"])[0]
    dets = []
    for n in np.nonzero(p >= cut)[0]:
        i, yy, xx = w["cands"][n][:3]
        r, c = divmod(i, 14)
        dets.append(dict(layer=int(lt[n]), row=r, col=c, x0=2 * (xx - HALF), y0=2 * (yy - HALF),
                         x1=2 * (xx + HALF), y1=2 * (yy + HALF), p=float(p[n])))
    out = f"{C}/det_E_{TEST}.json"
    json.dump(dets, open(out, "w"))
    print(f"train {ttrain:.0f}s, classify {time.time() - t1:.1f}s")
    res = subprocess.run([sys.executable, "../../score_detections.py", D, f"ml_{TEST}", out, "--mark", "first"],
                         capture_output=True, text=True).stdout
    print("TEST", res.strip())


if __name__ == "__main__":
    main()
