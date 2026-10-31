"""
Learned void filter for Detect Voids (die-to-die), trained from hand markings.

Faint voids can deviate from the golden chip no more than the layout does from chip to chip, so a
threshold on the deviation cannot separate them. Instead:

1. Candidates: local maxima of the (signed, averaged) deviation map of every chip, down to a low
   level (cand_threshold), at most MAX_CANDIDATES per chip; each grows into the region around it
   (same sign, above half the peak).
2. Features per candidate (FEATURES): how strong and how large, whether it is a round blob or a
   line (layout edges leaking into the difference), whether the difference follows the golden
   chip's edges, the contrast of the region against its ring in the chip and in the difference,
   the chip-to-chip spread there, distance to the chip border.
3. A boosted set of decision stumps (AdaBoost, numpy only) scores each candidate (Layer change:
   gradient-boosted trees, train_trees, which also combine features); candidates above
   the model's cut are voids.

Training ("Train" in Void Marking): the candidates of the open file are labelled from the hand
markings (a candidate inside a hand marking on the same layer and cell is a void; one over a hand
marking on another layer is left out), added to the training set (one entry per source file,
replaced when trained again) and the model is retrained on all of it. Cross-validation by chip
gives the expected recall / precision. Files: TRAIN_DIR (dataset.jsonl, void_classifier.json),
or config.json "void_detect": {"classifier": path} for a shared model.
"""
import hashlib
import json
import math
import os
import re
import time

import numpy as np
from PIL import Image, ImageFilter

MAX_CANDIDATES = 30 # Per chip and layer
NMS = 4 # Working px between candidate peaks
TRAIN_DIR = os.path.join(os.path.expanduser("~"), "TIF_Viewer_train")
DEFAULT_MODEL = os.path.join(TRAIN_DIR, "void_classifier.json")
DATASET = os.path.join(TRAIN_DIR, "dataset.jsonl")
# Layer change (sat_widgets/void_layer_change.py): its own candidates and features
LC_MODEL = os.path.join(TRAIN_DIR, "lc_classifier.json")
LC_DATASET = os.path.join(TRAIN_DIR, "lc_dataset.jsonl")

FEATURES = ["peak", "mean", "log_area", "width", "height", "aspect", "fill", "polarity",
            "blobness", "edge_corr", "on_edge", "img_contrast", "res_contrast", "gold_contrast",
            "spread", "border", "rank"]


def _maxf(a, size):
    return np.asarray(Image.fromarray(np.asarray(a, np.float32)).filter(ImageFilter.MaxFilter(size)))


def _grow(mask, seed):
    """Region of mask connected (4-neighbour) to seed (y, x)."""
    reg = np.zeros_like(mask)
    reg[seed] = True
    while True:
        nb = reg.copy()
        nb[1:] |= reg[:-1]
        nb[:-1] |= reg[1:]
        nb[:, 1:] |= reg[:, :-1]
        nb[:, :-1] |= reg[:, 1:]
        nb &= mask
        if (nb == reg).all():
            return reg
        reg = nb


def _dilate(mask, n):
    out = mask.copy()
    for _ in range(n):
        nb = out.copy()
        nb[1:] |= out[:-1]
        nb[:-1] |= out[1:]
        nb[:, 1:] |= out[:, :-1]
        nb[:, :-1] |= out[:, 1:]
        out = nb
    return out


class ChipContext:
    """Per chip arrays the candidates are measured on (working px)."""

    def __init__(self, z, m, golden, spread, sp_med):
        self.z = z # Signed deviation (as Detect Voids sees it)
        self.m = m # Matched chip
        self.g = golden
        self.res = m - golden
        self.spread = spread
        self.sp_med = max(sp_med, 1e-6)
        gy, gx = np.gradient(golden)
        self.gmag = np.hypot(gx, gy)
        self.gmag_med = float(np.median(self.gmag)) + 1e-6


def candidates(ctx, cand_threshold):
    """[(box (r0, r1, c0, c1), area, feature vector)] of the chip, strongest first."""
    a = np.abs(ctx.z)
    H, W = a.shape
    peaks = (a >= _maxf(a, 2 * NMS + 1)) & (a > cand_threshold)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-a[ys, xs])[:MAX_CANDIDATES]
    out = []
    R = max(12, int(0.12 * max(H, W)))
    for rank, i in enumerate(order):
        py, px = int(ys[i]), int(xs[i])
        y0, y1, x0, x1 = max(0, py - R), min(H, py + R + 1), max(0, px - R), min(W, px + R + 1)
        zw = ctx.z[y0:y1, x0:x1]
        sgn = 1.0 if ctx.z[py, px] > 0 else -1.0
        pk = float(a[py, px])
        reg = _grow(zw * sgn > 0.5 * pk, (py - y0, px - x0))
        rr, cc = np.nonzero(reg)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max()) + 1
        area = int(reg.sum())
        ring = _dilate(reg, 3) & ~reg
        if not ring.any():
            ring = ~reg
        m, res, g = ctx.m[y0:y1, x0:x1], ctx.res[y0:y1, x0:x1], ctx.g[y0:y1, x0:x1]
        gm = ctx.gmag[y0:y1, x0:x1]
        # Round blob vs line: Hessian of the smoothed signed deviation at the peak
        sig = max(1.0, math.sqrt(area) / 3.0)
        from .void_detect import _blur
        zs = _blur(zw * sgn, sig)
        cy, cx = py - y0, px - x0
        cy, cx = min(max(cy, 1), zs.shape[0] - 2), min(max(cx, 1), zs.shape[1] - 2)
        zxx = zs[cy, cx + 1] - 2 * zs[cy, cx] + zs[cy, cx - 1]
        zyy = zs[cy + 1, cx] - 2 * zs[cy, cx] + zs[cy - 1, cx]
        zxy = (zs[cy + 1, cx + 1] - zs[cy + 1, cx - 1] - zs[cy - 1, cx + 1] + zs[cy - 1, cx - 1]) / 4.0
        tr, det = zxx + zyy, zxx * zyy - zxy * zxy
        disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc
        blob = min(abs(l1), abs(l2)) / max(abs(l1), abs(l2), 1e-9)
        # Does the difference follow the layout edges (leak) rather than a shape of its own?
        ar, gg = np.abs(res).ravel(), gm.ravel()
        corr = float(np.corrcoef(ar, gg)[0, 1]) if ar.std() > 1e-9 and gg.std() > 1e-9 else 0.0
        bw, bh = c1 - c0, r1 - r0 + 1
        ring_std = float(res[ring].std()) + 0.05
        feats = [
            pk, float(np.abs(zw[reg]).mean()), math.log(area), float(bw), float(bh),
            max(bw, bh) / max(min(bw, bh), 1), area / float(bw * bh), sgn,
            blob, corr, float(gm[reg].mean()) / ctx.gmag_med,
            float(m[reg].mean() - m[ring].mean()) * sgn,
            float(res[reg].mean() - res[ring].mean()) * sgn / ring_std,
            float(g[reg].mean() - g[ring].mean()) * sgn,
            float(ctx.spread[y0:y1, x0:x1][reg].mean()) / ctx.sp_med,
            min(r0 + y0, c0 + x0, H - 1 - (r1 + y0), W - c1 - x0) / float(max(H, W)),
            float(rank),
        ]
        out.append(((r0 + y0, r1 + y0, c0 + x0, c1 + x0), area, np.asarray(feats, np.float32)))
    return out


# ----- boosted stumps -----
def train_stumps(X, y, rounds=200, bins=32):
    """
    AdaBoost with decision stumps. y in {0, 1}. Returns [(feature, threshold, polarity, alpha)].
    Thresholds are feature quantiles; each round reads the weighted error of every threshold from
    per-bin weight sums (np.bincount), so it stays fast for large training sets.
    """
    X = np.asarray(X, np.float64)
    t = np.where(np.asarray(y) > 0, 1.0, -1.0)
    n, d = X.shape
    pos = t > 0
    w = np.where(pos, 0.5 / max(pos.sum(), 1), 0.5 / max((~pos).sum(), 1)) # Classes weigh the same
    edges, idx = [], np.empty((d, n), np.int32)
    for j in range(d):
        e = np.unique(np.quantile(X[:, j], np.linspace(0.02, 0.98, bins)))
        edges.append(e)
        idx[j] = np.searchsorted(e, X[:, j], side="left") # Bin k: e[k-1] < x <= e[k]
    stumps = []
    for _ in range(rounds):
        best = None
        wp, wn = np.where(pos, w, 0.0), np.where(pos, 0.0, w)
        tot_p, tot_n = wp.sum(), wn.sum()
        for j in range(d):
            nb = len(edges[j]) + 1
            cp = np.cumsum(np.bincount(idx[j], wp, nb))[:-1] # Weight with x <= e[k]
            cn = np.cumsum(np.bincount(idx[j], wn, nb))[:-1]
            # Predict +1 where x > e[k]: errors = positives at or below + negatives above
            err = cp + (tot_n - cn)
            for pol, e_ in ((1.0, err), (-1.0, 1.0 - err)):
                k = int(np.argmin(e_))
                if best is None or e_[k] < best[0]:
                    best = (float(e_[k]), j, float(edges[j][k]), pol)
        e, j, th, pol = best
        e = min(max(e, 1e-9), 1 - 1e-9)
        if e >= 0.5:
            break
        alpha = 0.5 * math.log((1 - e) / e)
        h = pol * np.where(X[:, j] > th, 1.0, -1.0)
        w *= np.exp(-alpha * t * h)
        w /= w.sum()
        stumps.append((int(j), th, pol, float(alpha)))
    return stumps


def score(stumps, X):
    X = np.atleast_2d(np.asarray(X, np.float64))
    s = np.zeros(len(X))
    for j, th, pol, alpha in stumps:
        s += alpha * pol * np.where(X[:, j] > th, 1.0, -1.0)
    return s


def train_trees(X, y, rounds=400, depth=4, rate=0.03, bins=64, min_leaf=10, l2=1.0, rows=0.8, cols=0.8, seed=0):
    """
    Gradient-boosted decision trees (logistic loss, numpy only), like LightGBM: histogram splits on
    feature quantiles, trees grown level by level to depth, each round on a sample of the rows and
    of the features. Unlike the stumps they combine features (e.g. near the chip edge AND on one
    layer only). Returns {"base", "trees": [[[feature, threshold, left, right, value], ...]]}; a
    row goes left when x <= threshold; value is the leaf's logit step (feature -1 = leaf).
    """
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    edges, idx = [], np.empty((d, n), np.int32)
    for j in range(d):
        e = np.unique(np.quantile(X[:, j], np.linspace(0.01, 0.99, bins)))
        edges.append(e)
        idx[j] = np.searchsorted(e, X[:, j], side="left") # Bin k: e[k-1] < x <= e[k]
    p0 = min(max(y.mean(), 1e-6), 1 - 1e-6)
    base = math.log(p0 / (1 - p0))
    F = np.full(n, base)
    trees = []
    for _ in range(rounds):
        prob = 1.0 / (1.0 + np.exp(-F))
        g, h = prob - y, np.maximum(prob * (1 - prob), 1e-6)
        use = np.nonzero(rng.random(n) < rows)[0]
        feats = rng.choice(d, max(1, int(round(cols * d))), replace=False)
        nodes = [[-1, 0.0, -1, -1, 0.0]]
        level = [(0, use)]
        for _lv in range(depth + 1):
            nxt = []
            for node, r in level:
                G, H = g[r].sum(), h[r].sum()
                nodes[node][4] = float(-G / (H + l2) * rate)
                if _lv == depth or len(r) < 2 * min_leaf:
                    continue
                best = None
                for j in feats:
                    nb = len(edges[j]) + 1
                    b = idx[j, r]
                    gl = np.cumsum(np.bincount(b, g[r], nb))[:-1]
                    hl = np.cumsum(np.bincount(b, h[r], nb))[:-1]
                    cl = np.cumsum(np.bincount(b, None, nb))[:-1]
                    ok = (cl >= min_leaf) & (len(r) - cl >= min_leaf)
                    if not ok.any():
                        continue
                    gain = gl ** 2 / (hl + l2) + (G - gl) ** 2 / (H - hl + l2) - G ** 2 / (H + l2)
                    gain[~ok] = -np.inf
                    k = int(np.argmax(gain))
                    if gain[k] > 1e-9 and (best is None or gain[k] > best[0]):
                        best = (float(gain[k]), int(j), k)
                if best is None:
                    continue
                _, j, k = best
                left = r[idx[j, r] <= k]
                right = r[idx[j, r] > k]
                nodes[node][0], nodes[node][1] = j, float(edges[j][k])
                nodes[node][2], nodes[node][3] = len(nodes), len(nodes) + 1
                nodes += [[-1, 0.0, -1, -1, 0.0], [-1, 0.0, -1, -1, 0.0]]
                nxt += [(nodes[node][2], left), (nodes[node][3], right)]
            level = nxt
            if not level:
                break
        trees.append(nodes)
        F += _tree_values(nodes, X)
    return {"base": base, "trees": trees}


def _tree_values(nodes, X):
    at = np.zeros(len(X), np.int32)
    feat = np.array([nd[0] for nd in nodes])
    thr = np.array([nd[1] for nd in nodes])
    kids = np.array([[nd[2], nd[3]] for nd in nodes])
    val = np.array([nd[4] for nd in nodes])
    for _ in range(len(nodes)):
        f = feat[at]
        inner = f >= 0
        if not inner.any():
            break
        rows = np.nonzero(inner)[0]
        go_right = X[rows, f[inner]] > thr[at[inner]]
        at[rows] = kids[at[inner], go_right.astype(int)]
    return val[at]


def score_trees(model, X):
    """Logit of each row (cut in the same units)."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    s = np.full(len(X), float(model["base"]))
    for nodes in model["trees"]:
        s += _tree_values(nodes, X)
    return s


def _best_cut(s, y):
    """Cut on the score with the best F1."""
    best = (0.0, 0.0)
    for c in np.unique(np.quantile(s, np.linspace(0.5, 0.999, 120))):
        pred = s >= c
        tp = float((pred & (y > 0)).sum())
        if tp == 0:
            continue
        f1 = 2 * tp / (pred.sum() + (y > 0).sum())
        if f1 > best[0]:
            best = (f1, float(c))
    return best[1]


def cross_validate(X, y, groups, folds=5, rounds=200, trees=False, fp_per_file=None):
    """
    Out-of-fold scores -> (cut, recall, precision, false per file). trees: train_trees.
    groups: (source file, chip). Folds by chip; with fp_per_file and 3+ files by file (each file
    scored by a model that never saw it, like a new wafer). Cut: best F1, or with fp_per_file the
    lowest cut that leaves at most fp_per_file false candidates per file.
    """
    X, y = np.asarray(X), np.asarray(y)
    srcs = sorted({g[0] for g in groups})
    by_file = fp_per_file is not None and len(srcs) >= 3
    if by_file:
        folds = min(len(srcs), 10)
        keys = [g[0] for g in groups]
        fold = np.array([srcs.index(k) % folds if len(srcs) <= 10 else
                         int(hashlib.md5(str(k).encode()).hexdigest(), 16) % folds for k in keys])
    else:
        fold = np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % folds for g in groups])
    oof = np.zeros(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or (y[tr] > 0).sum() == 0:
            continue
        oof[te] = score_trees(train_trees(X[tr], y[tr], rounds), X[te]) if trees else \
            score(train_stumps(X[tr], y[tr], rounds), X[te])
    if fp_per_file is None:
        cut = _best_cut(oof, y)
    else:
        neg = np.sort(oof[y == 0])[::-1]
        allowed = int(math.floor(fp_per_file * max(len(srcs), 1)))
        cut = float(neg[allowed]) + 1e-9 if allowed < len(neg) else float(oof.min()) # Above the (allowed + 1)-th
    pred = oof >= cut
    tp = float((pred & (y > 0)).sum())
    fp = float((pred & (y == 0)).sum())
    return cut, tp / max((y > 0).sum(), 1), tp / max(pred.sum(), 1), fp / max(len(srcs), 1)


# ----- model / dataset files -----
class Model:
    def __init__(self, data):
        self.data = data
        self.stumps = [tuple(s) for s in data.get("stumps", [])]
        self.trees = data.get("trees") # Gradient-boosted trees (train_trees) instead of stumps
        self.cut = float(data["cut"])
        self.cand_threshold = float(data.get("cand_threshold", 1.0))

    @staticmethod
    def load(path, features=None):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("features") != (features or FEATURES):
            raise ValueError("model made for other features: train again")
        return Model(data)

    def scores(self, X):
        return score_trees(self.trees, X) if self.trees is not None else score(self.stumps, X)


def _product_dir(name):
    safe = re.sub(r"[^\w.-]+", "_", str(name)).strip("_") or "product"
    return os.path.join(TRAIN_DIR, "products", safe)


def model_path(params, method="die_to_die", product=None):
    """
    Model file of the method. Die-to-die: config "classifier", else DEFAULT_MODEL.
    Layer change, per product (product = (name, its products.json entry) or None): the product's
    "lc_classifier", else config "lc_classifier" ("{product}" = the product name, e.g. one shared
    folder for all products), else TRAIN_DIR/products/<name>/lc_classifier.json; no product:
    config "lc_classifier" or LC_MODEL.
    """
    if method == "layer_change":
        if product is not None:
            name, decl = product
            path = (decl or {}).get("lc_classifier") or params.get("lc_classifier")
            if path:
                return str(path).replace("{product}", str(name))
            return os.path.join(_product_dir(name), "lc_classifier.json")
        return params.get("lc_classifier") or LC_MODEL
    path = params.get("classifier")
    return path if path else DEFAULT_MODEL


def dataset_path(method="die_to_die", product=None):
    """Training set of the method (layer change: one per product, see model_path)."""
    if method == "layer_change":
        return os.path.join(_product_dir(product[0]), "lc_dataset.jsonl") if product is not None else LC_DATASET
    return DATASET


def add_to_dataset(source_key, rows, path=None):
    """Replaces the rows of source_key in the dataset (JSON lines). rows: [{"x", "y", "g"}]."""
    path = path or DATASET
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("src") != source_key:
                    keep.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(keep)
        for r in rows:
            f.write(json.dumps(dict(r, src=source_key)) + "\n")


def load_dataset(path=None, features=None):
    path, features = path or DATASET, features or FEATURES
    X, y, groups, sources = [], [], [], set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if len(r["x"]) != len(features):
                    continue # Rows of an older feature set
                X.append(r["x"])
                y.append(r["y"])
                groups.append((r["src"], r["g"]))
                sources.add(r["src"])
    return np.asarray(X, np.float64).reshape(-1, len(features)), np.asarray(y), groups, sources


MAX_NEGATIVES = 200000 # Training rows that are not voids (random subset beyond this)


def train_model(cand_threshold, path, rounds=200, dataset=None, features=None, trees=False, fp_per_file=None):
    """Trains on the whole dataset (trees: gradient-boosted trees, else stumps), saves the model;
    its cut: best F1, or at most fp_per_file false detections per file (cross_validate).
    Returns its summary dict."""
    features = features or FEATURES
    X, y, groups, sources = load_dataset(dataset, features)
    neg = np.nonzero(y == 0)[0]
    if len(neg) > MAX_NEGATIVES:
        keep = np.sort(np.concatenate([np.nonzero(y > 0)[0],
                                       np.random.default_rng(0).choice(neg, MAX_NEGATIVES, replace=False)]))
        X, y, groups = X[keep], y[keep], [groups[i] for i in keep]
    if (y > 0).sum() < 3 or (y == 0).sum() < 3:
        raise ValueError("not enough training data (at least 3 voids and 3 non-voids)")
    cut, rec, prec, fp_file = cross_validate(X, y, groups, rounds=rounds, trees=trees, fp_per_file=fp_per_file)
    used = {}
    if trees:
        model = {"trees": train_trees(X, y, rounds)}
        for nodes in model["trees"]["trees"]: # Splits per feature as its weight
            for nd in nodes:
                if nd[0] >= 0:
                    used[features[nd[0]]] = used.get(features[nd[0]], 0.0) + 1.0
    else:
        model = {"stumps": train_stumps(X, y, rounds)}
        for j, _, _, a in model["stumps"]:
            used[features[j]] = used.get(features[j], 0.0) + a
    data = {"version": 1, "features": features, **model, "cut": cut, "cand_threshold": cand_threshold,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": len(sources),
            "voids": int((y > 0).sum()), "others": int((y == 0).sum()),
            "cv_recall": rec, "cv_precision": prec, "cv_false_per_file": fp_file,
            "cut_rule": "best F1" if fp_per_file is None else f"<= {fp_per_file:g} false per file",
            "top_features": [k for k, _ in sorted(used.items(), key=lambda kv: -kv[1])[:5]]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return data


def label_rows(collected, truths, any_layer=False):
    """
    Training rows from collected candidates [(layer, cell, [(Deskewed box, area, features)])] and
    hand markings
    [{"layer", "cell", "box" (Deskewed x0, y0, x1, y1)}]: a candidate whose center lies in a hand
    marking (or holds its center) on the same layer and cell is a void (1); over a hand marking on
    another layer it is left out (any_layer: a void too, for a filter that decides the place only);
    otherwise 0. Returns (rows, markings covered by a candidate).
    """
    by_cell = {}
    for i, t in enumerate(truths):
        by_cell.setdefault(tuple(t["cell"]), []).append((i, t))
    rows, covered = [], set()
    for layer, cell, cands in collected:
        for b, _, feats in cands:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            label = 0
            for i, t in by_cell.get(tuple(cell), ()):
                tb = t["box"]
                tx, ty = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                # Margin: markings saved by older versions may sit a few px off
                mx, my = 0.25 * (tb[2] - tb[0]) + 3, 0.25 * (tb[3] - tb[1]) + 3
                tb = (tb[0] - mx, tb[1] - my, tb[2] + mx, tb[3] + my)
                if (tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]) or (b[0] <= tx <= b[2] and b[1] <= ty <= b[3]):
                    if any_layer or t["layer"] == layer:
                        label = 1
                        covered.add(i)
                        break
                    label = None # Over a marking of another layer: not used
            if label is not None:
                rows.append({"x": [round(float(v), 5) for v in feats], "y": label, "g": f"{cell[0]},{cell[1]}"})
    return rows, len(covered)
