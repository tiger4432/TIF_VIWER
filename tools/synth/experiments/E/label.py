import json, numpy as np
def label(cands, truth_path, grid=14):
    T = json.load(open(truth_path))
    tv = {}
    for k, (cx, cy, rx, ry, l0, l1, f) in T.items():
        r, c = map(int, k.split(',')); tv[r * grid + c] = (cx / 2, cy / 2, rx / 2, ry / 2, l0)
    y = np.zeros(len(cands), int); l0s = -np.ones(len(cands), int); vid = -np.ones(len(cands), int)
    for n, (i, yy, xx, s) in enumerate(cands):
        if i in tv:
            cx, cy, rx, ry, l0 = tv[i]
            # box-meet rule of the scorer with a small detection box (half size 3 px @140)
            if abs(xx - cx) <= rx + 4 + 1 and abs(yy - cy) <= ry + 4 + 1:
                y[n] = 1; l0s[n] = l0; vid[n] = i
    return y, l0s, vid, tv
