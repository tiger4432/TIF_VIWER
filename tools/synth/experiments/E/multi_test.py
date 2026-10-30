import sys, time, numpy as np, maps, features, label
s = sys.argv[1]
X = np.load(f'../../../../cache/X_{s}.npy')
t = time.time()
M = maps.wafer_multi(X); print('multi', time.time() - t)
np.save(f'../../../../cache/M_{s}.npy', M.astype(np.float16))
N = M.shape[0]
sp = np.quantile(np.abs(M), 0.75, axis=0) / 1.15
sp = maps.blur(sp, 1.5); sp = np.maximum(sp, 0.5 * np.median(sp, axis=(-1, -2), keepdims=True)) + 1e-6
Z = M / sp[None]
Z[:, 1, 0] = 0; Z[:, 2, -1] = 0
D = np.stack([maps.dog_bank(Z, [sc])[0] for sc in features.SCALES])  # (S, N, 3, L, H, W)
mad = np.median(np.abs(D[:, ::3, :, :, ::4, ::4]), axis=(1, 4, 5), keepdims=True) * 1.4826 + 1e-6
D = D / mad
y0 = label.label([], f'../../../../data/ml_{s}_truth.json')[3]
for m, name in enumerate(('loo', 'fwd', 'bwd')):
    A = maps.blur(np.abs(D[:, :, m]).max(axis=(0, 2)), 1.0)
    a = [A[i, int(cy), int(cx)] for i, (cx, cy, rx, ry, l) in y0.items()]
    print(name, 'A at voids pct', np.percentile(a, [10, 25, 50, 75]).round(1), 'bg 99/99.9', np.percentile(A[:, 10:-10, 10:-10], [99, 99.9]).round(1))
    # first-layer response: at void center, fwd residual per layer
    if m == 1:
        hits = 0
        for i, (cx, cy, rx, ry, l) in y0.items():
            prof = np.abs(D[:, i, 1, :, int(cy), int(cx)]).max(axis=0)
            hits += int(np.argmax(prof[1:]) + 1 == l)
        print('fwd argmax==l0', hits, len(y0))
