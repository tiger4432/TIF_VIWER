import sys, os, time, pickle, numpy as np, feats
C = '../../../../cache'
for s in sys.argv[1:]:
    t = time.time()
    X = np.load(f'{C}/X_{s}.npy')
    Rw = np.load(f'{C}/R_{s}.npy') if os.path.exists(f'{C}/R_{s}.npy') else None
    M = np.load(f'{C}/M_{s}.npy') if os.path.exists(f'{C}/M_{s}.npy') else None
    out = feats.wafer_features(X, True, Rw, M)
    pickle.dump(out, open(f'{C}/G_{s}.pkl', 'wb'))
    print(s, 'cands', len(out[0]), f'{time.time() - t:.0f}s', flush=True)
