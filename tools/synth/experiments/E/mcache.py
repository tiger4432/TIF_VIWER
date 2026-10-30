import sys, numpy as np, maps
for s in sys.argv[1:]:
    np.save(f'../../../../cache/M_{s}.npy', maps.wafer_multi(np.load(f'../../../../cache/X_{s}.npy')).astype(np.float16))
