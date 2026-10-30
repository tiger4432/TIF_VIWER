"""Grid search on the training wafers: python grid.py '<fixed params json>' <cache tag> '<grid json>'"""
import sys, time, numpy as np, json, itertools
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc, tune
D=os.environ.get('DATA', 'data')
p=json.loads(sys.argv[1]); tag=sys.argv[2]; grid=json.loads(sys.argv[3])
caches={}
for s in (201,202,203):
    X=lc.load(D,f'ml_{s}'); t=time.time()
    try:
        cache={'rem':np.load(f'{D}/rem_{tag}_{s}.npy'),'rem0':None}
    except Exception:
        cache={}
        lc.layer_scores(X,dict(lc.DEFAULTS,**p),4,cache); np.save(f'{D}/rem_{tag}_{s}.npy',cache['rem'])
    print(s,'time',round(time.time()-t,1), flush=True)
    caches[s]=(X,cache)
keys=list(grid); res=[]
for vals in itertools.product(*[grid[k] for k in keys]):
    q=dict(p,**dict(zip(keys,vals)))
    rows=[tune.score(tune.truth_of(D,f'ml_{s}'), lc.run(X,q,4,c)) for s,(X,c) in caches.items()]
    tt=tune.total(rows); res.append((tt['f1'],vals,tt)); print(dict(zip(keys,vals)), tt, flush=True)
res.sort(key=lambda r:-r[0]); print('BEST', res[:3])
