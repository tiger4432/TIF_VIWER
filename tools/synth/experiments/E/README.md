# Experiment E: candidates + gradient boosting

```bash
# from this directory, after generating data/ml_201..204 (see tools/synth/README.md)
python -c "import numpy as np; from tools.synth.chips import load_chips"   # (chips are cached by the steps below)
for s in 201 202 203 204; do python -c "import sys,numpy as np; sys.path.insert(0,'../../../..'); \
  from tools.synth.chips import load_chips; np.save('../../../../cache/X_$s.npy', load_chips('../../../../data','ml_$s',2))"; done
python -c "import numpy as np, maps; [np.save(f'../../../../cache/R_{s}.npy', maps.wafer_changes(np.load(f'../../../../cache/X_{s}.npy'))) for s in (201,202,203,204)]"
python mcache.py 201 202 203 204      # joint layer explanations (loo / fwd / bwd)
python build.py 201 202 203 204       # candidates + features (one at a time: ~6 GB peak)
python train.py                       # grouped CV on 201..203, fit, detect + score 204
```

- `maps.py`   change maps (numpy): pairwise R[k, j] (layer k explained by layer j, LayoutFit model) and
  joint M (layer k explained by all other / earlier / later layers of the chip, each with 4 bilinear
  weights + blur difference + x/y gradient for sub-pixel misregistration). FFT DoG bank.
- `feats.py`  normalisation over chips (per pixel spread), DoG at 1.5/3/6 px (140 px chip), candidates
  (16 local maxima per chip of max |loo|,|fwd| response), location features and per first-layer rows.
- `train.py`  LightGBM: void vs not (location features) + first layer (binary over (candidate, l) rows with
  relative before / at / after features, argmax). Cut on grouped-by-chip 5-fold CV F1.
  Env: `RANK=1` lambdarank layer model, `NOPW=1` ablation without pairwise maps, `CUT=` fixed cut.

Export: both models are LightGBM trees -> ONNX via onnxmltools.convert_lightgbm (or dump_model() and
evaluate the trees in numpy); feature extraction is numpy only.

Result on ml_204 (--mark first): ok 23, ly 14, ms 13, fp 12 of 50 (F1 0.465); baseline ok 0, fp 443.
