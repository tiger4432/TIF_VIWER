# Method D: DRAEM-style (reconstruction U-Net + discriminative U-Net)

```bash
pip install torch   # PyPI wheel: if the 3 GB CUDA deps are too slow, install the torch wheel --no-deps and
                    # the nvidia-* wheels in parallel (`pip download --no-deps` each), torch links them hard
python tools/synth/experiments/D/draem.py train --wafers ml_201 ml_202 --out runs/D/b --steps 2400 --base 16
python tools/synth/experiments/D/draem.py detect --wafer ml_203 --model runs/D/b/model.pt --save_probs runs/D/b/p203.npy
python tools/synth/experiments/D/tune.py data ml_203 runs/D/b/p203.npy          # pick post-processing on 203
python tools/synth/experiments/D/draem.py detect --wafer ml_204 --model runs/D/b/model.pt --out det204.json \
    --decode present --thr 0.3 --peak 0.95 --min_area 6 --frac 0.3
python tools/synth/score_detections.py data ml_204 det204.json --mark first
```

Trained on ml_201 + ml_202 (clean chips with synthetic voids pasted in + real truth voids), tuned on ml_203.

| run | ml_204 (--mark first) |
|---|---|
| a: rec L1 unweighted, decode "first" head | ok 2, ly 40, fp 36 (recall_any_layer 0.84) |
| b: rec L1 x30 in voids, decode earliest "present" layer | ok 9, ly 36, fp 36, f1 0.137 (recall_any_layer 0.90) |

Findings: the net localises voids well (90 % found on some layer, fp 36 vs 443 for die-to-die) but the
layer of first appearance is mostly wrong. The reconstruction net collapsed to the identity in both runs
(residual x - R(x) ~ 0 at voids), so D works on the raw stack alone and does not separate layers.
Next steps: a layer-symmetric discriminator (shared weights per layer, input [x_k, R_k, x_k - R_k, context])
and a reconstruction that never sees layer k (leave-one-out, or masked-in-paint) so it cannot copy the void.
