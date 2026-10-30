# Method B: whole-stack U-Net

Per chip, all 8 layers go into one small U-Net (~0.49 M params, base 16, 3 levels) that outputs 9 classes
per pixel: 0 = no void, 1 + k = void first appears on layer k. Boxes = connected components of
P(void) > thr (min area, factor-2 px), layer = argmax of the summed class probabilities in the component.

Input (26 channels, `common.features`), at factor 2 (140 x 140 per chip):
- per layer: log intensity, median / MAD normalised per chip-layer;
- per layer: robust residual of a least-squares fit of that layer by (cubic polynomial + top-40
  principal images of the whole wafer + all **shallower** layers of the chip) — what is new on the layer;
- per layer: same fit but by **all other** layers — what the chip's own stack cannot explain;
- x, y coordinate channels (the layout is the same on every chip, so the net can learn the pad / edges).

The wafer principal images (`wafer_pcs`, SVD over all chip-layers of the wafer being processed) are an
unsupervised template of the shared layout incl. its blur variants; voids sit at random places and do not
enter them. They took the void / worst-background contrast of the residual from 0.76 to 1.48 (median).

Training (`make_pool.py` + `train.py`): pre-computed pool of 80 x 80 feature crops; each chip gets
fresh pasted voids (generator model: multiplicative soft ellipse, persists to deeper layers with per-layer
shade incl. polarity and size x0.4-2.5) with p = 0.75; real truth voids are labelled on their base ellipse,
the possible deeper extent is ignored. Augmentation: per-layer sign flips (polarity). CE (void weight 4)
+ 0.5 soft dice. AdamW + OneCycle, 3000 steps x 24 crops.

```bash
python cache.py data ml_201 ml_202 ml_203 ml_204                       # chips at factor 2 -> *_chips2.npy
python make_pool.py --data data --train ml_201,ml_202,ml_203 --out pool.npz --variants 6 --seed 2
python train.py --pool pool.npz --out model.pt --steps 3000 --bs 24 --base 16
python infer.py --data data --name ml_204 --model model.pt --out dets.json --thr 0.8 --min-area 15
```

## Result (final model trained on ml_201..203; post-processing tuned on ml_203 with a 201+202 model)

`model_final.pt`, `--thr 0.8 --min-area 15`, ml_204, `--mark first`:
`{"voids": 50, "detections": 86, "ok": 23, "ly": 24, "ms": 3, "fp": 39, "recall": 0.46, "precision": 0.267, "f1": 0.338, "recall_any_layer": 0.94}`
(baseline die-to-die: ok 0, ly 3, fp 443). Train ~14 min (+ ~2 min pool) on 4 CPU cores; inference ~20 s / wafer.

Notes: first version (8 normalised layers + residuals by other layers only, with rot/flip augmentation,
no coordinates) found 97 % of the voids on some layer but put ~80 FPs per wafer on the pad / die edge;
wafer principal images + coordinate channels + no rot/flip cut FPs to ~30. The remaining weakness is the
layer of first appearance (often +-1..2 layers off: the first-layer shade can be very faint while deeper
layers are stronger / larger).
