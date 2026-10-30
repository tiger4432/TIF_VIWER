# Experiment A: the viewer's multi-layer U-Net, tuned

Trained on ml_201 + ml_202 (labels from their truth), thresholds / rules chosen on ml_203, reported on ml_204.

```bash
# data (+ per-layer void sizes for the "every layer" labels; same TIF as make_wafer.py)
for s in 201 202 203; do SEED=$s python tools/synth/experiments/A/make_wafer_layers.py data ml_$s; done
# app pipeline: voids.json + meta.json -> seg_data.extract_saved -> void_seg.write_source
python tools/synth/experiments/A/extract.py data results SEG_ALL --label all ml_201 ml_202 ml_203
# train (layer channels cached once per sample)
python tools/synth/experiments/A/train_cached.py --data SEG_ALL --sources ml_201,ml_202 --label all --stack 1 \
    --epochs 30 --steps 60 --out tools/synth/experiments/A/models/r3.onnx
# evaluate (viewer's Detect Voids)
python tools/synth/eval_app.py data ml_204 --seg-model tools/synth/experiments/A/models/r3.onnx \
    --param threshold=5 --param layer_rule=run --param min_layers=2
```

App changes (sat_widgets):
- `void_seg.stack_residual`: a layer of a chip fitted (robust least squares) from the chip's other layers
  `[1, layer, layer^2]` -> residual against all other layers and against the layers above. Every layer shows
  the same structures with its own per-structure contrast / polarity, so one layer is not a scaled copy of
  another and `layer_change` (single contrast fit) leaves the layout behind. Models whose sidecar has
  `"stack_residual": 1` get these two channels instead of the change against the layer above / below.
- `void_seg.change_spread` + sidecar `"change_norm"`: layer changes / their per-pixel chip-to-chip spread (not used by the best model).
- `layer_rule="run"` (+ `min_layers`, `run_gap`): spots of one place grouped across layers; marked where the run
  reaching the deepest find starts; groups on fewer than `min_layers` layers dropped unless on the last layer.

Results (score_detections.py --mark first):

| model / setting | wafer | ok | ly | fp | recall | recall any layer | f1 |
|---|---|---|---|---|---|---|---|
| die-to-die thr 6 (baseline) | 204 | 0 | 3 | 443 | 0 | 0.06 | 0 |
| die-to-die thr 6 | 203 | 2 | 7 | 760 | 0.03 | 0.13 | 0.005 |
| layer_change, labels first / all (r1) | val chips | | | | void F1 < 0.08 (stopped) | | |
| stack residual, labels all (r3), first, thr 2 | 203 | 11 | 38 | 1878 | 0.16 | 0.73 | 0.011 |
| r3, first, thr 4 | 203 | 8 | 29 | 978 | 0.12 | 0.55 | 0.015 |
| r3, first, thr 7 | 203 | 2 | 12 | 69 | 0.03 | 0.21 | 0.027 |
| r3, run, min_layers 2, thr 4 | 203 | 5 | 24 | 296 | 0.075 | 0.43 | 0.026 |
| **r3, run, min_layers 2, thr 5** (chosen on 203) | 203 | 4 | 14 | 142 | 0.06 | 0.27 | 0.035 |
| **r3, run, min_layers 2, thr 5** | **204** | 3 | 10 | 109 | 0.06 | 0.26 | 0.035 |
| r3, first, thr 4 | 204 | 8 | 23 | 960 | 0.16 | 0.62 | 0.015 |

Train 13 min (4 threads, 1800 batches of 16 x 128 px, base 16) + 1 min for the cached channels; Detect Voids
with the model about 90 s per 8-layer wafer (4 cores).

Notes: the stack residual finds voids (any layer) far better than die-to-die or layer_change, but mostly on
deeper layers where the void has grown (sizes x0.4-2.5 per layer); the first layer is often too faint, so the
correct-layer recall stays low. The stock trainer pastes synthetic voids with an absolute amplitude 0.3-3
(about 4-36x the chip-to-chip spread, median 0.08 here) while real voids are about 0.3-3x the spread;
train_cached.py scales them to the spread.
