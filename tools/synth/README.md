# Synthetic wafers for comparing void detection methods

Hard multi-layer synthetic SAT wafers (grid-line layout whose blur varies per chip and across the
chip; per layer brightness / contrast / polarity differ but the structure is the same; per chip
brightness gradient and wavy background; faint voids that appear on a layer and stay on every
deeper layer with per-layer shade and size) and one scoring rule for every method.

```bash
# data: 3 training wafers + 1 test wafer (deterministic per SEED; about 1 min each)
for s in 201 202 203 204; do SEED=$s python tools/synth/make_wafer.py data ml_$s; done

# score any method: write detections.json (see score_detections.py), then
python tools/synth/score_detections.py data ml_204 detections.json --mark first
# -> {"voids", "detections", "ok", "ly", "ms", "fp", "recall", "precision", "f1", "recall_any_layer"}

# chips for methods outside the viewer
python -c "from tools.synth.chips import load_chips; X = load_chips('data', 'ml_204', 2)"   # (8, 14, 14, 140, 140)

# the viewer's own methods
python tools/synth/eval_app.py data ml_204 --param threshold=6                     # die-to-die
python tools/synth/save_truth.py data ml_201 results/w201                           # truth -> voids.json + meta.json
python tools/synth/eval_app.py data ml_204 --seg-model models/void_unet.onnx --param threshold=5
```

Truth convention for comparisons: `--mark first` (a void counts on the layer where it first appears).
Train / tune only on ml_201..203; report on ml_204.
