# Experiment C: classical layer-change void detection (no learning)

```bash
python tools/synth/experiments/C/lc.py data ml_204 det.json        # ~1.7 min / 8-layer wafer, 4 cores
python tools/synth/score_detections.py data ml_204 det.json --mark first
# tuning (ml_201..203 only): python tools/synth/experiments/C/grid.py '{}' tag '{"thr":[9,10,11],"tlo":[2.5,3]}'
```

Result on ml_204 (tuned on ml_201..203 only):
`{"voids": 50, "detections": 55, "ok": 35, "ly": 6, "ms": 9, "fp": 14, "recall": 0.7, "precision": 0.636, "f1": 0.667, "recall_any_layer": 0.82}`
(baseline die-to-die: ok 0, ly 3, fp 443). Training wafers together: ok 116, ly 24, fp 89 of 198 (F1 0.54).

## Pipeline (`lc.py`)
1. **Chips** at full resolution, each layer **registered** to layer 0 per chip (sub-pixel shift from
   gradient-magnitude correlation: does not depend on polarity; applied with a cubic spline), then 2x box reduction.
2. **Explain layer k** of a chip by robust (iteratively clipped) least squares, linear intensity domain, from
   - the chip's own layers **above** (0..k-1): value, square, x/y derivative, blur(1.5) - value, each x 4 bilinear
     tent weights over the chip (smoothly varying fit);
   - the wafer's **golden chips of all 8 layers** (median over chips, void-free) and their blur difference, x tents;
   - region terms: 4 regions of the shared layout (k-means on the golden layer vectors), offset + source x region
     (the pad / die regions have their own per-layer coefficient);
   - a quadratic smooth brightness term.
   Layer 0 has no layer above: only the goldens explain it.
   Layers above the void's first layer are void-free, so before the first layer there is no leak.
3. **Normalise** the remainder per chip (robust spread) and per pixel by its chip-to-chip spread on that layer.
4. **Centre-surround** (Gaussian sigma minus 2 sigma) at sigma 2, 3, 4.5, 6.5 working px; |score| / robust spread.
5. **Place**: root-mean-square of the per-layer best score over the 8 layers (x sqrt(8) / 2), peaks > 10.
6. **First layer**: evidence e_k = best scale |score| at the place on each layer; first = argmax_l sum_{k>=l}(e_k - 3)
   (CUSUM change point). Box = 1.6 sigma of the first layer's best scale.

## What helped / what did not (training F1 or background tail)
- Explaining from the layer above only (k-1, or 0..k-1) leaves most of the layout: shallow layers have 1-2
  sources, a layer is a mix of 4-5 structures with independent per-layer coefficients. Per-layer detection: F1 < 0.02.
- Adding the **goldens of all layers** as void-free sources: background chip-max tail dropped, no leak before the
  first layer. Biggest single step together with the region terms (pad: 99 % tail 30-60 -> 11-19).
- Linear domain better than log; derivatives + sub-pixel registration remove bump dipoles (registration small gain;
  integer-pixel registration made it worse: the jitter is continuous).
- Explaining from **all other layers** (k-1, k+1, ...): best fit, but the void of deeper layers leaks into earlier
  layers (layer assignment breaks); `src=all`.
- Aggregating evidence over layers: max -> F1 0.36, RMS -> 0.42; first layer by threshold-crossing -> CUSUM: 0.54.
- Not done: per-chip parametric layout model, factor-1 processing, one-void-per-chip prior.
