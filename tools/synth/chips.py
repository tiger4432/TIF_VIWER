"""
Chip crops of a synthetic wafer for methods outside the viewer.

    from tools.synth.chips import load_chips
    X = load_chips(outdir, name, factor=2)   # float32 (layers, 14, 14, 280 // factor, 280 // factor)

Each chip is cut at its exact place (the 280 x 280 px bbox, rotation undone, bilinear) and reduced
by factor (box average). Box coordinates for score_detections.py are in full chip px (x factor).
"""
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sat_widgets.core_data import CoordinateTransform # noqa: E402

ORG, P, B, ANG, R = 100, 300, 280, 0.4, 14
M = (P - B) / 2


def load_chips(outdir, name, factor=2):
    with tifffile.TiffFile(os.path.join(outdir, name + ".tif")) as tf:
        pages = [pg.asarray().astype(np.float32) for pg in tf.pages]
    H, W = pages[0].shape
    ct = CoordinateTransform(ANG, W, H)
    yy, xx = np.mgrid[0:B, 0:B].astype(np.float32)
    out = np.zeros((len(pages), R, R, B // factor, B // factor), np.float32)
    for r in range(R):
        for c in range(R):
            rx, ry = ct.deskew_to_raw(ORG + c * P + M + xx, ORG + r * P + M + yy) # Grid -> raw image
            x0, y0 = np.floor(rx).astype(int), np.floor(ry).astype(int)
            fx, fy = rx - x0, ry - y0
            x0, y0 = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2)
            for k, img in enumerate(pages):
                v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
                     + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
                out[k, r, c] = v.reshape(B // factor, factor, B // factor, factor).mean(axis=(1, 3))
    return out
