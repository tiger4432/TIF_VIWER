"""
Generates a synthetic SAT multi-gate TIFF for testing (BigTIFF, one page per layer).

    python tools/make_sample_tif.py                         # 14900 x 14900, 12 layers, 8-bit gray, LZW (~1.7 GB)
    python tools/make_sample_tif.py --compress none         # uncompressed (~2.7 GB, memory-mapped by the viewer)
    python tools/make_sample_tif.py --dtype uint16          # 16-bit
    python tools/make_sample_tif.py --size 4000 --layers 3 --out small.tif

Content per layer (gate): a rotated grid of chips. Each cell has a package bbox, a die area inset
by the scribe line (SL), random voids that show up on a range of gates, and some corner
delaminations shaped like a quarter superellipse starting at an SL-shifted die corner.

The pixels are generated and written strip by strip, so memory use stays small for any size.
The printed grid values (origin, pitch, rows, cols, angle) line up with the generated chips in
the viewer (same rotation about the image center as the viewer uses).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import tifffile


def chip_params(rng, rows, cols, n_layers, die_lo, die_hi):
    """Random defects per chip (local coordinates inside the cell)."""
    shape = (rows, cols)
    span = die_hi - die_lo
    p = {
        # Void: ellipse inside the die, visible on gates [l0, l1]
        "void": rng.random(shape) < 0.35,
        "vx": die_lo + span * rng.uniform(0.15, 0.85, shape),
        "vy": die_lo + span * rng.uniform(0.15, 0.85, shape),
        "vrx": rng.uniform(6, 40, shape),
        "vry": rng.uniform(6, 40, shape),
        "vl0": rng.integers(0, n_layers, shape),
        # Corner delamination: quarter superellipse from one SL-shifted die corner
        "corner": rng.random(shape) < 0.2,
        "ck": rng.integers(0, 4, shape), # 0 TL, 1 TR, 2 BR, 3 BL
        "crx": rng.uniform(25, 0.35 * span, shape),
        "cry": rng.uniform(25, 0.35 * span, shape),
        "cp": rng.uniform(1.5, 6.0, shape),
        "cl0": rng.integers(0, n_layers, shape),
    }
    p["vl1"] = np.minimum(n_layers - 1, p["vl0"] + rng.integers(1, 4, shape))
    # Per gate: which chips show their defect
    p["void_on"] = [p["void"] & (p["vl0"] <= k) & (k <= p["vl1"]) for k in range(n_layers)]
    p["corner_on"] = [p["corner"] & (p["cl0"] <= k) for k in range(n_layers)]
    return p


def make_strip(y0, y1, width, layer, n_layers, geo, prm, rng, dtype, noise):
    cx_img, cy_img = geo["img_cx"], geo["img_cy"]
    c, s = geo["cos"], geo["sin"]
    ox, oy, pitch = geo["ox"], geo["oy"], geo["pitch"]
    rows, cols = geo["rows"], geo["cols"]
    m, bbox, sl = geo["margin"], geo["bbox"], geo["sl"]

    yy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy_img
    xx = np.arange(width, dtype=np.float32)[None, :] - cx_img
    # Raw -> Deskewed (grid space): rotation by -angle about the image center (as in the viewer)
    u = cx_img + xx * c + yy * s - ox
    v = cy_img - xx * s + yy * c - oy

    col = np.floor(u / pitch).astype(np.int32)
    row = np.floor(v / pitch).astype(np.int32)
    lx = u - col * pitch # Position inside the cell
    ly = v - row * pitch
    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)

    # Gate dependent levels (16-bit domain)
    t = layer / max(1, n_layers - 1)
    bg, pkg, die = 6000 + 2000 * t, 18000 + 6000 * t, 30000 + 12000 * math.sin(math.pi * t)
    img = np.full(lx.shape, bg, dtype=np.float32)

    in_bbox = inside & (lx >= m) & (lx < m + bbox) & (ly >= m) & (ly < m + bbox)
    lo, hi = m + sl, m + bbox - sl
    in_die = inside & (lx >= lo) & (lx < hi) & (ly >= lo) & (ly < hi)
    img[in_bbox] = pkg
    img[in_die] = die

    # Defects: evaluated only on die pixels of chips that have one on this gate
    iy, ix = np.nonzero(in_die)
    cell = row[iy, ix] * cols + col[iy, ix]
    px, py = lx[iy, ix], ly[iy, ix]
    g = lambda k, sel: prm[k].ravel()[cell[sel]]

    # Voids (dark ellipses) on their gate range
    sel = np.nonzero(prm["void_on"][layer].ravel()[cell])[0]
    dvx = (px[sel] - g("vx", sel)) / g("vrx", sel)
    dvy = (py[sel] - g("vy", sel)) / g("vry", sel)
    hit = sel[dvx * dvx + dvy * dvy <= 1.0]
    img[iy[hit], ix[hit]] = die * 0.35

    # Corner delamination (bright quarter superellipse) from the gate it starts on
    sel = np.nonzero(prm["corner_on"][layer].ravel()[cell])[0]
    ck = g("ck", sel)
    right = (ck == 1) | (ck == 2)
    bottom = ck >= 2
    qx = np.where(right, hi - px[sel], px[sel] - lo)
    qy = np.where(bottom, hi - py[sel], py[sel] - lo)
    pw = g("cp", sel)
    inq = (qx / g("crx", sel)) ** pw + (qy / g("cry", sel)) ** pw <= 1.0
    hit = sel[inq]
    img[iy[hit], ix[hit]] = min(60000.0, die * 1.45)

    img += rng.standard_normal(img.shape, dtype=np.float32) * np.float32(noise * 257.0)
    if dtype == np.uint8:
        img *= 1.0 / 257.0
        return np.clip(img, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sample_14900x14900x12_8bit.tif")
    ap.add_argument("--size", type=int, default=14900, help="width and height (px)")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dtype", choices=("uint8", "uint16"), default="uint8", help="single-channel gray")
    ap.add_argument("--pitch", type=float, default=500.0, help="cell pitch (px)")
    ap.add_argument("--bbox", type=float, default=480.0, help="package bbox size inside a cell (px)")
    ap.add_argument("--sl", type=float, default=20.0, help="scribe line width (px)")
    ap.add_argument("--origin", type=float, default=200.0, help="grid origin x = y (px)")
    ap.add_argument("--angle", type=float, default=0.5, help="grid rotation (deg)")
    ap.add_argument("--compress", choices=("lzw", "zlib", "none"), default="lzw",
                    help="lzw/zlib: decoded per layer by the viewer; none: memory-mapped (lzw needs imagecodecs)")
    ap.add_argument("--noise", type=float, default=3.0, help="noise std in 8-bit gray levels (sets the compressed size)")
    ap.add_argument("--rows-per-strip", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    width = height = a.size
    dtype = np.uint8 if a.dtype == "uint8" else np.uint16
    rows = int((height - a.origin) // a.pitch)
    cols = int((width - a.origin) // a.pitch)
    rad = math.radians(a.angle)
    geo = {
        "img_cx": width / 2.0, "img_cy": height / 2.0, "cos": math.cos(rad), "sin": math.sin(rad),
        "ox": a.origin, "oy": a.origin, "pitch": a.pitch, "rows": rows, "cols": cols,
        "margin": (a.pitch - a.bbox) / 2.0, "bbox": a.bbox, "sl": a.sl,
    }
    rng = np.random.default_rng(a.seed)
    prm = chip_params(rng, rows, cols, a.layers, geo["margin"] + a.sl, geo["margin"] + a.bbox - a.sl)

    size_gb = width * height * np.dtype(dtype).itemsize * a.layers / 1e9
    print(f"Writing {a.out}: {width}x{height}, {a.layers} layers, {a.dtype}, compress={a.compress} "
          f"(~{size_gb:.1f} GB uncompressed)")
    compression = None if a.compress == "none" else a.compress
    rps = a.rows_per_strip
    t0 = time.time()
    with tifffile.TiffWriter(a.out, bigtiff=True) as tw:
        for layer in range(a.layers):
            strips = (make_strip(y, min(y + rps, height), width, layer, a.layers, geo, prm, rng, dtype, a.noise)
                      for y in range(0, height, rps))
            if compression:
                # tifffile compresses whole pages: assemble this layer (one layer in memory)
                page = np.empty((height, width), dtype=dtype)
                for y, strip in zip(range(0, height, rps), strips):
                    page[y:y + strip.shape[0]] = strip
                tw.write(page, rowsperstrip=rps, compression=compression,
                         photometric="minisblack", contiguous=False, metadata=None)
                del page
            else:
                # Uncompressed: strips are streamed straight to the file
                tw.write(strips, shape=(height, width), dtype=dtype, rowsperstrip=rps,
                         photometric="minisblack", contiguous=False, metadata=None)
            print(f"  layer {layer + 1}/{a.layers} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"Done: {os.path.abspath(a.out)} ({os.path.getsize(a.out) / 1e9:.2f} GB, {time.time() - t0:.0f}s)")
    print("Grid for the viewer (Grid Configuration, or a products.json entry):")
    print(json.dumps({"grid": {"start_x": a.origin, "start_y": a.origin, "pitch_x": a.pitch, "pitch_y": a.pitch,
                               "rows": rows, "cols": cols, "angle": a.angle},
                      "bbox": {"w": a.bbox, "h": a.bbox},
                      "layout": {"design_size": [a.bbox, a.bbox], "SL": [a.sl, a.sl],
                                 "boxes": [{"name": "die", "inset": "SL"}]}}, indent=2))
    print(f"Reference bbox top-left inside each cell: ({geo['margin']:.0f}, {geo['margin']:.0f}) from the cell origin")


if __name__ == "__main__":
    sys.exit(main())
