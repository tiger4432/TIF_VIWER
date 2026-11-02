"""Multi-layer synthetic SAT wafer for method comparisons.

    python tools/synth/make_wafer.py OUTDIR NAME          (env SEED, NL, VOID_R; deterministic per SEED)
VOID_R="lo,hi": void radius range in chip px (default 6,25; e.g. 3,8 for small voids).
LAYOUT_FROM=k: layout (die, lines, bumps, pad) only from layer k on; the layers above show the
    brightness gradient, background and noise only (x LAYOUT_RESID of the layout, default 0.03), as
    real top layers do. Same random draws as without it: the same seed gives the same voids.
    -> OUTDIR/NAME.tif (8-bit pages = layers) and OUTDIR/NAME_truth.json

Grid: origin 100, pitch 300 px, 14 x 14 chips, angle 0.4 deg; chip (bbox) 280 px centred in the cell.
truth: "row,col" -> [cx, cy, rx, ry (chip px from the bbox corner), first layer, last layer, contrast].
Per chip (same on all layers): line strengths, blur varying across the chip, brightness gradient in
a random direction, wavy long-period background. Per chip AND layer: brightness / contrast /
polarity of each structure, gamma. Voids: appear on a layer and stay on every deeper layer, with
per-layer shade (incl. polarity) and size (x0.4-2.5)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
W, name = sys.argv[1], sys.argv[2]
import numpy as np, tifffile
from PIL import Image
from sat_widgets.core_data import CoordinateTransform
from sat_widgets.void_detect import _blur

t0 = time.time()
rng = np.random.default_rng(int(os.environ.get("SEED", 31)))
P, B, ORG, ANG = 300, 280, 100, 0.4
R = C = 14
NL = int(os.environ.get("NL", 8))
LAYOUT_FROM = int(os.environ.get("LAYOUT_FROM", 0))
LAYOUT_RESID = float(os.environ.get("LAYOUT_RESID", 0.03))
R_LO, R_HI = (float(v) for v in os.environ.get("VOID_R", "6,25").split(","))
SIZE = ORG * 2 + P * R
m = (P - B) / 2
yy, xx = np.mgrid[0:B, 0:B].astype(np.float32)
line_maps = []
for y0 in range(20, B - 10, 35):
    a = np.zeros((B, B), np.float32); a[(np.abs(yy - y0) < 2) & (xx > 10) & (xx < B - 10)] = 1; line_maps.append(a)
for x0 in range(25, B - 10, 35):
    a = np.zeros((B, B), np.float32); a[(np.abs(xx - x0) < 2) & (yy > 10) & (yy < B - 10)] = 1; line_maps.append(a)
bumps = np.zeros((B, B), np.float32)
for cx in range(42, B - 30, 35):
    for cy in range(37, B - 30, 35):
        bumps[(xx - cx) ** 2 + (yy - cy) ** 2 < 36] = 1
pad = np.zeros((B, B), np.float32); pad[(xx > 190) & (xx < 250) & (yy > 200) & (yy < 240)] = 1
die = np.zeros((B, B), np.float32); die[(xx > 8) & (xx < B - 8) & (yy > 8) & (yy < B - 8)] = 1


def wavy(scale, amp):
    g = rng.normal(0, 1, (scale, scale)).astype(np.float32)
    f = np.asarray(Image.fromarray(g).resize((B, B), Image.BICUBIC), np.float32)
    return _blur(f, B / scale / 3) * amp


def structure():
    lines = sum(lm * rng.uniform(0.4, 1.0) for lm in line_maps)
    s1, s2 = rng.uniform(1.0, 2.5), rng.uniform(3.0, 6.5)
    wgt = np.clip(0.5 + 0.5 * np.sin(xx / B * rng.uniform(1, 4) + rng.uniform(0, 6)) * np.cos(yy / B * rng.uniform(1, 4)), 0, 1)
    lines = (1 - wgt) * _blur(lines, s1) + wgt * _blur(lines, s2) * 1.6
    a = rng.uniform(0, 2 * np.pi)
    light = 1 + ((xx - B / 2) * np.cos(a) + (yy - B / 2) * np.sin(a)) / B * rng.uniform(0.1, 0.5) \
        + wavy(int(rng.integers(3, 6)), rng.uniform(0.04, 0.12))
    return lines, _blur(bumps, rng.uniform(0.8, 2.5)), light


def layer_mix(st, a=1.0):
    """a: layout visibility (0 = flat chip, background and gradient only)."""
    lines, bl, light = st
    v = (rng.uniform(0.35, 0.75) + a * (rng.uniform(-0.25, 0.25) * die + rng.uniform(-0.45, 0.35) * bl
         + rng.choice([-1, 1]) * rng.uniform(0.05, 0.35) * lines + rng.uniform(-0.4, 0.4) * pad))
    return np.clip(v * light, 0.02, None) ** rng.uniform(0.7, 1.4)


STRUCT = [[structure() for c in range(C)] for r in range(R)]
truth = {}
for r in range(R):
    for c in range(C):
        if rng.random() < 0.3:
            l0 = int(rng.integers(0, NL))
            truth[(r, c)] = (float(rng.uniform(35, B - 35)), float(rng.uniform(35, B - 35)),
                             float(rng.uniform(R_LO, R_HI)), float(rng.uniform(R_LO, R_HI)), l0, NL - 1,
                             float(rng.uniform(0.06, 0.14)))
ct = CoordinateTransform(ANG, SIZE, SIZE)
json.dump({f"{r},{c}": t for (r, c), t in truth.items()}, open(os.path.join(W, name + "_truth.json"), "w"))
ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
u, v = ct.raw_to_deskew(xs, ys)
del xs, ys
u -= ORG; v -= ORG
col = np.floor(u / P).astype(np.int32); row = np.floor(v / P).astype(np.int32)
lx = u - col * P - m; ly = v - row * P - m
del u, v
ok = (col >= 0) & (col < C) & (row >= 0) & (row < R) & (lx >= 0) & (ly >= 0) & (lx < B) & (ly < B)
rr, cc, lxo, lyo = row[ok], col[ok], lx[ok], ly[ok]
del row, col, lx, ly
cid = rr * C + cc
order = np.argsort(cid, kind="stable")
bounds = np.searchsorted(cid[order], np.arange(R * C + 1))
cell_idx = {(r, c): order[bounds[r * C + c]:bounds[r * C + c + 1]] for (r, c) in truth}
with tifffile.TiffWriter(os.path.join(W, name + ".tif")) as tw:
    for k in range(NL):
        jit = rng.normal(0, 0.6, (R, C, 2)); gain = rng.uniform(0.85, 1.15, (R, C))
        lxi = np.clip(np.round(lxo + jit[rr, cc, 0]).astype(np.int32), 0, B - 1)
        lyi = np.clip(np.round(lyo + jit[rr, cc, 1]).astype(np.int32), 0, B - 1)
        a = 1.0 if k >= LAYOUT_FROM else LAYOUT_RESID
        LAYS = np.stack([np.stack([layer_mix(STRUCT[r][c], a) for c in range(C)]) for r in range(R)])
        val = LAYS[rr, cc, lyi, lxi] * gain[rr, cc]
        for (r, c), (vx, vy, vrx, vry, l0, l1, f) in truth.items():
            if k >= l0:
                fk = 1 + rng.choice([-1, 1]) * rng.uniform(0.5, 1.3) * f
                sk = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5)))) if k > l0 else 1.0
                ix = cell_idx[(r, c)]
                d = ((lxo[ix] - vx) / (vrx * sk)) ** 2 + ((lyo[ix] - vy) / (vry * sk)) ** 2
                soft = np.clip((1.3 - d) / 0.6, 0, 1)
                val[ix] = val[ix] * (1 + (fk - 1) * soft)
        img = np.zeros((SIZE, SIZE), np.float32)
        img[ok] = val
        img = img * 160 + rng.normal(0, 4, img.shape).astype(np.float32)
        tw.write(np.clip(img, 0, 255).astype(np.uint8), contiguous=False)
print(f"written {name}: {len(truth)} voids, {NL} layers, {time.time() - t0:.0f}s")
