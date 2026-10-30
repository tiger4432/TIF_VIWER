"""
Experiment A: tools/train_seg.py (same U-Net, loss, ONNX export and sidecar) with the layer-change
channels computed once per sample instead of per batch (void_seg.layer_change is the slow part of
--neighbors 1). Synthetic voids are added to the cached change maps linearly: a void v pasted into
the chip adds v to both changes (the neighbour lacks it), a void v2 (other size / shade) pasted into
the neighbour too subtracts v2 (layer_change is the residual of chip - neighbour after a robust
layout fit, so it is close to linear in small local changes).

    python tools/synth/experiments/A/train_cached.py --data SEG_DIR --out models/x.onnx [--label all]
           [--epochs 40 --steps 100 --base 16 --crop 128 --change-norm 0]

--label all: the data marks a void on every layer it is visible on (extract.py --label all); then
synthetic voids are also pasted into the layer above half of the time (a void seen from above).
--change-norm 1: each change channel divided by its per-pixel chip-to-chip spread over the layer
(void_seg.change_spread; the viewer does the same when the sidecar says "change_norm").
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from sat_widgets import void_seg # noqa: E402
from sat_widgets.void_detect import _blur, _components, _translate # noqa: E402
if os.environ.get("A_CACHE_ONLY"): # Precompute the layer changes without torch installed
    import types
    _t = types.ModuleType("torch"); _t.nn = types.ModuleType("nn"); _t.nn.Module = object; _t.nn.functional = None
    sys.modules.update({"torch": _t, "torch.nn": _t.nn, "torch.nn.functional": types.ModuleType("F")})
import train_seg # noqa: E402
from train_seg import UNet, seg_loss, appearance, split # noqa: E402
import torch # noqa: E402


def _changes(args):
    m, above, below = args
    return [None if o is None else void_seg.layer_change(m, o).astype(np.float16) for o in (above, below)]


def chip_stacks(root):
    """Per sample (load_data order): (files of all layers of its chip, top first; index of its layer)."""
    out = []
    for folder, index in void_seg.list_sources(root):
        if not index.get("include", True):
            continue
        chips = {}
        for s in index["samples"]:
            chips.setdefault((s["row"], s["col"]), {})[s["layer"]] = s["id"]
        for s in index["samples"]:
            if not s.get("include", True):
                continue
            ls = sorted(chips[(s["row"], s["col"])])
            out.append(([os.path.join(folder, chips[(s["row"], s["col"])][lr] + ".npz") for lr in ls], ls.index(s["layer"])))
    return out


def _residuals(args):
    files, k = args
    return [c.astype(np.float16) for c in void_seg.stack_residual([np.load(f)["m"].astype(np.float32) for f in files], k)]


def load_cached(root, workers, change_norm, stack=False):
    """[(group, m, y, golden, spread, change a | None, change b | None, spread a, spread b)]; stack: the
    two channels are void_seg.stack_residual (all other layers, layers above) instead of layer_change."""
    cache = os.path.join(root, "_cache_residuals.npz" if stack else "_cache_changes.npz")
    samples = train_seg.load_data(root, not stack)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        ch = list(z["ch"])
    else:
        t0 = time.time()
        with Pool(workers) as pool:
            if stack:
                ch = pool.map(_residuals, chip_stacks(root), chunksize=16)
            else:
                ch = pool.map(_changes, [(s[1], *s[5]) for s in samples], chunksize=16)
        np.savez(cache, ch=np.array(ch, dtype=object))
        print(f"layer changes of {len(samples)} samples: {time.time() - t0:.0f}s", flush=True)
    out = [(s[0], s[1], s[2], s[3], s[4], c[0], c[1], 1.0, 1.0) for s, c in zip(samples, ch)]
    if change_norm:
        # Per source + layer + direction: per-pixel spread of the change maps over the chips
        groups = {}
        for i, s in enumerate(out):
            groups.setdefault((s[0].split("|")[0], id(s[3])), []).append(i)
        for idx in groups.values():
            spreads = [void_seg.change_spread([out[i][5 + k] for i in idx]) for k in (0, 1)]
            for i in idx:
                out[i] = out[i][:7] + tuple(spreads)
    return out


def soft_void(h, w, rng, v=None, scale=1.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if v is None:
        rx, ry = rng.uniform(3, min(30, w / 4)), rng.uniform(3, min(30, h / 4))
        v = {"cx": rng.uniform(rx, w - rx), "cy": rng.uniform(ry, h - ry), "rx": rx, "ry": ry,
             "p": float(rng.choice([2.0, 2.0, rng.uniform(1.2, 4.0)]))}
    t = (np.abs((xx - v["cx"]) / (v["rx"] * scale)) ** v["p"] + np.abs((yy - v["cy"]) / (v["ry"] * scale)) ** v["p"]) ** (1.0 / v["p"])
    soft = _blur(np.clip((1.25 - t) / 0.5, 0, 1).astype(np.float32), rng.uniform(0.5, 2.5))
    return v, t, soft


def make_batch(samples, n, crop, rng, synth_p, label_all, stack=False):
    xs, ys = [], []
    for _ in range(n):
        _, m, y, g, sp, ca, cb, sa, sb = samples[rng.integers(len(samples))]
        h, w = m.shape
        m = m.copy(); y = y.copy()
        ca = np.zeros((h, w), np.float32) if ca is None else ca.astype(np.float32)
        cb = np.zeros((h, w), np.float32) if cb is None else cb.astype(np.float32)
        has_a, has_b = bool(ca.any()), bool(cb.any())
        if stack:
            has_a = has_b = True
        if rng.random() < synth_p:
            noise = float(np.median(sp))
            for _ in range(rng.integers(1, 4)):
                v, t, soft = soft_void(h, w, rng)
                amp = math.exp(rng.uniform(math.log(0.3), math.log(3.0))) * rng.choice([-1.0, 1.0]) * noise
                m += amp * soft
                if stack:
                    # Residual channels (chip sigma units, about the spread): against all other layers the
                    # void is partly explained by its deeper copies; against the layers above only when
                    # it was there already (label all: seen from above, half of the time)
                    za = amp / noise * soft
                    zo = (lambda: za - amp / noise * rng.uniform(-1.0, 1.0)
                          * soft_void(h, w, rng, v, math.exp(rng.uniform(math.log(0.4), math.log(2.5))))[2])
                    ca += zo()
                    cb += zo() if (label_all and rng.random() < 0.5) else za
                    y[(t <= void_seg.CORE) & (y != 1)] = 1
                    y[(t > void_seg.CORE) & (t <= void_seg.BAND) & (y == 0)] = 255
                    continue
                da = amp * soft
                if label_all and has_a and rng.random() < 0.5:
                    # Seen from above too (other size / shade): the change against above is the difference
                    _, _, s2 = soft_void(h, w, rng, v, math.exp(rng.uniform(math.log(0.4), math.log(2.5))))
                    da = da - amp * rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5) * s2
                if has_a:
                    ca += da
                if has_b:
                    _, _, s2 = soft_void(h, w, rng, v, math.exp(rng.uniform(math.log(0.4), math.log(2.5))))
                    cb += amp * soft - amp * rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5) * s2
                y[(t <= void_seg.CORE) & (y != 1)] = 1
                y[(t > void_seg.CORE) & (t <= void_seg.BAND) & (y == 0)] = 255
        m_in, g_in = appearance(m, rng), appearance(g, rng)
        chans = [m_in, g_in, void_seg.diff(m_in, g_in, sp)]
        for c in (ca / sa, cb / sb):
            c = c * rng.uniform(0.8, 1.25)
            if rng.random() < 0.3:
                c = _blur(c.astype(np.float32), rng.uniform(0.3, 1.2))
            chans.append(c)
        x = np.stack([np.clip(c, -10, 10) for c in chans]).astype(np.float32) / 5.0
        ph, pw = max(0, crop - h), max(0, crop - w)
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            y = np.pad(y, ((0, ph), (0, pw)), constant_values=255)
            h, w = y.shape
        vy, vx = np.nonzero(y == 1)
        if len(vy) and rng.random() < 0.5:
            k = rng.integers(len(vy))
            r0 = int(np.clip(vy[k] - rng.integers(crop), 0, h - crop)); c0 = int(np.clip(vx[k] - rng.integers(crop), 0, w - crop))
        else:
            r0, c0 = int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1))
        x, y = x[:, r0:r0 + crop, c0:c0 + crop], y[r0:r0 + crop, c0:c0 + crop]
        k = rng.integers(4)
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys).astype(np.int64))


def cached_input(s):
    _, m, y, g, sp, ca, cb, sa, sb = s
    chans = [m, g, void_seg.diff(m, g, sp)] + [np.zeros_like(m) if c is None else c.astype(np.float32) / q for c, q in ((ca, sa), (cb, sb))]
    return np.stack([np.clip(c, -10, 10) for c in chans]).astype(np.float32) / 5.0


def evaluate(model, samples, thr=0.5):
    model.eval()
    found = total = det = det_ok = 0
    with torch.no_grad():
        for i in range(0, len(samples), 16):
            chunk = samples[i:i + 16]
            x = np.stack([cached_input(s) for s in chunk])
            h, w = x.shape[2:]
            ph, pw = (-h) % void_seg.PAD, (-w) % void_seg.PAD
            out = model(torch.from_numpy(np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="edge")))[:, 0, :h, :w].numpy()
            for s, o in zip(chunk, out):
                y = s[2]
                pred = o > math.log(thr / (1 - thr))
                t = y == 1
                comps_t = _components(t) if t.any() else []
                total += len(comps_t)
                found += sum(1 for r0, r1, c0, c1, _ in comps_t if pred[r0:r1 + 1, c0:c1].any())
                for r0, r1, c0, c1, area in (_components(pred) if pred.any() else []):
                    if area < 10:
                        continue
                    det += 1
                    det_ok += int((y[r0:r1 + 1, c0:c1] != 0).any())
    model.train()
    r, p = found / max(total, 1), det_ok / max(det, 1)
    return {"void_recall": r, "void_precision": p, "void_f1": 2 * r * p / max(r + p, 1e-9), "voids": total, "dets": det}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sources", default="", help="comma separated substrings of the sources to use (default all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", choices=["first", "all"], default="first")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--synth", type=float, default=0.5)
    ap.add_argument("--change-norm", type=int, default=0)
    ap.add_argument("--stack", type=int, default=0, help="1: residual against the other layers / the layers above (void_seg.stack_residual)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get("A_CACHE_ONLY"):
        load_cached(a.data, a.threads, 0, bool(a.stack))
        return
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    samples = load_cached(a.data, a.threads, a.change_norm, bool(a.stack))
    if a.sources:
        keys = a.sources.split(",")
        samples = [s for s in samples if any(k in s[0].split("|")[0] for k in keys)]
    train, val = split(samples, a.val, a.seed)
    n_void = sum(int((s[2] == 1).any()) for s in samples)
    print(f"{len(samples)} samples ({n_void} with voids): {len(train)} train, {len(val)} val", flush=True)
    model = UNet(5, a.base)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps)
    best, best_state = None, None
    t0 = time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for _ in range(a.steps):
            xb, yb = make_batch(train, a.batch, a.crop, rng, a.synth, a.label == "all", bool(a.stack))
            loss = seg_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        res = evaluate(model, val)
        if best is None or res["void_f1"] > best[0]:
            best = (res["void_f1"], ep, res)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / a.steps:.4f} val {json.dumps({k: round(v, 3) for k, v in res.items()})} "
              f"{time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    kw = dict(input_names=["input"], output_names=["output"], opset_version=17,
              dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}})
    torch.onnx.export(model, torch.zeros(1, 5, a.crop, a.crop), a.out, dynamo=False, **kw)
    info = {"mode": "seg", "inputs": "chip, golden, (chip - golden) / spread, " + ("residual against the other layers / the layers above"
                                     if a.stack else "layer change above / below") + "; clip +-10, / 5",
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "samples": len(samples), "samples_with_voids": n_void,
            "neighbor_layers": 1, "change_norm": a.change_norm, "stack_residual": a.stack, "channels": 5, "base": a.base, "epochs": a.epochs,
            "label": a.label, "best_epoch": best[1] + 1, "validation": best[2], "train_seconds": round(time.time() - t0),
            "suggested_threshold": 5}
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(f"best epoch {best[1] + 1}: {best[2]}\nSaved {a.out} (+ .json), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
