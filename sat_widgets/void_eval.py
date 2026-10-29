"""
Evaluation of Detect Voids against the hand markings, with the reason for every miss and false
detection, so the parameters (or the algorithm) can be tuned from the field.

Run from the Void Marking panel ("Evaluate"): the detection runs exactly like Detect Voids but
adds nothing; its result is compared with the markings drawn by hand (not the dashed detected
ones, not bboxes) on the loaded layers. Output:

- a short code (code) meant to be read on screen and typed on a phone (letters, digits, dots,
  spaces only),
  see docs/VOID_EVAL.md for every field,
- a folder with report.html (tables + chip images), eval.json and summary.txt.

Detection stages seen per chip (sat_widgets/void_detect.py): deviation map -> threshold -> 3x3
opening -> blobs merged -> size filter (min_size_px, max_area_frac) -> layer rule (a spot
found on several layers is marked on one of them: first, strongest or last) -> kept.
"""
import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from .void_detect import _components, _phase_shift as _shift

VERSION = 1
MAX_IMAGES = 150 # Chip images written to the report
SHIFT_SAMPLE = 40 # Chips per layer whose alignment is measured besides the ones of interest

# Miss reasons, in the order they are checked (first that applies)
MISS_REASONS = {
    "nl": "layer not searched (skipped, not loaded, or fewer than 3 chips)",
    "ni": "cell not inspected (not in the bonding map / rejected LEG)",
    "mg": "merged into one detection with another hand marking",
    "dup": "skipped: the spot is marked on another layer (Mark on layer rule)",
    "lg": "blob larger than max_area_frac (chip treated as misaligned / broken)",
    "sm": "blob smaller than min_size_px",
    "cl": "candidate rejected by the learned filter",
    "op": "above the threshold, but too thin / small for the 3x3 opening",
    "bt": "deviation below the threshold",
}
# False detection classes (first that applies)
FP_REASONS = {
    "ed": "touches the chip border (bbox / alignment)",
    "al": "chip still misaligned against the median chip after registration (> 1.5 working px)",
    "lg": "large blob (> 2% of the chip)",
    "le": "on a layout edge (chip-to-chip spread > 2x the layer median)",
    "cm": "common spot (> 15% of the chips deviate there)",
    "ot": "other (may be a real void that is not marked)",
}


def _fmt(x, digits=2):
    """Compact number: .62, 1.9, 12 (no leading zero, trailing zeros dropped)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "na"
    if abs(x) >= 10 or float(x).is_integer():
        return str(int(round(x)))
    s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return s[1:] if s.startswith("0.") else ("-" + s[2:] if s.startswith("-0.") else s)


def _med(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _boxes_meet(a, b, tol=0.0):
    return a[0] - tol <= b[2] and b[0] - tol <= a[2] and a[1] - tol <= b[3] and b[1] - tol <= a[3]


class Evaluation:
    """
    Collector passed to void_detect.detect_voids(diag=...), then finish() compares with the hand
    markings. truths: [{"layer", "cell": (row, col), "box": (x0, y0, x1, y1) Deskewed (the
    drawn extent), "type"}].
    """

    def __init__(self, truths, layers, method_label):
        self.truths = [dict(t, i=i) for i, t in enumerate(truths)]
        self.by_lc = {}
        for t in self.truths:
            self.by_lc.setdefault((t["layer"], tuple(t["cell"])), []).append(t)
        self.layers = list(layers)
        self.method = method_label
        self.layer_stats = {}
        self.dets = []
        self.kept_comps = [] # Blobs kept on their layer; the layer rule may still drop them (finish)
        self.chips = {} # (layer, cell) -> arrays for the images
        self.t0 = time.time()

    # ----- hooks called by detect_voids -----
    def setup(self, p, size, factor, rects, min_area, max_area, gap):
        self.p, self.size, self.factor, self.rects = dict(p), size, factor, rects
        self.min_area, self.max_area, self.gap = min_area, max_area, gap
        self.refs_frac = None

    def layer_start(self, layer, chips, threshold):
        self.layer = layer
        self.thr = threshold
        self.chip_src = chips
        keys = list(chips)
        idx = np.linspace(0, len(keys) - 1, min(len(keys), 48)).astype(int)
        self.ref = np.median(np.stack([chips[keys[i]].astype(np.float32) for i in idx]), axis=0)
        self.common = np.zeros(self.ref.shape, np.int32)
        self.gold = self.spread = None
        self.sp_med = None
        self.cur = {"n": len(chips), "res": [], "shift": [], "maxdev": [], "cand": 0, "kept": 0}
        self.cur_truth = []
        self.cur_dets = []
        self.n_shift = 0

    def golden(self, golden, spread):
        self.gold = np.asarray(golden, np.float32)
        self.spread = np.asarray(spread, np.float32)
        self.sp_med = float(np.median(self.spread))

    def chip(self, layer, cell, dev, signed, matched, mask0, mask, comps):
        thr = self.thr
        dev = np.asarray(dev, np.float32)
        self.common += dev > thr / 2.0
        st = self.cur
        st["maxdev"].append(float(dev.max()))
        st["cand"] += len(comps)
        img = matched if matched is not None else self.chip_src[cell].astype(np.float32)
        raw = self.chip_src[cell].astype(np.float32) # As cut by the bbox (before any registration)
        if self.gold is not None and matched is not None:
            st["res"].append(float(np.median(np.abs(matched - self.gold))))
        truths = self.by_lc.get((layer, cell), [])
        kept = [c for c in comps if c.get("status") == "kept"]
        interesting = bool(truths or comps)
        shift = None
        if interesting or self.n_shift < SHIFT_SAMPLE:
            self.n_shift += 1
            shift = _shift(raw, self.ref)
            st["shift"].append(math.hypot(*shift))
        # Misalignment left after the detector's own registration (classifies false detections)
        left = None
        if kept:
            left = _shift(matched, self.gold) if (matched is not None and self.gold is not None) else shift
        x0, y0, w, h = self.rects[cell]
        W, H = self.size
        sx, sy = w / W, h / H
        border = 1

        def to_work(box):
            return ((box[0] - x0) / sx, (box[1] - y0) / sy, (box[2] - x0) / sx, (box[3] - y0) / sy)

        def peak_in(c0, r0, c1, r1):
            c0, r0 = max(0, int(math.floor(c0))), max(0, int(math.floor(r0)))
            c1, r1 = min(W, int(math.ceil(c1))), min(H, int(math.ceil(r1)))
            if c1 <= c0 or r1 <= r0:
                return None
            sub = dev[r0:r1, c0:c1]
            iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
            return (r0 + iy, c0 + ix, float(sub[iy, ix]), (r0, r1, c0, c1))

        half = None

        def half_box(pr, pc):
            """Extent (Deskewed) of the region above thr / 2 around the peak."""
            nonlocal half
            if half is None:
                half = _components(dev > thr / 2.0)
            best = None
            for r0, r1, c0, c1, area in half:
                if r0 <= pr <= r1 and c0 <= pc < c1:
                    if best is None or area < best[4]:
                        best = (r0, r1, c0, c1, area)
            if best is None:
                return None
            r0, r1, c0, c1, _ = best
            return (x0 + c0 * sx, y0 + r0 * sy, x0 + c1 * sx, y0 + (r1 + 1) * sy)

        def at(arr, r, c):
            return None if arr is None else float(arr[r, c])

        for t in truths:
            wb = to_work(t["box"])
            pk = peak_in(wb[0] - 1, wb[1] - 1, wb[2] + 1, wb[3] + 1)
            rec = {"inspected": True, "size_w": max(wb[2] - wb[0], wb[3] - wb[1])}
            if pk is not None:
                pr, pc, pv, (r0, r1, c0, c1) = pk
                rec.update(peak=pv / thr, pos=(pr, pc),
                           above=bool(mask0[r0:r1, c0:c1].any()), opened=bool(mask[r0:r1, c0:c1].any()),
                           spread=None if self.spread is None else at(self.spread, pr, pc) / max(self.sp_med, 1e-6),
                           contrast=None if self.gold is None or matched is None else abs(at(matched, pr, pc) - at(self.gold, pr, pc)),
                           polarity=None if signed is None else ("bright" if signed[pr, pc] > 0 else "dark"),
                           half=half_box(pr, pc) if pv > thr / 2.0 else None)
                # Blobs over the marking; their final status is read in finish (layer rule)
                rec["comp_refs"] = [c for c in comps
                                    if c["box"][0] <= r1 - 1 and r0 <= c["box"][1] and c["box"][2] <= c1 - 1 and c0 <= c["box"][3] - 1]
            t["rec"] = rec
            self.cur_truth.append(t)

        for c in kept:
            r0, r1, c0, c1 = c["box"]
            sub = dev[r0:r1 + 1, c0:c1]
            iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
            pr, pc = r0 + iy, c0 + ix
            d = {"layer": layer, "cell": cell, "box": tuple(c["extent"]), "score": c["score"] / thr,
                 "area_frac": c["area"] / float(W * H), "merged": c["merged"], "type": c.get("type"),
                 "edge": r0 <= border or c0 <= border or r1 >= H - 1 - border or c1 >= W - border,
                 "shift": None if left is None else math.hypot(*left),
                 "spread": None if self.spread is None else at(self.spread, pr, pc) / max(self.sp_med, 1e-6),
                 "pos": (pr, pc), "half": half_box(pr, pc)}
            c["_det"] = d
            self.cur_dets.append(d)
            self.kept_comps.append(c)
        st["kept"] += len(kept)

        if interesting:
            # Arrays for the report images (written for the most useful chips at the end)
            self.chips[(layer, cell)] = {
                "img": np.asarray(img, np.float16), "dev": dev.astype(np.float16),
                "mask0": np.asarray(mask0, bool), "mask": np.asarray(mask, bool),
                "comps": comps, "thr": thr, "rect": (x0, y0, sx, sy), "shift": shift}

    def layer_end(self, layer):
        n = max(1, self.cur["n"])
        for t in self.cur_truth:
            pos = t["rec"].get("pos")
            t["rec"]["common"] = None if pos is None else float(self.common[pos]) / n
        for d in self.cur_dets:
            d["common"] = float(self.common[d["pos"]]) / n
        st = self.cur
        self.layer_stats[layer] = {
            "n": st["n"], "thr": self.thr,
            "spread": self.sp_med,
            "residual": _med(st["res"]),
            "shift": _med(st["shift"]),
            "shift95": float(np.percentile(st["shift"], 95)) if st["shift"] else None,
            "cand": st["cand"], "kept": st["kept"],
            "maxdev": _med(st["maxdev"]),
        }
        if self.gold is not None:
            self.layer_stats[layer]["_gold"] = self.gold.astype(np.float16)
            self.layer_stats[layer]["_spread"] = self.spread.astype(np.float16)
        self.chip_src = None

    # ----- comparison -----
    def finish(self, cells):
        """Matches detections with hand markings and assigns the reasons. Returns the summary dict."""
        cells = set(map(tuple, cells))
        # Detections: blobs still kept after the layer rule; statuses of the blobs over each marking
        self.dets = [c["_det"] for c in self.kept_comps if c.get("status") == "kept"]
        for t in self.truths:
            rec = t.get("rec")
            if rec and "comp_refs" in rec:
                refs = rec.pop("comp_refs")
                rec["comps"] = [c.get("status") for c in refs]
                rec["dup_layers"] = [c.get("dup_layer") for c in refs if c.get("status") == "dup"]
        tol = self.factor # One working pixel (Deskewed px, about)
        for t in self.truths:
            t["match"] = None
            t.setdefault("rec", None)
        for d in self.dets:
            d["match"] = None
        # 1. Same layer, same cell, boxes meet: nearest centers first
        pairs = []
        for t in self.truths:
            tc = ((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2)
            for j, d in enumerate(self.dets):
                if d["cell"] == tuple(t["cell"]) and _boxes_meet(t["box"], d["box"], tol):
                    dc = ((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2)
                    pairs.append((d["layer"] != t["layer"], math.hypot(tc[0] - dc[0], tc[1] - dc[1]), t["i"], j))
        pairs.sort()
        for other_layer, dist, i, j in pairs:
            t, d = self.truths[i], self.dets[j]
            if t["match"] is None and d["match"] is None:
                kind = "ly" if other_layer else "ok"
                t["match"], d["match"] = (kind, j), (kind, i)

        # 2. Reasons
        for t in self.truths:
            if t["match"] is not None:
                t["result"] = t["match"][0]
                continue
            rec = t["rec"]
            if t["layer"] not in self.layer_stats:
                reason = "nl"
            elif tuple(t["cell"]) not in cells or rec is None:
                reason = "ni"
            else:
                comps = rec.get("comps", [])
                if "kept" in comps:
                    reason = "mg"
                elif "dup" in comps:
                    reason = "dup"
                elif "large" in comps:
                    reason = "lg"
                elif "small" in comps:
                    reason = "sm"
                elif "cls" in comps:
                    reason = "cl"
                elif rec.get("above") and not rec.get("opened"):
                    reason = "op"
                else:
                    reason = "bt"
            t["result"] = "ms"
            t["reason"] = reason
        for d in self.dets:
            if d["match"] is not None:
                d["result"] = d["match"][0]
                continue
            d["result"] = "fp"
            if d["edge"]:
                d["reason"] = "ed"
            elif d["shift"] is not None and d["shift"] > 1.5:
                d["reason"] = "al"
            elif d["area_frac"] > 0.02:
                d["reason"] = "lg"
            elif d["spread"] is not None and d["spread"] > 2.0:
                d["reason"] = "le"
            elif d.get("common", 0) > 0.15:
                d["reason"] = "cm"
            else:
                d["reason"] = "ot"
        return self.summary()

    def summary(self):
        T, D = self.truths, self.dets
        ok = [t for t in T if t.get("result") == "ok"]
        ly = [t for t in T if t.get("result") == "ly"]
        ms = [t for t in T if t.get("result") == "ms"]
        fp = [d for d in D if d.get("result") == "fp"]
        size_r = []
        for t in ok:
            d = self.dets[t["match"][1]]
            tw, th = t["box"][2] - t["box"][0], t["box"][3] - t["box"][1]
            dw, dh = d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]
            hw = None if d["half"] is None else (d["half"][2] - d["half"][0]) / max(tw, 1e-6)
            size_r.append((dw / max(tw, 1e-6), dh / max(th, 1e-6), hw))
            t["size_ratio"] = (dw / max(tw, 1e-6), dh / max(th, 1e-6))
        same_type = sum(1 for t in ok if (self.dets[t["match"][1]].get("type") or "").lower() == str(t.get("type") or "").lower())
        return {"h": len(T), "ok": len(ok), "ly": len(ly), "ms": len(ms), "fp": len(fp), "det": len(D),
                "size": size_r, "same_type": same_type, "elapsed": time.time() - self.t0}

    # ----- outputs -----
    def short_code(self, s):
        """
        Two lines to type by hand (see docs/VOID_EVAL.md "짧은 코드"):
          S1 <method><thr> <ok> <ly> <ms> <fp> <top 2 miss reasons>
          <pk> <s> <c> <z> <top 2 false classes> <w> <h>
        pk, s, c, w, h x100 as integers (medians of the threshold misses / matched sizes), 0 = none.
        """
        p = self.p
        m = ("l" if p.get("learned_model") is not None else "d") if p.get("method") == "die_to_die" else "p"
        if p.get("seg_model") is not None:
            m = "s" if p["seg_model"].mode == "seg" else "r"
        thr = p.get("pc_threshold") if p.get("method") == "patchcore" else p.get("threshold")
        top = lambda d: [f"{k}{v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:2]]
        miss, false = {}, {}
        for t in self.truths:
            if t.get("result") == "ms":
                miss[t["reason"]] = miss.get(t["reason"], 0) + 1
        for d in self.dets:
            if d.get("result") == "fp":
                false[d["reason"]] = false.get(d["reason"], 0) + 1
        bt = [t["rec"] for t in self.truths if t.get("reason") == "bt" and t.get("rec")]
        i100 = lambda v: "0" if v is None else str(int(round(100 * v)))
        b = [i100(_med([r.get(k) for r in bt])) for k in ("peak", "spread", "contrast")]
        z = _med([r.get("size_w") for r in bt])
        b.append("0" if z is None else str(int(round(z))))
        w = _med([r[0] for r in s["size"]]) if s["size"] else None
        h = _med([r[1] for r in s["size"]]) if s["size"] else None
        line1 = " ".join([f"S1 {m}{_fmt(thr)}", str(s["ok"]), str(s["ly"]), str(s["ms"]), str(s["fp"])] + top(miss))
        line2 = " ".join(b + top(false) + [i100(w), i100(h)])
        return line1 + "\n" + line2

    def code(self, s):
        """Short text for the phone (see docs/VOID_EVAL.md)."""
        p = self.p
        m = {"die_to_die": "dd"}.get(p.get("method"), "pc" + ("r" if "resnet" in str(p.get("pc_model", "")).lower() else "m"))
        if p.get("seg_model") is not None:
            m = "sg" if p["seg_model"].mode == "seg" else "rc"
        thr = p.get("pc_threshold") if p.get("method") == "patchcore" else p.get("threshold")
        layers = sorted(self.layer_stats) or self.layers
        n_cells = len(self.rects)
        rule = {"first": "rf", "last": "rl", "strongest": "rs"}.get(p.get("layer_rule", "first"), "r" + str(p.get("layer_rule"))[:1])
        for k in sorted(int(x) for x in (p.get("skip_layers") or [])):
            rule += f" sk{k}"
        if p.get("learned_model") is not None:
            rule += " lm"
        if p.get("method") == "die_to_die":
            rule += f" sm{_fmt(p.get('smooth', 0))}" + (" lf" if p.get("layout_fit", True) else "") + (" rg" if p.get("register", True) else "")
        elif p.get("pc_position_norm", False):
            rule += " pn"
        lines = [f"VD{VERSION} {m} {rule} t{_fmt(thr)} f{self.factor} w{max(self.size)} m{_fmt(p.get('min_size_px'))} "
                 f"c{n_cells} b{self.refs_pct} L{min(layers)}to{max(layers)} h{s['h']}" if layers else f"VD{VERSION} {m} no layers"]
        R = round(100 * s["ok"] / s["h"]) if s["h"] else None
        P = round(100 * s["ok"] / s["det"]) if s["det"] else None
        lines.append(f"T ok{s['ok']} ly{s['ly']} ms{s['ms']} fp{s['fp']}" + ("" if R is None else f" R{R}") + ("" if P is None else f" P{P}"))
        for L in layers:
            tl = [t for t in self.truths if t["layer"] == L]
            dl = [d for d in self.dets if d["layer"] == L]
            parts = [f"L{L}"]
            cnt = {"h": len(tl), "ok": sum(t.get("result") == "ok" for t in tl), "ly": sum(t.get("result") == "ly" for t in tl),
                   "ms": sum(t.get("result") == "ms" for t in tl), "fp": sum(d.get("result") == "fp" for d in dl)}
            parts += [f"{k}{v}" for k, v in cnt.items() if v or k == "h"]
            reasons = {}
            for t in tl:
                if t.get("result") == "ms":
                    reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
            parts += [f"{k}{v}" for k, v in reasons.items()]
            pk = _med([t["rec"].get("peak") for t in tl if t.get("reason") == "bt" and t.get("rec")])
            if pk is not None:
                parts.append(f"pk{_fmt(pk)}")
            st = self.layer_stats.get(L)
            if st:
                q = [f"n{st['n']}"]
                if st["spread"] is not None:
                    q.append(f"s{_fmt(st['spread'])}")
                if st["residual"] is not None:
                    q.append(f"r{_fmt(st['residual'])}")
                if st["shift"] is not None:
                    q.append(f"a{_fmt(st['shift'] * self.factor, 1)}")
                q.append(f"x{_fmt(st['maxdev'] / st['thr'] if st['maxdev'] is not None else None)}")
                parts += q
            lines.append(" ".join(parts))
        moved = {}
        for t in self.truths:
            if t.get("result") == "ly": # Hand layer -> detected layer
                k = f"L{t['layer']}to{self.dets[t['match'][1]]['layer']}"
                moved[k] = moved.get(k, 0) + 1
        if moved:
            lines.append("D " + " ".join(f"{k}n{v}" for k, v in sorted(moved.items(), key=lambda kv: -kv[1])[:6]))
        reasons = {}
        for t in self.truths:
            if t.get("result") == "ms":
                reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        if reasons:
            lines.append("M " + " ".join(f"{k}{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
        bt = [t["rec"] for t in self.truths if t.get("reason") == "bt" and t.get("rec")]
        if bt:
            lines.append("B " + " ".join(f"{k}{_fmt(v)}" for k, v in (
                ("pk", _med([r.get("peak") for r in bt])), ("s", _med([r.get("spread") for r in bt])),
                ("c", _med([r.get("contrast") for r in bt])), ("g", _med([r.get("common") for r in bt])),
                ("z", _med([r.get("size_w") for r in bt]))) if v is not None))
        fr = {}
        for d in self.dets:
            if d.get("result") == "fp":
                fr[d["reason"]] = fr.get(d["reason"], 0) + 1
        if fr:
            lines.append("F " + " ".join(f"{k}{v}" for k, v in sorted(fr.items(), key=lambda kv: -kv[1])))
        if s["size"]:
            w = [r[0] for r in s["size"]]
            h = [r[1] for r in s["size"]]
            hf = _med([r[2] for r in s["size"]])
            lt = sum(1 for a, b, _ in s["size"] if max(a, b) < 0.7)
            gt = sum(1 for a, b, _ in s["size"] if min(a, b) > 1.4)
            lines.append(f"S w{_fmt(_med(w))} h{_fmt(_med(h))} hf{_fmt(hf)} lt{lt} gt{gt}")
        if s["ok"]:
            lines.append(f"Y{s['same_type']}")
        return "\n".join(lines)

    def write_report(self, folder, s, code, meta):
        os.makedirs(folder, exist_ok=True)
        img_dir = os.path.join(folder, "img")
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(folder, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(self.short_code(s) + "\n\n" + code + "\n")

        # Chip images, the most useful first: misses, false detections, layer mismatches, matches
        prio = {}
        for t in self.truths:
            k = (t["layer"], tuple(t["cell"]))
            prio[k] = min(prio.get(k, 9), {"ms": 0, "ly": 2, "ok": 3}.get(t.get("result"), 4))
        for d in self.dets:
            k = (d["layer"], tuple(d["cell"]))
            prio[k] = min(prio.get(k, 9), {"fp": 1, "ly": 2, "ok": 3}.get(d.get("result"), 4))
        written = {}
        for k in sorted((k for k in prio if k in self.chips), key=lambda k: (prio[k], k))[:MAX_IMAGES]:
            name = f"L{k[0]}_r{k[1][0]}_c{k[1][1]}.png"
            self._chip_image(k, os.path.join(img_dir, name))
            written[k] = "img/" + name
        for L, st in self.layer_stats.items():
            if "_gold" in st:
                _gray(st["_gold"]).save(os.path.join(img_dir, f"L{L}_golden.png"))
                _heat(st["_spread"], float(np.percentile(st["_spread"].astype(np.float32), 99))).save(os.path.join(img_dir, f"L{L}_spread.png"))

        data = {"version": VERSION, "meta": meta, "code": code, "params": {k: v for k, v in self.p.items() if k not in ("type_rules", "learned_model", "learn", "seg_model", "dump")},
                "factor": self.factor, "work_size": list(self.size), "min_area": self.min_area, "max_area": self.max_area,
                "layers": {str(L): {k: v for k, v in st.items() if not k.startswith("_")} for L, st in self.layer_stats.items()},
                "truths": [_clean(t) for t in self.truths], "detections": [_clean(d) for d in self.dets]}
        with open(os.path.join(folder, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=float)
        with open(os.path.join(folder, "report.html"), "w", encoding="utf-8") as f:
            f.write(self._html(s, code, meta, written))
        return os.path.join(folder, "report.html")

    def _chip_image(self, key, path):
        c = self.chips[key]
        img = c["img"].astype(np.float32)
        dev = c["dev"].astype(np.float32)
        thr = c["thr"]
        H, W = img.shape
        scale = max(1, int(round(256 / max(W, H))))
        panels = [_gray(img), _heat(dev, 2 * thr), _mask_img(c["mask0"], c["mask"])]
        gold = self.layer_stats.get(key[0], {}).get("_gold")
        if gold is not None:
            panels.insert(1, _gray(gold.astype(np.float32)))
        out = Image.new("RGB", (len(panels) * (W * scale + 6), H * scale + 16), (30, 30, 30))
        x0, y0, sx, sy = c["rect"]
        for k, pnl in enumerate(panels):
            pnl = pnl.convert("RGB").resize((W * scale, H * scale), Image.NEAREST)
            dr = ImageDraw.Draw(pnl)

            def rect(box, color, width=1):
                bx0, by0 = (box[0] - x0) / sx * scale, (box[1] - y0) / sy * scale
                bx1, by1 = (box[2] - x0) / sx * scale, (box[3] - y0) / sy * scale
                dr.rectangle([bx0, by0, bx1, by1], outline=color, width=width)

            for t in self.by_lc.get(key, []):
                rect(t["box"], (0, 255, 0))
            for comp in c["comps"]:
                r0, r1, c0, c1 = comp["box"]
                color = {"kept": (255, 40, 40), "dup": (80, 160, 255), "small": (255, 170, 0), "large": (200, 0, 255)}.get(comp.get("status"), (255, 255, 255))
                dr.rectangle([c0 * scale, r0 * scale, c1 * scale - 1, (r1 + 1) * scale - 1], outline=color)
            out.paste(pnl, (k * (W * scale + 6), 16))
        ImageDraw.Draw(out).text((2, 2), f"L{key[0]} r{key[1][0]} c{key[1][1]}  thr {thr:g}"
                                 + ("" if c["shift"] is None else f"  shift {c['shift'][0] * self.factor:+.1f},{c['shift'][1] * self.factor:+.1f}px"),
                                 fill=(230, 230, 230))
        out.save(path)

    def _html(self, s, code, meta, imgs):
        esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        f = lambda x: "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else esc(x))
        rows = []
        add = rows.append
        add("<!doctype html><meta charset='utf-8'><title>Void detection evaluation</title><style>"
            "body{font:13px sans-serif;background:#1e1e1e;color:#ddd;margin:16px}pre{background:#111;padding:10px;font-size:15px}"
            "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #444;padding:2px 6px;text-align:right}"
            "th{background:#2a2a2a}img{max-width:100%;image-rendering:pixelated}a{color:#7cf}"
            ".l{text-align:left}</style>")
        add(f"<h2>Void detection evaluation</h2><p>{esc(meta.get('source', ''))} · {esc(meta.get('time', ''))} · "
            f"{s['elapsed']:.0f} s</p><pre>{esc(code)}</pre>")
        add("<p>Hand markings: drawn by hand on the loaded layers (dashed detected voids and bboxes are not counted). "
            "Green box = hand marking, red = kept detection, orange = too small, purple = too large, blue = skipped by "
            "the layer rule (marked on another layer). Panels: chip (matched) · golden chip · deviation (0 .. 2x threshold) · "
            "mask (gray: above threshold, white: after the 3x3 opening).</p>")
        add("<h3>Layers</h3><table><tr><th>layer<th>chips<th>hand<th>ok<th>other layer<th>missed<th>false<th>candidates"
            "<th>spread med<th>residual med<th>shift med (px)<th>shift p95 (px)<th>max dev / thr (med)<th>golden<th>spread map</tr>")
        for L, st in sorted(self.layer_stats.items()):
            tl = [t for t in self.truths if t["layer"] == L]
            dl = [d for d in self.dets if d["layer"] == L]
            g = f"<a href='img/L{L}_golden.png'>golden</a>" if "_gold" in st else ""
            sp = f"<a href='img/L{L}_spread.png'>spread</a>" if "_gold" in st else ""
            sh = lambda v: None if v is None else v * self.factor
            add(f"<tr><td>{L}<td>{st['n']}<td>{len(tl)}<td>{sum(t.get('result') == 'ok' for t in tl)}"
                f"<td>{sum(t.get('result') == 'ly' for t in tl)}<td>{sum(t.get('result') == 'ms' for t in tl)}"
                f"<td>{sum(d.get('result') == 'fp' for d in dl)}<td>{st['cand']}<td>{f(st['spread'])}<td>{f(st['residual'])}"
                f"<td>{f(sh(st['shift']))}<td>{f(sh(st['shift95']))}<td>{f(None if st['maxdev'] is None else st['maxdev'] / st['thr'])}"
                f"<td>{g}<td>{sp}</tr>")
        add("</table>")

        add("<h3>Missed hand markings</h3><table><tr><th>reason<th class=l>meaning<th>layer<th>row<th>col<th>type"
            "<th>peak / thr<th>spread ratio<th>contrast<th>common<th>size (work px)<th>polarity<th>blobs<th>image</tr>")
        for t in sorted((t for t in self.truths if t.get("result") == "ms"), key=lambda t: (t["reason"], t["layer"], t["cell"])):
            r = t.get("rec") or {}
            k = (t["layer"], tuple(t["cell"]))
            add(f"<tr><td>{t['reason']}<td class=l>{esc(MISS_REASONS[t['reason']])}<td>{t['layer']}<td>{t['cell'][0]}<td>{t['cell'][1]}"
                f"<td>{esc(t.get('type'))}<td>{f(r.get('peak'))}<td>{f(r.get('spread'))}<td>{f(r.get('contrast'))}<td>{f(r.get('common'))}"
                f"<td>{f(r.get('size_w'))}<td>{f(r.get('polarity'))}<td class=l>{esc(','.join(map(str, r.get('comps', []))))}"
                f"<td>{_img_link(imgs, k)}</tr>")
        add("</table>")

        add("<h3>False detections (no hand marking there)</h3><table><tr><th>class<th class=l>meaning<th>layer<th>row<th>col<th>type"
            "<th>score / thr<th>area %<th>merged<th>spread ratio<th>common<th>chip shift (px)<th>image</tr>")
        for d in sorted((d for d in self.dets if d.get("result") == "fp"), key=lambda d: (d["reason"], d["layer"], d["cell"])):
            k = (d["layer"], tuple(d["cell"]))
            add(f"<tr><td>{d['reason']}<td class=l>{esc(FP_REASONS[d['reason']])}<td>{d['layer']}<td>{d['cell'][0]}<td>{d['cell'][1]}"
                f"<td>{esc(d.get('type'))}<td>{f(d['score'])}<td>{f(100 * d['area_frac'])}<td>{d['merged']}<td>{f(d['spread'])}"
                f"<td>{f(d.get('common'))}<td>{f(None if d['shift'] is None else d['shift'] * self.factor)}<td>{_img_link(imgs, k)}</tr>")
        add("</table>")

        add("<h3>Found on another layer</h3><table><tr><th>hand layer<th>detected layer<th>row<th>col<th>image</tr>")
        for t in (t for t in self.truths if t.get("result") == "ly"):
            d = self.dets[t["match"][1]]
            add(f"<tr><td>{t['layer']}<td>{d['layer']}<td>{t['cell'][0]}<td>{t['cell'][1]}<td>{_img_link(imgs, (t['layer'], tuple(t['cell'])))} "
                f"{_img_link(imgs, (d['layer'], tuple(d['cell'])))}</tr>")
        add("</table>")

        add("<h3>Matched: size</h3><table><tr><th>layer<th>row<th>col<th>hand type<th>detected type<th>width ratio<th>height ratio"
            "<th>width at thr/2<th>merged blobs<th>image</tr>")
        for t in (t for t in self.truths if t.get("result") == "ok"):
            d = self.dets[t["match"][1]]
            tw = t["box"][2] - t["box"][0]
            hw = None if d["half"] is None else (d["half"][2] - d["half"][0]) / max(tw, 1e-6)
            add(f"<tr><td>{t['layer']}<td>{t['cell'][0]}<td>{t['cell'][1]}<td>{esc(t.get('type'))}<td>{esc(d.get('type'))}"
                f"<td>{f(t['size_ratio'][0])}<td>{f(t['size_ratio'][1])}<td>{f(hw)}<td>{d['merged']}"
                f"<td>{_img_link(imgs, (t['layer'], tuple(t['cell'])))}</tr>")
        add("</table>")

        add("<h3>Chip images</h3>")
        for k, path in sorted(imgs.items()):
            add(f"<p id='{esc(path)}'><img loading=lazy src='{esc(path)}'></p>")
        return "\n".join(rows)


def _img_link(imgs, k):
    p = imgs.get(k)
    return f"<a href='#{p}'>view</a>" if p else ""


def _clean(x):
    out = {}
    for k, v in x.items():
        if k in ("rec",) and isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items()}
        if isinstance(v, tuple):
            v = list(v)
        out[k] = v
    return out


def _gray(a):
    a = np.asarray(a, np.float32)
    lo, hi = np.percentile(a, [1, 99])
    return Image.fromarray(np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8))


def _heat(a, top):
    """0 .. top as black -> red -> yellow -> white."""
    t = np.clip(np.asarray(a, np.float32) / max(top, 1e-6), 0, 1)
    rgb = np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _mask_img(mask0, mask):
    out = np.zeros(mask0.shape, np.uint8)
    out[mask0] = 110
    out[mask] = 255
    return Image.fromarray(out)
