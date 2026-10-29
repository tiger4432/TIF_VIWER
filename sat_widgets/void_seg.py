"""
Void segmentation with a small U-Net (trained offline with tools/train_seg.py, run with
onnxruntime): per chip, a void probability map replaces the die-to-die deviation map; the rest of
Detect Voids (blobs, sizes, layer rule, type rules) is unchanged. See docs/SEGMENTATION.md.

Model input (inputs()): the chip as Detect Voids sees it (registered, matched to the golden chip):
the chip, the golden chip, (chip - golden) / chip-to-chip spread; multi-layer models add what
changed against the same chip on the layer above and below (layer_change: brightness / contrast
differ per layer, the structure does not; a person flips through the layers too). Each
clipped to +-10 and divided by 5. Output: one logit per pixel.

Training data (Seg Data in Void Marking): per source file a folder in DATA_DIR with
  index.json            source, meta, samples [{id, layer, row, col, voids, include}], include
  L<layer>_ref.npz      golden, spread of the layer (shared by its chips)
  <id>.npz              m (matched chip, float16), y (mask: 0 background, 1 void, 255 unknown)
  <id>.png              thumbnail for the review (chip, void green, unknown yellow)
The mask comes from the markings: inside a hand marking (core) = void, a band around its outline
= unknown (markings approximate the outline), markings of other layers and unreviewed detections
= unknown, the rest = background.
"""
import hashlib
import json
import os
import re
import shutil

import numpy as np
from PIL import Image

DATA_DIR = os.path.join(os.path.expanduser("~"), "TIF_Viewer_seg")
DEFAULT_MODEL = "models/void_unet.onnx"
CORE, BAND = 0.75, 1.35 # Mask: void inside CORE x the marking, unknown up to BAND x
PAD = 8 # Input sizes are padded to a multiple of this (U-Net with 3 poolings)


def diff(m, golden, spread):
    """Die-to-die difference of a matched chip: (chip - golden) / spread."""
    return (np.asarray(m, np.float32) - np.asarray(golden, np.float32)) / np.maximum(np.asarray(spread, np.float32), 1e-3)


def layer_change(m, m_other):
    """
    What changed between two layers of the same chip: the layers of a scan are aligned (and the
    chips registered with one shift for all layers), only brightness / contrast differ while the
    structure is the same. The other layer's chip is fitted to this one like the layout fit (its
    structure x contrast incl. inverted, blurred structure, brightness; smoothly varying across
    the chip) and subtracted: the chip's own layout cancels, what appeared or disappeared stays.
    """
    from .void_detect import _LayoutFit
    m = np.asarray(m, np.float32)
    o = np.asarray(m_other, np.float32)
    return _LayoutFit(o).apply(m) - o


def inputs(m, golden, spread, neighbors=None):
    """
    Model input (C, H, W) float32: the matched chip, the golden chip, the difference, and for
    multi-layer models (neighbors = [the same chip on the layer above, below], None = no such
    layer) what changed against each neighbouring layer (layer_change). Each clipped to +-10, / 5.
    """
    m = np.asarray(m, np.float32)
    g = np.asarray(golden, np.float32)
    chans = [m, g, diff(m, g, spread)]
    for o in (neighbors or []):
        chans.append(np.zeros_like(m) if o is None else layer_change(m, o))
    return np.stack([np.clip(c, -10, 10) for c in chans]) / 5.0


def rasterize(h, w, shapes):
    """
    Mask (uint8: 0 background, 1 void, 255 unknown) of shapes in working px:
    [{"cx", "cy", "rx", "ry", "shape", "p", "quadrant", "kind": "void" | "unknown"}].
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.zeros((h, w), np.uint8)
    unknown = np.zeros((h, w), bool)
    void = np.zeros((h, w), bool)
    for s in shapes:
        dx = (xx - s["cx"]) / max(s["rx"], 0.5)
        dy = (yy - s["cy"]) / max(s["ry"], 0.5)
        shape = s.get("shape", "ellipse")
        if shape == "rectangle":
            t = np.maximum(np.abs(dx), np.abs(dy))
        else:
            p = float(s.get("p") or 2.0) if shape == "curve" else 2.0
            t = (np.abs(dx) ** p + np.abs(dy) ** p) ** (1.0 / p)
        q = s.get("quadrant")
        if q:
            t = np.where((dx * q[0] >= 0) & (dy * q[1] >= 0), t, np.inf)
        if s.get("kind") == "void":
            void |= t <= CORE
        unknown |= t <= BAND
    mask[unknown] = 255
    mask[void] = 1
    return mask


def thumbnail(m, mask, size=128):
    """RGB thumbnail: chip in gray, void green, unknown yellow (translucent)."""
    m = np.asarray(m, np.float32)
    lo, hi = np.percentile(m, [1, 99])
    gray = np.clip((m - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    rgb = np.stack([gray] * 3, axis=-1)
    for value, color in ((255, (255, 220, 0)), (1, (0, 255, 0))):
        sel = mask == value
        rgb[sel] = 0.55 * rgb[sel] + 0.45 * np.array(color, np.float32)
    img = Image.fromarray(rgb.astype(np.uint8))
    img.thumbnail((size, size), Image.BILINEAR)
    return img


def source_key(source):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(source.rstrip("/\\")))[0])[:40]
    return f"{stem}_{hashlib.md5(source.encode('utf-8')).hexdigest()[:8]}"


def write_source(source, meta, refs, samples, data_dir=None):
    """
    Replaces the training data of one source file. refs: {layer: (golden, spread)};
    samples: [{"layer", "row", "col", "m", "y"}]. Returns the folder.
    """
    folder = os.path.join(data_dir or DATA_DIR, source_key(source))
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for layer, (g, spread) in refs.items():
        np.savez_compressed(os.path.join(folder, f"L{layer}_ref.npz"), golden=g.astype(np.float16), spread=spread.astype(np.float16))
    index = []
    for s in samples:
        sid = f"L{s['layer']}_r{s['row']}_c{s['col']}"
        np.savez_compressed(os.path.join(folder, sid + ".npz"), m=np.asarray(s["m"], np.float16), y=s["y"])
        thumbnail(s["m"], s["y"]).save(os.path.join(folder, sid + ".png"))
        index.append({"id": sid, "layer": s["layer"], "row": s["row"], "col": s["col"],
                      "voids": int((s["y"] == 1).any()), "include": True})
    with open(os.path.join(folder, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"source": source, "meta": meta, "include": True, "samples": index}, f, indent=1, ensure_ascii=False)
    return folder


def list_sources(data_dir=None):
    """[(folder, index dict)] of the stored training data."""
    root = data_dir or DATA_DIR
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name, "index.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    out.append((os.path.join(root, name), json.load(f)))
    return out


def save_index(folder, index):
    with open(os.path.join(folder, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)


def load_sample(folder, sample, _ref_cache={}):
    """(input (3, H, W) float32, mask uint8) of a stored sample."""
    key = (folder, sample["layer"])
    ref = _ref_cache.get(key)
    if ref is None:
        r = np.load(os.path.join(folder, f"L{sample['layer']}_ref.npz"))
        ref = (r["golden"].astype(np.float32), r["spread"].astype(np.float32))
        if len(_ref_cache) > 64:
            _ref_cache.clear()
        _ref_cache[key] = ref
    d = np.load(os.path.join(folder, sample["id"] + ".npz"))
    return inputs(d["m"].astype(np.float32), *ref), d["y"]


# ----- inference -----
def model_file(path):
    if not path:
        path = DEFAULT_MODEL
    if os.path.isabs(path):
        return path
    from .app_paths import app_file
    return app_file(path)


def available(path=None):
    """'' if the segmentation model can run, else the reason."""
    import importlib.util
    if importlib.util.find_spec("onnxruntime") is None:
        return "onnxruntime is not installed (pip install -r requirements-patchcore.txt)"
    if not os.path.exists(model_file(path)):
        return f"model not found: {model_file(path)} (train with tools/train_seg.py)"
    return ""


class Model:
    """
    A trained U-Net (tools/train_seg.py) and its sidecar <model>.json: "mode" is "seg" (output = void
    logit) or "recon" (output = the chip redrawn without voids, in input units / 5).
    """

    def __init__(self, path=None):
        import onnxruntime as ort
        path = model_file(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.info = {}
        side = os.path.splitext(path)[0] + ".json"
        if os.path.exists(side):
            with open(side, "r", encoding="utf-8") as f:
                self.info = json.load(f)
        self.mode = self.info.get("mode", "recon" if "recon" in os.path.basename(path).lower() else "seg")
        self.neighbors = int(self.info.get("neighbor_layers", 0)) # 1: the layers above / below as extra channels

    def predict(self, x):
        """seg: void probability (H, W); recon: the redrawn chip (H, W), for one input (3, H, W)."""
        c, h, w = x.shape
        ph, pw = (-h) % PAD, (-w) % PAD
        xp = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")[None].astype(np.float32)
        out = self.sess.run(None, {self.input: xp})[0][0, 0, :h, :w]
        if self.mode == "recon":
            return (out * 5.0).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-out))).astype(np.float32)
