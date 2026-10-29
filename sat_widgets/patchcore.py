"""
Optional PatchCore scorer for Detect Voids (CPU, onnxruntime; see docs/PATCHCORE.md).

Each aligned chip is described by mid-level features of an ImageNet CNN (ONNX model in models/,
built by tools/build_patchcore_onnx.py, no download at run time). A feature cell is anomalous when
it is far from every feature of the OTHER chips at the same position (+-1 cell): "position-aware"
PatchCore, which suits wafers (aligned, repeated chips) better than the classic position-free
memory bank. Scores are robustly normalized per layer (median / MAD) so the threshold reads like
the die-to-die one.

Not used unless config.json "void_detect": {"method": "patchcore"} (or the Method box) selects it
and onnxruntime is installed; the viewer runs without it.
"""
import importlib.util
import os

import numpy as np
from PIL import Image

from .app_paths import app_file

DEFAULTS = {
    "pc_model": "models/patchcore_mobilenetv2.onnx", # Relative to the program folder
    "pc_input": 224, # CNN input size (pixels); feature grid = input / 8
    "pc_dims": 128, # Random projection of the features
    "pc_bank_chips": 64, # Other chips compared per position (speed)
    "pc_threshold": 10.0, # Score above the layer median, in MADs
    "pc_match": False, # Match brightness / contrast / shading to the golden chip first (tested: no gain)
    "pc_position_norm": False, # Normalize each position by its own chip-to-chip spread (see score_layer).
                               # Field data (20 chips): border false detections 57 -> 5 but found 20 -> 2
                               # (real voids at the chip border were normalized away): off by default
}


def model_path(p):
    path = p.get("pc_model", DEFAULTS["pc_model"])
    return path if os.path.isabs(path) else app_file(path)


def available(p=None):
    """'' if PatchCore can run, else the reason."""
    # find_spec: checked without importing onnxruntime (start-up time)
    if importlib.util.find_spec("onnxruntime") is None:
        return "onnxruntime is not installed (pip install -r requirements-patchcore.txt)"
    path = model_path(p or DEFAULTS)
    if not os.path.exists(path):
        return f"model not found: {path}"
    return ""


class PatchCore:
    def __init__(self, p):
        import onnxruntime as ort
        self.p = dict(DEFAULTS)
        self.p.update(p or {})
        path = model_path(self.p)
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.resnet = "resnet" in os.path.basename(path).lower()
        self._proj = None

    def _features(self, chips):
        """(N, h, w, D) features of a list of normalized chips (median 0, MAD 1)."""
        size = int(self.p["pc_input"])
        out = []
        for i in range(0, len(chips), 16):
            imgs = []
            for ch in chips[i:i + 16]:
                u8 = np.clip(np.asarray(ch, np.float32) * 25.0 + 128.0, 0, 255) # Chip contrast -> 0..255
                imgs.append(np.asarray(Image.fromarray(u8).resize((size, size), Image.BILINEAR)))
            x = np.repeat(np.stack(imgs)[:, None], 3, axis=1).astype(np.float32)
            if self.resnet:
                x -= np.array([103.939, 116.779, 123.68], np.float32)[None, :, None, None]
            else:
                x = x / 127.5 - 1.0
            f8, f16 = self.sess.run(None, {"input": x})
            f16 = f16.repeat(2, axis=2).repeat(2, axis=3)[:, :, :f8.shape[2], :f8.shape[3]]
            f = np.concatenate([f8, f16], axis=1)
            # 3x3 neighbourhood average (PatchCore's local aggregation)
            fp = np.pad(f, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
            h, w = f.shape[2:]
            f = sum(fp[:, :, dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9.0
            if self._proj is None:
                rng = np.random.default_rng(0)
                self._proj = (rng.standard_normal((f.shape[1], int(self.p["pc_dims"]))) /
                              np.sqrt(f.shape[1])).astype(np.float32)
            out.append(np.einsum("nchw,cd->nhwd", f, self._proj))
        return np.concatenate(out)

    def score_layer(self, chips, size):
        """{cell: score map at the chip working size (normalized: 0 = layer median, 1 = MAD)}."""
        keys = list(chips)
        F = self._features([chips[k] for k in keys])
        N, H, W, D = F.shape
        rng = np.random.default_rng(0)
        bank_ids = np.arange(N) if N <= self.p["pc_bank_chips"] else \
            np.sort(rng.choice(N, int(self.p["pc_bank_chips"]), replace=False))
        scores = np.zeros((N, H, W), np.float32)
        for y in range(H):
            for x in range(W):
                win = F[bank_ids, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                k = win.shape[1] * win.shape[2]
                bank = win.reshape(-1, D)
                q = F[:, y, x]
                d2 = (q ** 2).sum(1)[:, None] - 2.0 * (q @ bank.T) + (bank ** 2).sum(1)[None, :]
                d2[np.repeat(bank_ids, k)[None, :] == np.arange(N)[:, None]] = np.inf # Not itself
                scores[:, y, x] = np.sqrt(np.maximum(d2.min(1), 0.0))
        if self.p.get("pc_position_norm", False) and N >= 5:
            # Per position: a place that scores high on every chip (chip border, layout edges that
            # shift a little from chip to chip) is normal there; only a chip standing out from the
            # others at that place counts. The spread has a floor (median spread of all positions).
            med = np.median(scores, axis=0)
            mad = np.median(np.abs(scores - med), axis=0) * 1.4826
            mad = np.maximum(mad, float(np.median(mad))) + 1e-6
        else:
            med = float(np.median(scores))
            mad = float(np.median(np.abs(scores - med))) * 1.4826 + 1e-6
        out = {}
        for i, cell in enumerate(keys):
            s = ((scores[i] - med) / mad).astype(np.float32)
            out[cell] = np.asarray(Image.fromarray(s).resize(size, Image.BILINEAR))
        return out
