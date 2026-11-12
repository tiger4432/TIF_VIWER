import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable limit
from PySide6.QtGui import QColor
import json
import os
import math
from collections import OrderedDict

# tifffile reads TIFF pages without PIL's extra copies and can memory-map
# uncompressed pages, so only the pixels that are actually touched are read.
try:
    import tifffile
except ImportError:
    tifffile = None

# PIL mode -> numpy dtype (fallback path only)
# PIL modes read as they are (gray, 16-bit, RGB(A)); any other mode (palette PNG, gray + alpha,
# CMYK JPG, 1-bit) is converted to RGB first, then to gray like every color image
_PIL_DIRECT = ("L", "I;16", "I;16L", "I;16B", "I", "F", "RGB", "RGBA")
_PIL_ONLY = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp") # Not TIFF: read with PIL directly

_PIL_DTYPES = {
    'I;16': np.uint16, 'I;16L': np.uint16, 'I;16B': np.uint16,
    'I': np.int32, 'F': np.float32, '1': np.bool_,
}


def to_gray(arr: np.ndarray) -> np.ndarray:
    """
    Reduces a (H, W), (H, W, C) or planar (C, H, W) array to a single channel.
    uint8 / uint16 are kept as is (no copy); RGB(A) -> luminance in the source bit depth;
    any other dtype -> uint16 (clipped).
    """
    a = arr
    if a.ndim == 3:
        if a.shape[-1] not in (3, 4) and a.shape[0] in (3, 4):
            a = np.moveaxis(a, 0, -1) # Planar (C, H, W) -> (H, W, C)
        if a.shape[-1] in (3, 4):
            rgb = a[..., :3]
            gray = rgb[..., 0] * np.float32(0.299)
            gray += rgb[..., 1] * np.float32(0.587)
            gray += rgb[..., 2] * np.float32(0.114)
            if a.dtype == np.uint8:
                return np.clip(np.rint(gray), 0, 255).astype(np.uint8)
            return np.clip(np.rint(gray), 0, 65535).astype(np.uint16)
        a = a[..., 0]

    if a.dtype == np.uint8 or a.dtype == np.uint16:
        return a
    if a.dtype == np.bool_:
        return a.astype(np.uint8) * np.uint8(255)
    return np.clip(a, 0, 65535).astype(np.uint16)


def to_uint16(arr: np.ndarray) -> np.ndarray:
    """Gray uint16 in the 16-bit window/level domain (8-bit data is scaled by 257)."""
    a = to_gray(arr)
    if a.dtype == np.uint8:
        out = a.astype(np.uint16)
        out *= 257
        return out
    return a


def to_gray2d_uint16(arr, z_index: int = 0) -> np.ndarray:
    """
    Extracts a single 2D grayscale layer as (H, W) uint16.
    Accepts an ImageStack or a numpy array (Z, H, W), (Z, H, W, C), (H, W, C) or (H, W).
    Prefer ImageStack.read_region / get_crop for large images: this returns the full layer.
    """
    if isinstance(arr, ImageStack):
        return arr.get_crop(z_index, 0, 0, arr.width, arr.height)

    frame = arr
    if isinstance(arr, list):
        frame = arr[z_index]
    elif arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] not in (3, 4)):
        frame = arr[z_index]
    return to_uint16(np.asarray(frame))


def _slice2d(frame, x, y, w, h, step=1):
    if frame.ndim == 3 and frame.shape[-1] not in (3, 4) and frame.shape[0] in (3, 4):
        return frame[:, y:y + h:step, x:x + w:step] # Planar (C, H, W)
    return frame[y:y + h:step, x:x + w:step]


class ImageStack:
    """
    Common interface of the layer stacks.
    Layers are addressed by a relative index z (0 .. len-1).
    layer_number(z) is the absolute layer number (used for void data and file names).
    """
    path = None
    width = 0
    height = 0
    frame_count = 0
    dtype = np.dtype(np.uint16)
    ndim = 3

    @property
    def shape(self):
        return (self.frame_count, self.height, self.width)

    def __len__(self):
        return self.frame_count

    def __getitem__(self, z):
        # Full layer as uint16. Kept for compatibility; prefer region reads.
        return self.get_crop(z, 0, 0, self.width, self.height)

    def layer_number(self, z):
        raise NotImplementedError

    def read_region(self, z, x, y, w, h, step=1):
        """
        Native-dtype region of layer z, clipped to the image (may be smaller than w x h).
        May be a view of a memory map or of a cached frame: do not modify it.
        """
        raise NotImplementedError

    def release_cache(self):
        """Drops decoded frame data kept for repeated region reads."""
        pass

    supports_reopen = False

    def reopen(self):
        """Independent reader of the same layers (for a background thread)."""
        raise NotImplementedError

    def close(self):
        pass

    def get_tile(self, z, x, y, w, h):
        """(h, w) contiguous single-channel uint8/uint16 block for GPU upload (zero padded)."""
        region = to_gray(self.read_region(z, x, y, w, h))
        if region.shape != (h, w):
            padded = np.zeros((h, w), dtype=region.dtype)
            padded[:region.shape[0], :region.shape[1]] = region
            return padded
        return np.ascontiguousarray(region)

    def get_crop(self, z, x, y, w, h):
        """(h, w) uint16 gray crop at (x, y), zero padded outside the image."""
        out = np.zeros((h, w), dtype=np.uint16)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(x + w, self.width), min(y + h, self.height)
        if x0 >= x1 or y0 >= y1:
            return out
        region = to_uint16(self.read_region(z, x0, y0, x1 - x0, y1 - y0))
        rh, rw = region.shape
        out[y0 - y:y0 - y + rh, x0 - x:x0 - x + rw] = region
        return out

    def get_sample(self, z, max_items=10000):
        """Evenly subsampled uint16 gray view of layer z (for auto-leveling)."""
        step = int(max(1, np.sqrt((self.width * self.height) / max_items)))
        return to_uint16(self.read_region(z, 0, 0, self.width, self.height, step))

    def get_poly_crop(self, z, points):
        """
        Extracts a patch defined by 4 points (TL, TR, BR, BL) in Raw Coordinates.
        Returns a straightened (deskewed) uint16 array.
        """
        poly = np.asarray(points, dtype=np.float64) # [[x,y], ...]

        # 1. Bounding Box in Raw Space (clamped to the image)
        min_x = max(0, int(np.floor(poly[:, 0].min())))
        min_y = max(0, int(np.floor(poly[:, 1].min())))
        max_x = min(self.width, int(np.ceil(poly[:, 0].max())))
        max_y = min(self.height, int(np.ceil(poly[:, 1].max())))
        if max_x <= min_x or max_y <= min_y:
            return None

        # Size of the target patch (edge lengths)
        p0, p1, p2, p3 = poly
        dst_w = int(round(np.hypot(*(p1 - p0))))
        dst_h = int(round(np.hypot(*(p3 - p0))))
        if dst_w <= 0 or dst_h <= 0:
            return None

        # 2. Extract bounding rect (Raw)
        region = to_gray(self.read_region(z, min_x, min_y, max_x - min_x, max_y - min_y))

        # 3. Straighten with a QUAD transform. PIL's bilinear resampling is broken for
        # 16-bit ('I;16') images, so resample in float ('F') mode.
        # QUAD data: upper-left, lower-left, lower-right, upper-right corners of the source,
        # relative to the extracted rect. Our points are TL, TR, BR, BL -> p0, p3, p2, p1.
        quad = []
        for p in (p0, p3, p2, p1):
            quad += [p[0] - min_x, p[1] - min_y]

        src = Image.fromarray(region.astype(np.float32))
        out = np.asarray(src.transform((dst_w, dst_h), Image.QUAD, tuple(quad), resample=Image.BILINEAR))
        if region.dtype == np.uint8:
            out = out * 257.0
        return np.clip(np.rint(out), 0, 65535).astype(np.uint16)


class TiffStack(ImageStack):
    """
    Multi-page TIFF with region reads.
    - tifffile, uncompressed native-order pages: memory-mapped (only touched pixels are read).
    - tifffile, other pages: the page is decoded once into a single array and cached
      (one frame at a time).
    - no tifffile: PIL fallback, also one cached frame.
    """
    MAX_MEMMAPS = 8

    def __init__(self, path, start=0, count=None):
        self.path = path
        self._tf = None
        self._volume = None # (N, H, W[, C]) when all frames live in one page/block
        self._memmaps = OrderedDict() # phys_idx -> np.memmap
        self._frame_cache = None # (phys_idx, ndarray)
        # Sparse reads: a region of a compressed, striped page is decoded from only the strips it
        # covers (no frame cache). For readers doing a few small reads per layer (previews).
        self.sparse_reads = False

        n = w = h = None
        dtype = None
        if tifffile is not None and not path.lower().endswith(_PIL_ONLY):
            try:
                self._tf = tifffile.TiffFile(path)
                n = len(self._tf.pages)
                page = self._tf.pages[0]
                h, w = page.imagelength, page.imagewidth
                dtype = page.dtype
                if n == 1:
                    n = self._open_volume(page) or 1
            except Exception as e:
                print(f"tifffile could not open {path} ({e}); falling back to PIL")
                if self._tf is not None:
                    self._tf.close()
                self._tf = None
        if self._tf is None: # Also single-layer PNG / JPG / BMP ... (PIL)
            with Image.open(path) as img:
                n = getattr(img, 'n_frames', 1)
                w, h = img.size
                dtype = _PIL_DTYPES.get(img.mode, np.uint8) if img.mode in _PIL_DIRECT else np.uint8

        self.n_frames = int(n)
        self.width, self.height = int(w), int(h)
        self.dtype = np.dtype(dtype if dtype is not None else np.uint16)
        self.backend = "tifffile" if self._tf is not None else "PIL"
        self.frame_start = 0
        self.frame_count = self.n_frames
        self.set_range(start, self.n_frames if count is None else count)

    def _open_volume(self, page):
        """
        Single-IFD files can still hold several frames (ImageJ > 4 GB, tifffile 'shaped',
        SGI image depth). Returns the frame count if so, else None.
        """
        series = self._tf.series[0] if self._tf.series else None
        if series is None:
            return None
        frame_ndim = 3 if page.samplesperpixel > 1 else 2
        n = int(np.prod(series.shape[:-frame_ndim], dtype=np.int64)) if len(series.shape) > frame_ndim else 1
        if n <= 1:
            return None
        try:
            vol = tifffile.memmap(self.path, series=0, mode='r')
        except Exception:
            vol = series.asarray() # Not memory-mappable: the data is one block anyway
        self._volume = vol.reshape((n,) + tuple(series.shape[-frame_ndim:]))
        return n

    def set_range(self, start, count):
        self.frame_start = min(max(0, start), self.n_frames - 1)
        self.frame_count = max(1, min(count, self.n_frames - self.frame_start))

    def layer_number(self, z):
        return self.frame_start + z

    def _phys(self, z):
        if z < 0:
            z += self.frame_count
        if z < 0 or z >= self.frame_count:
            raise IndexError(f"Layer {z} out of range (0..{self.frame_count - 1})")
        return self.frame_start + z

    def _frame(self, idx):
        if self._volume is not None:
            return self._volume[idx]
        mm = self._memmaps.get(idx)
        if mm is not None:
            self._memmaps.move_to_end(idx)
            return mm
        if self._frame_cache is not None and self._frame_cache[0] == idx:
            return self._frame_cache[1]

        if self._tf is not None:
            page = self._tf.pages[idx]
            if getattr(page, 'is_memmappable', False):
                try:
                    mm = tifffile.memmap(self.path, page=idx, mode='r')
                    self._memmaps[idx] = mm
                    while len(self._memmaps) > self.MAX_MEMMAPS:
                        self._memmaps.popitem(last=False)
                    return mm
                except Exception:
                    pass # e.g. non-native byte order -> decode
            self._frame_cache = None # Free the previous frame before decoding the next one
            arr = page.asarray()
        else:
            self._frame_cache = None
            with Image.open(self.path) as img:
                img.seek(idx)
                if img.mode not in _PIL_DIRECT: # Palette, gray + alpha, CMYK, 1-bit ...: values as seen
                    img = img.convert("RGB")
                arr = np.array(img)
        self._frame_cache = (idx, arr)
        return arr

    def read_region(self, z, x, y, w, h, step=1):
        x0, y0 = max(0, x), max(0, y)
        if self.sparse_reads and step == 1:
            region = self._read_strips(self._phys(z), x0, y0, x + w - x0, y + h - y0)
            if region is not None:
                return region
        frame = self._frame(self._phys(z))
        return _slice2d(frame, x0, y0, x + w - x0, y + h - y0, step)

    def _read_strips(self, idx, x, y, w, h):
        """Region decoded from the strips covering rows y .. y+h, or None if not applicable."""
        if self._tf is None or self._volume is not None or idx in self._memmaps:
            return None
        if self._frame_cache is not None and self._frame_cache[0] == idx:
            return None # Already decoded
        page = self._tf.pages[idx]
        if (page.is_tiled or page.samplesperpixel != 1 or getattr(page, 'is_memmappable', False)
                or len(page.dataoffsets) < 2):
            return None
        rps = page.rowsperstrip
        y1 = min(y + h, page.imagelength)
        if y >= y1:
            return None
        fh = self._tf.filehandle
        parts = []
        for i in range(y // rps, (y1 - 1) // rps + 1):
            fh.seek(page.dataoffsets[i])
            seg, _, _ = page.decode(fh.read(page.databytecounts[i]), i)
            parts.append(seg.reshape(-1, page.imagewidth))
        block = parts[0] if len(parts) == 1 else np.concatenate(parts)
        s0 = (y // rps) * rps
        return block[y - s0:y1 - s0, x:x + w]

    def release_cache(self):
        self._frame_cache = None

    supports_reopen = True

    def reopen(self):
        return TiffStack(self.path, self.frame_start, self.frame_count)

    def close(self):
        self._frame_cache = None
        self._memmaps.clear()
        self._volume = None
        if self._tf is not None:
            self._tf.close()
            self._tf = None


# Backward compatible name
LazyTiffStack = TiffStack


class MosaicTiffStack(ImageStack):
    """
    Virtual Stack acting as the ORIGINAL RAW Image, rebuilt from an exported project folder.
    Strategy: "Pre-assemble Deskewed -> Warp to Raw".
    1. Parses patches.json.
    2. Assembles a layer on demand by pasting the patches at 'bbox_deskewed' and warping the
       canvas into the original raw frame (same size and rotation center as the exported image,
       so voids.json and the grid line up). Only MAX_CACHED_LAYERS layers are kept in memory.
    """
    MAX_CACHED_LAYERS = 2

    def __init__(self, patches_json_path):
        self.path = patches_json_path
        self.base_dir = os.path.dirname(patches_json_path)
        with open(patches_json_path, 'r') as f:
            data = json.load(f)
        self.grid_cfg = data.get("grid_config", {})
        self.patches = data.get("patches", [])
        self.angle = self.grid_cfg.get("angle", 0.0)

        self.patches_by_layer = {}
        for p in self.patches:
            self.patches_by_layer.setdefault(p["layer"], []).append(p)
        self.layer_ids = sorted(self.patches_by_layer) or [0]
        self.n_frames = self.frame_count = len(self.layer_ids)
        self.layer_offset = self.layer_ids[0]
        self.dtype = np.dtype(np.uint8)

        # The layers are rebuilt in the ORIGINAL raw frame (same size and rotation center), so
        # the raw void coordinates of voids.json and the grid line up exactly
        self.width, self.height = self._original_size()
        self.transform = CoordinateTransform(self.angle, self.width, self.height)
        max_dx = max((pt[0] for p in self.patches for pt in p.get("bbox_deskewed", [])), default=0)
        max_dy = max((pt[1] for p in self.patches for pt in p.get("bbox_deskewed", [])), default=0)
        self.deskew_w = int(max(max_dx, self.width) + 2)
        self.deskew_h = int(max(max_dy, self.height) + 2)

        self.layer_cache = OrderedDict() # layer_id -> (H, W) uint8
        print(f"[Mosaic] {self.n_frames} layers, size {self.width}x{self.height}, angle {self.angle}")

    def _original_size(self):
        """
        Size of the image the patches were exported from: patches.json "image_size", or (older
        exports) solved from the patches' raw / deskewed corners, which are related by a rotation
        about the image center: raw = c + R (deskew - c).
        """
        size = self.grid_cfg.get("image_size")
        if size:
            return int(size[0]), int(size[1])
        pairs = [(np.asarray(p["bbox_raw"], float), np.asarray(p["bbox_deskewed"], float))
                 for p in self.patches if p.get("bbox_raw") and p.get("bbox_deskewed")]
        if not pairs:
            return 1, 1
        raw = np.concatenate([r for r, _ in pairs])
        desk = np.concatenate([d for _, d in pairs])
        a = np.radians(self.angle)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        if abs(a) > 1e-6:
            A = np.tile(np.eye(2) - R, (len(raw), 1))
            b = (raw - desk @ R.T).reshape(-1)
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            return max(1, int(round(2 * c[0]))), max(1, int(round(2 * c[1])))
        return int(np.ceil(raw[:, 0].max())), int(np.ceil(raw[:, 1].max()))

    def layer_number(self, z):
        return self.layer_ids[z]

    def _build_layer(self, layer_id):
        print(f"Building Layer {layer_id} from patches...")
        canvas = Image.new("L", (self.deskew_w, self.deskew_h), 0) # Deskewed (grid) space

        for p in self.patches_by_layer.get(layer_id, []):
            bd = p.get("bbox_deskewed", [])
            if not bd: continue

            path = os.path.join(self.base_dir, p["filename"])
            try:
                with Image.open(path) as img:
                    if img.height > 40:
                        img = img.crop((0, 40, img.width, img.height)) # Title bar
                    if img.mode != "L":
                        img = img.convert("L")
                    pts = np.array(bd)
                    canvas.paste(img, (int(round(pts[:, 0].min())), int(round(pts[:, 1].min()))))
            except Exception as e:
                print(f"[Mosaic] Failed to load patch {path}: {e}")

        # Deskewed canvas -> raw frame: every raw pixel samples the canvas at raw_to_deskew(x, y)
        t = self.transform
        cx, cy, ci, si = t.img_cx, t.img_cy, t.cos_inv, t.sin_inv
        coeffs = (ci, -si, cx - ci * cx + si * cy,
                  si, ci, cy - si * cx - ci * cy)
        raw = canvas.transform((self.width, self.height), Image.AFFINE, coeffs, resample=Image.BILINEAR)
        return np.array(raw)

    def _layer(self, z):
        if z < 0:
            z += self.frame_count
        layer_id = self.layer_ids[z]
        arr = self.layer_cache.get(layer_id)
        if arr is None:
            arr = self._build_layer(layer_id)
            self.layer_cache[layer_id] = arr
            while len(self.layer_cache) > self.MAX_CACHED_LAYERS:
                self.layer_cache.popitem(last=False)
        else:
            self.layer_cache.move_to_end(layer_id)
        return arr

    def deskew_to_raw(self, gx, gy):
        return self.transform.deskew_to_raw(gx, gy)

    def raw_to_deskew(self, rx, ry):
        return self.transform.raw_to_deskew(rx, ry)

    def read_region(self, z, x, y, w, h, step=1):
        arr = self._layer(z)
        x0, y0 = max(0, x), max(0, y)
        return arr[y0:y + h:step, x0:x + w:step]

    supports_reopen = True

    def reopen(self):
        return MosaicTiffStack(self.path)

    def close(self):
        self.layer_cache.clear()


class GridConfig:
    def __init__(self):
        self.visible = False
        self.start_x = 100.0
        self.start_y = 100.0
        self.pitch_x = 200.0
        self.pitch_y = 200.0
        self.rows = 1
        self.cols = 1
        self.angle = 0.0      # Rotation in degrees
        self.line_width = 1.0 # Reduced from 2.0
        self.opacity = 0.5    # Default 50% opacity
        
        # Calibration
        self.use_calib = False
        self.target_mean = 128.0
        self.target_std = 40.0

    def cell_at(self, gx, gy):
        """
        (row, col) of the cell containing the Deskewed (grid space) point (gx, gy).
        May lie outside 0..rows-1 / 0..cols-1; see contains().
        """
        col = int(np.floor((gx - self.start_x) / self.pitch_x))
        row = int(np.floor((gy - self.start_y) / self.pitch_y))
        return row, col

    def contains(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def cell_origin(self, row, col):
        """Deskewed top-left corner of a cell."""
        return self.start_x + col * self.pitch_x, self.start_y + row * self.pitch_y

class BondingMap:
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.data_map = [] 
        self.unique_keys = {} 
        self.colors_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (255, 128, 0), (128, 255, 0), (0, 128, 255),
            (128, 0, 255), (255, 0, 128), (128, 128, 128)
        ]

    def parse_text(self, text: str, reject_values=()):
        """
        Parses a pasted (tab separated) map. Cells whose value is in reject_values are treated
        as not bonded (empty), so they are skipped everywhere a bonded cell is required.
        """
        reject = {str(v).strip() for v in reject_values if str(v).strip()}
        self.source_text = text # Saved with exports (bonding_map.txt)
        self.reject_values = sorted(reject)
        # Only line breaks at the ends: a leading tab is an empty first cell (empty wafer corners)
        lines = [ln for ln in text.replace('\r\n', '\n').strip('\n').split('\n')]
        if not lines: return

        raw_grid = []
        for line in lines:
            line = line.rstrip('\r\n')
            parts = line.split('\t')
            raw_grid.append([p.strip() for p in parts])
            
        if not raw_grid: return

        # Header Detection
        has_col_header = False
        try:
            ints_found = 0
            for cell in raw_grid[0]:
                if cell.isdigit(): ints_found += 1
            if ints_found > len(raw_grid[0]) * 0.5: has_col_header = True
        except: pass

        has_row_header = False
        try:
            ints_found = 0
            for row in raw_grid:
                if row and row[0].isdigit(): ints_found += 1
            if ints_found > len(raw_grid) * 0.5: has_row_header = True
        except: pass
            
        start_r = 1 if has_col_header else 0
        start_c = 1 if has_row_header else 0
        
        # Extract Data
        self.data_map = []
        unique_set = set()
        max_cols = 0
        extracted_rows = []
        for r in range(start_r, len(raw_grid)):
            row_items = raw_grid[r]
            if start_c < len(row_items):
                data_items = row_items[start_c:]
            else:
                data_items = []
            extracted_rows.append(data_items)
            max_cols = max(max_cols, len(data_items))
            
        self.rows = len(extracted_rows)
        self.cols = max_cols
        
        self.data_map = []
        for r_idx, row_data in enumerate(extracted_rows):
            row_data = ["" if v in reject else v for v in row_data]
            padded = row_data + [""] * (max_cols - len(row_data))
            self.data_map.append(padded)
            for val in padded:
                if val: unique_set.add(val)

        # Assign colors
        self.unique_keys = {}
        sorted_keys = sorted(list(unique_set))
        for i, k in enumerate(sorted_keys):
            rgb = self.colors_palette[i % len(self.colors_palette)]
            self.unique_keys[k] = QColor(rgb[0], rgb[1], rgb[2], 100) 
        
    def get_color(self, r, c) -> QColor:
        if r < 0 or r >= len(self.data_map): return None
        row = self.data_map[r]
        if c < 0 or c >= len(row): return None
        key = row[c]
        if not key: return None 
        return self.unique_keys.get(key, None)

    def get_key(self, r, c) -> str:
        if r < 0 or r >= len(self.data_map): return ""
        row = self.data_map[r]
        if c < 0 or c >= len(row): return ""
        return row[c]

class CoordinateTransform:
    def __init__(self, angle_deg, img_w, img_h):
        self.angle = angle_deg
        self.img_w = img_w
        self.img_h = img_h
        self.img_cx = img_w / 2.0
        self.img_cy = img_h / 2.0
        rad_deskew = np.radians(-self.angle)
        self.cos_inv = np.cos(rad_deskew)
        self.sin_inv = np.sin(rad_deskew)
        rad_raw = np.radians(self.angle)
        self.cos_raw = np.cos(rad_raw)
        self.sin_raw = np.sin(rad_raw)

    def raw_to_deskew(self, rx, ry):
        dx = rx - self.img_cx
        dy = ry - self.img_cy
        rot_x = dx * self.cos_inv - dy * self.sin_inv
        rot_y = dx * self.sin_inv + dy * self.cos_inv
        return self.img_cx + rot_x, self.img_cy + rot_y

    def deskew_to_raw(self, gx, gy):
        dx = gx - self.img_cx
        dy = gy - self.img_cy
        rot_x = dx * self.cos_raw - dy * self.sin_raw
        rot_y = dx * self.sin_raw + dy * self.cos_raw
        return self.img_cx + rot_x, self.img_cy + rot_y

class ImageNormalizer:
    @staticmethod
    def process(data: np.ndarray, win_lo: float, win_hi: float, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
        if data.dtype != np.float32 and data.dtype != np.float64:
             data = data.astype(np.float32)
        win_range = max(1.0, win_hi - win_lo)
        res = (data - win_lo) / win_range
        res = res * scale + offset
        res = np.clip(res, 0.0, 1.0)
        return (res * 255.0).astype(np.uint8)
