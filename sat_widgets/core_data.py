import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable limit
from PySide6.QtGui import QColor
import json
import os
import math

# Use cv2 for speed if available, else fallback? 
# Let's try to stick to PIL/Numpy to avoid dependency issues unless confirmed.
# PIL transform is decent.

def to_gray2d_uint16(arr: np.ndarray, z_index: int = 0) -> np.ndarray:
    """
    Extracts a single 2D grayscale layer from a multi-dim array.
    Returns (H, W) uint16.
    """
    if hasattr(arr, 'get_crop') and not hasattr(arr, '__getitem__'):
        raise TypeError("Cannot slice Virtual Stack. Use get_crop or set_image(stack).")

    current_frame = arr
    if hasattr(arr, 'ndim'):
        if arr.ndim == 4:
            current_frame = arr[z_index]
        elif arr.ndim == 3:
            if arr.shape[-1] not in (3, 4):
                 current_frame = arr[z_index]
            else:
                 pass 
    elif isinstance(arr, list):
         current_frame = arr[z_index]
            
    # Now current_frame is (H, W) or (H, W, C)
    
    # 2. Convert RGB to Gray if needed
    if hasattr(current_frame, 'ndim') and current_frame.ndim == 3 and current_frame.shape[-1] in (3, 4):
        rgb = current_frame[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if current_frame.dtype == np.uint8:
            gray = gray * 257.0 
        current_frame = np.clip(gray, 0, 65535).astype(np.uint16)
        
    if hasattr(current_frame, 'ndim') and current_frame.ndim != 2:
         pass

    if hasattr(current_frame, 'dtype') and current_frame.dtype == np.uint8:
        return (current_frame.astype(np.uint16) * 257)

    return np.array(current_frame).astype(np.uint16, copy=False)

class LazyTiffStack:
    """Wraps a PIL.Image instance to provide lazy-loading of layers."""
    def __init__(self, source):
        if isinstance(source, str):
            self.img = Image.open(source)
            self.path = source
            self.owned = True
        elif isinstance(source, Image.Image):
            self.img = source
            self.path = getattr(source, 'filename', None)
            self.owned = False
        else:
            raise ValueError("Source must be a file path or PIL Image object")
            
        self.n_frames = getattr(self.img, 'n_frames', 1)
        self.width, self.height = self.img.size
        self.frame_start = 0
        self.frame_count = self.n_frames
        self.shape = (self.frame_count, self.height, self.width)
        self.ndim = 3
        
        if self.img.mode == 'I;16': self.dtype = np.uint16
        elif self.img.mode == 'I': self.dtype = np.uint32
        elif self.img.mode == 'F': self.dtype = np.float32
        else: self.dtype = np.uint8
            
    def set_range(self, start, count):
        self.frame_start = max(0, start)
        self.frame_count = min(count, self.n_frames - self.frame_start)
        self.shape = (self.frame_count, self.height, self.width)
        
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0: key += self.frame_count
            if key < 0 or key >= self.frame_count: raise IndexError("Index out of bounds")
            phys_idx = self.frame_start + key
            self.img.seek(phys_idx)
            return np.array(self.img)
        if isinstance(key, slice):
            start, stop, step = key.indices(self.frame_count)
            res = []
            for i in range(start, stop, step):
                phys_idx = self.frame_start + i
                self.img.seek(phys_idx)
                res.append(np.array(self.img))
            if not res: return np.zeros((0, self.height, self.width), dtype=self.dtype)
            return np.stack(res)
        if isinstance(key, tuple):
            z = key[0]
            if isinstance(z, int):
                phys_idx = self.frame_start + z
                self.img.seek(phys_idx)
                arr = np.array(self.img)
                return arr[key[1:]]
        return np.array(self.img)
    
    def get_crop(self, z, x, y, w, h):
        phys_idx = self.frame_start + z
        if phys_idx < 0: phys_idx += self.n_frames
        self.img.seek(phys_idx)
        img_w, img_h = self.width, self.height
        x = max(0, x); y = max(0, y)
        x2 = min(x + w, img_w); y2 = min(y + h, img_h)
        if x >= x2 or y >= y2: return np.zeros((h, w), dtype=self.dtype)
        region = self.img.crop((x, y, x2, y2))
        arr = np.array(region)
        out = np.zeros((h, w), dtype=arr.dtype)
        out_h, out_w = arr.shape[:2]
        out[0:out_h, 0:out_w] = arr
        if out.dtype == np.uint8: return (out.astype(np.uint16) * 257)
        return out

    def get_poly_crop(self, z, points):
        """
        Extracts a patch defined by 4 points (TL, TR, BR, BL) in Raw Coordinates.
        Returns a straightened (deskewed) numpy array.
        """
        # 1. Bounding Box in Raw Space
        poly = np.array(points) # [[x,y], ...]
        min_x = int(np.floor(poly[:, 0].min()))
        min_y = int(np.floor(poly[:, 1].min()))
        max_x = int(np.ceil(poly[:, 0].max()))
        max_y = int(np.ceil(poly[:, 1].max()))
        
        # Clamp to image bounds
        img_w, img_h = self.width, self.height
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)
        
        if max_x <= min_x or max_y <= min_y:
            return None
            
        # 2. Extract bounding rect (Raw)
        phys_idx = self.frame_start + z
        if phys_idx < 0: phys_idx += self.n_frames
        self.img.seek(phys_idx)
        raw_crop_pil = self.img.crop((min_x, min_y, max_x, max_y))
        
        # 3. Rotate and Straighten
        # Calculate angle from TL->TR vector
        # points expected order: TL, TR, BR, BL
        p0 = points[0]
        p1 = points[1]
        p3 = points[3] # BL
        
        # Width/Height of the target patch (Euclidean dist)
        dst_w = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        dst_h = int(np.hypot(p3[0] - p0[0], p3[1] - p0[1]))
        if dst_w <= 0 or dst_h <= 0: return None
        
        # Angle of the edge in Raw Space
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # We need to rotate the RAW content by -angle_deg to make it horizontal
        # PIL rotate is CCW. 
        # If line is angled +30 deg (down-right). We rotate by +30 to make it upright? 
        # No, if line is +30, we rotate by +30 to align X axis? 
        # Wait. Visualizing: Line is \ (positive slope in Y-down?).
        # If we Rotate image by Angle, we align axis.
        # Let's trust standard deskew logic: Rotate by Angle.
        # But `raw_crop_pil` center is NOT the patch center.
        # We need to handle translation.
        
        # Robust Approach: Use Affine Transform on the BBox Crop
        # Center of ROI in Raw Coords
        cx = np.mean(poly[:, 0])
        cy = np.mean(poly[:, 1])
        
        # Center of the Crop Image
        crop_cx = cx - min_x
        crop_cy = cy - min_y
        
        # Target Center (center of dst)
        tgt_cx = dst_w / 2.0
        tgt_cy = dst_h / 2.0
        
        # We want to map: Output(xy) -> Input(xy)
        # Transform Matrix M maps Output -> Input
        # 1. Output Center -> Origin
        # 2. Rotate
        # 3. Translate to Input Center
        
        # Using PIL.Image.transform with QUAD is easiest for 4 points
        # quad argument: 4 points in the Input Image (raw_crop_pil)
        # mapped to the corners of the Output Image (0,0, w,0, w,h, 0,h)
        
        # The points passed to `transform` must be relative to the image being transformed (raw_crop_pil)
        # So we shift `points` by (min_x, min_y)
        local_poly = []
        for p in points:
            local_poly.append(p[0] - min_x)
            local_poly.append(p[1] - min_y)
            
        # Flatten for quad (TL, BL, BR, TR) order?
        # PIL.Image.transform method=QUAD
        # data = (x0, y0, x1, y1, x2, y2, x3, y3) - NW, SW, SE, NE?
        # Doc: "The QUAD transform maps a quadrilateral (a region defined by four corners) in the *given image* to a rectangle of the given size."
        # "Data is an 8-tuple (x0, y0, x1, y1, x2, y2, x3, y3) which contain the upper left, lower left, lower right, and upper right corners of the source quadrilateral."
        # ORDER: TL, BL, BR, TR ??
        # Wait, standard is usually TL, TR, BR, BL?
        # Let's check PIL Docs or assume standard order.
        # StackOverflow: "NW, SW, SE, NE". So TL, BL, BR, TR.
        # My points are TL, TR, BR, BL.
        # So I need: p0, p3, p2, p1.
        
        quad_data = (
            local_poly[0], local_poly[1], # TL
            local_poly[6], local_poly[7], # BL (Index 3 * 2 = 6,7)
            local_poly[4], local_poly[5], # BR (Index 2)
            local_poly[2], local_poly[3]  # TR (Index 1)
        )
        print(quad_data, dst_w, dst_h, raw_crop_pil.size)
        out_pil = raw_crop_pil.transform(
            (dst_w, dst_h),
            method=3, # Image.QUAD (cannot import Image here easily if not at top)
            # method 3 is QUAD
            data=quad_data,
            resample=2 # Image.BILINEAR or BICUBIC
        )
        out = np.array(out_pil)
        
        if out.dtype == np.uint8:
            return (out.astype(np.uint16) * 257)
            
        return out

    def __len__(self):
        return self.frame_count

    def close(self): 
        if self.owned: 
            self.img.close()


class MosaicTiffStack:
    """
    Virtual Stack acting as the ORIGINAL RAW Image.
    Strategy: "Pre-assemble Deskewed -> Warp to Raw".
    1. Parses patches.json.
    2. Assembles 'Deskewed Layer Images' in memory (Lazy) by pasting cropped patches at 'bbox_deskewed'.
    3. Serves Raw Crops by warping (rotating) regions from the Deskewed Layer.
    """
    def __init__(self, patches_json_path):
        self.path = patches_json_path
        self.base_dir = os.path.dirname(patches_json_path)
        with open(patches_json_path, 'r') as f:
            data = json.load(f)
        self.grid_cfg = data.get("grid_config", {})
        self.patches = data.get("patches", [])
        self.angle = self.grid_cfg.get("angle", 0.0)
        
        # 1. Analyze Geometry
        self.layers = set()
        self.patches_by_layer = {}
        
        max_dx, max_dy = 0, 0
        max_rx, max_ry = 0, 0
        
        # Sample for Alignment (Grid/Deskewed -> Raw)
        offsets_x, offsets_y = [], []
        rad = np.radians(self.angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        for p in self.patches:
            z = p["layer"]
            self.layers.add(z)
            if z not in self.patches_by_layer: self.patches_by_layer[z] = []
            self.patches_by_layer[z].append(p)
            
            # Deskewed Bounds
            bd = p.get("bbox_deskewed", [])
            if bd:
                pts = np.array(bd)
                max_dx = max(max_dx, pts[:, 0].max())
                max_dy = max(max_dy, pts[:, 1].max())
                
                # Alignment Calculation
                br = p.get("bbox_raw", [])
                if br:
                    pts_r = np.array(br)
                    max_rx = max(max_rx, pts_r[:, 0].max())
                    max_ry = max(max_ry, pts_r[:, 1].max())
                    
                    # Center Comparison
                    rcx = (pts_r[:, 0].min() + pts_r[:, 0].max()) / 2.0
                    rcy = (pts_r[:, 1].min() + pts_r[:, 1].max()) / 2.0
                    
                    dcx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
                    dcy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
                    
                    rdx = dcx * cos_a - dcy * sin_a
                    rdy = dcx * sin_a + dcy * cos_a
                    
                    offsets_x.append(rcx - rdx)
                    offsets_y.append(rcy - rdy)
        
        # Canvas Sizes (Full Raw Extent from 0,0)
        self.width = int(max_rx + 2000) 
        self.height = int(max_ry + 2000)
        
        self.deskew_w = int(max_dx + 2000)
        self.deskew_h = int(max_dy + 2000)
        
        self.n_frames = len(self.layers) if self.layers else 1
        self.shape = (self.n_frames, self.height, self.width)
        self.ndim = 3
        self.dtype = np.uint8 
        self.layer_offset = min(list(self.layers))
        
        # Alignment Params
        if offsets_x:
            self.off_x = np.median(offsets_x)
            self.off_y = np.median(offsets_y)
        else:
            self.off_x = 0.0
            self.off_y = 0.0
            
        print(f"[MosaicDebug] Raw Size: {self.width}x{self.height}")
        print(f"[MosaicDebug] Deskew Size: {self.deskew_w}x{self.deskew_h}")
        print(f"[MosaicDebug] Offset: ({self.off_x:.2f}, {self.off_y:.2f})")
            
        self.cos_a = cos_a
        self.sin_a = sin_a
        self.layer_cache = {} 

        for i in list(self.layers):
            self._build_deskewed_layer(i)
        
    def _build_deskewed_layer(self, z):
        if z in self.layer_cache: return self.layer_cache[z]
        
        print(f"Building Deskewed Layer {z}...")
        canvas = Image.new("L", (self.deskew_w, self.deskew_h), 0)
        patches = self.patches_by_layer.get(z, [])
        first_patch = True
        
        for p in patches:
            bd = p.get("bbox_deskewed", [])
            if not bd: continue
            
            try:
                path = os.path.join(self.base_dir, p["filename"])
                img = Image.open(path)
                
                # Crop Title Bar
                if img.height > 40:
                    img = img.crop((0, 40, img.width, img.height))
                    
                
                pts = np.array(bd)
                min_x = int(pts[:, 0].min())
                min_y = int(pts[:, 1].min())
                w_box = int(pts[:, 0].max() - min_x)
                h_box = int(pts[:, 1].max() - min_y)
                print(p['filename'], min_x, min_y, w_box, h_box)

                if first_patch:
                    print(f"[MosaicDebug] P[0] Img: {img.size} vs BBox: {w_box}x{h_box}")
                    first_patch = False
                
                # Paste
                canvas.paste(img, (min_x, min_y))
                
            except Exception as e:
                pass
                
        self.layer_cache[z] = canvas.rotate(-self.angle, expand=True)
        return self.layer_cache[z]

    def deskew_to_raw(self, gx, gy):
        rx = (gx * self.cos_a - gy * self.sin_a) + self.off_x
        ry = (gx * self.sin_a + gy * self.cos_a) + self.off_y
        return rx, ry

    def raw_to_deskew(self, rx, ry):
        dx = rx - self.off_x
        dy = ry - self.off_y
        gx = dx * self.cos_a + dy * self.sin_a
        gy = -dx * self.sin_a + dy * self.cos_a
        return gx, gy

    def get_crop(self, z, x, y, w, h):
        phys_idx =  z
        self.img = self.layer_cache[phys_idx]
        img_w, img_h = self.width, self.height
        x = max(0, x); y = max(0, y)
        x2 = min(x + w, img_w); y2 = min(y + h, img_h)
        if x >= x2 or y >= y2: return np.zeros((h, w), dtype=self.dtype)
        region = self.img.crop((x, y, x2, y2))
        arr = np.array(region)
        out = np.zeros((h, w), dtype=arr.dtype)
        out_h, out_w = arr.shape[:2]
        out[0:out_h, 0:out_w] = arr
        if out.dtype == np.uint8: return (out.astype(np.uint16) * 257)
        return out

    def get_poly_crop(self, z, points):
        """
        Extracts a patch defined by 4 points (TL, TR, BR, BL) in Raw Coordinates.
        Returns a straightened (deskewed) numpy array.
        """
        # 1. Bounding Box in Raw Space
        poly = np.array(points) # [[x,y], ...]
        min_x = int(np.floor(poly[:, 0].min()))
        min_y = int(np.floor(poly[:, 1].min()))
        max_x = int(np.ceil(poly[:, 0].max()))
        max_y = int(np.ceil(poly[:, 1].max()))
        
        # Clamp to image bounds
        img_w, img_h = self.width, self.height
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)
        
        if max_x <= min_x or max_y <= min_y:
            return None
            
        # 2. Extract bounding rect (Raw)
        phys_idx = z - self.layer_offset
        self.img = self.layer_cache[phys_idx]

        raw_crop_pil = self.img.crop((min_x, min_y, max_x, max_y))

        # 3. Rotate and Straighten
        # Calculate angle from TL->TR vector
        # points expected order: TL, TR, BR, BL
        p0 = points[0]
        p1 = points[1]
        p3 = points[3] # BL
        
        # Width/Height of the target patch (Euclidean dist)
        dst_w = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        dst_h = int(np.hypot(p3[0] - p0[0], p3[1] - p0[1]))

        if dst_w <= 0 or dst_h <= 0: return None
        
        # Angle of the edge in Raw Space
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # We need to rotate the RAW content by -angle_deg to make it horizontal
        # PIL rotate is CCW. 
        # If line is angled +30 deg (down-right). We rotate by +30 to make it upright? 
        # No, if line is +30, we rotate by +30 to align X axis? 
        # Wait. Visualizing: Line is \ (positive slope in Y-down?).
        # If we Rotate image by Angle, we align axis.
        # Let's trust standard deskew logic: Rotate by Angle.
        # But `raw_crop_pil` center is NOT the patch center.
        # We need to handle translation.
        
        # Robust Approach: Use Affine Transform on the BBox Crop
        # Center of ROI in Raw Coords
        cx = np.mean(poly[:, 0])
        cy = np.mean(poly[:, 1])
        
        # Center of the Crop Image
        crop_cx = cx - min_x
        crop_cy = cy - min_y
        
        # Target Center (center of dst)
        tgt_cx = dst_w / 2.0
        tgt_cy = dst_h / 2.0
        
        # We want to map: Output(xy) -> Input(xy)
        # Transform Matrix M maps Output -> Input
        # 1. Output Center -> Origin
        # 2. Rotate
        # 3. Translate to Input Center
        
        # Using PIL.Image.transform with QUAD is easiest for 4 points
        # quad argument: 4 points in the Input Image (raw_crop_pil)
        # mapped to the corners of the Output Image (0,0, w,0, w,h, 0,h)
        
        # The points passed to `transform` must be relative to the image being transformed (raw_crop_pil)
        # So we shift `points` by (min_x, min_y)
        local_poly = []
        for p in points:
            local_poly.append(p[0] - min_x)
            local_poly.append(p[1] - min_y)
            
        quad_data = (
            local_poly[0], local_poly[1], # TL
            local_poly[6], local_poly[7], # BL (Index 3 * 2 = 6,7)
            local_poly[4], local_poly[5], # BR (Index 2)
            local_poly[2], local_poly[3]  # TR (Index 1)
        )

        print(quad_data, dst_w, dst_h, raw_crop_pil.size)
        
        out_pil = raw_crop_pil.transform(
            (dst_w, dst_h),
            method=Image.QUAD, # Image.QUAD (cannot import Image here easily if not at top)
            # method 3 is QUAD
            data=quad_data,
            resample=2 # Image.BILINEAR or BICUBIC
        )
        
        out = np.array(out_pil)
        Image.fromarray(out).save("out_pil.png")
        if out.dtype == np.uint8:
            return (out.astype(np.uint16) * 257)
            
        return out

    def __getitem__(self, key):
        phy_idx = key + self.layer_offset
        return np.array(self.layer_cache[phy_idx])


    def __len__(self): return self.n_frames
    def close(self): pass


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

    def parse_text(self, text: str):
        lines = text.strip().split('\n')
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
