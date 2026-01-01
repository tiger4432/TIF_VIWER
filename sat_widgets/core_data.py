import numpy as np
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable limit for large satellite images
from PySide6.QtGui import QColor
from PySide6.QtGui import QColor
import colorsys

# ==================================================================================
# Core Data Classes
# ==================================================================================

def to_gray2d_uint16(arr: np.ndarray, z_index: int = 0) -> np.ndarray:
    """
    Extracts a single 2D grayscale layer from a multi-dim array.
    Returns (H, W) uint16.
    """
    current_frame = arr
    if arr.ndim == 4:
        # (Z, H, W, C)
        current_frame = arr[z_index]
    elif arr.ndim == 3:
        if arr.shape[-1] not in (3, 4):
            # (Z, H, W) - Gray Z-stack
            current_frame = arr[z_index]
        else:
            # (H, W, RGB) - Single RGB image
            pass  # No Z slicing needed, but z_index should technically be 0
            
    # Now current_frame is (H, W) or (H, W, C)
    
    # 2. Convert RGB to Gray if needed
    if current_frame.ndim == 3 and current_frame.shape[-1] in (3, 4):
        # RGB(A) to Gray
        # Use float32 for precision
        rgb = current_frame[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        
        # Scale if it was uint8
        if current_frame.dtype == np.uint8:
            gray = gray * 257.0 # 0-255 -> 0-65535 map approximately
            
        current_frame = np.clip(gray, 0, 65535).astype(np.uint16)
        
    # 3. Ensure 2D (H, W)
    if current_frame.ndim != 2:
        raise ValueError(f"Could not convert to (H,W) uint16. Shape: {current_frame.shape}")
        
    # Check if we need to scale uint8 -> uint16
    if current_frame.dtype == np.uint8:
        return (current_frame.astype(np.uint16) * 257)

    return current_frame.astype(np.uint16, copy=False)

class LazyTiffStack:
    """
    Wraps a PIL.Image instance to provide lazy-loading of layers.
    Mimics a 3D numpy array (Z, H, W).
    """
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
        
        # Limit Logic
        self.frame_start = 0
        self.frame_count = self.n_frames
        
        # Determine dtype/shape helper
        self.shape = (self.frame_count, self.height, self.width)
        self.ndim = 3
        
    def set_range(self, start, count):
        self.frame_start = max(0, start)
        self.frame_count = min(count, self.n_frames - self.frame_start)
        self.shape = (self.frame_count, self.height, self.width)
        
        # Check mode for dtype
        # Common modes: 'L' (8-bit), 'I;16' (16-bit unsigned), 'I' (32-bit signed), 'F' (32-bit float)
        if self.img.mode == 'I;16':
            self.dtype = np.uint16
        elif self.img.mode == 'I':
            self.dtype = np.uint32 # Often used for 16-bit data in some PIL versions, but let's assume it maps to int/uint
        elif self.img.mode == 'F':
            self.dtype = np.float32
        else:
            self.dtype = np.uint8 # Default fallback (L, RGB, etc)
            
    def __getitem__(self, key):
        # Handle Integer Index [z]
        if isinstance(key, int):
            if key < 0: key += self.frame_count
            if key < 0 or key >= self.frame_count: raise IndexError("Index out of bounds")
            
            # Map logical index to physical index
            phys_idx = self.frame_start + key
            self.img.seek(phys_idx)
            return np.array(self.img)
            
        # Handle Slice [a:b]
        if isinstance(key, slice):
            # Indices relative to frame_count
            start, stop, step = key.indices(self.frame_count)
            res = []
            for i in range(start, stop, step):
                phys_idx = self.frame_start + i
                self.img.seek(phys_idx)
                res.append(np.array(self.img))
            if not res: return np.array([], dtype=self.dtype).reshape((0, self.height, self.width))
            return np.stack(res)
            
        # Handle tuple [z, y, x]
        if isinstance(key, tuple):
            z = key[0]
            if isinstance(z, int):
                # Optimization
                if z < 0: z += self.frame_count
                
                # Check Bounds
                if z < 0 or z >= self.frame_count: raise IndexError("Index out of bounds")
                
                phys_idx = self.frame_start + z
                self.img.seek(phys_idx)
                arr = np.array(self.img)
                return arr[key[1:]]
            else:
                 # Slice
                 layer = self[z] 
                 return layer[key[1:]]

        # Fallback
        return np.array(self.img)
    
    def get_crop(self, z, x, y, w, h):
        """Efficiently extracts a crop from a specific layer without loading full image."""
        phys_idx = self.frame_start + z
        if phys_idx < 0: phys_idx += self.n_frames
        if phys_idx < 0 or phys_idx >= self.n_frames:
             return np.zeros((h, w), dtype=self.dtype)

        self.img.seek(phys_idx)
        # PIL crop is lazy-ish. Converting to numpy triggers the load of just that region (mostly).
        # Bounds check
        if x < 0: x = 0
        if y < 0: y = 0
        # If crop exceeds bounds, PIL handles it or we should clamp?
        # PIL crop handles cropping outside? No, it pads or cuts.
        # Safe crop:
        img_w, img_h = self.width, self.height
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        
        if x >= x2 or y >= y2:
             return np.zeros((h, w), dtype=self.dtype)
             
        region = self.img.crop((x, y, x2, y2))
        arr = np.array(region)
        
        # Determine output array size (might be smaller if clipped)
        out = np.zeros((h, w), dtype=arr.dtype)
        out_h, out_w = arr.shape[:2]
        
        # Place in output (top-left alignment)
        out[0:out_h, 0:out_w] = arr
        
        # Promote uint8 to uint16 (match to_gray2d_uint16 logic)
        if out.dtype == np.uint8:
            return (out.astype(np.uint16) * 257)
            
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
    """Parses and stores the bonding map data (keys/labels in a grid).
    Assigns colors to unique keys.
    """
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.data_map = [] # List of lists (row-major)
        self.unique_keys = {} # key -> QColor
        self.colors_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (255, 128, 0), (128, 255, 0), (0, 128, 255),
            (128, 0, 255), (255, 0, 128), (128, 128, 128)
        ]

    def parse_text(self, text: str):
        """
        Parses tab/newline separated text (Excel copy-paste).
        Handles optional row/column headers (0, 1, 2...).
        """
        lines = text.strip().split('\n')
        if not lines: return

        # Pre-process into list of lists (preserving empty cells)
        raw_grid = []
        for line in lines:
            line = line.rstrip('\r\n')
            # Use split('\t') to preserve empty columns (Excel style)
            parts = line.split('\t')
            raw_grid.append([p.strip() for p in parts])
            
        if not raw_grid: return

        # Header Detection Heuristic
        # Check Row 0: Are they mostly sequential integers?
        has_col_header = False
        try:
            # Check a few items in first row
            ints_found = 0
            for cell in raw_grid[0]:
                if cell.isdigit():
                    ints_found += 1
            if ints_found > len(raw_grid[0]) * 0.5: # > 50% are digits
                has_col_header = True
        except:
            pass

        # Check Col 0: Are they mostly sequential integers?
        has_row_header = False
        try:
            ints_found = 0
            for row in raw_grid:
                if row and row[0].isdigit():
                    ints_found += 1
            if ints_found > len(raw_grid) * 0.5:
                has_row_header = True
        except:
            pass
            
        # Determine start indices
        start_r = 1 if has_col_header else 0
        start_c = 1 if has_row_header else 0
        
        # Extract Data
        self.data_map = []
        unique_set = set()
        
        # Scan to find max dimensions
        max_cols = 0
        
        # Read data
        extracted_rows = []
        for r in range(start_r, len(raw_grid)):
            row_items = raw_grid[r]
            # Slice off header column
            if start_c < len(row_items):
                data_items = row_items[start_c:]
            else:
                data_items = []
                
            extracted_rows.append(data_items)
            max_cols = max(max_cols, len(data_items))
            
        self.rows = len(extracted_rows)
        self.cols = max_cols
        
        # Normalize to rectangular grid
        self.data_map = []
        for r_idx, row_data in enumerate(extracted_rows):
            padded = row_data + [""] * (max_cols - len(row_data))
            self.data_map.append(padded)
            # Collect unique keys
            for val in padded:
                if val: unique_set.add(val)

        # Assign colors
        self.unique_keys = {}
        sorted_keys = sorted(list(unique_set))
        for i, k in enumerate(sorted_keys):
            rgb = self.colors_palette[i % len(self.colors_palette)]
            self.unique_keys[k] = QColor(rgb[0], rgb[1], rgb[2], 100) # Alpha 100 for overlay

    def get_color(self, r, c) -> QColor:
        if r < 0 or r >= len(self.data_map): return None
        row = self.data_map[r]
        if c < 0 or c >= len(row): return None
        key = row[c]
        if not key: return None # Empty string -> No color
        return self.unique_keys.get(key, None)

    def get_key(self, r, c) -> str:
        if r < 0 or r >= len(self.data_map): return ""
        row = self.data_map[r]
        if c < 0 or c >= len(row): return ""
        return row[c]

# ==================================================================================
# Coordinate Transform Logic
# ==================================================================================
class CoordinateTransform:
    """
    Central Logic for transforming coordinates between:
    1. Raw Image Space (Pixel Coordinates, u, v) - Where data is stored.
    2. Deskewed Space (Grid Coordinates, x, y) - Where data is visualized/aligned.
    
    Transformation:
    - Raw -> Deskewed: Rotate by -Angle around Image Center.
    - Deskewed -> Raw: Rotate by +Angle around Image Center.
    """
    def __init__(self, angle_deg, img_w, img_h):
        self.angle = angle_deg
        self.img_w = img_w
        self.img_h = img_h
        self.img_cx = img_w / 2.0
        self.img_cy = img_h / 2.0
        
        # Precompute for Deskew (-Angle)
        rad_deskew = np.radians(-self.angle)
        self.cos_inv = np.cos(rad_deskew)
        self.sin_inv = np.sin(rad_deskew)
        
        # Precompute for Raw (+Angle)
        rad_raw = np.radians(self.angle)
        self.cos_raw = np.cos(rad_raw)
        self.sin_raw = np.sin(rad_raw)

    def raw_to_deskew(self, rx, ry):
        """
        Transforms Raw Coordinates (rx, ry) to Deskewed Space (gx, gy).
        Formula:
        1. Translate to Center
        2. Rotate by -Angle
        3. Translate back
        """
        dx = rx - self.img_cx
        dy = ry - self.img_cy
        
        rot_x = dx * self.cos_inv - dy * self.sin_inv
        rot_y = dx * self.sin_inv + dy * self.cos_inv
        
        return self.img_cx + rot_x, self.img_cy + rot_y

    def deskew_to_raw(self, gx, gy):
        """
        Transforms Deskewed Coordinates (gx, gy) to Raw Space (rx, ry).
        Formula:
        1. Translate to Center
        2. Rotate by +Angle
        3. Translate back
        """
        dx = gx - self.img_cx
        dy = gy - self.img_cy
        
        rot_x = dx * self.cos_raw - dy * self.sin_raw
        rot_y = dx * self.sin_raw + dy * self.cos_raw
        
        return self.img_cx + rot_x, self.img_cy + rot_y

class ImageNormalizer:
    """
    Unified Image Processing Logic.
    Ensures consistency between Viewer (Shader) and Exporter (CPU).
    Pipeline: Normalization (Window Level) -> Calibration (Scale/Offset) -> Clip -> Uint8
    """
    @staticmethod
    def process(data: np.ndarray, win_lo: float, win_hi: float, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
        # Cast to float for precision
        if data.dtype != np.float32 and data.dtype != np.float64:
             data = data.astype(np.float32)
             
        # 1. Apply Window Leveling (Brightness/Contrast)
        # Matches Shader: normalized = (pixel - win_lo) / win_range
        win_range = max(1.0, win_hi - win_lo)
        res = (data - win_lo) / win_range
        
        # 2. Apply Calibration (Scale/Offset)
        # Matches Shader: final = normalized * scale + offset
        res = res * scale + offset
        
        # 3. Clip
        res = np.clip(res, 0.0, 1.0)
        
        # 4. Convert to uint8 (0-255)
        return (res * 255.0).astype(np.uint8)
