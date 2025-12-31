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
            self.owned = True
        elif isinstance(source, Image.Image):
            self.img = source
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
