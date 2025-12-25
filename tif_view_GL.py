import sys
import numpy as np
import tifffile
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QSlider, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
    QSpinBox, QSplitter
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QAction
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QSurfaceFormat, QAction, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QSlider, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
    QSpinBox, QSplitter, QDockWidget, QPlainTextEdit, QPushButton,
    QGroupBox, QFormLayout, QDoubleSpinBox
)
import re

from OpenGL.GL import *

# ==================================================================================
# 1. Utils
# ==================================================================================
def to_gray2d_uint16(arr: np.ndarray, z_index: int = 0) -> np.ndarray:
    """
    Extracts a single 2D grayscale layer from a multi-dim array.
    Returns (H, W) uint16.
    """
    arr = np.asarray(arr)
    
    # Handle Z-stack or Time dimensions by slicing
    # Expected inputs: (H,W), (H,W,C), (Z,H,W), (Z,H,W,C)
    
    # Simple logic: flatten until we have [Z_remaining, H, W, C_maybe]
    # This is a heuristic. For strict TIF handling, we assume:
    # If ndim=3 and last dim is not 3/4 -> (Z, H, W)
    # If ndim=3 and last dim is 3/4 -> (H, W, RGB)
    # If ndim=4 -> (Z, H, W, RGB) or (T, Z, H, W) ... let's assume (Z, H, W, C) for now
    
    # 1. Select Z
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


# ==================================================================================
# 2. Tiling System
# ==================================================================================
class Tile:
    """
    Represents a single GPU texture tile.
    """
    def __init__(self, x, y, width, height, data):
        self.x = x  # Global X position
        self.y = y  # Global Y position
        self.w = width
        self.h = height
        self.data = np.ascontiguousarray(data, dtype=np.uint16) # CPU copy
        self.tex_id = None
        self.is_uploaded = False

    def upload(self):
        if self.is_uploaded: 
            return
            
        if self.tex_id is None:
            self.tex_id = glGenTextures(1)
            
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        # 16-bit single channel texture
        # Internal format: GL_R16 (16-bit Red/Gray)
        # Format: GL_RED
        # Type: GL_UNSIGNED_SHORT
        
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R16, 
            self.w, self.h, 0, 
            GL_RED, GL_UNSIGNED_SHORT, self.data
        )
        
        self.is_uploaded = True
        
    def cleanup(self):
        if self.tex_id:
            glDeleteTextures([self.tex_id])
            self.tex_id = None
        self.is_uploaded = False

class TiledImage:
    """
    Manages the grid of tiles for a single image layer.
    """
    TILE_SIZE = 4096 # Safe for most modern GPUs (limit is usually 8192 or 16384)

    def __init__(self, full_image: np.ndarray):
        """
        full_image: (H, W) uint16 numpy array
        """
        self.h, self.w = full_image.shape
        self.tiles = []
        
        # Create tiles
        for y in range(0, self.h, self.TILE_SIZE):
            for x in range(0, self.w, self.TILE_SIZE):
                h_chunk = min(self.TILE_SIZE, self.h - y)
                w_chunk = min(self.TILE_SIZE, self.w - x)
                
                # Extract sub-region
                sub_img = full_image[y:y+h_chunk, x:x+w_chunk]
                
                tile = Tile(x, y, w_chunk, h_chunk, sub_img)
                self.tiles.append(tile)

    def upload_all(self):
        for t in self.tiles:
            t.upload()

    def cleanup(self):
        for t in self.tiles:
            t.cleanup()



# ==================================================================================
# 2.1 Bonding Map & Grid System
# ==================================================================================
class BondingMap:
    """
    Parses and stores the bonding map data (keys/labels in a grid).
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
            # Usually starts with empty or '0' or whatever
            # Let's count how many look like ints
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
        valid_rows = 0
        
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
            padded = row_data + [""] * (max_cols - len(row_items))
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


class GridConfig:
    def __init__(self):
        self.visible = False
        self.start_x = 100.0
        self.start_y = 100.0
        self.pitch_x = 200.0
        self.pitch_y = 200.0
        self.rows = 1
        self.cols = 1
        self.line_width = 1.0 # Reduced from 2.0
        self.opacity = 0.5    # Default 50% opacity
        
# ==================================================================================
# 3. OpenGL Widget
# ==================================================================================
class GLImageWidget(QOpenGLWidget):
    grid_params_changed = Signal()

    def __init__(self):
        super().__init__()
        self.tiled_image: TiledImage = None
        
        self.win_lo = 0.0
        self.win_hi = 1.0
        
        # Grid & Map
        self.grid_cfg = GridConfig()
        self.bonding_map = None # Single BondingMap Instance
        self.map_tex = None
        
        self.setFocusPolicy(Qt.StrongFocus) # Enable keyboard events
        
        # View transform
        # Image coordinates: (0,0) is top-left, (W,H) is bottom-right.
        # We map this to Normalized Device Coords (NDC) in shader.
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        self._last_mouse = None
        

        # Shader vars
        self.prog = None
        
    def update_map_texture(self):
        """Creates/Updates the texture for the bonding map labels"""
        if not self.bonding_map: 
            return
            
        rows = self.bonding_map.rows
        cols = self.bonding_map.cols
        if rows == 0 or cols == 0:
            return
            
        # Create a tiny texture (cols x rows)
        # Format: GL_RGBA, GL_FLOAT? Or GL_UNSIGNED_BYTE
        # We'll use simple RGBA uint8
        
        # Build buffer
        arr = np.zeros((rows, cols, 4), dtype=np.uint8)
        
        for r in range(rows):
            for c in range(cols):
                color = self.bonding_map.get_color(r, c)
                if color:
                    arr[r, c] = [color.red(), color.green(), color.blue(), color.alpha()]
                else:
                    arr[r, c] = [0, 0, 0, 0] # Transparent
                    
        # Upload
        self.makeCurrent()
        if self.map_tex:
            glDeleteTextures([self.map_tex])
            
        self.map_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.map_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Note: rows is height, cols is width. 
        # But OpenGL expects width, height.
        # arr shape is (H, W, 4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr)
        
        self.doneCurrent()
        self.update()

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # Enable basic GL states
        # Enable blending for overlay
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        
        # Create Shader
        self.vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vs, """
        #version 330
        
        layout(location = 0) in vec2 aPos; // 0..1 quad
        layout(location = 1) in vec2 aTex; 
        
        uniform vec2 tileOffset; // In Image Pixels
        uniform vec2 tileSize;   // In Image Pixels
        uniform vec2 imageSize;  // Total Image Size
        
        uniform float zoom;
        uniform vec2 pan;     
        uniform vec2 viewSize; 
        
        out vec2 uv;
        out vec2 pixelPos; // Pass global pixel position to fragment for grid
        
        void main() {
            pixelPos = tileOffset + aPos * tileSize;
            
            vec2 imgCenter = imageSize * 0.5;
            vec2 viewCenter = imgCenter - pan; 
            
            vec2 finalPos = (pixelPos - viewCenter) * zoom / (viewSize * 0.5);
            finalPos.y = -finalPos.y; 
            
            gl_Position = vec4(finalPos, 0.0, 1.0);
            uv = aTex;
        }
        """)
        
        self.fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fs, """
        #version 330
        in vec2 uv;
        in vec2 pixelPos;
        
        out vec4 color;
        
        uniform sampler2D tex;
        uniform float win_lo;
        uniform float win_hi;
        
        // Grid / Map Params
        uniform int show_grid;
        uniform float grid_x;
        uniform float grid_y;
        uniform float pitch_x;
        uniform float pitch_y;
        uniform int grid_rows;
        uniform int grid_cols;
        uniform float line_width;
        uniform float zoom; // to adjust line width?
        
        uniform sampler2D map_tex;
        uniform int has_map;
        uniform float grid_opacity;
        
        void main() {
            // 1. Base Image
            float val = texture(tex, uv).r; 
            float res = (val - win_lo) / (win_hi - win_lo);
            res = clamp(res, 0.0, 1.0);
            
            vec4 baseColor = vec4(res, res, res, 1.0);
            
            // 2. Grid & Overlay
            vec4 overlayColor = vec4(0.0);
            
            if (show_grid == 1) {
                // Determine if we are inside the grid bounding box
                float relX = pixelPos.x - grid_x;
                float relY = pixelPos.y - grid_y;
                
                float totalW = float(grid_cols) * pitch_x;
                float totalH = float(grid_rows) * pitch_y;
                
                // Check bounds
                if (relX >= 0.0 && relX <= totalW && relY >= 0.0 && relY <= totalH) {
                    
                    // Grid Cell Index
                    int col = int(relX / pitch_x);
                    int row = int(relY / pitch_y);
                    
                    // Local coords in cell
                    float localX = relX - float(col) * pitch_x;
                    float localY = relY - float(row) * pitch_y;
                    
                    // Compute borders
                    // We want constant screen-space line width
                    // Shader 'line_width' is in pixels? 
                    // pixelPos is in Image Pixels.
                    // Screen Width = Image Width * Zoom.
                    // So 1 Image Pixel = Zoom Screen Pixels.
                    // We want N Screen Pixel width -> N / Zoom Image Pixels.
                    
                    float thresh = line_width / max(zoom, 0.001);
                    
                    bool is_border = false;
                    if (localX < thresh || localX > (pitch_x - thresh)) is_border = true;
                    if (localY < thresh || localY > (pitch_y - thresh)) is_border = true;
                    
                    if (is_border) {
                        overlayColor = vec4(1.0, 1.0, 0.0, 0.6); // Yellow Grid lines
                    } else {
                        // Inside Cell - Check Map
                        if (has_map == 1) {
                            // Fetch Map Texture
                            // Texture Coords: (col / grid_cols, row / grid_rows)
                            // Note: standard UV origin (0,0) is usually Top-Left?
                            // In OpenGL Texture (0,0) is Bottom-Left.
                            // But our map is Row 0 at Top? 
                            // Let's assume Row 0 (top) maps to V=0? Or V=1?
                            // Ideally: texelFetch is safer for integer grid
                            
                            // Let's use texelFetch
                            // texelFetch(tex, ivec2(x, y), lod)
                            // y needs to be inverted if texture is loaded upside down?
                            // We loaded it linearly. GL convention: row 0 is bottom.
                            // But usually we want row 0 at top.
                            // So let's invert row index for look up: (rows - 1 - row)
                            
                            ivec2 mapCoord = ivec2(col, grid_rows - 1 - row);
                            vec4 mapVal = texelFetch(map_tex, mapCoord, 0);
                            
                            if (mapVal.a > 0.0) {
                                overlayColor = mapVal;
                            }
                        }
                    }
                }
            }
            
            // Blend
            // Apply global opacity to the overlay
            overlayColor.a *= grid_opacity;
            
            // Manual blend: src * src_a + dst * (1-src_a)
            // dst is baseColor
            
            color = vec4(mix(baseColor.rgb, overlayColor.rgb, overlayColor.a), 1.0);
        }
        """)
        
        self.prog = glCreateProgram()
        glAttachShader(self.prog, self.vs)
        glAttachShader(self.prog, self.fs)
        glLinkProgram(self.prog)
        
        # Check errors
        if not glGetProgramiv(self.prog, GL_LINK_STATUS):
            print("Link Error:", glGetProgramInfoLog(self.prog))
            
        # Create a simple Quad VAO (0,0) to (1,1)
        # We will scale this in the vertex shader using 'tileOffset' and 'tileSize'
        vertices = np.array([
            0.0, 0.0,  0.0, 0.0,
            1.0, 0.0,  1.0, 0.0,
            0.0, 1.0,  0.0, 1.0,
            1.0, 1.0,  1.0, 1.0,
        ], dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Attr 0: Pos
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Attr 1: UV
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*4, ctypes.c_void_p(2*4))
        glEnableVertexAttribArray(1)
        

    def set_tiled_image(self, tiled_img: TiledImage):
        # Cleanup old
        if self.tiled_image:
            self.tiled_image.cleanup()
        
        self.tiled_image = tiled_img
        
        # Context must be current for upload
        self.makeCurrent()
        self.tiled_image.upload_all()
        self.doneCurrent()
        
        self.update()

    def set_window_level(self, lo, hi):
        self.win_lo = lo
        self.win_hi = hi
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        if not self.tiled_image:
            return
            
        glUseProgram(self.prog)
        
        # Global uniforms
        w = self.width()
        h = self.height()
        img_w = self.tiled_image.w
        img_h = self.tiled_image.h
        
        glUniform1f(glGetUniformLocation(self.prog, "zoom"), self.zoom)
        glUniform2f(glGetUniformLocation(self.prog, "pan"), self.pan_x, self.pan_y)
        glUniform2f(glGetUniformLocation(self.prog, "viewSize"), w, h)
        glUniform2f(glGetUniformLocation(self.prog, "imageSize"), img_w, img_h)
        glUniform1f(glGetUniformLocation(self.prog, "win_lo"), self.win_lo)
        glUniform1f(glGetUniformLocation(self.prog, "win_hi"), self.win_hi)
        

        glUniform1i(glGetUniformLocation(self.prog, "tex"), 0)
        
        # Grid Uniforms
        glUniform1i(glGetUniformLocation(self.prog, "show_grid"), 1 if self.grid_cfg.visible else 0)
        glUniform1f(glGetUniformLocation(self.prog, "grid_x"), self.grid_cfg.start_x)
        glUniform1f(glGetUniformLocation(self.prog, "grid_y"), self.grid_cfg.start_y)
        glUniform1f(glGetUniformLocation(self.prog, "pitch_x"), self.grid_cfg.pitch_x)
        glUniform1f(glGetUniformLocation(self.prog, "pitch_y"), self.grid_cfg.pitch_y)
        glUniform1i(glGetUniformLocation(self.prog, "grid_rows"), self.grid_cfg.rows)
        glUniform1i(glGetUniformLocation(self.prog, "grid_cols"), self.grid_cfg.cols)
        glUniform1f(glGetUniformLocation(self.prog, "line_width"), self.grid_cfg.line_width)
        glUniform1f(glGetUniformLocation(self.prog, "grid_opacity"), self.grid_cfg.opacity)
        
        glUniform1i(glGetUniformLocation(self.prog, "has_map"), 1 if (self.bonding_map is not None and self.map_tex is not None) else 0)
        
        if self.map_tex:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.map_tex)
            glUniform1i(glGetUniformLocation(self.prog, "map_tex"), 1)
        
        glBindVertexArray(self.vao)
        glActiveTexture(GL_TEXTURE0)
        
        # Draw each tile
        for tile in self.tiled_image.tiles:
            if not tile.is_uploaded:
                continue
                
            glBindTexture(GL_TEXTURE_2D, tile.tex_id)
            
            glUniform2f(glGetUniformLocation(self.prog, "tileOffset"), tile.x, tile.y)
            glUniform2f(glGetUniformLocation(self.prog, "tileSize"), tile.w, tile.h)
            
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            
    # -----------------------------------------------
    # Mouse Interaction
    # -----------------------------------------------
    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta == 0: return
        
        factor = 1.1 if delta > 0 else 0.9
        
        # Mouse pos in widget
        mx = e.position().x()
        my = e.position().y()
        
        # We need to adjust pan so that the point under mouse stays stationary.
        # Current Screen Pos X = (PixelX - CenterX) * Zoom * (2/Width)
        # New Screen Pos X should be same.
        
        # It's easier to calculate "World Point under Mouse" before zoom
        # then adjust pan so "World Point" is still at Mouse after zoom.
        
        # But for now, simple centered zoom + drift approach (ImageJ styleish)
        # To strictly follow ImageJ 'zoom to mouse':
        #   WorldX = (ScreenX / (Zoom * 2/Width)) + CenterX
        #   ...
        
        # Let's stick to the simpler accumulation logic from before but careful
        self.zoom *= factor
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._last_mouse = e.position()
            self.setFocus() # Ensure we get keyboard focus

    def mouseMoveEvent(self, e):
        if self._last_mouse is None:
            return
            
        dx = e.position().x() - self._last_mouse.x()
        dy = e.position().y() - self._last_mouse.y()
        
        # Check modifiers
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.ShiftModifier and self.grid_cfg.visible:
            # Move Grid
            # dx is screen pixels.
            # grid_x/y are Image Pixels.
            # We need to map Screen Delta -> Image Delta.
            # 1 Image Pixel = Zoom Screen Pixels
            # DeltaImage = DeltaScreen / Zoom
            
            img_dx = dx / self.zoom
            img_dy = dy / self.zoom
            
            self.grid_cfg.start_x += img_dx
            self.grid_cfg.start_y += img_dy
            
            # Notify main window
            self.grid_params_changed.emit()
            
        else:
            # Pan
            self.pan_x += dx / self.zoom
            self.pan_y += dy / self.zoom 
        
        self._last_mouse = e.position()
        self.update()
        
    def keyPressEvent(self, e):
        if not self.grid_cfg.visible:
            super().keyPressEvent(e)
            return

        # Nudge amount (in pixels)
        step = 1.0 
        if e.modifiers() == Qt.ShiftModifier:
            step = 10.0
            
        if e.key() == Qt.Key_Left:
            self.grid_cfg.start_x -= step
        elif e.key() == Qt.Key_Right:
            self.grid_cfg.start_x += step
        elif e.key() == Qt.Key_Up:
            self.grid_cfg.start_y -= step
        elif e.key() == Qt.Key_Down:
            self.grid_cfg.start_y += step
        else:
            super().keyPressEvent(e)
            return
            
        self.grid_params_changed.emit()
        self.update()


    def mouseReleaseEvent(self, e):
        self._last_mouse = None


# ==================================================================================
# 4. Main Window
# ==================================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tiled TIF Viewer (OpenGl Mosaic)")
        self.resize(1280, 720)
        
        self.full_data = None #(Z, H, W) or (H, W)
        self.current_z = 0
        
        self.glw = GLImageWidget()
        
        # Controls
        self.spin_layer = QSpinBox()
        self.spin_layer.setRange(0, 0)
        self.spin_layer.setEnabled(False)
        self.spin_layer.valueChanged.connect(self.on_layer_changed)
        
        self.slider_lo = QSlider(Qt.Horizontal)
        self.slider_lo.setRange(0, 65535)
        self.slider_lo.setValue(0)
        self.slider_lo.valueChanged.connect(self.on_window_changed)
        
        self.slider_hi = QSlider(Qt.Horizontal)
        self.slider_hi.setRange(0, 65535)
        self.slider_hi.setValue(65535)
        self.slider_hi.valueChanged.connect(self.on_window_changed)
        
        # Layouts
        ctrl_widget = QWidget()
        h = QHBoxLayout(ctrl_widget)
        h.addWidget(QLabel("Layer:"))
        h.addWidget(self.spin_layer)
        h.addWidget(QLabel("Min:"))
        h.addWidget(self.slider_lo)
        h.addWidget(QLabel("Max:"))
        h.addWidget(self.slider_hi)
        
        root = QWidget()
        v = QVBoxLayout(root)
        v.addWidget(self.glw, 1)
        v.addWidget(ctrl_widget)
        self.setCentralWidget(root)
        
        # Toolbar
        tb = QToolBar()
        self.addToolBar(tb)
        a = QAction("Open TIF", self)
        a.triggered.connect(self.open_file)
        tb.addAction(a)

        # ---------------------------------------------------------
        # Dock Widget for Grid/Map Controls
        # ---------------------------------------------------------
        dock = QDockWidget("Grid & Map Tools", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        
        # 1. Grid Controls
        gb_grid = QGroupBox("Grid Configuration")
        form = QFormLayout(gb_grid)
        
        self.chk_grid_show = QPushButton("Show Grid: OFF")
        self.chk_grid_show.setCheckable(True)
        self.chk_grid_show.toggled.connect(self.toggle_grid)
        form.addRow(self.chk_grid_show)
        
        self.sb_grid_x = QDoubleSpinBox(); self.sb_grid_x.setRange(-10000, 50000); self.sb_grid_x.setValue(100)
        self.sb_grid_y = QDoubleSpinBox(); self.sb_grid_y.setRange(-10000, 50000); self.sb_grid_y.setValue(100)
        form.addRow("Origin X:", self.sb_grid_x)
        form.addRow("Origin Y:", self.sb_grid_y)
        
        self.sb_pitch_x = QDoubleSpinBox(); self.sb_pitch_x.setRange(1, 10000); self.sb_pitch_x.setValue(200)
        self.sb_pitch_y = QDoubleSpinBox(); self.sb_pitch_y.setRange(1, 10000); self.sb_pitch_y.setValue(200)
        form.addRow("Pitch X:", self.sb_pitch_x)
        form.addRow("Pitch Y:", self.sb_pitch_y)
        
        self.sb_rows = QSpinBox(); self.sb_rows.setRange(1, 1000); self.sb_rows.setValue(5)
        self.sb_cols = QSpinBox(); self.sb_cols.setRange(1, 1000); self.sb_cols.setValue(5)
        form.addRow("Rows:", self.sb_rows)
        form.addRow("Cols:", self.sb_cols)
        
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(50)
        form.addRow("Opacity:", self.slider_opacity)
        
        # Connect signals
        for w in [self.sb_grid_x, self.sb_grid_y, self.sb_pitch_x, self.sb_pitch_y]:
            w.valueChanged.connect(self.update_grid_params)
        for w in [self.sb_rows, self.sb_cols]:
            w.valueChanged.connect(self.update_grid_params)
            
        self.slider_opacity.valueChanged.connect(self.update_grid_params)
            
        self.glw.grid_params_changed.connect(self.on_grid_moved_by_input)
            
        dock_layout.addWidget(gb_grid)
        
        # 2. Bonding Map Paste
        gb_map = QGroupBox("Bonding Map (Paste Excel)")
        v_map = QVBoxLayout(gb_map)
        self.txt_map = QPlainTextEdit()
        self.txt_map.setPlaceholderText("Paste grid data here (tab separated)...")
        v_map.addWidget(self.txt_map)
        
        btn_import_map = QPushButton("Import Map Data")
        btn_import_map.clicked.connect(self.import_map_data)
        v_map.addWidget(btn_import_map)
        
        dock_layout.addWidget(gb_map)
        
        # 3. Extraction
        gb_extract = QGroupBox("Extraction")
        v_ext = QVBoxLayout(gb_extract)
        btn_extract = QPushButton("Extract Patches")
        btn_extract.clicked.connect(self.extract_patches)
        v_ext.addWidget(btn_extract)
        dock_layout.addWidget(gb_extract)
        
        dock_layout.addStretch()
        dock.setWidget(dock_content)

    def toggle_grid(self, checked):
        self.chk_grid_show.setText(f"Show Grid: {'ON' if checked else 'OFF'}")
        self.glw.grid_cfg.visible = checked
        self.glw.update()

    def update_grid_params(self):
        cfg = self.glw.grid_cfg
        cfg.start_x = self.sb_grid_x.value()
        cfg.start_y = self.sb_grid_y.value()
        cfg.pitch_x = self.sb_pitch_x.value()
        cfg.pitch_y = self.sb_pitch_y.value()
        cfg.rows = self.sb_rows.value()
        cfg.cols = self.sb_cols.value()
        cfg.opacity = self.slider_opacity.value() / 100.0
        self.glw.update()

    def on_grid_moved_by_input(self):
        # Update spinboxes without triggering recursive updates
        cfg = self.glw.grid_cfg
        self.sb_grid_x.blockSignals(True)
        self.sb_grid_y.blockSignals(True)
        
        self.sb_grid_x.setValue(cfg.start_x)
        self.sb_grid_y.setValue(cfg.start_y)
        
        self.sb_grid_x.blockSignals(False)
        self.sb_grid_y.blockSignals(False)

    def import_map_data(self):
        text = self.txt_map.toPlainText()
        if not text.strip():
            return
            
        bmap = BondingMap()
        bmap.parse_text(text)
        
        if bmap.rows > 0 and bmap.cols > 0:
            self.glw.bonding_map = bmap
            self.glw.update_map_texture()
            
            # Auto-update grid dimensions to match map
            self.sb_rows.setValue(bmap.rows)
            self.sb_cols.setValue(bmap.cols)
            
            # Enable grid if not
            if not self.chk_grid_show.isChecked():
                self.chk_grid_show.setChecked(True)
                
            print(f"Map Imported: {bmap.rows}x{bmap.cols}, Unique Keys: {len(bmap.unique_keys)}")
            
    def extract_patches(self):
        if self.full_data is None:
            print("No image loaded.")
            return

        # Choose directory
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir:
            return
            
        print(f"Extracting to {out_dir}...")
        cfg = self.glw.grid_cfg
        
        import os
        # import cv2 # Removed

        
        # Ensure we have data to extract from
        # self.full_data is either (Z, H, W) or (H, W) or (H, W, C)
        # We need the current layer.
        
        # Reuse to_gray2d helper or operate on current layer?
        # Let's operate on the currently displayed layer for now, or all layers?
        # User request: "Extract patch of bonded chips" -> Usually implies the visual layer.
        # But maybe they want raw data? Let's implement extraction of the CURRENT layer first.
        
        # Get current layer data
        try:
            current_layer_img = to_gray2d_uint16(self.full_data, self.current_z)
        except:
            print("Failed to get current layer for extraction.")
            return

        h, w = current_layer_img.shape
        
        count = 0
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                # Calculate box
                x0 = int(cfg.start_x + c * cfg.pitch_x)
                y0 = int(cfg.start_y + r * cfg.pitch_y)
                x1 = int(x0 + cfg.pitch_x)
                y1 = int(y0 + cfg.pitch_y)
                
                # Check bounds (partial clipping ok, or skip?)
                # Let's clip coordinates
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(w, x1); y1 = min(h, y1)
                
                if x1 <= x0 or y1 <= y0:
                    continue
                    
                patch = current_layer_img[y0:y1, x0:x1]
                
                # Filename: layer_row_col_label.tif
                label = "unknown"
                if self.glw.bonding_map:
                    k = self.glw.bonding_map.get_key(r, c)
                    if k: label = k
                
                fname = f"L{self.current_z}_R{r}_C{c}_{label}.tif"
                fpath = os.path.join(out_dir, fname)
                
                tifffile.imwrite(fpath, patch)
                count += 1
                
        print(f"Extracted {count} patches.")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "TIFF (*.tif *.tiff)")
        if not path:
            return
            
        print(f"Loading {path}...")
        try:
            # Memory map if possible for huge files? 
            # tifffile.memmap is good but let's try standard read first (user has RAM)
            data = tifffile.imread(path)
            print(f"Shape: {data.shape}, Dtype: {data.dtype}")
            
            self.full_data = data
            
            # Detect layers
            if self.full_data.ndim > 2 and self.full_data.shape[0] > 1:
                # Assume dim 0 is Layers if not RGB
                 # Refine layer detection
                is_rgb = (self.full_data.ndim == 3 and self.full_data.shape[-1] in (3,4))
                if not is_rgb:
                    num_layers = self.full_data.shape[0]
                    self.spin_layer.setRange(0, num_layers - 1)
                    self.spin_layer.setValue(0)
                    self.spin_layer.setEnabled(True)
                else:
                    self.spin_layer.setEnabled(False)
            else:
                self.spin_layer.setEnabled(False)
                
            self.load_layer(0)
            
        except Exception as e:
            print(f"Error loading file: {e}")

    def load_layer(self, z_index):
        if self.full_data is None:
            return
            
        print(f"Processing Layer {z_index}...")
        img_2d = to_gray2d_uint16(self.full_data, z_index)
        
        print("Creating tiles...")
        tiled = TiledImage(img_2d)
        
        print("Uploading to GPU...")
        self.glw.set_tiled_image(tiled)
        
        # Reset view if first load?
        # self.glw.zoom = 1.0 ... maybe preserve view across layers

    def on_layer_changed(self, val):
        self.current_z = val
        self.load_layer(val)

    def on_window_changed(self):
        lo = self.slider_lo.value()
        hi = self.slider_hi.value()
        if hi <= lo: hi = lo + 1
        
        self.glw.set_window_level(lo / 65535.0, hi / 65535.0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
