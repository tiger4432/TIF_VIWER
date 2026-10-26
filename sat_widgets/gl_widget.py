import sys
import math
import ctypes
import numpy as np

from OpenGL.GL import *
from OpenGL.GL import shaders

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QAction, QColor, QImage, QPainter, QPen, QCursor, QPixmap
from PySide6.QtCore import Qt, Signal, QRectF, QTimer, QPointF, Slot
from PySide6.QtWidgets import QApplication, QToolTip

from .core_data import GridConfig, BondingMap, CoordinateTransform
from .products import layout_boxes


# ==================================================================================
# 2. Tiling System
# ==================================================================================
class Tile:
    """
    Represents a single GPU texture tile.
    Pixel data is pulled from `loader` at upload time and released right after,
    so at most one tile is held in CPU memory while a layer is uploaded.
    """
    def __init__(self, x, y, width, height, loader):
        self.x = x  # Global X position
        self.y = y  # Global Y position
        self.w = width
        self.h = height
        self.loader = loader # () -> (h, w) uint8 / uint16 array
        self.tex_id = None
        self.is_uploaded = False
        self.nbytes = 0 # GPU memory used

    def upload(self):
        if self.is_uploaded:
            return

        data = self.loader()
        if data.dtype == np.uint8:
            # 8-bit: half the VRAM. Sampled values are v/255, which the shader scales by 65535
            # -> v * 257, the same 16-bit domain as before.
            internal_fmt, gl_type = GL_R8, GL_UNSIGNED_BYTE
        else:
            data = data.astype(np.uint16, copy=False)
            internal_fmt, gl_type = GL_R16, GL_UNSIGNED_SHORT
        data = np.ascontiguousarray(data)

        if self.tex_id is None:
            self.tex_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.tex_id)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glTexImage2D(
            GL_TEXTURE_2D, 0, internal_fmt,
            self.w, self.h, 0,
            GL_RED, gl_type, data
        )

        # Restore the default: Qt uploads its glyph textures (overlay text) with this state
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)

        self.is_uploaded = True
        self.nbytes = data.nbytes

    def cleanup(self):
        if self.tex_id:
            glDeleteTextures([self.tex_id])
            self.tex_id = None
        self.is_uploaded = False
        self.nbytes = 0

class TiledImage:
    """
    Manages the grid of tiles for a single image layer.
    """
    TILE_SIZE = 4096 # Safe for most modern GPUs (limit is usually 8192 or 16384)

    def __init__(self, width, height, read_tile):
        """
        read_tile(x, y, w, h) -> (h, w) uint8/uint16 array. Called once per tile at upload.
        """
        self.w = width
        self.h = height
        self.tiles = []

        for y in range(0, self.h, self.TILE_SIZE):
            for x in range(0, self.w, self.TILE_SIZE):
                h_chunk = min(self.TILE_SIZE, self.h - y)
                w_chunk = min(self.TILE_SIZE, self.w - x)
                loader = (lambda x=x, y=y, w=w_chunk, h=h_chunk: read_tile(x, y, w, h))
                self.tiles.append(Tile(x, y, w_chunk, h_chunk, loader))

    @classmethod
    def from_stack(cls, stack, z):
        """Tiles of layer z of an ImageStack, read region by region (never the full layer)."""
        return cls(stack.width, stack.height, lambda x, y, w, h: stack.get_tile(z, x, y, w, h))

    @classmethod
    def from_array(cls, image: np.ndarray):
        """(H, W) uint8/uint16 array."""
        h, w = image.shape[:2]
        return cls(w, h, lambda x, y, tw, th: image[y:y + th, x:x + tw])

    @property
    def nbytes(self):
        return sum(t.nbytes for t in self.tiles)

    @property
    def estimated_nbytes(self):
        return sum(t.w * t.h for t in self.tiles) * 2

    def upload_all(self):
        for t in self.tiles:
            t.upload()

    def cleanup(self):
        for t in self.tiles:
            t.cleanup()

# ==================================================================================
# 3. OpenGL Widget
# ==================================================================================
class GLImageWidget(QOpenGLWidget):
    grid_params_changed = Signal()
    layer_wheel_changed = Signal(int) # Delta (+1 or -1)
    cursor_moved = Signal(int, int) # Col, Row
    navigation_requested = Signal(int, int) # Col, Row (Request to move view)
    voids_changed = Signal() # Void data edited (count labels etc.)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True) # Enable hover events
        self.tiled_image: TiledImage = None
        
        self.win_lo = 0.0
        self.win_hi = 1.0
        
        # Grid & Map
        self.grid_cfg = GridConfig()
        self.bonding_map = None # Single BondingMap Instance
        self.map_tex = None
        
        # Void Marking
        self.void_manager = None
        self.void_mode = False # Active State: "DRAW" | "EDIT" | "ERASE" | False
        self.selected_void_tool = "DRAW" # Selected Tool: "DRAW" | "EDIT" | "ERASE"
        self.active_type_id = 0
        self.current_layer = 0 # Absolute layer number
        self.active_void = None
        self.active_void_action = None
        self._void_txn = False # Inside a VoidManager change (mouse press .. release)
        self._snap_hint = None
        self._resize_anchor = (None, None)
        self._pan_only = False
        
        # Chip layout of the selected WF product, drawn inside each cell's reference bbox
        self.layout = None
        self.setCursor(Qt.OpenHandCursor) # Starts in view mode

        # Auto-Calibration
        self.calib_tex = None # GL Texture for (Scale, Offset) per cell
        self.calib_data_shape = (0, 0) # (Rows, Cols)
        
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
        
        # Coordinate Transform Logic
        self.coord_transform = None

        
        
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

    def update_calib_texture(self, calib_data: np.ndarray):
        """
        Updates the calibration texture.
        calib_data: (rows, cols, 2) float32 array, where [..., 0] is scale, [..., 1] is offset.
        """
        if calib_data is None or calib_data.size == 0:
            if self.calib_tex:
                self.makeCurrent()
                glDeleteTextures([self.calib_tex])
                self.calib_tex = None
                self.doneCurrent()
            self.calib_data_shape = (0, 0)
            self.update()
            return

        rows, cols, _ = calib_data.shape
        self.calib_data_shape = (rows, cols)

        self.makeCurrent()
        if self.calib_tex:
            glDeleteTextures([self.calib_tex])
            
        self.calib_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.calib_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Upload as RG float texture
        # calib_data is (rows, cols, 2)
        # OpenGL expects (width, height) -> (cols, rows)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, cols, rows, 0, GL_RG, GL_FLOAT, calib_data)
        
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
        
        uniform float img_rotation; // Added: Rotation in Radians
        
        out vec2 uv;
        out vec2 pixelPos; // This will now be "Deskewed" position
        
        void main() {
            // 1. Raw Position in Image Space
            vec2 posRaw = tileOffset + aPos * tileSize;
            
            // 2. Rotate around Image Center
            vec2 center = imageSize * 0.5;
            vec2 rel = posRaw - center;
            float c = cos(img_rotation);
            float s = sin(img_rotation);
            vec2 posRot;
            posRot.x = rel.x * c - rel.y * s;
            posRot.y = rel.x * s + rel.y * c;
            vec2 posDeskewed = center + posRot; 
            
            pixelPos = posDeskewed; // Pass deskewed coordinate to fragment (Grid Space)
            
            // 3. View Transform (Pan/Zoom in Deskewed Space)
            vec2 viewCenter = center - pan; 
            
            vec2 finalPos = (posDeskewed - viewCenter) * zoom / (viewSize * 0.5);
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
        uniform float grid_angle; // Degrees
        uniform float line_width;
        uniform float zoom; // to adjust line width?
        
        uniform sampler2D map_tex;
        uniform int has_map;
        uniform float grid_opacity;
        
        uniform sampler2D calib_tex;
        uniform int use_calib;
        
        void main() {
            // 1. Base Image
            float val = texture(tex, uv).r; 

            // Apply Window Leveling
            float pixel_val = val * 65535.0;
            float normalized = (pixel_val - win_lo) / (win_hi - win_lo);
            
            // Grid Calculation Wrapper
            
            // Calculate Grid Coords with Rotation
            float rad = radians(grid_angle);
            float s = sin(rad);
            float c = cos(rad);
            
            
            vec2 rel = pixelPos - vec2(grid_x, grid_y);
            // Grid is now Axis-Aligned in this visual space (Deskewed)
            // No rotation needed here because we rotated the Image World to match the Grid.
            // Or effectively, we are rendering in "Grid Space".
            
            vec2 rotPos = rel; 
            // rotPos.x = rel.x * c + rel.y * s; // Removed
            // rotPos.y = -rel.x * s + rel.y * c; // Removed
            
            float gx = rotPos.x / pitch_x;
            float gy = rotPos.y / pitch_y;
            
            int c_idx = int(floor(gx));
            int r_idx = int(floor(gy));
    
            // Auto-Calibration
            if (use_calib == 1) {
                if (c_idx >= 0 && c_idx < grid_cols && r_idx >= 0 && r_idx < grid_rows) {
                     // Sample Calib Texture
                     // Center of the texel
                     // Data array[0] (Row 0, Top) is uploaded to V=0 (Bottom).
                     // So we want r=0 to map to V=0.
                     vec2 calibUV = vec2((float(c_idx) + 0.5) / float(grid_cols), (float(r_idx) + 0.5) / float(grid_rows));
                     vec2 params = texture(calib_tex, calibUV).rg;
                     
                     normalized = normalized * params.r + params.g;
                }
            }
            
            vec3 baseColor = vec3(clamp(normalized, 0.0, 1.0));
            
            // 2. Grid & Overlay
            vec4 overlayColor = vec4(0.0);
            
            if (show_grid == 1) {
                // Determine if we are inside the grid bounding box
                // Check bounds in Grid Space (gx, gy)
                if (gx >= 0.0 && gx <= float(grid_cols) && gy >= 0.0 && gy <= float(grid_rows)) {
                    
                    int c = int(floor(gx));
                    int r = int(floor(gy));
                    
                    // Borders
                    // Distance to nearest integer
                    float dx = abs(gx - round(gx)) * pitch_x;
                    float dy = abs(gy - round(gy)) * pitch_y;
                    
                    // Zoom-dependent width?
                    float lw = line_width / zoom; 
                    if (lw < 1.0) lw = 1.0;
                    
                    if (dx < lw || dy < lw) {
                        overlayColor = vec4(1.0, 1.0, 0.0, 0.6); // Yellow Grid lines
                    }
                    
                    // Map Params (Colors)
                    if (has_map == 1) {
                            if (c >= 0 && c < grid_cols && r >= 0 && r < grid_rows) {
                                  // Map Lookup: Flip Y for texture lookup?
                                  // We previously handled map vertically?
                                  // Lets assume (row, col) matches Map Texture (row, col) with V inverted?
                                  // Re-use logic: map is uploaded row-by-row. R=0 at V=0?
                                  // If BondingMap.data_map[0] is R=0.
                                  // If we upload it, Row 0 goes to V=0 (Bottom).
                                  // So mapUV should be (r+0.5)/rows => V=Low.
                                  // So same as Calib.
                                  vec2 mapUV = vec2((float(c) + 0.5) / float(grid_cols), (float(r) + 0.5) / float(grid_rows));
                                  vec4 mapCol = texture(map_tex, mapUV);
                                  if (mapCol.a > 0.0) {
                                      // Blend map color
                                      overlayColor = mix(overlayColor, mapCol, 0.5); // Tint
                                  }
                            }
                    }
                }
            }
            
            // Blend
            overlayColor.a *= grid_opacity;
            color = vec4(mix(baseColor.rgb, overlayColor.rgb, overlayColor.a), 1.0);
        }
        """)

        glCompileShader(self.vs)
        if not glGetShaderiv(self.vs, GL_COMPILE_STATUS):
            print("VS Compile Error:", glGetShaderInfoLog(self.vs))
            
        glCompileShader(self.fs)
        if not glGetShaderiv(self.fs, GL_COMPILE_STATUS):
            print("FS Compile Error:", glGetShaderInfoLog(self.fs))

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
        

    def set_tiled_image(self, tiled_img: TiledImage, cleanup_old: bool = True):
        # Cleanup old if requested
        if cleanup_old and self.tiled_image and self.tiled_image != tiled_img:
            self.tiled_image.cleanup()
        
        self.tiled_image = tiled_img
        
        # Init Transform (Only if changed)
        if self.coord_transform is None or \
           self.coord_transform.img_w != self.tiled_image.w or \
           self.coord_transform.img_h != self.tiled_image.h:
               self.coord_transform = CoordinateTransform(self.grid_cfg.angle, self.tiled_image.w, self.tiled_image.h)
        
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
        
        # Image Rotation (Deskewing)
        # We rotate the Image Texture by -GridAngle so it aligns with the screen axes.
        # This makes the Chips "Upright" on screen.
        rad_angle = np.radians(self.grid_cfg.angle)
        glUniform1f(glGetUniformLocation(self.prog, "img_rotation"), -rad_angle) # Negative angle to deskew
        
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
        glUniform1f(glGetUniformLocation(self.prog, "grid_angle"), self.grid_cfg.angle)
        glUniform1f(glGetUniformLocation(self.prog, "line_width"), self.grid_cfg.line_width)
        glUniform1f(glGetUniformLocation(self.prog, "grid_opacity"), self.grid_cfg.opacity)
        
        glUniform1i(glGetUniformLocation(self.prog, "use_calib"), 1 if self.grid_cfg.use_calib and self.calib_tex else 0)
        
        if self.calib_tex:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.calib_tex)
            glUniform1i(glGetUniformLocation(self.prog, "calib_tex"), 2)
        
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

        # Leave a clean GL state for the QPainter overlay drawn afterwards (text uses glyph textures)
        for unit in (GL_TEXTURE2, GL_TEXTURE1, GL_TEXTURE0):
            glActiveTexture(unit)
            glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)
        glUseProgram(0)
            
    # -----------------------------------------------
    # Coordinate Transforms (Deskew Logic)
    # -----------------------------------------------
    def raw_to_deskew(self, rx, ry):
        """
        Transforms Raw Image Coordinates (u, v) -> Deskewed Space (x, y)
        Delegates to CoordinateTransform
        """
        if self.coord_transform:
            return self.coord_transform.raw_to_deskew(rx, ry)
        return rx, ry

    def deskew_to_raw(self, gx, gy):
        """
        Transforms Deskewed/Screen Coordinates (x, y) -> Raw Image Coordinates (u, v)
        Delegates to CoordinateTransform
        """
        if self.coord_transform:
            return self.coord_transform.deskew_to_raw(gx, gy)
        return gx, gy

    def raw_to_cell(self, rx, ry, inside_only=False):
        """
        Grid cell (row, col) containing the Raw (absolute image) point (rx, ry).
        The grid is rotated by grid_cfg.angle relative to the raw image.
        inside_only: return None when the point is outside the grid.
        """
        gx, gy = self.raw_to_deskew(rx, ry)
        row, col = self.grid_cfg.cell_at(gx, gy)
        if inside_only and not self.grid_cfg.contains(row, col):
            return None
        return row, col
            
    # -----------------------------------------------
    # Mouse Interaction
    # -----------------------------------------------
    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        modifiers = e.modifiers()
        
        # Edit tool (E) + hovering a curve void: the wheel adjusts its exponent p
        if self._wheel_edit_curve(e, delta, modifiers):
            return

        # Reconfiguration (User Request 2025-12-28)
        # No Modifier: Zoom
        # Ctrl or Shift: Layer Change
        
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            # Layer Change
            # Wheel Down (delta < 0) -> Increase Layer Index -> Next Layer (steps=1)
            # Wheel Up (delta > 0) -> Decrease Layer Index -> Prev Layer (steps=-1)
            steps = 1 if delta < 0 else -1
            self.layer_wheel_changed.emit(steps)
            
        else:
            # Zoom Logic (Default)
            mx = e.position().x()
            my = e.position().y()
            
            view_w = self.width()
            view_h = self.height()
            cx = view_w / 2
            cy = view_h / 2
            
            # Factor
            factor = 1.1
            if delta < 0:
                factor = 1.0 / 1.1
                
            # Mouse relative to screen center
            dx = mx - cx
            dy = my - cy
            
            new_zoom = self.zoom * factor
            
            # Limit zoom
            if new_zoom < 0.001: new_zoom = 0.001
            if new_zoom > 1000.0: new_zoom = 1000.0
            
            # Adjust Pan to zoom towards mouse
            self.pan_x += (dx / self.zoom) - (dx / new_zoom)
            self.pan_y += (dy / self.zoom) - (dy / new_zoom)
            
            self.zoom = new_zoom
            self.update()

    def _wheel_edit_curve(self, e, delta, modifiers):
        if (self.selected_void_tool != "EDIT" or not self.void_manager or not self.tiled_image
                or modifiers & (Qt.ControlModifier | Qt.ShiftModifier) or delta == 0):
            return False
        gx, gy = self.get_image_coords(e.position())
        v, _ = self.void_manager.hit_test(self.current_layer, gx, gy, margin=10.0/self.zoom, coord_transform=self.raw_to_deskew, quadrant_ctx=self._quadrant_ctx())
        if v is None or self.void_manager.shape_of(v) != "curve":
            return False
        p = self.void_manager.set_p(v, self.void_manager.p_of(v) * (1.1 ** (delta / 120.0)))
        QToolTip.showText(e.globalPosition().toPoint(), f"p = {p:.2f}", self)
        self.voids_changed.emit()
        self.update()
        return True

    def mouseDoubleClickEvent(self, e):
        if not self.grid_cfg.visible or self.tiled_image is None:
            super().mouseDoubleClickEvent(e)
            return

        # 1. Map Mouse to Image Coords
        # Screen Center
        view_w = self.width()
        view_h = self.height()
        cx = view_w / 2
        cy = view_h / 2
        
        # Mouse relative to center
        mx = e.position().x()
        my = e.position().y()
        dx = mx - cx
        dy = my - cy
        
        # Current View Center in Image Space
        img_w = self.tiled_image.w
        img_h = self.tiled_image.h
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        # Pan is (imgCenter - viewCenter)
        # viewCenter = imgCenter - pan
        view_center_x = img_cx - self.pan_x
        view_center_y = img_cy - self.pan_y
        
        # Click position in Image Space
        click_img_x = view_center_x + dx / self.zoom
        click_img_y = view_center_y + dy / self.zoom
        
        # 2. Check Grid intersection
        gx = click_img_x - self.grid_cfg.start_x
        gy = click_img_y - self.grid_cfg.start_y
        
        if gx < 0 or gy < 0: 
            return # Clicked before grid start
            
        col = int(gx / self.grid_cfg.pitch_x)
        row = int(gy / self.grid_cfg.pitch_y)
        
        if col >= self.grid_cfg.cols or row >= self.grid_cfg.rows:
            return # Clicked outside grid
            
        if col >= self.grid_cfg.cols or row >= self.grid_cfg.rows:
            return # Clicked outside grid
            
        self.fit_to_cell(col, row)

    def fit_to_cell(self, col, row):
        """Zooms and Pans to fit the specified grid cell in view."""
        # Target Cell Center
        cell_cx = self.grid_cfg.start_x + (col + 0.5) * self.grid_cfg.pitch_x
        cell_cy = self.grid_cfg.start_y + (row + 0.5) * self.grid_cfg.pitch_y
        
        # Image Center
        if self.tiled_image:
            img_w = self.tiled_image.w
            img_h = self.tiled_image.h
        else:
            img_w = 1000
            img_h = 1000
            
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        self.coord_transform = CoordinateTransform(self.grid_cfg.angle, img_w, img_h)
        
        # 1. Calculate Zoom to fit Pitch in View
        view_w = self.width()
        view_h = self.height()
        
        margin = 0.95
        zoom_x = view_w / self.grid_cfg.pitch_x
        zoom_y = view_h / self.grid_cfg.pitch_y
        self.zoom = min(zoom_x, zoom_y) * margin
        
        # 2. Calculate Pan to center target
        # Pan = imgCenter - viewCenter
        # We want viewCenter to be cellCenter
        self.pan_x = img_cx - cell_cx
        self.pan_y = img_cy - cell_cy
        
        self.update()

    # -----------------------------------------------
    # Overlay Rendering (QPainter)
    # -----------------------------------------------

    # Signal for shortcuts
    void_type_shortcut_triggered = Signal(int) # Type ID
    
    def create_colored_cross_cursor(self, color_tuple):
        """
        Creates a Crosshair cursor with the specified RGB color.
        """
        size = 32
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        col = QColor(color_tuple[0], color_tuple[1], color_tuple[2])
        pen = QPen(col)
        pen.setWidth(2)
        painter.setPen(pen)
        
        center = size // 2
        # Horizontal
        painter.drawLine(0, center, size, center)
        # Vertical
        painter.drawLine(center, 0, center, size)
        
        # Optional: Add white outline for contrast?
        # Maybe complex to draw properly on 1px width.
        # Let's stick to simple colored cross first.
        
        painter.end()
        
        return QCursor(pixmap, center, center)

    def create_erase_cursor(self):
        """Red circle with an X (erase mode)."""
        size = 32
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        for color, width in ((QColor(0, 0, 0, 160), 4), (QColor(255, 60, 60), 2)):
            pen = QPen(color)
            pen.setWidth(width)
            painter.setPen(pen)
            painter.drawEllipse(QRectF(5, 5, 22, 22))
            painter.drawLine(QPointF(10, 10), QPointF(22, 22))
            painter.drawLine(QPointF(22, 10), QPointF(10, 22))
        painter.end()
        return QCursor(pixmap, size // 2, size // 2)

    def update_cursor_style(self):
        """
        Mode cursor: View = open hand, Draw = crosshair in the type color, Edit = arrow
        (changes over voids, see _update_edit_hover_cursor), Erase = red circle with X.
        """
        if not self.void_mode:
            self.setCursor(Qt.OpenHandCursor)
        elif self.void_mode == "DRAW":
            # Use Colored Crosshair
            tid = self.active_type_id
            if self.void_manager and tid in self.void_manager.types:
                col = self.void_manager.types[tid]["color"]
                cursor = self.create_colored_cross_cursor(col)
                self.setCursor(cursor)
            else:
                self.setCursor(Qt.CrossCursor)
        elif self.void_mode == "ERASE":
            self.setCursor(self.create_erase_cursor())
        elif self.void_mode == "EDIT":
            self.setCursor(Qt.ArrowCursor)

    def _update_edit_hover_cursor(self, gx, gy):
        """Edit mode: move cursor over a void, resize arrows over its edge, arrow elsewhere."""
        if not self.void_manager or not self.tiled_image:
            return
        v, action = self.void_manager.hit_test(self.current_layer, gx, gy, margin=10.0/self.zoom,
                                               coord_transform=self.raw_to_deskew, quadrant_ctx=self._quadrant_ctx())
        if v is None:
            shape = Qt.ArrowCursor
        elif action == 'center':
            shape = Qt.SizeAllCursor
        else:
            fx, fy = self._resize_anchor_for(v, gx, gy)
            if fx is not None and fy is not None:
                cx, cy = self.raw_to_deskew(v["globalCX"], v["globalCY"])
                shape = Qt.SizeFDiagCursor if (gx - cx) * (gy - cy) > 0 else Qt.SizeBDiagCursor
            elif fx is not None:
                shape = Qt.SizeHorCursor
            else:
                shape = Qt.SizeVerCursor
        if self.cursor().shape() != shape:
            self.setCursor(shape)

    view_center_changed = Signal(int, int) # Col, Row

    def update_view_center_info(self):
        # Calculate Center (Col, Row) and emit
        if self.tiled_image:
             img_w = self.tiled_image.w
             img_h = self.tiled_image.h
        else:
             img_w = 1000
    def update_view_center_info(self):
        # Calculate Center (Col, Row) and emit
        if self.coord_transform:
             img_cx = self.coord_transform.img_cx
             img_cy = self.coord_transform.img_cy
        else:
             img_cx = 500
             img_cy = 500
        
        gx = img_cx - self.pan_x
        gy = img_cy - self.pan_y
        
        cfg = self.grid_cfg
        dx = gx - cfg.start_x
        dy = gy - cfg.start_y
        
        rx = dx
        ry = dy
        
        c = int(np.floor(rx / cfg.pitch_x))
        r = int(np.floor(ry / cfg.pitch_y))
        
        self.view_center_changed.emit(c, r)
        
    @Slot()
    def update_grid_params(self):
        # Update transform when grid changes
        if self.tiled_image:
             self.coord_transform = CoordinateTransform(self.grid_cfg.angle, self.tiled_image.w, self.tiled_image.h)
        self.update()

    def paintEvent(self, e):
        # 1. Draw OpenGL content
        super().paintEvent(e)
        
        # Emit View Center Info
        self.update_view_center_info()
        
        if self.void_manager and self.tiled_image:
             painter = QPainter(self)
             painter.setRenderHint(QPainter.Antialiasing)
             
             # Chip layout inside each cell's reference bbox
             self._draw_layout(painter)

             # First Pass: Ghost Voids (Other Layers), then Current Layer Voids on top
             vm = self.void_manager
             refs = vm.reference_bboxes(self.grid_cfg, self.coord_transform)

             def paint(v, style):
                 # Raw (storage) -> Deskewed (visual, applies the grid rotation) -> Screen
                 dx, dy = self.raw_to_deskew(v["globalCX"], v["globalCY"])
                 sx, sy = self.deskew_to_screen(dx, dy)
                 vm.paint_void(painter, v, sx, sy, self.zoom, style=style,
                               quadrant=vm.quadrant_of(v, dx, dy, refs, self.grid_cfg))

             for layer_idx, v_list in vm.voids.items():
                 if layer_idx == self.current_layer:
                     continue
                 for v in v_list:
                     paint(v, "ghost")

             active = getattr(self, 'active_void', None)
             for v in vm.voids.get(self.current_layer, []):
                 paint(v, "active" if v is active else "normal")


             # Center cell info (top-left, yellow)
             self._draw_center_info(painter)
             painter.end()

    def center_cell(self):
        """(row, col) of the grid cell at the view center (grid space = Deskewed)."""
        if self.coord_transform:
            img_cx, img_cy = self.coord_transform.img_cx, self.coord_transform.img_cy
        else:
            img_cx, img_cy = 500, 500
        return self.grid_cfg.cell_at(img_cx - self.pan_x, img_cy - self.pan_y)

    def _draw_center_info(self, painter):
        r, c = self.center_cell()
        if not self.grid_cfg.contains(r, c):
            text = "X: -  Y: -  LEG: -"
        else:
            leg = self.bonding_map.get_key(r, c) if self.bonding_map else ""
            text = f"X: {c:02d}  Y: {r:02d}  LEG: {leg or '-'}"
        font = painter.font()
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        # Dark outline for readability on bright images
        painter.setPen(QColor(0, 0, 0, 200))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
            painter.drawText(QPointF(12 + dx, 28 + dy), text)
        painter.setPen(QColor(255, 255, 0))
        painter.drawText(QPointF(12, 28), text)

    def deskew_to_screen(self, dx, dy):
        """Deskewed (grid space) point -> widget pixel position."""
        if self.coord_transform:
            img_cx, img_cy = self.coord_transform.img_cx, self.coord_transform.img_cy
        else:
            img_cx, img_cy = 500, 500
        return ((dx - img_cx + self.pan_x) * self.zoom + self.width() / 2,
                (dy - img_cy + self.pan_y) * self.zoom + self.height() / 2)

    def _draw_layout(self, painter):
        """
        Draws the selected product's layout (dashed sub-boxes, see products.json) inside every
        cell's reference bbox. Layout coordinates are in layout.design_size units relative to the
        bbox top-left and are scaled to the actual bbox (the drawn bbox is the reference).
        """
        layout = self.layout
        if not layout or not self.void_manager:
            return
        boxes = layout_boxes(layout)
        if not boxes:
            return
        style = layout.get("style", {})
        col = style.get("color", [0, 200, 255])
        pen = QPen(QColor(col[0], col[1], col[2], style.get("alpha", 200)))
        pen.setWidthF(style.get("width", 1.0))
        dash = style.get("dash", [4, 4])
        if dash:
            pen.setDashPattern(dash)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        refs = self.void_manager.reference_bboxes(self.grid_cfg, self.coord_transform)
        for (x0, y0, w, h) in refs.values():
            dw, dh = layout.get("design_size", (w, h))
            sx, sy = w / dw, h / dh
            for (bx, by, bw, bh) in boxes:
                p0 = self.deskew_to_screen(x0 + bx * sx, y0 + by * sy)
                p1 = self.deskew_to_screen(x0 + (bx + bw) * sx, y0 + (by + bh) * sy)
                painter.drawRect(QRectF(QPointF(*p0), QPointF(*p1)))

    # -----------------------------------------------
    # Mouse Interaction
    # -----------------------------------------------
    def get_image_coords(self, pos):
        # Converts screen position to Global Image Coordinates
        mx = pos.x()
        my = pos.y()
        
        view_w = self.width()
        view_h = self.height()
        cx = view_w / 2
        cy = view_h / 2
        
        dx = mx - cx
        dy = my - cy
        
        if self.coord_transform:
             img_cx = self.coord_transform.img_cx
             img_cy = self.coord_transform.img_cy
        else:
             img_cx = 500
             img_cy = 500
        
        view_center_x = img_cx - self.pan_x
        view_center_y = img_cy - self.pan_y
        
        gx = view_center_x + dx / self.zoom
        gy = view_center_y + dy / self.zoom
        
        return gx, gy

    def mousePressEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.RightButton):
            # Pan in every mode
            self._last_mouse = e.position()
            self._pan_only = True
            self.setCursor(Qt.ClosedHandCursor)
            self.setFocus()
            return
        if e.button() == Qt.LeftButton:
            self._last_mouse = e.position()
            self._pan_only = False
            self.setFocus()
            if not self.void_mode:
                self.setCursor(Qt.ClosedHandCursor) # View mode: left drag pans
            
            # Void Interaction (one undo step per press .. release)
            if self.void_mode and self.void_manager and self.tiled_image:
                gx, gy = self.get_image_coords(e.position())
                self._begin_void_change()
                
                if self.void_mode == "DRAW":
                    if self.void_manager and self.current_layer is not None:
                        # CREATE VOID
                        # 1. Start Point comes in Deskewed Space (gx, gy)
                        # 2. We store in Raw Space.
                        rx, ry = self.deskew_to_raw(gx, gy)
                        
                        self._void_start = e.position() # Store Screen Start
                        self._void_start_gx = gx # Store Deskewed Start
                        self._void_start_gy = gy
                        
                        def_w = 0
                        def_h = 0
                        
                        shape = "ellipse"
                        if self.active_type_id in self.void_manager.types:
                             tdata = self.void_manager.types[self.active_type_id]
                             shape = tdata.get("shape", "ellipse")
                             def_w = tdata.get("def_w", 0)
                             def_h = tdata.get("def_h", 0)
                        
                        is_sizing = (def_w <= 0 or def_h <= 0)
                        
                        if not is_sizing:
                            # Predefined Size (Click-to-Draw)
                            # Size is Width/Height in Deskewed Frame (Grid Axes)
                            # Stored as radiusX/radiusY
                            
                            if shape == "rectangle":
                                # Rectangle: Center/Radius Logic (Unified)
                                # Click Point = Top-Left of Box -> Calculate Center
                                cx_deskew = gx + def_w / 2.0
                                cy_deskew = gy + def_h / 2.0
                                cx_raw, cy_raw = self.deskew_to_raw(cx_deskew, cy_deskew)
                                
                                # Store Center. radiusX = Half-Width
                                self.active_void = self.void_manager.add_void(self.current_layer, cx_raw, cy_raw, def_w/2.0, def_h/2.0, self.active_type_id)
                            else:
                                # Ellipse: Schema Center, Radius
                                # Click Point = Top-Left of Bounding Box
                                # So Center (Deskewed) = gx + w/2, gy + h/2.
                                # We need Center (Raw).
                                cx_deskew = gx + def_w / 2.0
                                cy_deskew = gy + def_h / 2.0
                                cx_raw, cy_raw = self.deskew_to_raw(cx_deskew, cy_deskew)
                                
                                self.active_void = self.void_manager.add_void(self.current_layer, cx_raw, cy_raw, def_w/2.0, def_h/2.0, self.active_type_id)
                                
                            # Snap types: attach to the bbox corner nearest to the click
                            self._snap_hint = (gx, gy)
                            self.void_manager.snap_void(self.active_void, self.grid_cfg, self.coord_transform, hint=self._snap_hint)
                            
                            # Switch to Move mode immediately
                            self.active_void_action = 'center'
                            
                        else:
                            # Drag-to-Size
                            # Initialize 1x1 void
                            # Center/Pos is rx, ry
                            self.active_void = self.void_manager.add_void(self.current_layer, rx, ry, 1.0, 1.0, self.active_type_id)
                            self.active_void_action = 'sizing'
                            self._snap_hint = (gx, gy) # Snap types: corner nearest to the press point

                    self.voids_changed.emit()
                    self.update()
                    return # Consume event
                    
                elif self.void_mode == "EDIT":
                    # Hit Test on the current layer (ghost voids of other layers are reference only)
                    # Mouse (gx, gy) is Deskewed -> pass the Raw -> Deskew transform.
                    v, action = self.void_manager.hit_test(self.current_layer, gx, gy, margin=10.0/self.zoom, coord_transform=self.raw_to_deskew, quadrant_ctx=self._quadrant_ctx())
                    if v:
                        self.active_void = v
                        self.active_void_action = action # 'center' or 'edge'
                        v.pop("auto", None) # Reviewed by hand (no longer "estimated")
                        # Snap types keep their anchored corner while resizing
                        self._snap_hint = None
                        if action == 'edge':
                            self._snap_hint = self.void_manager.snap_corner_point(v, self.grid_cfg, self.coord_transform)
                            self._resize_anchor = self._resize_anchor_for(v, gx, gy)
                        self.update()
                        return

                elif self.void_mode == "ERASE":
                    if self._erase_at(gx, gy):
                        self.active_void_action = 'erasing' # Keep erasing while dragging
                        self.voids_changed.emit()
                        self.update()
                        return

    def mouseMoveEvent(self, e):
        # 0. Always calculate grid pos and emit signal
        gx, gy = self.get_image_coords(e.position())
        
        # Guard against zero pitch logic/init issues
        if self.grid_cfg.pitch_x > 0 and self.grid_cfg.pitch_y > 0:
             c = int(np.floor((gx - self.grid_cfg.start_x) / self.grid_cfg.pitch_x))
             r = int(np.floor((gy - self.grid_cfg.start_y) / self.grid_cfg.pitch_y))
        self.cursor_moved.emit(gx, gy)

        if self._last_mouse is None:
            # Hover only
            if self.void_mode == "EDIT":
                self._update_edit_hover_cursor(gx, gy)
            return
            
            
        dx = e.position().x() - self._last_mouse.x()
        dy = e.position().y() - self._last_mouse.y()
        
        # Void Interaction
        if self.void_mode and self.active_void_action and not getattr(self, '_pan_only', False):
             # Handle Drag
             gx, gy = self.get_image_coords(e.position())
             
             if self.active_void_action == 'sizing' and self.active_void:
                 # Corner-to-Corner Logic
                 start_x = getattr(self, '_void_start_gx', gx)
                 start_y = getattr(self, '_void_start_gy', gy)
                 
                 shape = self.void_manager.types.get(self.active_void.get("type_id", 0), {}).get("shape", "ellipse")
                 
                 if shape == "rectangle":
                     # Rectangle Sizing: Defined by Center and Radii
                     # Start and Current define bbox
                     min_x = min(start_x, gx)
                     max_x = max(start_x, gx)
                     min_y = min(start_y, gy)
                     max_y = max(start_y, gy)
                     
                     # Center
                     cx = (min_x + max_x) / 2.0
                     cy = (min_y + max_y) / 2.0
                     
                     # Radii (Half-Dims)
                     rx = (max_x - min_x) / 2.0
                     ry = (max_y - min_y) / 2.0
                     
                     self.active_void["radiusX"] = max(1.0, rx) # Half-Width
                     self.active_void["radiusY"] = max(1.0, ry) # Half-Height
                     
                     # Store Center in RAW
                     cx_raw, cy_raw = self.deskew_to_raw(cx, cy)
                     self.active_void["globalCX"] = cx_raw
                     self.active_void["globalCY"] = cy_raw
                     
                 else:
                     # Ellipse Sizing (Center/Radius)
                     # Start and Current define bbox
                     min_x = min(start_x, gx)
                     max_x = max(start_x, gx)
                     min_y = min(start_y, gy)
                     max_y = max(start_y, gy)
                     
                     # Center
                     cx = (min_x + max_x) / 2.0
                     cy = (min_y + max_y) / 2.0
                     
                     # Radii
                     rx = (max_x - min_x) / 2.0
                     ry = (max_y - min_y) / 2.0
                     
                     self.active_void["radiusX"] = max(1.0, rx)
                     self.active_void["radiusY"] = max(1.0, ry)
                     
                     # Store Center in RAW
                     cx_raw, cy_raw = self.deskew_to_raw(cx, cy)
                     self.active_void["globalCX"] = cx_raw
                     self.active_void["globalCY"] = cy_raw
                 
                 # Snap types: span from the bbox corner nearest to the press point to the mouse
                 self.void_manager.snap_void(self.active_void, self.grid_cfg, self.coord_transform,
                                             hint=self._snap_hint, drag_point=(gx, gy))
                 self.update()
                 
             elif self.active_void_action == 'center' and self.active_void:
                 # Move Center
                 # img_dx, img_dy are Deskewed Deltas
                 img_dx = dx / self.zoom
                 img_dy = dy / self.zoom
                 
                 # Rotate Delta by +Angle to get Raw Delta
                 rad = np.radians(self.grid_cfg.angle)
                 c = np.cos(rad)
                 s = np.sin(rad)
                 
                 raw_dx = img_dx * c - img_dy * s
                 raw_dy = img_dx * s + img_dy * c
                 
                 self.active_void["globalCX"] += raw_dx
                 self.active_void["globalCY"] += raw_dy
                 # Snap types: re-attach to the nearest bbox corner
                 self.void_manager.snap_void(self.active_void, self.grid_cfg, self.coord_transform)
                 self.update()
                 
             elif self.active_void_action == 'edge' and self.active_void:
                 shape = self.void_manager.types.get(self.active_void.get("type_id", 0), {}).get("shape", "ellipse")
                 
                 # Get Center in Deskewed Space (Reference)
                 cx_deskew, cy_deskew = self.raw_to_deskew(self.active_void["globalCX"], self.active_void["globalCY"])
                 
                 if self._snap_hint is not None:
                     # Snap types: anchored corner stays, size follows the mouse
                     self.void_manager.snap_void(self.active_void, self.grid_cfg, self.coord_transform,
                                                 hint=self._snap_hint, drag_point=(gx, gy))
                 else:
                     # The side opposite to the grabbed one stays fixed; the grabbed side follows the mouse
                     fx, fy = self._resize_anchor
                     if fx is not None:
                         cx_deskew = (fx + gx) / 2.0
                         self.active_void["radiusX"] = max(1.0, abs(gx - fx) / 2.0)
                     if fy is not None:
                         cy_deskew = (fy + gy) / 2.0
                         self.active_void["radiusY"] = max(1.0, abs(gy - fy) / 2.0)
                     self.active_void["globalCX"], self.active_void["globalCY"] = self.deskew_to_raw(cx_deskew, cy_deskew)
                 
                 self.update()
                 
             elif self.active_void_action == 'erasing':
                 # Continuous erase
                 if self._erase_at(gx, gy):
                     self.voids_changed.emit()
                     self.update()
                 
             self._last_mouse = e.position()
             return

        # Check modifiers

        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.ControlModifier and self.grid_cfg.visible and not getattr(self, '_pan_only', False):
            # Move Grid
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

    def _resize_anchor_for(self, v, gx, gy):
        """
        Fixed opposite side(s) for resizing from the Deskewed grab point: (fixed x or None,
        fixed y or None). Near a corner both axes resize, near a side only that axis.
        """
        cx, cy = self.raw_to_deskew(v["globalCX"], v["globalCY"])
        rx, ry = v["radiusX"], v["radiusY"]
        nx, ny = (gx - cx) / max(rx, 1e-6), (gy - cy) / max(ry, 1e-6)
        use_x = abs(nx) >= 0.5 * max(abs(nx), abs(ny))
        use_y = abs(ny) >= 0.5 * max(abs(nx), abs(ny))
        fx = cx - math.copysign(rx, nx) if use_x else None
        fy = cy - math.copysign(ry, ny) if use_y else None
        return fx, fy

    def _quadrant_ctx(self):
        return (self.void_manager.reference_bboxes(self.grid_cfg, self.coord_transform), self.grid_cfg)

    def _erase_at(self, gx, gy):
        """Deletes the topmost void of the current layer under the Deskewed point (gx, gy)."""
        v, _ = self.void_manager.hit_test(self.current_layer, gx, gy, margin=10.0/self.zoom, coord_transform=self.raw_to_deskew, quadrant_ctx=self._quadrant_ctx())
        return v is not None and self.void_manager.remove_void(self.current_layer, v)

    def mouseReleaseEvent(self, e):
        # Clear interaction state
        self._last_mouse = None
        if getattr(self, '_pan_only', False):
            self._pan_only = False
            self.update_cursor_style()
            return

        # A click without dragging in Drag-to-Size mode leaves a 1px void -> discard it
        if self.active_void_action == 'sizing' and self.active_void is not None:
            start = getattr(self, '_void_start', None)
            if start is not None:
                moved = abs(e.position().x() - start.x()) + abs(e.position().y() - start.y())
                if moved < 3:
                    self.void_manager.remove_void(self.current_layer, self.active_void)
        
        changed = self._end_void_change()
        if not self.void_mode:
            self.setCursor(Qt.OpenHandCursor)
        if self.void_mode or self.active_void_action:
            self.active_void = None
            self.active_void_action = None
            self._snap_hint = None
            self.update()
        if changed:
            self.voids_changed.emit()

    def _begin_void_change(self):
        if not self._void_txn:
            self.void_manager.begin_change()
            self._void_txn = True

    def _end_void_change(self):
        """Closes the press .. release change. Returns True if it changed void data."""
        if not self._void_txn:
            return False
        self._void_txn = False
        return self.void_manager.end_change()
        
    
    def set_void_tool(self, tool_name):
        """Activates a marking tool ("DRAW" / "EDIT" / "ERASE"), or None for view mode."""
        if self.active_void_action:
            return # Not in the middle of a drag
        if tool_name:
            self.selected_void_tool = tool_name
        self.void_mode = tool_name or False
        self.update_cursor_style()
        self.update()

    def keyPressEvent(self, e):
        modifiers = e.modifiers()
        
        # 1. Ctrl + Arrows : Grid Nudge
        if modifiers & Qt.ControlModifier:
            if not self.grid_cfg.visible:
                return
                
            step = 1.0
            if modifiers & Qt.ShiftModifier: step = 10.0 # Ctrl+Shift = Fast Nudge
            
            if e.key() == Qt.Key_Left:
                self.grid_cfg.start_x -= step
                self.grid_params_changed.emit()
                self.update()
                return
            elif e.key() == Qt.Key_Right:
                self.grid_cfg.start_x += step
                self.grid_params_changed.emit()
                self.update()
                return
            elif e.key() == Qt.Key_Up:
                self.grid_cfg.start_y -= step
                self.grid_params_changed.emit()
                self.update()
                return
            elif e.key() == Qt.Key_Down:
                self.grid_cfg.start_y += step
                self.grid_params_changed.emit()
                self.update()
                return
        
        # Shortcuts for Void Types (No Modifier or Shift is OK)
        # 1-9 -> Index 0-8
        # 0 -> Index 9
        key = e.key()
        if Qt.Key_0 <= key <= Qt.Key_9:
             idx = -1
             if key == Qt.Key_0: idx = 9
             else: idx = key - Qt.Key_1
             
             if idx >= 0 and self.void_manager:
                 types = sorted(self.void_manager.types.keys())
                 if idx < len(types):
                     new_id = types[idx]
                     self.active_type_id = new_id
                     print(f"Shortcut: Switch to Void Type ID {new_id}")
                     
                     # Emit Signal to sync UI
                     self.void_type_shortcut_triggered.emit(new_id)
                     
                     # Update Cursor if needed
                     if self.void_mode:
                         self.update_cursor_style()
                     
                     self.update()
                     
             # Don't return, allow other handlers (though digits usually don't do nav)
             return
        
        # 2. Plain Arrows
        # Left/Right: Navigate Patch
        # Up/Down: Navigate Layer
        if e.key() == Qt.Key_Left or e.key() == Qt.Key_Right:
             # Calculate current cell index based on center of view
             view_w = self.width()
             view_h = self.height()
             
             if self.tiled_image:
                 img_w = self.tiled_image.w
                 img_h = self.tiled_image.h
             else:
                 img_w = 1000
                 img_h = 1000
             
             img_cx = img_w / 2
             img_cy = img_h / 2
             
             # Current pan is (imgCenter - viewCenter)
             # viewCenter = imgCenter - pan
             vc_x = img_cx - self.pan_x
             vc_y = img_cy - self.pan_y
             
             # Convert to Grid Coords
             gx = (vc_x - self.grid_cfg.start_x) / self.grid_cfg.pitch_x
             gy = (vc_y - self.grid_cfg.start_y) / self.grid_cfg.pitch_y
             
             # c = int(round(gx)) --> Bug with .5 rounding to even
             # Use floor to get the integer index of the cell containing the center
             c = int(np.floor(gx))
             r = int(np.floor(gy))
             
             direction = -1 if e.key() == Qt.Key_Left else 1
             
             # Smart Navigation (Bonding Map Aware)
             if self.bonding_map is not None:
                 # Search for next valid key
                 # Flattened search or row-based?
                 # User likely wants "Next Bonding Patch".
                 # Simple logic: Stay in same row? Or wrap? 
                 # Let's try simple row search first. 
                 # "Next bonding patch" implies skipping NG/Empty ones.
                 
                 found = False
                 search_c = c
                 search_r = r
                 
                 # Limit search to reasonable bounds (e.g. whole grid)
                 max_steps = self.grid_cfg.cols * self.grid_cfg.rows
                 steps = 0
                 
                 while steps < max_steps:
                     search_c += direction
                     
                     # Simple Row Wrap Logic
                     if search_c >= self.grid_cfg.cols:
                         search_c = 0
                         search_r += 1
                     elif search_c < 0:
                         search_c = self.grid_cfg.cols - 1
                         search_r -= 1
                         
                     if search_r < 0 or search_r >= self.grid_cfg.rows:
                         break # End of grid
                         
                     key = self.bonding_map.get_key(search_r, search_c)

                     if key: # Check truthy (skips "" and None)
                         # Found one!
                         c = search_c
                         r = search_r
                         found = True
                         break
                     steps += 1
        
                 if found:
                     self.navigation_requested.emit(c, r)
                 return
                 
             else:
                 # Standard Navigation (Standard Grid Cell)
                 c += direction
                 self.navigation_requested.emit(c, r)
                 return

        elif e.key() == Qt.Key_Up:
             # Prev Layer (Lower Index? Logic inverted relative to scroll?)
             # Scroll Up (Positive) usually means Prev Item
             self.layer_wheel_changed.emit(-1)
             return
        elif e.key() == Qt.Key_Down:
             # Next Layer
             self.layer_wheel_changed.emit(1)
             return

        super().keyPressEvent(e)

    def move_hovered_void_to_current_layer(self):
        """Q: moves the void of another layer under the mouse (ghost) to the current layer."""
        if not self.void_manager or not self.tiled_image:
            return False
        gx, gy = self.get_image_coords(QPointF(self.mapFromGlobal(QCursor.pos())))
        ctx = self._quadrant_ctx()
        for layer in sorted(self.void_manager.voids):
            if layer == self.current_layer:
                continue
            v, _ = self.void_manager.hit_test(layer, gx, gy, margin=10.0/self.zoom,
                                              coord_transform=self.raw_to_deskew, quadrant_ctx=ctx)
            if v is not None and self.void_manager.move_void_to_layer(v, layer, self.current_layer):
                self.voids_changed.emit()
                self.update()
                return True
        return False

    def clear_current_chip_voids(self):
        """
        Clears voids in the chip currently at the center of the view.
        """
        if not self.void_manager: return
        
        # 1. Get View Center in Global Coords
        # 1. Get View Center in Global Coords
        if self.coord_transform:
             img_cx = self.coord_transform.img_cx
             img_cy = self.coord_transform.img_cy
        else:
             img_cx = 500
             img_cy = 500
        
        # View Center in Deskewed (Grid) Space - pan is defined in this space
        gx = img_cx - self.pan_x
        gy = img_cy - self.pan_y

        # 2. Determine Grid Cell (grid is axis-aligned in Deskewed space)
        r, c = self.grid_cfg.cell_at(gx, gy)
        
        print(f"Clearing voids in Chip ({c}, {r})")
        
        # 3. Delete voids in ALL layers for this chip (void centers are stored in Raw space)
        total_count = self.void_manager.remove_voids_where(
            lambda layer, v: self.raw_to_cell(v["globalCX"], v["globalCY"]) == (r, c))
                 
        if total_count > 0:
            print(f"Cleared {total_count} voids in Chip ({c}, {r}) across all layers")
            self.update()
                 

