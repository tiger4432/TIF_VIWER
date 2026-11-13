import sys
import math
import time
import ctypes
import numpy as np

from OpenGL.GL import *
from OpenGL.GL import shaders

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QAction, QColor, QImage, QPainter, QPen, QCursor, QPixmap
from PySide6.QtCore import Qt, Signal, QRectF, QTimer, QPointF, Slot
from PySide6.QtWidgets import QApplication, QToolTip

from .core_data import GridConfig, BondingMap, CoordinateTransform


# ==================================================================================
# 2. Tiling System
# ==================================================================================
class Tile:
    """
    Represents a single GPU texture tile.
    Pixel data is pulled from `loader` at upload time and released right after,
    so at most one tile is held in CPU memory while a layer is uploaded.
    """
    def __init__(self, x, y, width, height, loader, one_shot=False, linear=False):
        self.x = x  # Global X position
        self.y = y  # Global Y position
        self.w = width
        self.h = height
        self.loader = loader # () -> uint8 / uint16 array; None = data not available yet
        self.one_shot = one_shot # Drop the loader (and its data) after the upload
        self.linear = linear # Linear filtering (downscaled preview textures)
        self.tex_id = None
        self.is_uploaded = False
        self.nbytes = 0 # GPU memory used

    def upload(self):
        if self.is_uploaded or self.loader is None:
            return

        data = self.loader()
        if self.one_shot:
            self.loader = None
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
        filt = GL_LINEAR if self.linear else GL_NEAREST
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # The texture may be smaller than the area it covers (preview): drawn over (x, y, w, h)
        glTexImage2D(
            GL_TEXTURE_2D, 0, internal_fmt,
            data.shape[1], data.shape[0], 0,
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
                loader = (lambda x=x, y=y, w=w_chunk, h=h_chunk: read_tile(x, y, w, h)) if read_tile else None
                self.tiles.append(Tile(x, y, w_chunk, h_chunk, loader))

    @classmethod
    def from_stack(cls, stack, z):
        """Tiles of layer z of an ImageStack, read region by region (never the full layer)."""
        return cls(stack.width, stack.height, lambda x, y, w, h: stack.get_tile(z, x, y, w, h))

    @classmethod
    def pending(cls, width, height):
        """Tiles without data yet (set_tile_data when the background decode is done)."""
        return cls(width, height, None)

    def set_tile_data(self, arrays):
        """arrays: one (h, w) array per tile, in tile order. Each is released after its upload."""
        for t, a in zip(self.tiles, arrays):
            if not t.is_uploaded:
                t.loader = (lambda a=a: a)
                t.one_shot = True

    @classmethod
    def proxy(cls, width, height, image: np.ndarray):
        """Low-resolution preview covering the full width x height with one (small) texture."""
        obj = cls.__new__(cls)
        obj.w, obj.h = width, height
        obj.tiles = [Tile(0, 0, width, height, (lambda: image), one_shot=True, linear=True)]
        return obj

    @property
    def fully_uploaded(self):
        return all(t.is_uploaded for t in self.tiles)

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
    P_STEP = 0.02 # Wheel step of the curve exponent p (Edit tool over a curve void)
    cursor_moved = Signal(int, int) # Col, Row
    navigation_requested = Signal(int, int) # Col, Row (Request to move view)
    voids_changed = Signal() # Void data edited (count labels etc.)
    measure_done = Signal(float) # Measure (scale): distance in image pixels between the two clicked points
    cell_step_requested = Signal(int) # Right click without dragging: +1 next / -1 previous (Shift) cell

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True) # Enable hover events
        self.tiled_image: TiledImage = None
        self.proxy_image: TiledImage = None # Preview of the layer while its tiles are loading
        
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
        self.active_type = None # Void type name used for drawing
        self.bbox_layer = 0 # Absolute layer whose bboxes can be hovered / edited (topmost loaded)
        self.hover_void = None # Void under the mouse (drawn translucent)
        self._hover_state = None # (void, action, axis) for the edit cursor (hysteresis)
        self._digits = "" # Type number being typed (two digits when typed quickly)
        self._digit_time = 0.0
        self._number_started = 0.0
        self._pending_tool = False # Tool change requested during a drag (applied on release)
        self.current_layer = 0 # Absolute layer number
        self.active_void = None
        self.active_void_action = None
        self._void_txn = False # Inside a VoidManager change (mouse press .. release)
        self.edit_locked = False # Detect Voids running in the background: viewing only
        self.detect_cells = None # {(row, col): share done 0..1} of the last Detect Voids run
        self.show_detect_state = True # Its circles at the top-right of each cell
        self.void_line_width = 2.0 # Void outlines on screen (px; config.json ui_defaults "void_line_width")
        self.void_min_px = 5.0 # Voids drawn at least this radius on screen when zoomed out ("void_min_px")
        self.show_outlines = False # Detected shapes (debug)
        self._snap_hint = None # Snap types: press point while drawing (bbox corner nearest to it)
        self._snap_drag_point = None # Snap types: last mouse point while drawing (size on release)
        self._free_quadrant = None # Center-anchored snap type being drawn: quadrant press -> mouse
        self._bbox_followers = [] # Corner voids attached to the bbox being moved (they move along)
        self._resize_plan = None
        self._pan_only = False
        
        # Chip layout of the selected WF product, drawn inside each cell's reference bbox
        self.layout = None

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
                            // Exact map cell (row r, col c), as BondingMap.get_key: the map may
                            // have another size than the grid, so it is not stretched over it
                            ivec2 mapSize = textureSize(map_tex, 0);
                            if (c >= 0 && c < mapSize.x && r >= 0 && r < mapSize.y) {
                                  vec4 mapCol = texelFetch(map_tex, ivec2(c, r), 0);
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
        

    def set_tiled_image(self, tiled_img: TiledImage, cleanup_old: bool = True, upload: bool = True):
        # Cleanup old if requested
        if cleanup_old and self.tiled_image and self.tiled_image != tiled_img:
            self.tiled_image.cleanup()
        
        self.tiled_image = tiled_img
        
        # Init Transform (Only if changed)
        if self.coord_transform is None or \
           self.coord_transform.img_w != self.tiled_image.w or \
           self.coord_transform.img_h != self.tiled_image.h:
               self.coord_transform = CoordinateTransform(self.grid_cfg.angle, self.tiled_image.w, self.tiled_image.h)
        
        # Context must be current for upload (upload=False: the caller uploads tile by tile)
        if upload:
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
        
        # Low-resolution preview under tiles that are not uploaded yet (layer still loading)
        tiles = self.tiled_image.tiles
        if self.proxy_image is not None and not self.tiled_image.fully_uploaded:
            tiles = self.proxy_image.tiles + tiles

        # Draw each tile
        for tile in tiles:
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
        if (self.void_mode != "EDIT" or not self.void_manager or not self.tiled_image
                or modifiers & (Qt.ControlModifier | Qt.ShiftModifier) or delta == 0):
            return False
        gx, gy = self.get_image_coords(e.position())
        v, _, _ = self._hit_any(gx, gy)
        if v is None or self.void_manager.shape_of(v) != "curve":
            return False
        p = self.void_manager.set_p(v, round(round((self.void_manager.p_of(v) + self.P_STEP * delta / 120.0) / self.P_STEP) * self.P_STEP, 4))
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
    void_type_shortcut_triggered = Signal(str) # Type name
    
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
        Mode cursor, only while a mode is active: Draw = crosshair in the type color, Edit = arrow
        (changes over voids, see _update_hover), Erase = red circle with X. View: default cursor.
        """
        if not self.void_mode:
            self.unsetCursor()
        elif self.void_mode == "DRAW":
            # Use Colored Crosshair
            t = self.void_manager.types.get(self.active_type) if self.void_manager else None
            if t is not None:
                col = t["color"]
                cursor = self.create_colored_cross_cursor(col)
                self.setCursor(cursor)
            else:
                self.setCursor(Qt.CrossCursor)
        elif self.void_mode == "ERASE":
            self.setCursor(self.create_erase_cursor())
        elif self.void_mode == "EDIT":
            self.setCursor(Qt.ArrowCursor)

    def _update_hover(self, gx, gy):
        """
        Hover in any mode: the void under the mouse (current layer first, then ghosts) is drawn
        translucent; in edit mode the cursor shows move / resize direction. Hysteresis keeps the
        cursor from flickering at the edge band and direction boundaries.
        """
        if not self.void_manager or not self.tiled_image:
            return
        vm = self.void_manager
        margin = 10.0 / self.zoom
        ctx = self._quadrant_ctx()
        v, action = vm.hit_test(self.current_layer, gx, gy, margin=margin, coord_transform=self.raw_to_deskew, quadrant_ctx=ctx, allow_bbox=self.bbox_recognized())

        prev = self._hover_state
        if prev is not None and prev[0] is not None and prev[1] == "edge":
            # Stay on the edge of the previous void until clearly away from it (1.6x band)
            pv = prev[0]
            pcx, pcy = self.raw_to_deskew(pv["globalCX"], pv["globalCY"])
            shape = vm.hit_shape(pv)
            dist, inside = vm.boundary_distance(shape, gx - pcx, gy - pcy, pv["radiusX"], pv["radiusY"],
                                                vm.p_of(pv) if shape == "curve" else 2.0)
            if (any(pv is x for x in vm.voids.get(self.current_layer, [])) and dist <= 1.6 * margin and (v is None or v is pv)
                    and (self.bbox_recognized() or not vm.is_bbox(pv))):
                v, action = pv, "edge"
        if v is None:
            # Ghost (other layer) voids: highlighted, edited (E) / erased (R) like the current
            # layer's; Q moves them here
            v, action, _ = self._hit_any(gx, gy, margin)

        axis = None
        if v is not None and action == "edge":
            axis = self._grab_axis(v, gx, gy, prev[2] if (prev and prev[0] is v and prev[1] == "edge") else "unset")
        state = (v, action, axis)
        if v is not self.hover_void:
            self.hover_void = v
            self.update()
        self._hover_state = state

        if self.void_mode != "EDIT":
            return
        if v is None or action is None:
            shape = Qt.ArrowCursor
        elif action == "center":
            shape = Qt.SizeAllCursor
        elif axis == "x":
            shape = Qt.SizeHorCursor
        elif axis == "y":
            shape = Qt.SizeVerCursor
        else:
            cx, cy = self.raw_to_deskew(v["globalCX"], v["globalCY"])
            shape = Qt.SizeFDiagCursor if (gx - cx) * (gy - cy) > 0 else Qt.SizeBDiagCursor
        if self.cursor().shape() != shape:
            self.setCursor(shape)

    def _paint_outline(self, painter, v):
        """Detected shape (debug, Shapes in Void Detection > Advanced): raw offsets from the centre,
        scaled with the radii (the void may have been resized or moved since)."""
        r0x, r0y = v["outline_r"]
        kx, ky = v["radiusX"] / max(r0x, 1e-6), v["radiusY"] / max(r0y, 1e-6)
        cx, cy = v["globalCX"], v["globalCY"]
        pts = []
        for dx, dy in v["outline"]:
            px, py = self.deskew_to_screen(*self.raw_to_deskew(cx + dx * kx, cy + dy * ky))
            pts.append(QPointF(px, py))
        if len(pts) < 3:
            return
        painter.save()
        for color, width in ((QColor(0, 0, 0, 150), 3.0), (QColor(255, 80, 255, 230), 1.5)):
            pen = QPen(color); pen.setWidthF(width)
            painter.setPen(pen); painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(pts)
        painter.restore()

    def refresh_hover(self):
        """Re-evaluates the hovered void at the mouse position (after undo, load, layer change)."""
        self.hover_void = None
        self._hover_state = None
        pos = self.mapFromGlobal(QCursor.pos())
        if self.void_manager and self.tiled_image and self.rect().contains(pos):
            self._update_hover(*self.get_image_coords(QPointF(pos)))
        self.update()

    def _live_hover(self):
        """The hovered void, or None; a void replaced by undo / load is re-evaluated first."""
        hv = self.hover_void
        if hv is not None and not any(hv is x for vs in self.void_manager.voids.values() for x in vs):
            self.refresh_hover()
            hv = self.hover_void
        return hv

    def _grab_axis(self, v, gx, gy, prev_axis="unset"):
        """
        Resize direction for a grab at (gx, gy) on the outline: "x", "y" or None (both / uniform),
        the same rule the resize uses. With prev_axis, a wider threshold keeps the previous
        direction (hysteresis).
        """
        vm = self.void_manager
        shape = vm.hit_shape(v)
        p = vm.p_of(v) if shape == "curve" else 2.0
        cx, cy = self.raw_to_deskew(v["globalCX"], v["globalCY"])
        ux = abs(gx - cx) / max(v["radiusX"], 1e-9)
        uy = abs(gy - cy) / max(v["radiusY"], 1e-9)
        if shape == "rectangle":
            bx, by = min(ux, 1.0), min(uy, 1.0)
            corner = 0.70 if prev_axis is None else 0.80 # Corner zone: both axes
            if bx > corner and by > corner:
                return None
            if prev_axis in ("x", "y") and abs(bx - by) < 0.1:
                return prev_axis
            return "x" if bx >= by else "y"
        t = (ux ** p + uy ** p) ** (1.0 / p) or 1.0
        bx, by = ux / t, uy / t
        lo = {"x": 0.47, "y": 0.47}.get(prev_axis, 0.40)
        if by < (lo if prev_axis == "x" else 0.40):
            return "x"
        if bx < (lo if prev_axis == "y" else 0.40):
            return "y"
        return None

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
        if getattr(self, "measure_points", None):
            mp = QPainter(self)
            self._paint_measure(mp)
            mp.end()
        
        if self.void_manager and self.tiled_image:
             painter = QPainter(self)
             painter.setRenderHint(QPainter.Antialiasing)
             
             # Chip layout inside each cell's reference bbox
             self._draw_layout(painter)

             # First Pass: Ghost Voids (Other Layers), then Current Layer Voids on top
             vm = self.void_manager
             refs = vm.reference_bboxes(self.grid_cfg, self.coord_transform)

             vw, vh = self.width(), self.height()
             hover = self.hover_void

             def paint(v, style):
                 if not vm.passes(v):
                     return # Below the SNR filter
                 # Raw (storage) -> Deskewed (visual, applies the grid rotation) -> Screen
                 dx, dy = self.raw_to_deskew(v["globalCX"], v["globalCY"])
                 sx, sy = self.deskew_to_screen(dx, dy)
                 rr = max(v["radiusX"], v["radiusY"]) * self.zoom + 2
                 if sx + rr < 0 or sy + rr < 0 or sx - rr > vw or sy - rr > vh:
                     return # Off screen
                 if v is hover and style != "active":
                     style = "hover"
                 quadrant = self._free_quadrant if (v is self.active_void and self._free_quadrant) \
                     else vm.quadrant_of(v, dx, dy, refs, self.grid_cfg)
                 ghost = style == "ghost"
                 vm.paint_void(painter, v, sx, sy, self.zoom, style=style, quadrant=quadrant, badges=True,
                               width=1.0 if ghost else self.void_line_width, halo=not ghost,
                               min_radius=0.0 if vm.is_bbox(v) else self.void_min_px)
                 if self.show_outlines and not ghost and v.get("outline") and v.get("outline_r"):
                     self._paint_outline(painter, v)

             for layer_idx, v_list in vm.voids.items():
                 if layer_idx == self.current_layer:
                     continue
                 for v in v_list:
                     paint(v, "ghost")

             active = getattr(self, 'active_void', None)
             for v in vm.voids.get(self.current_layer, []):
                 paint(v, "active" if v is active else "normal")


             self._draw_detect_state(painter)
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

    DETECT_R = 6 # Radius (screen px) of the per-cell detection circle

    def _draw_detect_state(self, painter):
        """
        Detect Voids progress per cell: a circle at the top-right corner of each inspected cell,
        empty = pending, pie = share done, filled with a check = done. Constant screen size; not
        drawn when the cells are small on screen or the grid is hidden.
        """
        cells, cfg = self.detect_cells, self.grid_cfg
        r = self.DETECT_R
        if not cells or not self.show_detect_state or not cfg.visible or \
                min(cfg.pitch_x, cfg.pitch_y) * self.zoom < 8 * r:
            return
        vw, vh = self.width(), self.height()
        dark, green = QColor(0, 0, 0, 170), QColor(60, 220, 90)
        painter.save()
        for (row, col), frac in cells.items():
            x0, y0 = cfg.cell_origin(row, col)
            sx, sy = self.deskew_to_screen(x0 + cfg.pitch_x, y0)
            cx, cy = sx - r - 4, sy + r + 4
            if cx < -r or cy < -r or cx > vw + r or cy > vh + r:
                continue
            box = QRectF(cx - r, cy - r, 2 * r, 2 * r)
            painter.setPen(QPen(dark, 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(box)
            if frac >= 1.0:
                painter.setPen(QPen(green, 1.5))
                painter.setBrush(green)
                painter.drawEllipse(box)
                painter.setPen(QPen(QColor(0, 0, 0), 1.6))
                painter.drawPolyline([QPointF(cx - 0.55 * r, cy), QPointF(cx - 0.15 * r, cy + 0.45 * r),
                                      QPointF(cx + 0.55 * r, cy - 0.45 * r)])
                continue
            if frac > 0:
                painter.setPen(Qt.NoPen)
                painter.setBrush(green)
                painter.drawPie(box, 90 * 16, -int(360 * 16 * frac))
            painter.setPen(QPen(QColor(235, 235, 235), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(box)
        painter.restore()

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
        if not self.layout or not self.void_manager:
            return
        refs = self.void_manager.reference_bboxes(self.grid_cfg, self.coord_transform)
        vw, vh = self.width(), self.height()
        for (x0, y0, w, h) in refs.values():
            s0 = self.deskew_to_screen(x0, y0)
            s1 = self.deskew_to_screen(x0 + w, y0 + h)
            if max(s0[0], s1[0]) < 0 or max(s0[1], s1[1]) < 0 or min(s0[0], s1[0]) > vw or min(s0[1], s1[1]) > vh:
                continue # Off screen
            self.void_manager.paint_layout(painter, (x0, y0, w, h), self.deskew_to_screen)

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

    # ---- Measure: two clicked points -> their distance in image pixels (Scale from a known length) ----
    def start_measure(self):
        self.measure_points = []
        self._measure_mouse = None
        self.setCursor(Qt.CrossCursor)
        self.setFocus()
        self.update()

    def cancel_measure(self):
        self.measure_points = None
        self._measure_mouse = None
        self.update_cursor_style()
        self.update()

    def _image_to_screen(self, gx, gy):
        """Inverse of get_image_coords (deskewed image coordinates -> widget pixels)."""
        if self.coord_transform:
            img_cx, img_cy = self.coord_transform.img_cx, self.coord_transform.img_cy
        else:
            img_cx = img_cy = 500
        return (QPointF(self.width() / 2 + (gx - (img_cx - self.pan_x)) * self.zoom,
                        self.height() / 2 + (gy - (img_cy - self.pan_y)) * self.zoom))

    def _paint_measure(self, painter):
        pts = list(getattr(self, "measure_points", None) or [])
        if not pts:
            return
        if len(pts) == 1 and getattr(self, "_measure_mouse", None) is not None:
            pts.append(self._measure_mouse)
        sp = [self._image_to_screen(*p) for p in pts]
        painter.setRenderHint(QPainter.Antialiasing)
        for color, width in ((QColor(0, 0, 0, 200), 4), (QColor(0, 255, 255), 2)):
            painter.setPen(QPen(color, width))
            if len(sp) == 2:
                painter.drawLine(sp[0], sp[1])
            for q in sp:
                painter.drawLine(QPointF(q.x() - 8, q.y()), QPointF(q.x() + 8, q.y()))
                painter.drawLine(QPointF(q.x(), q.y() - 8), QPointF(q.x(), q.y() + 8))
        if len(pts) == 2:
            d = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            painter.setPen(QColor(0, 255, 255))
            painter.drawText(sp[1] + QPointF(10, -10), f"{d:.1f} px")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and getattr(self, "measure_points", None) is not None:
            self.measure_points.append(self.get_image_coords(e.position()))
            self.update()
            if len(self.measure_points) == 2:
                (x0, y0), (x1, y1) = self.measure_points
                self.cancel_measure()
                self.measure_done.emit(math.hypot(x1 - x0, y1 - y0)) # Deskewed = raw distance (rotation)
            return
        if e.button() in (Qt.MiddleButton, Qt.RightButton):
            # Pan in every mode (a right click without dragging steps to the next cell, see release)
            self._last_mouse = e.position()
            self._pan_only = True
            self._press_pos = e.position()
            self.setFocus()
            return
        if e.button() == Qt.LeftButton:
            self._last_mouse = e.position()
            self._pan_only = False
            self.setFocus()
            
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
                        
                        if self.active_type in self.void_manager.types:
                             tdata = self.void_manager.types[self.active_type]
                             def_w = tdata.get("def_w", 0)
                             def_h = tdata.get("def_h", 0)
                        
                        is_sizing = (def_w <= 0 or def_h <= 0)
                        
                        if not is_sizing:
                            # Predefined Size (Click-to-Draw): the click is the top-left corner
                            # of the box in the Deskewed frame (grid axes); stored as Raw center
                            # + half sizes (VoidManager.place_void, as automatic detection).
                            # Snap types follow the mouse freely and attach to the bbox corner
                            # when the mouse is released.
                            cx_raw, cy_raw = self.deskew_to_raw(gx + def_w / 2.0, gy + def_h / 2.0)
                            self._snap_hint = None
                            self.active_void = self.void_manager.place_void(
                                self.current_layer, cx_raw, cy_raw, def_w / 2.0, def_h / 2.0, self.active_type,
                                self.grid_cfg, self.coord_transform, snap=False)

                            # Switch to Move mode immediately
                            self.active_void_action = 'center'
                            
                        else:
                            # Drag-to-Size
                            # Initialize 1x1 void
                            # Center/Pos is rx, ry
                            self.active_void = self.void_manager.add_void(self.current_layer, rx, ry, 1.0, 1.0, self.active_type)
                            self.active_void_action = 'sizing'
                            self._snap_hint = (gx, gy) # Snap types: corner nearest to the press point
                        if self.active_void is not None and self.void_manager.is_bbox(self.active_void):
                            self.active_void["hand"] = True # Placed by hand: Refine BBOX keeps it, aligns to it

                    self.voids_changed.emit()
                    self.update()
                    return # Consume event
                    
                elif self.void_mode == "EDIT":
                    # Hit Test on the current layer (ghost voids of other layers are reference only)
                    # Mouse (gx, gy) is Deskewed -> pass the Raw -> Deskew transform.
                    # Current layer first, else a ghost of another layer (edited on its own layer)
                    v, action, _ = self._hit_any(gx, gy)
                    if v:
                        self.active_void = v
                        self.active_void_action = action # 'center' or 'edge'
                        v.pop("auto", None) # Reviewed by hand (no longer "estimated")
                        if self.void_manager.is_bbox(v):
                            v["hand"] = True # Refine BBOX keeps it and aligns the others to it
                        self._snap_hint = None # Snap types re-attach when the mouse is released
                        if action == 'center' and self.void_manager.is_bbox(v):
                            self._bbox_followers = self.void_manager.snapped_to(v, self.grid_cfg, self.coord_transform)
                        if action == 'edge':
                            # Alt held: from the centre (both sides); else the opposite side stays
                            self._resize_plan = self._make_resize_plan(v, gx, gy, centered=bool(e.modifiers() & Qt.AltModifier))
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
        if getattr(self, "measure_points", None):
            self._measure_mouse = (gx, gy) # Rubber line from the first measure point
            self.update()
        
        # Guard against zero pitch logic/init issues
        if self.grid_cfg.pitch_x > 0 and self.grid_cfg.pitch_y > 0:
             c = int(np.floor((gx - self.grid_cfg.start_x) / self.grid_cfg.pitch_x))
             r = int(np.floor((gy - self.grid_cfg.start_y) / self.grid_cfg.pitch_y))
        self.cursor_moved.emit(gx, gy)

        if self._last_mouse is None:
            # Hover only
            self._update_hover(gx, gy)
            return
            
            
        dx = e.position().x() - self._last_mouse.x()
        dy = e.position().y() - self._last_mouse.y()
        
        # Void Interaction
        if self.void_mode and self.active_void_action and not getattr(self, '_pan_only', False):
             self.void_manager.touch() # Fields are edited directly below
             # Handle Drag
             gx, gy = self.get_image_coords(e.position())
             
             if self.active_void_action == 'sizing' and self.active_void:
                 # Corner-to-Corner Logic
                 start_x = getattr(self, '_void_start_gx', gx)
                 start_y = getattr(self, '_void_start_gy', gy)
                 
                 # The drawn box spans press point -> mouse
                 bx0, bx1 = sorted((start_x, gx))
                 by0, by1 = sorted((start_y, gy))
                 v = self.active_void
                 vm = self.void_manager
                 anchor = vm.snap_anchor(v)
                 # Corner-snap types: the bbox corner nearest to the FIRST click (press point), whatever
                 # the drag direction; on release the shape spans that corner -> the box point farthest
                 # from it (see _snap_on_release)
                 k = vm.corner_for_box((start_x, start_y, start_x, start_y), self.grid_cfg, self.coord_transform) if anchor else None
                 if k is not None:
                     _, _, ix, iy = k
                     near = (bx0 if ix > 0 else bx1, by0 if iy > 0 else by1)
                     far = (bx1 if ix > 0 else bx0, by1 if iy > 0 else by0)
                     self._snap_hint, self._snap_drag_point = (start_x, start_y), far
                 if k is not None and anchor == "center":
                     # Quadrant shape inside the drawn box, oriented as it will attach
                     cx, cy = near
                     rx, ry = bx1 - bx0, by1 - by0
                     self._free_quadrant = (ix, iy)
                 else:
                     cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
                     rx, ry = (bx1 - bx0) / 2.0, (by1 - by0) / 2.0
                 v["radiusX"], v["radiusY"] = max(1.0, rx), max(1.0, ry)
                 v["globalCX"], v["globalCY"] = self.deskew_to_raw(cx, cy)
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
                 for f in self._bbox_followers: # Corner voids move with their bbox
                     f["globalCX"] += raw_dx
                     f["globalCY"] += raw_dy
                 # Snap types move freely; they re-attach when the mouse is released
                 self.update()
                 
             elif self.active_void_action == 'edge' and self.active_void:
                 shape = self.void_manager.shape_of(self.active_void)
                 
                 # Get Center in Deskewed Space (Reference)
                 cx_deskew, cy_deskew = self.raw_to_deskew(self.active_void["globalCX"], self.active_void["globalCY"])
                 
                 self._apply_resize(self.active_void, gx, gy) # Snap types re-attach on release
                 
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

    def _make_resize_plan(self, v, gx, gy, centered=False):
        """
        How a drag from the grab point (Deskewed gx, gy) on the outline resizes the void, so the
        grabbed outline point follows the mouse. The opposite side / point stays and only the
        grabbed side moves (centered=True, Alt held: the center stays and both sides move):
        - rectangle: grabbed side(s) ("anchored" / "axes");
        - ellipse / curve near the end of an axis: only that axis ("anchored" / "axes");
        - ellipse / curve on the curved part: uniform scaling about the opposite outline point
          ("uniform", "a") / the center;
        - quadrant voids (center on a bbox corner): the corner stays; x end / y end / arc
          ("quadrant");
        - corner-snap types anchored by their edge: the anchored corner stays ("anchored").
        """
        vm = self.void_manager
        shape = vm.hit_shape(v) # Curve / snap types: their box
        p = vm.p_of(v) if shape == "curve" else 2.0
        cx, cy = self.raw_to_deskew(v["globalCX"], v["globalCY"])
        rx, ry = v["radiusX"], v["radiusY"]
        q = vm.quadrant_of(v, cx, cy, *self._quadrant_ctx())
        dx, dy = gx - cx, gy - cy

        # Grabbed point on the unit outline; the direction shown by the hover cursor
        ux, uy = abs(dx) / max(rx, 1e-9), abs(dy) / max(ry, 1e-9)
        if shape == "rectangle":
            bx, by = min(ux, 1.0), min(uy, 1.0)
        else:
            t = (ux ** p + uy ** p) ** (1.0 / p) or 1.0
            bx, by = ux / t, uy / t
        hs = self._hover_state
        axis = hs[2] if (hs and hs[0] is v and hs[1] == "edge") else self._grab_axis(v, gx, gy)

        if q:
            return {"kind": "quadrant", "q": q, "c": (cx, cy), "r0": (rx, ry), "p": p, "axis": axis, "shape": shape}
        if vm.snap_anchor(v) == "edge":
            # Corner-snap types: the corner at the bbox corner stays, the grabbed side(s) follow
            k = vm.snap_corner_point(v, self.grid_cfg, self.coord_transform)
            if k is not None:
                fixed = min(((cx + a * rx, cy + b * ry) for a in (-1, 1) for b in (-1, 1)),
                            key=lambda c: math.hypot(c[0] - k[0], c[1] - k[1]))
                return {"kind": "anchored", "fixed": fixed, "x": axis in (None, "x"), "y": axis in (None, "y")}
        sx = 1.0 if dx >= 0 else -1.0
        sy = 1.0 if dy >= 0 else -1.0
        if shape == "rectangle" or axis is not None:
            if not centered: # The opposite side(s) stay
                return {"kind": "anchored", "fixed": (cx - sx * rx, cy - sy * ry),
                        "x": axis in (None, "x"), "y": axis in (None, "y")}
            return {"kind": "axes", "c": (cx, cy), "x": axis in (None, "x"), "y": axis in (None, "y")}
        B = (sx * rx * bx, sy * ry * by) # Center -> grabbed outline point
        return {"kind": "uniform", "c": (cx, cy), "r0": (rx, ry), "B": B,
                "a": None if centered else (cx - B[0], cy - B[1])} # Opposite outline point stays

    def _apply_resize(self, v, gx, gy):
        """Resizes v following the mouse (Deskewed gx, gy) according to the plan from the press."""
        plan = self._resize_plan
        if plan is None:
            return
        kind = plan["kind"]
        if kind == "axes": # Center fixed: the grabbed side follows the mouse
            cx, cy = plan["c"]
            if plan["x"]:
                v["radiusX"] = max(1.0, abs(gx - cx))
            if plan["y"]:
                v["radiusY"] = max(1.0, abs(gy - cy))
        elif kind == "anchored":
            cx, cy = self.raw_to_deskew(v["globalCX"], v["globalCY"])
            fx, fy = plan["fixed"]
            if plan["x"]:
                v["radiusX"] = max(1.0, abs(gx - fx) / 2.0)
                cx = fx + math.copysign(v["radiusX"], gx - fx)
            if plan["y"]:
                v["radiusY"] = max(1.0, abs(gy - fy) / 2.0)
                cy = fy + math.copysign(v["radiusY"], gy - fy)
            v["globalCX"], v["globalCY"] = self.deskew_to_raw(cx, cy)
        elif kind == "uniform": # Scale so the grabbed point follows the mouse
            (cx, cy), (bx, by), (r0x, r0y) = plan["c"], plan["B"], plan["r0"]
            s_min = 1.0 / max(min(r0x, r0y), 1e-9)
            if plan.get("a") is None: # About the center
                s = ((gx - cx) * bx + (gy - cy) * by) / max(bx * bx + by * by, 1e-9)
                v["radiusX"], v["radiusY"] = r0x * max(s, s_min), r0y * max(s, s_min)
            else: # About the opposite outline point a: grabbed point a + 2B -> a + 2sB, center a + sB
                ax, ay = plan["a"]
                s = max(((gx - ax) * bx + (gy - ay) * by) / max(2.0 * (bx * bx + by * by), 1e-9), s_min)
                v["radiusX"], v["radiusY"] = r0x * s, r0y * s
                v["globalCX"], v["globalCY"] = self.deskew_to_raw(ax + s * bx, ay + s * by)
        else: # quadrant: the corner (center) stays
            (cx, cy), (qx, qy), (r0x, r0y) = plan["c"], plan["q"], plan["r0"]
            mx = max(0.0, (gx - cx) * qx)
            my = max(0.0, (gy - cy) * qy)
            axis = plan["axis"]
            if axis == "x":
                v["radiusX"] = max(1.0, mx)
            elif axis == "y":
                v["radiusY"] = max(1.0, my)
            elif plan["shape"] == "rectangle":
                v["radiusX"], v["radiusY"] = max(1.0, mx), max(1.0, my)
            else:
                p = plan["p"]
                s = ((mx / r0x) ** p + (my / r0y) ** p) ** (1.0 / p)
                s = max(s, 1.0 / max(min(r0x, r0y), 1e-9))
                v["radiusX"], v["radiusY"] = r0x * s, r0y * s

    def bbox_recognized(self):
        """
        Bboxes are recognized (hover, edit, erase, Q, type keys) only on the topmost loaded layer, and
        not while locked (bbox_locked: the markings inside are edited without catching the bbox).
        """
        return self.current_layer == self.bbox_layer and not getattr(self, "bbox_locked", False)

    def _quadrant_ctx(self):
        return (self.void_manager.reference_bboxes(self.grid_cfg, self.coord_transform), self.grid_cfg)

    def _hit_any(self, gx, gy, margin=None):
        """
        Void under the Deskewed point: the current layer first, else a ghost of another layer (the
        nearest layer first), so edit (E) and erase (R) work on every marking seen, hand-drawn or
        detected. Returns (void, action, layer) or (None, None, None).
        """
        vm = self.void_manager
        margin = 10.0 / self.zoom if margin is None else margin
        ctx = self._quadrant_ctx()
        layers = sorted(vm.voids, key=lambda l: (l != self.current_layer, abs(l - self.current_layer)))
        for layer in layers:
            v, action = vm.hit_test(layer, gx, gy, margin=margin, coord_transform=self.raw_to_deskew, quadrant_ctx=ctx,
                                    allow_bbox=self.bbox_recognized() and layer == self.current_layer)
            if v is not None:
                return v, action, layer
        return None, None, None

    def _erase_at(self, gx, gy):
        """Deletes the void under the Deskewed point (current layer first, else the nearest ghost)."""
        v, _, layer = self._hit_any(gx, gy)
        return v is not None and self.void_manager.remove_void(layer, v)

    def mouseReleaseEvent(self, e):
        # Clear interaction state
        self._last_mouse = None
        if getattr(self, '_pan_only', False):
            self._pan_only = False
            self.update_cursor_style()
            p0 = getattr(self, '_press_pos', None)
            if e.button() == Qt.RightButton and p0 is not None and \
                    abs(e.position().x() - p0.x()) + abs(e.position().y() - p0.y()) < 4:
                self.cell_step_requested.emit(-1 if e.modifiers() & Qt.ShiftModifier else 1)
            return

        # A click without dragging in Drag-to-Size mode leaves a 1px void -> discard it
        if self.active_void_action == 'sizing' and self.active_void is not None:
            start = getattr(self, '_void_start', None)
            if start is not None:
                moved = abs(e.position().x() - start.x()) + abs(e.position().y() - start.y())
                if moved < 3:
                    self.void_manager.remove_void(self.current_layer, self.active_void)
                    self.active_void = None
        self._snap_on_release()

        changed = self._end_void_change()
        if self._pending_tool is not False:
            tool, self._pending_tool = self._pending_tool, False
            self.active_void_action = None
            self.set_void_tool(tool)
        if self.void_mode or self.active_void_action:
            self.active_void = None
            self.active_void_action = None
            self._snap_hint = None
            self.update()
        if changed:
            self.voids_changed.emit()

    def _snap_on_release(self):
        """
        Snap types are drawn / moved / resized as the mouse goes; they attach to the bbox corner
        only when the mouse is released: drawn ones from the corner nearest the press point to
        the last mouse point, moved or resized ones to the corner nearest to them (size kept).
        """
        v, action = self.active_void, self.active_void_action
        drag, hint = self._snap_drag_point, self._snap_hint
        self._snap_drag_point = self._free_quadrant = None
        followers, self._bbox_followers = self._bbox_followers, []
        if followers: # Corner voids of a moved bbox: exactly on its corners again
            self.void_manager.resnap(self.grid_cfg, self.coord_transform, followers)
        if v is None or action not in ('sizing', 'center', 'edge') or self.void_manager.snap_anchor(v) is None:
            return
        if action == 'sizing' and drag is not None:
            self.void_manager.snap_void(v, self.grid_cfg, self.coord_transform, hint=hint, drag_point=drag)
        else:
            self.void_manager.snap_void(v, self.grid_cfg, self.coord_transform)
        self.update()

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
        if self.edit_locked:
            tool_name = None
        if self.active_void_action or getattr(self, '_void_txn', False):
            self._pending_tool = tool_name # Applied when the drag ends
            return
        if tool_name:
            self.selected_void_tool = tool_name
        self.void_mode = tool_name or False
        self.update_cursor_style()
        self.update()

    def type_number_key(self, digit):
        if not self.void_manager:
            return
        names = list(self.void_manager.types)
        now = time.monotonic()
        combined = self._digits + str(digit)
        if self._digits and now - self._digit_time < 0.4 and 1 <= int(combined) <= len(names):
            self._digits = combined # Second digit of the same number
        else:
            self._digits = str(digit)
            self._number_started = now
        self._digit_time = now
        number = int(self._digits) or 10
        if not 1 <= number <= len(names):
            return
        name = names[number - 1]
        hv = None if self.edit_locked else self._live_hover()
        if hv is not None:
            # The digits of one number are one undo step
            if self.void_manager.set_void_type(hv, name, coalesce_key=("type", id(hv), self._number_started)):
                self.voids_changed.emit()
        else:
            self.active_type = name
            self.void_type_shortcut_triggered.emit(name) # Sync UI
            if self.void_mode:
                self.update_cursor_style()
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and getattr(self, "measure_points", None) is not None:
            self.cancel_measure()
            return
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
        
        # Type numbers (declaration order, 1-based; 0 alone = 10). Digits typed within 0.4 s
        # form a two-digit number. Over a void: change its type; else: the type to draw with.
        key = e.key()
        if Qt.Key_0 <= key <= Qt.Key_9:
            self.type_number_key(key - Qt.Key_0)
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

    def toggle_hovered_layer0(self):
        """B: sets / clears the "visible on layer 0" flag of the void under the mouse (any layer, undoable)."""
        if not self.void_manager or not self.tiled_image or self.edit_locked:
            return None
        hv = self._live_hover()
        if hv is None:
            return None
        state = self.void_manager.toggle_layer0(hv)
        self.voids_changed.emit()
        self.update()
        return state

    def move_hovered_void_to_current_layer(self):
        """Q: moves the void of another layer under the mouse (ghost) to the current layer."""
        if not self.void_manager or not self.tiled_image or self.edit_locked:
            return False
        hv = self._live_hover()
        if hv is not None:
            layer = next((l for l, vs in self.void_manager.voids.items() if any(hv is x for x in vs)), None)
            if layer is not None and layer != self.current_layer:
                if self.void_manager.move_void_to_layer(hv, layer, self.current_layer):
                    self.voids_changed.emit()
                    self.update()
                    return True
                return False
        gx, gy = self.get_image_coords(QPointF(self.mapFromGlobal(QCursor.pos())))
        ctx = self._quadrant_ctx()
        for layer in sorted(self.void_manager.voids):
            if layer == self.current_layer:
                continue
            v, _ = self.void_manager.hit_test(layer, gx, gy, margin=10.0/self.zoom,
                                              coord_transform=self.raw_to_deskew, quadrant_ctx=ctx, allow_bbox=self.bbox_recognized())
            if v is not None and self.void_manager.move_void_to_layer(v, layer, self.current_layer):
                self.voids_changed.emit()
                self.update()
                return True
        return False

    def clear_current_chip_voids(self):
        """
        Clears voids in the chip currently at the center of the view.
        """
        if not self.void_manager or self.edit_locked: return
        
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
                 

