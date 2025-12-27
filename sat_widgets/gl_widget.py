import sys
import ctypes
import numpy as np

from OpenGL.GL import *
from OpenGL.GL import shaders

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QAction, QColor, QImage, QPainter, QPen
from PySide6.QtCore import Qt, Signal, QRectF, QTimer, QPointF
from PySide6.QtWidgets import QApplication

from .core_data import GridConfig, BondingMap

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
        
        # MEMORY OPTIMIZATION: Release CPU copy immediately
        # We don't need it after upload. If context is lost, we reload from disk (LazyStack).
        self.data = None

    def cleanup(self):
        if self.tex_id:
            glDeleteTextures([self.tex_id])
            self.tex_id = None
        self.is_uploaded = False
        self.data = None # Ensure clear

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
# 3. OpenGL Widget
# ==================================================================================
class GLImageWidget(QOpenGLWidget):
    grid_params_changed = Signal()
    layer_wheel_changed = Signal(int) # Delta (+1 or -1)
    cursor_moved = Signal(int, int) # Col, Row
    navigation_requested = Signal(int, int) # Col, Row (Request to move view)
    
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
        self.current_layer = 0
        
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
            // Rotate by -angle (World to Grid)
            // x_rot = x*cos(-a) - y*sin(-a) = x*c + y*s
            // y_rot = x*sin(-a) + y*cos(-a) = -x*s + y*c
            vec2 rotPos;
            rotPos.x = rel.x * c + rel.y * s;
            rotPos.y = -rel.x * s + rel.y * c;
            
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
                                      overlayColor = mix(overlayColor, mapCol, 0.3); // Tint
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
            
    # -----------------------------------------------
    # Mouse Interaction
    # -----------------------------------------------
    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        modifiers = e.modifiers()
        
        # Swap Logic (User Request 2025-12-28)
        # Ctrl + Wheel: Zoom
        # Wheel: Layer Change
        
        if modifiers & Qt.ControlModifier:
            # Zoom Logic
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
            
        else:
            # Layer Change
            steps = 1 if delta > 0 else -1
            self.layer_wheel_changed.emit(steps)

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
    def paintEvent(self, e):
        # 1. Draw OpenGL content
        super().paintEvent(e)
        
        # 2. Draw 2D Overlay (Voids)
    view_center_changed = Signal(int, int) # Col, Row

    def update_view_center_info(self):
        # Calculate Center (Col, Row) and emit
        if self.tiled_image:
             img_w = self.tiled_image.w
             img_h = self.tiled_image.h
        else:
             img_w = 1000
             img_h = 1000
             
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        gx = img_cx - self.pan_x
        gy = img_cy - self.pan_y
        
        cfg = self.grid_cfg
        dx = gx - cfg.start_x
        dy = gy - cfg.start_y
        
        rad = np.radians(-cfg.angle)
        sin_a = np.sin(rad)
        cos_a = np.cos(rad)
        
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        c = int(np.floor(rx / cfg.pitch_x))
        r = int(np.floor(ry / cfg.pitch_y))
        
        self.view_center_changed.emit(c, r)

    def paintEvent(self, e):
        # 1. Draw OpenGL content
        super().paintEvent(e)
        
        # Emit View Center Info
        self.update_view_center_info()
        
        if self.void_manager and self.tiled_image:
             painter = QPainter(self)
             painter.setRenderHint(QPainter.Antialiasing)
             
             # Current Layer Voids
             layer_voids = self.void_manager.voids.get(self.current_layer, [])
             
             # Draw Helper
             def to_screen(gx, gy):
                 # Inverse of get_image_coords
                 view_w = self.width()
                 view_h = self.height()
                 cx = view_w / 2
                 cy = view_h / 2
                 
                 if self.tiled_image:
                     img_w = self.tiled_image.w
                     img_h = self.tiled_image.h
                 else:
                     img_w = 1000
                     img_h = 1000
                 
                 img_cx = img_w / 2
                 img_cy = img_h / 2
                 
                 screen_x = (gx - img_cx + self.pan_x) * self.zoom + cx
                 screen_y = (gy - img_cy + self.pan_y) * self.zoom + cy
                 return screen_x, screen_y

             # Iterate All Layers
             # First Pass: Ghost Voids (Other Layers)
             for layer_idx, v_list in self.void_manager.voids.items():
                 if layer_idx == self.current_layer:
                     continue
                     
                 for v in v_list:
                     sx, sy = to_screen(v["globalCX"], v["globalCY"])
                     srx = v["radiusX"] * self.zoom
                     sry = v["radiusY"] * self.zoom
                     
                     tid = v.get("type_id", 0)
                     if tid not in self.void_manager.types: tid = 0
                     type_data = self.void_manager.types.get(tid, {"color": (255, 255, 0)})
                     col_rgb = type_data["color"]
                     
                     # Ghost Style: Dotted, Lower Opacity
                     pen = QPen(QColor(col_rgb[0], col_rgb[1], col_rgb[2], 128))
                     pen.setStyle(Qt.DotLine)
                     pen.setWidth(1)
                     painter.setPen(pen)
                     painter.setBrush(Qt.NoBrush)
                     
                     painter.drawEllipse(QPointF(sx, sy), srx, sry)

             # Second Pass: Current Layer Voids (On Top)
             layer_voids = self.void_manager.voids.get(self.current_layer, [])
             
             for v in layer_voids:
                 sx, sy = to_screen(v["globalCX"], v["globalCY"])
                 srx = v["radiusX"] * self.zoom
                 sry = v["radiusY"] * self.zoom
                 
                 # Determine Color
                 tid = v.get("type_id", 0)
                 # Safety check
                 if tid not in self.void_manager.types: tid = 0
                 type_data = self.void_manager.types.get(tid, {"color": (255, 255, 0)})
                 col_rgb = type_data["color"]
                 
                 if v == getattr(self, 'active_void', None):
                     # Active: Red border, Type fill
                     pen = QPen(QColor(255, 0, 0, 255))
                     pen.setStyle(Qt.SolidLine)
                     painter.setPen(pen) 
                     painter.setBrush(QColor(col_rgb[0], col_rgb[1], col_rgb[2], 100))
                 else:
                     # Normal: Type border, no fill
                     pen = QPen(QColor(col_rgb[0], col_rgb[1], col_rgb[2], 255))
                     pen.setStyle(Qt.SolidLine)
                     painter.setPen(pen)
                     painter.setBrush(Qt.NoBrush)
                     
                 # Draw Ellipse
                 painter.drawEllipse(QPointF(sx, sy), srx, sry)

             painter.end()

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
        
        if self.tiled_image:
             img_w = self.tiled_image.w
             img_h = self.tiled_image.h
        else:
             img_w = 1000
             img_h = 1000
             
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        view_center_x = img_cx - self.pan_x
        view_center_y = img_cy - self.pan_y
        
        gx = view_center_x + dx / self.zoom
        gy = view_center_y + dy / self.zoom
        
        return gx, gy

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._last_mouse = e.position()
            self.setFocus()
            
            # Void Interaction
            if self.void_mode and self.void_manager and self.tiled_image:
                gx, gy = self.get_image_coords(e.position())
                
                if self.void_mode == "DRAW":
                    # Start New Void
                    # Corner-to-Corner logic:
                    # Anchor point is the start. Center will move as we drag.
                    self.drag_start_void_pos = (gx, gy)
                    
                    # Create void initially at anchor with 1.0 radius (tiny)
                    self.active_void = self.void_manager.add_void(self.current_layer, gx, gy, 1.0, 1.0, self.active_type_id)
                    self.active_void_action = 'sizing'
                    self.update()
                    return # Consume event
                    
                elif self.void_mode == "EDIT":
                    # Hit Test (Global)
                    v, action, layer = self.void_manager.hit_test_all(gx, gy, margin=10.0/self.zoom, priority_layer=self.current_layer)
                    if v:
                        self.active_void = v
                        self.active_void_action = action # 'center' or 'edge'
                        # Note: We don't necessarily switch self.current_layer.
                        # The void object 'v' is a reference, so modifying it works across layers.
                        self.update()
                        return
                        
                elif self.void_mode == "ERASE":
                    # Hit Test (Global) to find what to delete
                    v, action, layer = self.void_manager.hit_test_all(gx, gy, margin=10.0/self.zoom, priority_layer=self.current_layer)
                    
                    if v and layer is not None:
                        deleted = self.void_manager.delete_void_at(layer, gx, gy)
                        if deleted:
                            self.update()
                            self.active_void_action = 'erasing'
                            return

    def mouseMoveEvent(self, e):
        # 0. Always calculate grid pos and emit signal
        gx, gy = self.get_image_coords(e.position())
        
        # Guard against zero pitch logic/init issues
        if self.grid_cfg.pitch_x > 0 and self.grid_cfg.pitch_y > 0:
             c = int(np.floor((gx - self.grid_cfg.start_x) / self.grid_cfg.pitch_x))
             r = int(np.floor((gy - self.grid_cfg.start_y) / self.grid_cfg.pitch_y))
             self.cursor_moved.emit(c, r)

        if self._last_mouse is None:
            # Hover only
            return
            
            
        dx = e.position().x() - self._last_mouse.x()
        dy = e.position().y() - self._last_mouse.y()
        
        # Void Interaction
        if self.void_mode and getattr(self, 'active_void_action', None):
             # Handle Drag
             gx, gy = self.get_image_coords(e.position())
             
             if self.active_void_action == 'sizing' and self.active_void:
                 # Corner-to-Corner Logic
                 start_x, start_y = getattr(self, 'drag_start_void_pos', (gx, gy))
                 
                 # Current is gx, gy
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
                 
                 # Update Active
                 self.active_void["globalCX"] = cx
                 self.active_void["globalCY"] = cy
                 self.active_void["radiusX"] = max(1.0, rx)
                 self.active_void["radiusY"] = max(1.0, ry)
                 self.update()
                 
             elif self.active_void_action == 'center' and self.active_void:
                 # Move Center
                 # Simple delta
                 img_dx = dx / self.zoom
                 img_dy = dy / self.zoom
                 self.active_void["globalCX"] += img_dx
                 self.active_void["globalCY"] += img_dy
                 self.update()
                 
             elif self.active_void_action == 'edge' and self.active_void:
                 # Resize (Independent Axes)
                 dcx = abs(gx - self.active_void["globalCX"])
                 dcy = abs(gy - self.active_void["globalCY"])
                 self.active_void["radiusX"] = max(1.0, dcx)
                 self.active_void["radiusY"] = max(1.0, dcy)
                 self.update()
                 
             elif self.active_void_action == 'erasing':
                 # Continuous erase?
                 self.void_manager.delete_void_at(self.current_layer, gx, gy)
                 self.update()
                 
             self._last_mouse = e.position()
             return

        # Check modifiers

        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.ControlModifier and self.grid_cfg.visible:
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

    def mouseReleaseEvent(self, e):
        # Clear interaction state
        self._last_mouse = None
        
        if self.void_mode:
            self.active_void = None
            self.active_void_action = None
            self.update()
        
    
    def set_void_tool(self, tool_name):
        self.selected_void_tool = tool_name
        # If currently active (Shift held), update immediate mode
        if self.void_mode:
             self.void_mode = tool_name
             self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Shift:
            if not self.void_mode:
                self.void_mode = self.selected_void_tool
                print(f"Void Mode Active: {self.void_mode}")
                self.update()
            super().keyPressEvent(e)
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

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Shift:
            if self.void_mode:
                self.void_mode = False
                print("Void Mode Deactivated")
                self.update()
        super().keyReleaseEvent(e)

    def clear_current_chip_voids(self):
        """
        Clears voids in the chip currently at the center of the view.
        """
        if not self.void_manager: return
        
        # 1. Get View Center in Global Coords
        if self.tiled_image:
             img_w = self.tiled_image.w
             img_h = self.tiled_image.h
        else:
             img_w = 1000
             img_h = 1000
             
        img_cx = img_w / 2
        img_cy = img_h / 2
        
        # View Center in Global Space
        gx = img_cx - self.pan_x
        gy = img_cy - self.pan_y
        
        # 2. Determine Grid Cell
        cfg = self.grid_cfg
        
        dx = gx - cfg.start_x
        dy = gy - cfg.start_y
        
        rad = np.radians(-cfg.angle)
        sin_a = np.sin(rad)
        cos_a = np.cos(rad)
        
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        c = int(np.floor(rx / cfg.pitch_x))
        r = int(np.floor(ry / cfg.pitch_y))
        
        print(f"Clearing voids in Chip ({c}, {r})")
        
        # 3. Delete voids
        # 3. Delete voids in ALL layers for this chip
        rad_inv = np.radians(-cfg.angle)
        sin_inv = np.sin(rad_inv)
        cos_inv = np.cos(rad_inv)
        
        total_count = 0
        
        # Iterate over copy of item keys since we modify the lists
        for layer, current_voids in list(self.void_manager.voids.items()):
            to_remove = []
            for v in current_voids:
                 vx = v["globalCX"]
                 vy = v["globalCY"]
                 
                 vdx = vx - cfg.start_x
                 vdy = vy - cfg.start_y
                 
                 vrx = vdx * cos_inv - vdy * sin_inv
                 vry = vdx * sin_inv + vdy * cos_inv
                 
                 vc = int(np.floor(vrx / cfg.pitch_x))
                 vr = int(np.floor(vry / cfg.pitch_y))
                 
                 if vc == c and vr == r:
                     to_remove.append(v)
            
            for v in to_remove:
                self.void_manager.voids[layer].remove(v)
                total_count += 1
                 
        if total_count > 0:
            print(f"Cleared {total_count} voids in Chip ({c}, {r}) across all layers")
            self.update()
                 

