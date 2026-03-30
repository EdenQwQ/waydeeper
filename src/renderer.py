"""OpenGL renderer for waydeeper using GTK4 Layer Shell."""

import logging
import signal
from dataclasses import dataclass

import numpy as np
from PIL import Image
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader

from ctypes import CDLL

try:
    CDLL("libgtk4-layer-shell.so")
except OSError:
    raise Error("GTK4 Layer Shell library not found. Please install gtk4-layer-shell.")

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk, Gdk, GLib, Gio
    from gi.repository import Gtk4LayerShell
except ImportError as import_error:
    raise ImportError(
        "GTK4 or Gtk4LayerShell not found. Please install gtk4-layer-shell."
    ) from import_error

from src.depth_estimator import load_depth_map

logger = logging.getLogger(__name__)

vertex_shader_source = """
#version 300 es
precision mediump float;

layout(location = 0) in vec2 vertex_position;
layout(location = 1) in vec2 texture_coordinate;

out vec2 fragment_texture_coordinate;

void main() {
    gl_Position = vec4(vertex_position, 0.0, 1.0);
    fragment_texture_coordinate = texture_coordinate;
}
"""

fragment_shader_source = """
#version 300 es
precision mediump float;

in vec2 fragment_texture_coordinate;
out vec4 output_color;

uniform sampler2D wallpaper_texture;
uniform sampler2D depth_texture;
uniform vec2 mouse_position;
uniform vec2 screen_resolution;
uniform vec2 parallax_strength;
uniform float zoom_level;
uniform vec2 image_dimensions;
uniform bool inverse_depth;

void main() {
    float depth_value = texture(depth_texture, fragment_texture_coordinate).r;
    
    // Invert depth if requested (for models where white=far instead of white=near)
    if (inverse_depth) {
        depth_value = 1.0 - depth_value;
    }
    
    vec2 mouse_offset_from_center = mouse_position - vec2(0.5);
    vec2 parallax_offset = mouse_offset_from_center * depth_value * parallax_strength;
    
    float screen_aspect = screen_resolution.x / screen_resolution.y;
    float image_aspect = image_dimensions.x / image_dimensions.y;
    
    vec2 uv_scale;
    if (image_aspect > screen_aspect) {
        uv_scale.x = screen_aspect / image_aspect;
        uv_scale.y = 1.0;
    } else {
        uv_scale.x = 1.0;
        uv_scale.y = image_aspect / screen_aspect;
    }
    
    uv_scale /= zoom_level;
    
    vec2 centered_position = fragment_texture_coordinate - vec2(0.5);
    vec2 scaled_position = centered_position * uv_scale;
    vec2 sample_coordinate = scaled_position + vec2(0.5) + parallax_offset;
    vec2 clamped_coordinate = clamp(sample_coordinate, 0.001, 0.999);
    
    output_color = texture(wallpaper_texture, clamped_coordinate);
}
"""


@dataclass
class Textures:
    wallpaper: int | None = None
    depth: int | None = None


class WallpaperRenderer(Gtk.GLArea):
    def __init__(
        self,
        smooth_animation=True,
        animation_speed=0.02,
        fps=60,
        active_delay_ms=150.0,
        idle_timeout_ms=500.0,
        inverse_depth=False,
    ):
        super().__init__()

        self.shader_program = None
        self.vertex_array_object = None
        self.vertex_buffer_object = None
        self.element_buffer_object = None
        self.textures = Textures()

        self.image_dimensions = (1, 1)
        self.mouse_position = (0.5, 0.5)
        self.current_mouse_position = (0.5, 0.5)
        self.parallax_strength_x = 0.02
        self.parallax_strength_y = 0.02
        self.inverse_depth = inverse_depth

        self.smooth_animation = smooth_animation
        self.animation_speed = animation_speed

        self.fps = fps
        self.frame_interval_ms = int(1000 / fps)
        self.active_delay_ms = active_delay_ms
        self.idle_timeout_ms = idle_timeout_ms

        self.mouse_in_window = False
        self.mouse_active_start_time = None
        self.last_mouse_movement_time = 0.0
        self.is_animating = False
        self.has_met_active_delay = False

        self.animation_timer_id = None
        self.inactivity_timer_id = None
        self.last_frame_time = 0.0

        self.pending_wallpaper_path = None
        self.pending_depth_path = None

        self.set_auto_render(True)
        self.set_has_depth_buffer(False)
        self.set_has_stencil_buffer(False)
        self.set_allowed_apis(Gdk.GLAPI.GL | Gdk.GLAPI.GLES)

        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("map", self.on_map)

    def on_map(self, widget):
        logger.debug("Renderer mapped, triggering initial render")
        self.queue_render()

    def calculate_zoom_level(self):
        max_shift = max(self.parallax_strength_x, self.parallax_strength_y)
        parallax_margin = 1.0 + max_shift * 2.5
        return parallax_margin

    def on_realize(self, area):
        self.make_current()

        try:
            version = gl.glGetString(gl.GL_VERSION).decode("utf-8")
            renderer = gl.glGetString(gl.GL_RENDERER).decode("utf-8")
            logger.info(f"OpenGL version: {version}, renderer: {renderer}")
        except Exception as error:
            logger.warning(f"Could not query OpenGL info: {error}")

        self.compile_shader_program()
        self.create_fullscreen_quad()

        if self.pending_wallpaper_path and self.pending_depth_path:
            self.load_wallpaper_files(
                self.pending_wallpaper_path, self.pending_depth_path
            )
            self.pending_wallpaper_path = None
            self.pending_depth_path = None

        self.queue_render()

    def compile_shader_program(self):
        vertex_shader = compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)
        self.shader_program = compileProgram(vertex_shader, fragment_shader)
        logger.info("Shader program compiled successfully")

    def create_fullscreen_quad(self):
        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        self.vertex_array_object = gl.glGenVertexArrays(1)
        self.vertex_buffer_object = gl.glGenBuffers(1)
        self.element_buffer_object = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vertex_array_object)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer_object)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW
        )

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.element_buffer_object)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW
        )

        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, None)
        gl.glEnableVertexAttribArray(0)

        gl.glVertexAttribPointer(
            1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.GLvoidp(2 * 4)
        )
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

    def on_unrealize(self, area):
        self.stop_all_timers()
        self.make_current()

        if self.textures.wallpaper:
            gl.glDeleteTextures([self.textures.wallpaper])
        if self.textures.depth:
            gl.glDeleteTextures([self.textures.depth])
        if self.shader_program:
            gl.glDeleteProgram(self.shader_program)
        if self.vertex_array_object:
            gl.glDeleteVertexArrays(1, [self.vertex_array_object])
        if self.vertex_buffer_object:
            gl.glDeleteBuffers(1, [self.vertex_buffer_object])
        if self.element_buffer_object:
            gl.glDeleteBuffers(1, [self.element_buffer_object])

    def start_animation_timer(self):
        if self.animation_timer_id is None:
            self.animation_timer_id = GLib.timeout_add(
                self.frame_interval_ms, self.on_animation_frame
            )
            self.is_animating = True
            logger.debug(f"Animation timer started ({self.fps} FPS)")

    def stop_animation_timer(self):
        if self.animation_timer_id is not None:
            GLib.source_remove(self.animation_timer_id)
            self.animation_timer_id = None
            self.is_animating = False
            logger.debug("Animation timer stopped")

    def start_inactivity_timer(self):
        if self.inactivity_timer_id is None:
            self.inactivity_timer_id = GLib.timeout_add(50, self.check_inactivity)

    def stop_inactivity_timer(self):
        if self.inactivity_timer_id is not None:
            GLib.source_remove(self.inactivity_timer_id)
            self.inactivity_timer_id = None

    def stop_all_timers(self):
        self.stop_animation_timer()
        self.stop_inactivity_timer()

    def check_inactivity(self):
        current_time = GLib.get_monotonic_time() / 1000.0

        if not self.mouse_in_window:
            return True

        time_since_movement = current_time - self.last_mouse_movement_time

        if time_since_movement > self.idle_timeout_ms:
            if self.is_animating:
                logger.debug("Mouse idle, stopping animation")
                self.stop_animation_timer()
                self.has_met_active_delay = False
        else:
            if not self.is_animating:
                if self.has_met_active_delay:
                    logger.debug("Mouse active again, restarting animation")
                    self.start_animation_timer()
                elif self.mouse_active_start_time is not None:
                    active_duration = current_time - self.mouse_active_start_time
                    if active_duration >= self.active_delay_ms:
                        self.has_met_active_delay = True
                        logger.debug("Active delay met, starting animation")
                        self.start_animation_timer()

        return True

    def on_animation_frame(self):
        position_changed = self.update_smooth_position()

        if position_changed:
            self.queue_render()

        if not self.mouse_in_window:
            dx = 0.5 - self.current_mouse_position[0]
            dy = 0.5 - self.current_mouse_position[1]
            distance = max(abs(dx), abs(dy))

            if distance < 0.0001:
                self.stop_animation_timer()
                return False

        return True

    def update_smooth_position(self):
        if not self.smooth_animation:
            if self.current_mouse_position != self.mouse_position:
                self.current_mouse_position = self.mouse_position
                return True
            return False

        current_x, current_y = self.current_mouse_position
        target_x, target_y = self.mouse_position

        dx = target_x - current_x
        dy = target_y - current_y
        distance = max(abs(dx), abs(dy))

        if distance < 0.0001:
            if self.current_mouse_position != self.mouse_position:
                self.current_mouse_position = self.mouse_position
                return True
            return False

        lerp_factor = 0.02 + (self.animation_speed * 0.28)
        new_x = current_x + dx * lerp_factor
        new_y = current_y + dy * lerp_factor

        self.current_mouse_position = (new_x, new_y)
        return True

    def on_render(self, area, context):
        if self.shader_program is None or self.textures.wallpaper is None:
            return False

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(self.shader_program)

        screen_width = self.get_allocated_width()
        screen_height = self.get_allocated_height()

        if not hasattr(self, "render_count"):
            self.render_count = 0
        self.render_count += 1
        if self.render_count == 1:
            logger.info(
                f"Rendering to {screen_width}x{screen_height} with image {self.image_dimensions}"
            )

        zoom_level = self.calculate_zoom_level()

        self.bind_textures()
        self.set_shader_uniforms(screen_width, screen_height, zoom_level)

        gl.glBindVertexArray(self.vertex_array_object)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        return True

    def bind_textures(self):
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.wallpaper)
        wallpaper_location = gl.glGetUniformLocation(
            self.shader_program, "wallpaper_texture"
        )
        gl.glUniform1i(wallpaper_location, 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.depth)
        depth_location = gl.glGetUniformLocation(self.shader_program, "depth_texture")
        gl.glUniform1i(depth_location, 1)

    def set_shader_uniforms(self, screen_width, screen_height, zoom_level):
        gl.glUniform2f(
            gl.glGetUniformLocation(self.shader_program, "mouse_position"),
            self.current_mouse_position[0],
            self.current_mouse_position[1],
        )
        gl.glUniform2f(
            gl.glGetUniformLocation(self.shader_program, "screen_resolution"),
            float(screen_width),
            float(screen_height),
        )
        gl.glUniform2f(
            gl.glGetUniformLocation(self.shader_program, "parallax_strength"),
            self.parallax_strength_x,
            self.parallax_strength_y,
        )
        gl.glUniform1f(
            gl.glGetUniformLocation(self.shader_program, "zoom_level"), zoom_level
        )
        gl.glUniform2f(
            gl.glGetUniformLocation(self.shader_program, "image_dimensions"),
            float(self.image_dimensions[0]),
            float(self.image_dimensions[1]),
        )
        gl.glUniform1i(
            gl.glGetUniformLocation(self.shader_program, "inverse_depth"),
            1 if self.inverse_depth else 0,
        )

    def set_wallpaper(self, wallpaper_path, depth_path):
        if not self.get_realized():
            self.pending_wallpaper_path = wallpaper_path
            self.pending_depth_path = depth_path
            return

        self.load_wallpaper_files(wallpaper_path, depth_path)

    def load_wallpaper_files(self, wallpaper_path, depth_path):
        self.make_current()

        logger.info(f"Loading wallpaper: {wallpaper_path}")

        image = Image.open(wallpaper_path).convert("RGBA")
        self.image_dimensions = image.size
        image_data = np.flipud(np.array(image))

        depth_map = load_depth_map(depth_path)
        depth_data = np.flipud((depth_map * 255).astype(np.uint8))

        self.textures.wallpaper = self.create_texture(image_data, gl.GL_RGBA)
        self.textures.depth = self.create_texture(depth_data, gl.GL_RED)

        self.queue_render()
        logger.info(f"Wallpaper loaded successfully: {wallpaper_path}")

    def create_texture(self, data, format_type):
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        height, width = data.shape[:2]
        if format_type == gl.GL_RGBA:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA,
                width,
                height,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                data,
            )
        else:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RED,
                width,
                height,
                0,
                gl.GL_RED,
                gl.GL_UNSIGNED_BYTE,
                data,
            )

        return texture_id

    def update_mouse_position(self, normalized_x, normalized_y):
        current_time = GLib.get_monotonic_time() / 1000.0

        self.mouse_position = (normalized_x, normalized_y)
        self.last_mouse_movement_time = current_time

        self.start_inactivity_timer()

    def on_mouse_enter(self):
        current_time = GLib.get_monotonic_time() / 1000.0
        self.mouse_in_window = True
        self.mouse_active_start_time = current_time
        self.last_mouse_movement_time = current_time
        logger.debug("Mouse entered window, active timer started")

    def on_mouse_leave(self):
        self.mouse_in_window = False
        self.mouse_active_start_time = None
        self.has_met_active_delay = False

        self.mouse_position = (0.5, 0.5)

        if not self.is_animating:
            self.start_animation_timer()

        logger.debug("Mouse left window, returning to center")

    def set_parallax_strength(self, strength_x, strength_y):
        logger.debug(f"Setting parallax strength: X={strength_x}, Y={strength_y}")
        self.parallax_strength_x = strength_x
        self.parallax_strength_y = strength_y
        self.queue_render()

    def set_inverse_depth(self, inverse_depth):
        self.inverse_depth = inverse_depth
        self.queue_render()


class WallpaperWindow(Gtk.ApplicationWindow):
    def __init__(
        self,
        application,
        monitor=0,
        smooth_animation=True,
        animation_speed=0.02,
        fps=60,
        active_delay_ms=150.0,
        idle_timeout_ms=500.0,
        inverse_depth=False,
    ):
        super().__init__(application=application)

        self.monitor = str(monitor)
        self.renderer = None
        self.smooth_animation = smooth_animation
        self.animation_speed = animation_speed
        self.fps = fps
        self.active_delay_ms = active_delay_ms
        self.idle_timeout_ms = idle_timeout_ms
        self.inverse_depth = inverse_depth

        self.setup_layer_shell_window()

    def setup_layer_shell_window(self):
        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.BACKGROUND)

        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.LEFT, True)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.RIGHT, True)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.TOP, True)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.BOTTOM, True)

        Gtk4LayerShell.set_exclusive_zone(self, -1)
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)

        display = Gdk.Display.get_default()
        if display:
            monitors = display.get_monitors()
            target_monitor = None

            for index in range(monitors.get_n_items()):
                monitor = monitors.get_item(index)
                if monitor:
                    monitor_name = None
                    if hasattr(monitor, "get_model"):
                        monitor_name = monitor.get_model()
                    if hasattr(monitor, "get_connector"):
                        monitor_name = monitor.get_connector()

                    if monitor_name == self.monitor or str(index) == self.monitor:
                        target_monitor = monitor
                        break

            if target_monitor is None:
                try:
                    idx = int(self.monitor)
                    if idx < monitors.get_n_items():
                        target_monitor = monitors.get_item(idx)
                except ValueError:
                    pass

            if target_monitor is None and monitors.get_n_items() > 0:
                target_monitor = monitors.get_item(0)
                logger.warning(f"Monitor '{self.monitor}' not found, using monitor 0")

            if target_monitor:
                Gtk4LayerShell.set_monitor(self, target_monitor)
                geometry = target_monitor.get_geometry()
                scale = target_monitor.get_scale_factor()
                self.set_default_size(geometry.width, geometry.height)
                logger.info(
                    f"Using monitor {self.monitor}: {geometry.width}x{geometry.height} (scale: {scale})"
                )

        self.renderer = WallpaperRenderer(
            smooth_animation=self.smooth_animation,
            animation_speed=self.animation_speed,
            fps=self.fps,
            active_delay_ms=self.active_delay_ms,
            idle_timeout_ms=self.idle_timeout_ms,
            inverse_depth=self.inverse_depth,
        )
        self.renderer.set_vexpand(True)
        self.renderer.set_hexpand(True)
        self.set_child(self.renderer)

        self.setup_mouse_tracking()

    def setup_mouse_tracking(self):
        motion_controller = Gtk.EventControllerMotion.new()
        motion_controller.connect("motion", self.on_mouse_moved)
        motion_controller.connect("enter", self.on_mouse_enter)
        motion_controller.connect("leave", self.on_mouse_leave)
        self.add_controller(motion_controller)
        self.set_can_target(False)

    def on_mouse_enter(self, controller, x_position, y_position):
        if self.renderer:
            self.renderer.on_mouse_enter()

    def on_mouse_leave(self, controller):
        if self.renderer:
            self.renderer.on_mouse_leave()

    def on_mouse_moved(self, controller, x_position, y_position):
        if self.renderer:
            window_width = self.get_width()
            window_height = self.get_height()
            if window_width > 0 and window_height > 0:
                normalized_x = x_position / window_width
                normalized_y = 1.0 - (y_position / window_height)
                self.renderer.update_mouse_position(normalized_x, normalized_y)

    def set_wallpaper(self, wallpaper_path, depth_path):
        if self.renderer:
            self.renderer.set_wallpaper(wallpaper_path, depth_path)

    def set_parallax_strength(self, strength_x, strength_y):
        if self.renderer:
            self.renderer.set_parallax_strength(strength_x, strength_y)

    def set_inverse_depth(self, inverse_depth):
        if self.renderer:
            self.renderer.set_inverse_depth(inverse_depth)


class WallpaperApplication(Gtk.Application):
    def __init__(
        self,
        wallpaper_path,
        depth_path,
        monitor=0,
        strength_x=0.02,
        strength_y=0.02,
        smooth_animation=True,
        animation_speed=0.02,
        fps=60,
        active_delay_ms=150.0,
        idle_timeout_ms=500.0,
        inverse_depth=False,
        ready_callback=None,
    ):
        monitor_id = str(monitor).replace("-", "_").replace(".", "_")
        super().__init__(
            application_id=f"com.waydeeper.daemon.{monitor_id}",
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )

        self.wallpaper_path = wallpaper_path
        self.depth_path = depth_path
        self.monitor = monitor
        self.strength_x = strength_x
        self.strength_y = strength_y
        self.smooth_animation = smooth_animation
        self.animation_speed = animation_speed
        self.fps = fps
        self.active_delay_ms = active_delay_ms
        self.idle_timeout_ms = idle_timeout_ms
        self.inverse_depth = inverse_depth
        self.ready_callback = ready_callback
        self.window = None
        self._ready_called = False

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.quit()

    def do_activate(self):
        if self.window is not None:
            self.window.present()
            return

        logger.info(
            f"Activating wallpaper: strength_x={self.strength_x}, strength_y={self.strength_y}, fps={self.fps}"
        )

        self.window = WallpaperWindow(
            self,
            self.monitor,
            self.smooth_animation,
            self.animation_speed,
            self.fps,
            self.active_delay_ms,
            self.idle_timeout_ms,
            self.inverse_depth,
        )
        self.window.set_parallax_strength(self.strength_x, self.strength_y)
        self.window.set_wallpaper(self.wallpaper_path, self.depth_path)
        self.window.present()

        # Signal that we're ready (window is shown)
        if self.ready_callback and not self._ready_called:
            self._ready_called = True
            self.ready_callback()

        self.hold()
        GLib.timeout_add(1000, lambda: True)


def run_wallpaper(
    wallpaper_path,
    depth_path,
    monitor=0,
    strength_x=0.02,
    strength_y=0.02,
    smooth_animation=True,
    animation_speed=0.02,
    fps=60,
    active_delay_ms=150.0,
    idle_timeout_ms=500.0,
    inverse_depth=False,
    ready_callback=None,
):
    logger.info(
        f"Starting wallpaper on monitor {monitor}: strength_x={strength_x}, strength_y={strength_y}, fps={fps}, inverse_depth={inverse_depth}"
    )

    app = WallpaperApplication(
        wallpaper_path,
        depth_path,
        monitor,
        strength_x,
        strength_y,
        smooth_animation,
        animation_speed,
        fps,
        active_delay_ms,
        idle_timeout_ms,
        inverse_depth,
        ready_callback,
    )
    return app.run([])
