use anyhow::{anyhow, Result};
use glow::HasContext;
use std::ffi::{c_char, c_int, c_void, CString};

use crate::math;
use crate::mesh::Mesh;

pub struct RendererConfig {
    pub wallpaper_path: String,
    pub depth_path: String,
    pub monitor: String,
    pub strength_x: f64,
    pub strength_y: f64,
    pub smooth_animation: bool,
    pub animation_speed: f64,
    pub fps: u32,
    pub active_delay_ms: f64,
    pub idle_timeout_ms: f64,
    pub invert_depth: bool,
    pub ply_path: Option<String>,
}

pub struct MouseState {
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub current_x: f64,
    pub current_y: f64,
    pub mouse_in_window: bool,
    pub last_mouse_time: std::time::Instant,
    pub mouse_active_start: Option<std::time::Instant>,
    pub is_animating: bool,
    pub has_met_delay: bool,
    pub surface_width: f64,
    pub surface_height: f64,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            mouse_x: 0.5,
            mouse_y: 0.5,
            current_x: 0.5,
            current_y: 0.5,
            mouse_in_window: false,
            last_mouse_time: std::time::Instant::now(),
            mouse_active_start: None,
            is_animating: false,
            has_met_delay: false,
            surface_width: 1.0,
            surface_height: 1.0,
        }
    }
}

#[repr(C)]
struct EglContext {
    display: *mut c_void,
    context: *mut c_void,
    config: *mut c_void,
}

extern "C" {
    fn egl_init_from_wl_display(wl_display: *mut c_void, out: *mut EglContext) -> c_int;
    fn egl_create_surface(
        ctx: *mut EglContext,
        wl_surface: *mut c_void,
        w: c_int,
        h: c_int,
    ) -> *mut c_void;
    fn egl_swap_buffers(ctx: *const EglContext, surface: *mut c_void) -> c_int;
    fn egl_get_proc_address(name: *const c_char) -> *mut c_void;
    fn egl_destroy_surface(ctx: *mut EglContext, surface: *mut c_void);
    fn egl_destroy_ctx(ctx: *mut EglContext);
}

fn gl_error(message: String) -> anyhow::Error {
    anyhow!(message)
}

// ---------------------------------------------------------------------------
// Renderer (handles both flat-depth and mesh modes)
// ---------------------------------------------------------------------------

pub struct EglRenderer {
    egl_context: EglContext,
    egl_surface: *mut c_void,
    gl_context: Option<glow::Context>,

    // --- Flat depth-warp mode ---
    flat_shader: Option<glow::Program>,
    flat_vao: Option<glow::VertexArray>,
    flat_vbo: Option<glow::Buffer>,
    flat_ebo: Option<glow::Buffer>,
    wallpaper_texture: Option<glow::Texture>,
    depth_texture: Option<glow::Texture>,
    image_width: u32,
    image_height: u32,

    // --- Mesh mode ---
    mesh_shader: Option<glow::Program>,
    mesh_vao: Option<glow::VertexArray>,
    mesh_pos_vbo: Option<glow::Buffer>,
    mesh_col_vbo: Option<glow::Buffer>,
    mesh_uv_vbo: Option<glow::Buffer>,
    mesh_ebo: Option<glow::Buffer>,
    mesh_index_count: u32,
    /// Whether the loaded PLY has per-vertex UVs (new format).
    mesh_has_uvs: bool,
    /// Camera Z offset (always 0 — mesh is already in camera space).
    mesh_camera_z: f32,
    /// Absolute Z of the nearest mesh face.  Used to bound parallax travel so
    /// near-field pixels never shift more than a small fraction of screen width.
    mesh_near_z: f32,
    /// Mesh XY half-extent (kept for logging; no longer used for travel).
    mesh_xy_half: f32,
    /// Intrinsic vertical FoV baked into the mesh (read from PLY comment).
    mesh_fov_y_deg: f32,
    /// Source image W/H aspect ratio (read from PLY comment).
    /// Used to compute the cover FoV when screen aspect ≠ image aspect.
    mesh_image_aspect: f32,

    screen_width: u32,
    screen_height: u32,
    pub config: RendererConfig,
    pub mouse: MouseState,
}

impl EglRenderer {
    pub fn new(wayland_display: *mut c_void, config: RendererConfig) -> Result<Self> {
        let mut egl_context = EglContext {
            display: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
            config: std::ptr::null_mut(),
        };
        if unsafe { egl_init_from_wl_display(wayland_display, &mut egl_context) } != 0 {
            return Err(anyhow!("EGL init failed"));
        }
        Ok(Self {
            egl_context,
            egl_surface: std::ptr::null_mut(),
            gl_context: None,

            flat_shader: None,
            flat_vao: None,
            flat_vbo: None,
            flat_ebo: None,
            wallpaper_texture: None,
            depth_texture: None,
            image_width: 1,
            image_height: 1,

            mesh_shader: None,
            mesh_vao: None,
            mesh_pos_vbo: None,
            mesh_col_vbo: None,
            mesh_ebo: None,
            mesh_index_count: 0,
            mesh_camera_z: 0.0,
            mesh_near_z: 1.0,
            mesh_xy_half: 1.0,
            mesh_fov_y_deg: 60.0,
            mesh_image_aspect: 1.0,
            mesh_uv_vbo: None,
            mesh_has_uvs: false,

            screen_width: 1,
            screen_height: 1,
            config,
            mouse: MouseState::default(),
        })
    }

    pub fn create_surface(
        &mut self,
        wayland_surface: *mut c_void,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let surface = unsafe {
            egl_create_surface(
                &mut self.egl_context,
                wayland_surface,
                width as i32,
                height as i32,
            )
        };
        if surface.is_null() {
            return Err(anyhow!("EGL surface creation failed"));
        }
        self.egl_surface = surface;
        self.screen_width = width.max(1);
        self.screen_height = height.max(1);

        let gl = unsafe {
            glow::Context::from_loader_function(|name| {
                let c_string = CString::new(name).unwrap();
                egl_get_proc_address(c_string.as_ptr())
            })
        };
        self.gl_context = Some(gl);
        log::info!("EGL surface created {}x{}", width, height);

        if self.config.ply_path.is_some() {
            // Initialise both mesh AND flat resources.
            // The flat quad is drawn first as a background layer so that holes
            // left by back-face culling show the original image rather than black.
            self.init_gl_flat()?;
            self.init_gl_mesh()
        } else {
            self.init_gl_flat()
        }
    }

    // -----------------------------------------------------------------------
    // Flat depth-warp mode init
    // -----------------------------------------------------------------------

    fn init_gl_flat(&mut self) -> Result<()> {
        let gl = self.gl_context.as_ref().unwrap();
        unsafe {
            let program = compile_program(gl, FLAT_VERT, FLAT_FRAG)?;
            self.flat_shader = Some(program);

            #[rustfmt::skip]
            let vertices: [f32; 16] = [
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                 1.0,  1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 1.0,
            ];
            let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

            let vao = gl.create_vertex_array().map_err(gl_error)?;
            gl.bind_vertex_array(Some(vao));

            let vbo = gl.create_buffer().map_err(gl_error)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&vertices),
                glow::STATIC_DRAW,
            );

            let ebo = gl.create_buffer().map_err(gl_error)?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&indices),
                glow::STATIC_DRAW,
            );

            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);
            gl.enable_vertex_attrib_array(1);
            gl.bind_vertex_array(None);

            self.flat_vao = Some(vao);
            self.flat_vbo = Some(vbo);
            self.flat_ebo = Some(ebo);
        }
        log::info!("Flat GL resources initialized");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Mesh mode init
    // -----------------------------------------------------------------------

    fn init_gl_mesh(&mut self) -> Result<()> {
        let gl = self.gl_context.as_ref().unwrap();
        unsafe {
            let program = compile_program(gl, MESH_VERT, MESH_FRAG)?;
            self.mesh_shader = Some(program);

            // Depth test for correct occlusion between layers.
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LESS);

            // Back-face culling: discard triangles whose winding order (and therefore
            // normal) faces away from the camera.  At depth discontinuities the mesh
            // curves and some triangles face sideways or backward — culling them
            // eliminates the "silk" ripple artefact caused by back-facing polygons
            // rendering on top of front-facing ones.
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::BACK);
            gl.front_face(glow::CCW);
        }
        log::info!("Mesh GL resources initialized");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Load textures (flat mode)
    // -----------------------------------------------------------------------

    pub fn load_textures(&mut self) -> Result<()> {
        let gl = self.gl_context.as_ref().ok_or_else(|| anyhow!("no GL"))?;
        let wallpaper_raw = image::open(&self.config.wallpaper_path)?.to_rgba8();
        self.image_width = wallpaper_raw.width();
        self.image_height = wallpaper_raw.height();
        let depth_raw = image::open(&self.config.depth_path)?.to_luma8();

        let mut wallpaper = wallpaper_raw;
        image::imageops::flip_vertical_in_place(&mut wallpaper);
        let mut depth = depth_raw;
        image::imageops::flip_vertical_in_place(&mut depth);

        log::info!(
            "Textures: wallpaper {}x{}, depth {}x{}",
            self.image_width,
            self.image_height,
            depth.width(),
            depth.height()
        );

        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

            let wallpaper_tex = gl.create_texture().map_err(gl_error)?;
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(wallpaper_tex));
            set_texture_params(gl);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                self.image_width as i32,
                self.image_height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                Some(&wallpaper),
            );
            gl.generate_mipmap(glow::TEXTURE_2D);
            self.wallpaper_texture = Some(wallpaper_tex);

            let depth_tex = gl.create_texture().map_err(gl_error)?;
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(depth_tex));
            set_texture_params(gl);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R8 as i32,
                depth.width() as i32,
                depth.height() as i32,
                0,
                glow::RED,
                glow::UNSIGNED_BYTE,
                Some(depth.as_raw()),
            );
            gl.generate_mipmap(glow::TEXTURE_2D);
            self.depth_texture = Some(depth_tex);
        }
        log::info!("Textures loaded");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Load PLY mesh (mesh mode)
    // -----------------------------------------------------------------------

    pub fn load_mesh(&mut self, ply_path: &str) -> Result<()> {
        let gl = self.gl_context.as_ref().ok_or_else(|| anyhow!("no GL"))?;

        let mesh = Mesh::load_ply(std::path::Path::new(ply_path))?;

        // With the OpenGL convention from reproject (Z negated), the mesh sits at
        // negative Z.  aabb[5] = max_z is the nearest face (least negative, ~0).
        // aabb[4] = min_z is the farthest face (most negative).
        // Place the camera slightly in front of the nearest face so the full
        // depth range is visible.
        // The mesh is already in OpenGL camera space (origin = camera, -Z = scene).
        // Camera sits at Z=0; no tz offset needed.
        self.mesh_camera_z   = 0.0;
        // aabb[5] = max_z = nearest face (least negative Z, e.g. ~-1.0).
        // Store its absolute value as the reference for travel scaling.
        self.mesh_near_z     = mesh.aabb[5].abs().max(0.1);
        self.mesh_xy_half    = mesh.xy_half_extent().max(0.001);
        // Read the intrinsic FoV and image aspect baked into the mesh.
        self.mesh_fov_y_deg    = mesh.fov_y_deg;
        self.mesh_image_aspect = mesh.image_aspect.max(0.01);
        self.mesh_index_count  = mesh.indices.len() as u32;
        self.mesh_has_uvs      = mesh.has_uvs;

        log::info!(
            "Mesh: {} verts, near_z={:.3}, xy_half={:.3}, depth_range={:.3}, fov_y={:.2}°, image_aspect={:.4}",
            mesh.vertex_count, self.mesh_near_z, self.mesh_xy_half,
            mesh.depth_range(), self.mesh_fov_y_deg, self.mesh_image_aspect
        );

        unsafe {
            let vao = gl.create_vertex_array().map_err(gl_error)?;
            gl.bind_vertex_array(Some(vao));

            // Position VBO  (vec3 f32)
            let pos_vbo = gl.create_buffer().map_err(gl_error)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(pos_vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&mesh.positions),
                glow::STATIC_DRAW,
            );
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.enable_vertex_attrib_array(0);

            // Color VBO  (vec4 u8 normalised)  — attrib location 1
            let col_vbo = gl.create_buffer().map_err(gl_error)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(col_vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                &mesh.colors,
                glow::STATIC_DRAW,
            );
            gl.vertex_attrib_pointer_f32(1, 4, glow::UNSIGNED_BYTE, true, 4, 0);
            gl.enable_vertex_attrib_array(1);

            // UV VBO  (vec2 f32)  — attrib location 2 (only when UVs are present)
            let uv_vbo = if mesh.has_uvs && !mesh.uvs.is_empty() {
                let vbo = gl.create_buffer().map_err(gl_error)?;
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&mesh.uvs),
                    glow::STATIC_DRAW,
                );
                gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 8, 0);
                gl.enable_vertex_attrib_array(2);
                Some(vbo)
            } else {
                None
            };

            // Index EBO
            let ebo = gl.create_buffer().map_err(gl_error)?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&mesh.indices),
                glow::STATIC_DRAW,
            );

            gl.bind_vertex_array(None);

            self.mesh_vao     = Some(vao);
            self.mesh_pos_vbo = Some(pos_vbo);
            self.mesh_col_vbo = Some(col_vbo);
            self.mesh_uv_vbo  = uv_vbo;
            self.mesh_ebo     = Some(ebo);
        }

        log::info!(
            "Mesh uploaded: {} vertices, {} triangles",
            mesh.vertex_count,
            mesh.triangle_count
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Mouse update (shared)
    // -----------------------------------------------------------------------

    pub fn update_mouse(&mut self) {
        let now = std::time::Instant::now();
        let mouse = &mut self.mouse;

        if mouse.mouse_in_window {
            let idle_ms = now.duration_since(mouse.last_mouse_time).as_millis() as f64;

            if idle_ms > self.config.idle_timeout_ms {
                if mouse.is_animating {
                    mouse.is_animating = false;
                    mouse.has_met_delay = false;
                }
                mouse.mouse_active_start = None;
            } else {
                if mouse.mouse_active_start.is_none() {
                    mouse.mouse_active_start = Some(now);
                }
                if !mouse.is_animating {
                    if let Some(start) = mouse.mouse_active_start {
                        let active_ms = now.duration_since(start).as_millis() as f64;
                        if active_ms >= self.config.active_delay_ms {
                            mouse.has_met_delay = true;
                            mouse.is_animating = true;
                        }
                    }
                }
            }
        }

        if mouse.is_animating {
            let lerp = if self.config.smooth_animation {
                0.02 + self.config.animation_speed * 0.28
            } else {
                1.0
            };
            mouse.current_x += (mouse.mouse_x - mouse.current_x) * lerp;
            mouse.current_y += (mouse.mouse_y - mouse.current_y) * lerp;
        }

        if !mouse.mouse_in_window {
            let lerp = 0.05;
            mouse.current_x += (0.5 - mouse.current_x) * lerp;
            mouse.current_y += (0.5 - mouse.current_y) * lerp;
            mouse.is_animating = false;
            mouse.has_met_delay = false;
            mouse.mouse_active_start = None;
        }
    }

    // -----------------------------------------------------------------------
    // Draw dispatch
    // -----------------------------------------------------------------------

    pub fn draw(&self) -> Result<()> {
        if self.config.ply_path.is_some() {
            self.draw_mesh()
        } else {
            self.draw_flat()
        }
    }

    // -----------------------------------------------------------------------
    // Flat depth-warp draw
    // -----------------------------------------------------------------------

    fn draw_flat(&self) -> Result<()> {
        let gl = match &self.gl_context { Some(c) => c, None => return Ok(()) };
        let program = match self.flat_shader { Some(p) => p, None => return Ok(()) };
        let wallpaper_tex = match self.wallpaper_texture { Some(t) => t, None => return Ok(()) };
        let depth_tex = match self.depth_texture { Some(t) => t, None => return Ok(()) };

        unsafe {
            gl.viewport(0, 0, self.screen_width as i32, self.screen_height as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(program));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(wallpaper_tex));
            gl.uniform_1_i32(gl.get_uniform_location(program, "wallpaper_texture").as_ref(), 0);

            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(depth_tex));
            gl.uniform_1_i32(gl.get_uniform_location(program, "depth_texture").as_ref(), 1);

            gl.uniform_2_f32(
                gl.get_uniform_location(program, "mouse_position").as_ref(),
                self.mouse.current_x as f32,
                self.mouse.current_y as f32,
            );
            gl.uniform_2_f32(
                gl.get_uniform_location(program, "screen_resolution").as_ref(),
                self.screen_width as f32,
                self.screen_height as f32,
            );
            gl.uniform_2_f32(
                gl.get_uniform_location(program, "parallax_strength").as_ref(),
                self.config.strength_x as f32,
                self.config.strength_y as f32,
            );
            let zoom = (1.0 + self.config.strength_x.max(self.config.strength_y) * 2.0) as f32;
            gl.uniform_1_f32(gl.get_uniform_location(program, "zoom_level").as_ref(), zoom);
            gl.uniform_2_f32(
                gl.get_uniform_location(program, "image_dimensions").as_ref(),
                self.image_width as f32,
                self.image_height as f32,
            );
            gl.uniform_1_i32(
                gl.get_uniform_location(program, "invert_depth").as_ref(),
                if self.config.invert_depth { 1 } else { 0 },
            );
            // No mesh parallax in flat-only mode
            gl.uniform_2_f32(
                gl.get_uniform_location(program, "mesh_parallax_travel").as_ref(),
                0.0, 0.0,
            );

            gl.bind_vertex_array(self.flat_vao);
            gl.draw_elements(glow::TRIANGLES, 6, glow::UNSIGNED_INT, 0);
            gl.bind_vertex_array(None);
        }
        unsafe { egl_swap_buffers(&self.egl_context, self.egl_surface); }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Mesh perspective draw
    // -----------------------------------------------------------------------

    fn draw_mesh(&self) -> Result<()> {
        let gl = match &self.gl_context { Some(c) => c, None => return Ok(()) };
        let program = match self.mesh_shader { Some(p) => p, None => return Ok(()) };
        let vao = match self.mesh_vao { Some(v) => v, None => return Ok(()) };

        // Camera parallax travel.
        //
        // Base travel is set so that the nearest pixel shifts at most ~3% of the
        // screen half-width at strength=0.02:
        //   shift_near = travel × 2.0 / near_z ≤ 0.03
        //   travel     = near_z × 0.015
        //
        // Strength scales travel linearly (default 0.02 → factor 1.0).
        // Separate strength_x and strength_y allow independent horizontal/vertical parallax.
        let travel_x = self.mesh_near_z * 0.015 * (self.config.strength_x as f32 / 0.02).max(0.0);
        let travel_y = self.mesh_near_z * 0.015 * (self.config.strength_y as f32 / 0.02).max(0.0);

        // Mouse [0,1] → offset [-0.5, +0.5] → world camera shift.
        // Both axes negated: camera moves opposite to mouse so the scene
        // (geometry at -Z) appears to follow the mouse, matching flat mode.
        let tx = -(self.mouse.current_x as f32 - 0.5) * 2.0 * travel_x;
        let ty = -(self.mouse.current_y as f32 - 0.5) * 2.0 * travel_y;
        let tz = self.mesh_camera_z; // 0.0 — camera is at the origin

        // Compute a "cover" FoV so the mesh fills the entire screen regardless of
        // whether the source image is portrait on a landscape screen (or vice-versa).
        //
        // At depth near_z the mesh spans:
        //   y_half = tan(fov_y/2) * near_z   (fits exactly by construction)
        //   x_half = image_aspect * y_half
        //
        // The screen half-extents at depth near_z with FoV fov_y are:
        //   y_screen = tan(fov_y/2) * near_z  (= y_half)
        //   x_screen = aspect * y_screen
        //
        // For cover we need x_half ≥ x_screen AND y_half ≥ y_screen simultaneously.
        // If image_aspect < screen_aspect (portrait image on landscape screen),
        // x_half < x_screen → black bars on sides.  Reduce fov_y until x fills:
        //   fov_y_for_x: the fov_y that makes x_half = x_screen
        //     tan(fov_y_for_x/2) = x_half / (near_z * aspect) = image_aspect * tan(fov_y/2) / aspect
        //   fov_y_cover = min(fov_y_intrinsic, fov_y_for_x)   ← zoom in enough to cover
        let screen_aspect = self.screen_width as f32 / self.screen_height as f32;
        let fov_y_intrinsic = self.mesh_fov_y_deg.to_radians();
        // Horizontal half-tan at the intrinsic FoV
        let x_half_tan = self.mesh_image_aspect * (fov_y_intrinsic * 0.5).tan();
        // The fov_y required so x_half fills screen width
        let fov_y_for_x = 2.0 * (x_half_tan / screen_aspect).atan();
        // Cover = smallest fov_y (= most zoomed in, fills both axes)
        let fov_y_cover = fov_y_intrinsic.min(fov_y_for_x);

        // Reduce FoV based on strength to prevent seeing past mesh edges.
        // Same formula as flat mode zoom: divide by (1 + strength * 2).
        let strength_max = self.config.strength_x.max(self.config.strength_y) as f32;
        let fov_rad = fov_y_cover / (1.0 + strength_max * 2.0);

        let aspect  = screen_aspect;
        // near: just inside the nearest vertex (near_z ≈ 1.0 → near = 0.5).
        // far: past the deepest border vertex (border max depth ≈ 5.0 × 1.3 = 6.5 → far = 50).
        let near = (self.mesh_near_z * 0.5).min(0.5);
        let far  = 50.0_f32;

        let proj = math::perspective(fov_rad, aspect, near, far);
        let view = math::translation(tx, ty, tz);

        unsafe {
            gl.viewport(0, 0, self.screen_width as i32, self.screen_height as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

            // --- Pass 1: flat background quad (fills the whole screen behind mesh) ---
            // Drawn with depth test disabled and depth writes disabled so it sits
            // behind everything.  Any hole left by the mesh shows this background.
            // Uses the same parallax as the mesh pass so the transition is smooth.
            if let (Some(flat_prog), Some(flat_vao), Some(wt), Some(dt)) = (
                self.flat_shader,
                self.flat_vao,
                self.wallpaper_texture,
                self.depth_texture,
            ) {
                gl.disable(glow::DEPTH_TEST);
                gl.depth_mask(false);
                gl.use_program(Some(flat_prog));

                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(wt));
                gl.uniform_1_i32(gl.get_uniform_location(flat_prog, "wallpaper_texture").as_ref(), 0);
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(dt));
                gl.uniform_1_i32(gl.get_uniform_location(flat_prog, "depth_texture").as_ref(), 1);

                // Use actual mouse position so the flat background shifts with parallax
                gl.uniform_2_f32(gl.get_uniform_location(flat_prog, "mouse_position").as_ref(),
                    self.mouse.current_x as f32, self.mouse.current_y as f32);
                gl.uniform_2_f32(gl.get_uniform_location(flat_prog, "screen_resolution").as_ref(),
                    self.screen_width as f32, self.screen_height as f32);
                gl.uniform_2_f32(gl.get_uniform_location(flat_prog, "parallax_strength").as_ref(), 0.0, 0.0);
                gl.uniform_1_f32(gl.get_uniform_location(flat_prog, "zoom_level").as_ref(), 1.0);
                gl.uniform_2_f32(gl.get_uniform_location(flat_prog, "image_dimensions").as_ref(),
                    self.image_width as f32, self.image_height as f32);
                gl.uniform_1_i32(gl.get_uniform_location(flat_prog, "invert_depth").as_ref(),
                    if self.config.invert_depth { 1 } else { 0 });
                // Pass mesh parallax travel so the flat background matches the mesh movement
                gl.uniform_2_f32(gl.get_uniform_location(flat_prog, "mesh_parallax_travel").as_ref(),
                    travel_x, travel_y);

                gl.bind_vertex_array(Some(flat_vao));
                gl.draw_elements(glow::TRIANGLES, 6, glow::UNSIGNED_INT, 0);
                gl.bind_vertex_array(None);

                // Re-enable depth test and writes for the mesh pass
                gl.enable(glow::DEPTH_TEST);
                gl.depth_mask(true);
            }

            // --- Pass 2: 3D mesh with back-face culling ---
            gl.use_program(Some(program));

            gl.uniform_matrix_4_f32_slice(
                gl.get_uniform_location(program, "projection").as_ref(),
                false,
                &proj,
            );
            gl.uniform_matrix_4_f32_slice(
                gl.get_uniform_location(program, "view").as_ref(),
                false,
                &view,
            );

            // When UVs are available, sample the full-resolution wallpaper texture
            // instead of using the low-res baked vertex colors.
            if self.mesh_has_uvs {
                if let Some(wt) = self.wallpaper_texture {
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(wt));
                    gl.uniform_1_i32(gl.get_uniform_location(program, "wallpaper_texture").as_ref(), 0);
                }
                gl.uniform_1_i32(gl.get_uniform_location(program, "use_texture").as_ref(), 1);
            } else {
                gl.uniform_1_i32(gl.get_uniform_location(program, "use_texture").as_ref(), 0);
            }

            gl.bind_vertex_array(Some(vao));
            gl.draw_elements(glow::TRIANGLES, self.mesh_index_count as i32, glow::UNSIGNED_INT, 0);
            gl.bind_vertex_array(None);
        }
        unsafe { egl_swap_buffers(&self.egl_context, self.egl_surface); }
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen_width = width.max(1);
        self.screen_height = height.max(1);
    }

    pub fn reload_textures(&mut self, wallpaper_path: &str, depth_path: &str) -> Result<()> {
        let gl = self.gl_context.as_ref().ok_or_else(|| anyhow!("no GL"))?;
        let wallpaper_raw = image::open(wallpaper_path)?.to_rgba8();
        self.image_width = wallpaper_raw.width();
        self.image_height = wallpaper_raw.height();
        let depth_raw = image::open(depth_path)?.to_luma8();

        let mut wallpaper = wallpaper_raw;
        image::imageops::flip_vertical_in_place(&mut wallpaper);
        let mut depth = depth_raw;
        image::imageops::flip_vertical_in_place(&mut depth);

        log::info!(
            "Reload textures: wallpaper {}x{}, depth {}x{}",
            self.image_width,
            self.image_height,
            depth.width(),
            depth.height()
        );

        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

            if let Some(old_tex) = self.wallpaper_texture.take() {
                gl.delete_texture(old_tex);
            }
            if let Some(old_tex) = self.depth_texture.take() {
                gl.delete_texture(old_tex);
            }

            let wallpaper_tex = gl.create_texture().map_err(gl_error)?;
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(wallpaper_tex));
            set_texture_params(gl);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                self.image_width as i32,
                self.image_height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                Some(&wallpaper),
            );
            gl.generate_mipmap(glow::TEXTURE_2D);
            self.wallpaper_texture = Some(wallpaper_tex);

            let depth_tex = gl.create_texture().map_err(gl_error)?;
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(depth_tex));
            set_texture_params(gl);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R8 as i32,
                depth.width() as i32,
                depth.height() as i32,
                0,
                glow::RED,
                glow::UNSIGNED_BYTE,
                Some(depth.as_raw()),
            );
            gl.generate_mipmap(glow::TEXTURE_2D);
            self.depth_texture = Some(depth_tex);
        }
        log::info!("Textures reloaded");
        Ok(())
    }

    pub fn reload_mesh(&mut self, ply_path: &str) -> Result<()> {
        let gl = self.gl_context.as_ref().ok_or_else(|| anyhow!("no GL"))?;

        let mesh = Mesh::load_ply(std::path::Path::new(ply_path))?;

        self.mesh_camera_z   = 0.0;
        self.mesh_near_z     = mesh.aabb[5].abs().max(0.1);
        self.mesh_xy_half    = mesh.xy_half_extent().max(0.001);
        self.mesh_fov_y_deg    = mesh.fov_y_deg;
        self.mesh_image_aspect = mesh.image_aspect.max(0.01);
        self.mesh_index_count  = mesh.indices.len() as u32;
        self.mesh_has_uvs      = mesh.has_uvs;

        log::info!(
            "Mesh reloaded: {} verts, near_z={:.3}, xy_half={:.3}, depth_range={:.3}, fov_y={:.2}°, image_aspect={:.4}",
            mesh.vertex_count, self.mesh_near_z, self.mesh_xy_half,
            mesh.depth_range(), self.mesh_fov_y_deg, self.mesh_image_aspect
        );

        unsafe {
            if let Some(old) = self.mesh_vao.take() { gl.delete_vertex_array(old); }
            if let Some(old) = self.mesh_pos_vbo.take() { gl.delete_buffer(old); }
            if let Some(old) = self.mesh_col_vbo.take() { gl.delete_buffer(old); }
            if let Some(old) = self.mesh_uv_vbo.take() { gl.delete_buffer(old); }
            if let Some(old) = self.mesh_ebo.take() { gl.delete_buffer(old); }

            let vao = gl.create_vertex_array().map_err(gl_error)?;
            gl.bind_vertex_array(Some(vao));

            let pos_vbo = gl.create_buffer().map_err(gl_error)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(pos_vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&mesh.positions),
                glow::STATIC_DRAW,
            );
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.enable_vertex_attrib_array(0);

            let col_vbo = gl.create_buffer().map_err(gl_error)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(col_vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                &mesh.colors,
                glow::STATIC_DRAW,
            );
            gl.vertex_attrib_pointer_f32(1, 4, glow::UNSIGNED_BYTE, true, 4, 0);
            gl.enable_vertex_attrib_array(1);

            let uv_vbo = if mesh.has_uvs && !mesh.uvs.is_empty() {
                let vbo = gl.create_buffer().map_err(gl_error)?;
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&mesh.uvs),
                    glow::STATIC_DRAW,
                );
                gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 8, 0);
                gl.enable_vertex_attrib_array(2);
                Some(vbo)
            } else {
                None
            };

            let ebo = gl.create_buffer().map_err(gl_error)?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&mesh.indices),
                glow::STATIC_DRAW,
            );

            gl.bind_vertex_array(None);

            self.mesh_vao     = Some(vao);
            self.mesh_pos_vbo = Some(pos_vbo);
            self.mesh_col_vbo = Some(col_vbo);
            self.mesh_uv_vbo  = uv_vbo;
            self.mesh_ebo     = Some(ebo);
        }

        log::info!(
            "Mesh reuploaded: {} vertices, {} triangles",
            mesh.vertex_count,
            mesh.triangle_count
        );
        Ok(())
    }
}

impl Drop for EglRenderer {
    fn drop(&mut self) {
        unsafe {
            if !self.egl_surface.is_null() {
                egl_destroy_surface(&mut self.egl_context, self.egl_surface);
            }
            egl_destroy_ctx(&mut self.egl_context);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

unsafe fn compile_program(
    gl: &glow::Context,
    vert_src: &str,
    frag_src: &str,
) -> Result<glow::Program> {
    let vert = gl.create_shader(glow::VERTEX_SHADER).map_err(gl_error)?;
    gl.shader_source(vert, vert_src);
    gl.compile_shader(vert);
    if !gl.get_shader_compile_status(vert) {
        return Err(anyhow!("vertex shader: {}", gl.get_shader_info_log(vert)));
    }

    let frag = gl.create_shader(glow::FRAGMENT_SHADER).map_err(gl_error)?;
    gl.shader_source(frag, frag_src);
    gl.compile_shader(frag);
    if !gl.get_shader_compile_status(frag) {
        return Err(anyhow!("fragment shader: {}", gl.get_shader_info_log(frag)));
    }

    let program = gl.create_program().map_err(gl_error)?;
    gl.attach_shader(program, vert);
    gl.attach_shader(program, frag);
    gl.link_program(program);
    if !gl.get_program_link_status(program) {
        return Err(anyhow!("link: {}", gl.get_program_info_log(program)));
    }
    gl.delete_shader(vert);
    gl.delete_shader(frag);
    Ok(program)
}

unsafe fn set_texture_params(gl: &glow::Context) {
    // Use trilinear filtering (LINEAR_MIPMAP_LINEAR) for minification so that
    // high-resolution wallpaper images (e.g. 8K → 1600px display) are properly
    // downsampled without aliasing/blurring artefacts.
    // generate_mipmap() must be called after tex_image_2d.
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
}

// ---------------------------------------------------------------------------
// Shaders — flat depth-warp mode (unchanged from original)
// ---------------------------------------------------------------------------

const FLAT_VERT: &str = r#"#version 300 es
precision mediump float;
layout(location=0) in vec2 position;
layout(location=1) in vec2 uv;
out vec2 fragment_uv;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    fragment_uv = uv;
}
"#;

const FLAT_FRAG: &str = r#"#version 300 es
precision mediump float;
in vec2 fragment_uv;
out vec4 output_color;
uniform sampler2D wallpaper_texture;
uniform sampler2D depth_texture;
uniform vec2 mouse_position;
uniform vec2 screen_resolution;
uniform vec2 parallax_strength;
uniform float zoom_level;
uniform vec2 image_dimensions;
uniform bool invert_depth;
uniform vec2 mesh_parallax_travel;

void main() {
    float depth = texture(depth_texture, fragment_uv).r;
    if (invert_depth) depth = 1.0 - depth;
    
    // Parallax offset: use mesh-consistent parallax when in mesh mode
    // (mesh_parallax_travel > 0), otherwise use the flat-mode strength formula.
    vec2 mouse_offset = mouse_position - vec2(0.5);
    vec2 parallax_offset;
    if (mesh_parallax_travel.x > 0.0 || mesh_parallax_travel.y > 0.0) {
        // Mesh mode: camera translates by -(mouse - 0.5) * 2 * travel.
        // This makes the scene FOLLOW the mouse (objects move right when mouse
        // is right). To match, the texture UV must shift positively:
        //   sample_coord += mouse_offset * 2 * travel  (follows cursor)
        // The NEGATION is needed because camera translation is opposite to
        // the visual texture shift direction.
        parallax_offset = -mouse_offset * 2.0 * mesh_parallax_travel;
    } else {
        // Flat mode: original depth-based parallax
        float parallax_amount = 1.0 - depth;
        parallax_offset = mouse_offset * parallax_amount * parallax_strength;
    }

    float screen_aspect = screen_resolution.x / screen_resolution.y;
    float image_aspect  = image_dimensions.x / image_dimensions.y;

    vec2 scale;
    if (image_aspect > screen_aspect) {
        scale = vec2(screen_aspect / image_aspect, 1.0);
    } else {
        scale = vec2(1.0, image_aspect / screen_aspect);
    }
    scale /= zoom_level;

    vec2 sample_coord = (fragment_uv - 0.5) * scale + 0.5 + parallax_offset;
    output_color = texture(wallpaper_texture, clamp(sample_coord, 0.001, 0.999));
}
"#;

// ---------------------------------------------------------------------------
// Shaders — mesh perspective mode
// ---------------------------------------------------------------------------

const MESH_VERT: &str = r#"#version 300 es
precision highp float;
layout(location=0) in vec3 position;
layout(location=1) in vec4 color;
layout(location=2) in vec2 uv;
out vec4 v_color;
out vec2 v_uv;
out float v_world_z;
uniform mat4 projection;
uniform mat4 view;
void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    v_color = color;
    v_uv = uv;
    v_world_z = position.z;
}
"#;

const MESH_FRAG: &str = r#"#version 300 es
precision mediump float;
in vec4 v_color;
in vec2 v_uv;
in float v_world_z;
out vec4 out_color;
uniform sampler2D wallpaper_texture;
uniform int use_texture;
void main() {
    if (use_texture == 1) {
        if (v_uv.x < 0.0 || v_uv.x > 1.0 || v_uv.y < 0.0 || v_uv.y > 1.0)
            discard;
        out_color = vec4(texture(wallpaper_texture, v_uv).rgb, 1.0);
    } else {
        out_color = vec4(v_color.rgb, 1.0);
    }
}
"#;
