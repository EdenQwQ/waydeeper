use anyhow::{anyhow, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_pointer, delegate_registry,
    delegate_seat,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        pointer::{PointerEvent, PointerEventKind, PointerHandler},
        Capability, SeatHandler, SeatState,
    },
    shell::{
        wlr_layer::{Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface, LayerSurfaceConfigure},
        WaylandSurface,
    },
};
use std::ffi::c_void;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_output, wl_pointer, wl_seat, wl_surface},
    Connection, Dispatch, Proxy, QueueHandle,
};
use wayland_protocols::wp::{
    fractional_scale::v1::client::{wp_fractional_scale_manager_v1, wp_fractional_scale_v1},
    viewporter::client::{wp_viewport, wp_viewporter},
};

use crate::daemon::DepthWallpaperDaemon;
use crate::ipc::{ReloadParams, ReloadResult, ReloadState};
use crate::renderer::{EglRenderer, PendingReload, RendererConfig};

pub struct App {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    layer: LayerSurface,
    pointer: Option<wl_pointer::WlPointer>,

    #[allow(dead_code)]
    viewporter: Option<wp_viewporter::WpViewporter>,
    viewport: Option<wp_viewport::WpViewport>,
    fractional_scale: Option<wp_fractional_scale_v1::WpFractionalScaleV1>,
    fractional_scale_value: Option<f64>,

    renderer: EglRenderer,
    running: Arc<AtomicBool>,
    first_configure: bool,
    frame_counter: u64,
}

fn find_output(output_state: &OutputState, monitor: &str) -> Option<wl_output::WlOutput> {
    for output in output_state.outputs() {
        if let Some(info) = output_state.info(&output) {
            if info.name.as_deref() == Some(monitor) {
                return Some(output);
            }
        }
    }
    None
}

pub fn run(config: RendererConfig, running: Arc<AtomicBool>, reload_state: Arc<ReloadState>) -> Result<()> {
    log::info!(
        "renderer: starting on monitor '{}' (strength_x={}, strength_y={}, fps={})",
        config.monitor, config.strength_x, config.strength_y, config.fps,
    );

    let connection = Connection::connect_to_env()
        .map_err(|error| anyhow!("Wayland connect: {}", error))?;
    let (globals, mut event_queue) = registry_queue_init(&connection)
        .map_err(|error| anyhow!("registry_queue_init: {:?}", error))?;
    let queue_handle = event_queue.handle();

    let compositor = CompositorState::bind(&globals, &queue_handle)
        .map_err(|_| anyhow!("wl_compositor not available"))?;
    let layer_shell = LayerShell::bind(&globals, &queue_handle)
        .map_err(|_| anyhow!("wlr-layer-shell not available"))?;

    let viewporter_global = globals.contents().with_list(|list| {
        list.iter()
            .find(|global| global.interface == "wp_viewporter")
            .map(|global| global.name)
    });
    let viewporter = if let Some(name) = viewporter_global {
        let vp: wp_viewporter::WpViewporter = globals.registry().bind(name, 1, &queue_handle, ());
        log::info!("Bound wp_viewporter");
        Some(vp)
    } else {
        log::warn!("wp_viewporter not available");
        None
    };

    let fractional_scale_global = globals.contents().with_list(|list| {
        list.iter()
            .find(|global| global.interface == "wp_fractional_scale_manager_v1")
            .map(|global| global.name)
    });
    let fractional_scale_manager = if let Some(name) = fractional_scale_global {
        let manager: wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1 =
            globals.registry().bind(name, 1, &queue_handle, ());
        log::info!("Bound wp_fractional_scale_manager_v1");
        Some(manager)
    } else {
        log::warn!("wp_fractional_scale_manager_v1 not available");
        None
    };

    let output_state = OutputState::new(&globals, &queue_handle);
    let seat_state = SeatState::new(&globals, &queue_handle);
    let registry_state = RegistryState::new(&globals);
    let fps = config.fps.max(1) as u64;

    let temp_surface = compositor.create_surface(&queue_handle);
    let mut app = App {
        registry_state,
        seat_state,
        output_state,
        layer: layer_shell.create_layer_surface(
            &queue_handle,
            temp_surface,
            Layer::Background,
            Some("waydeeper"),
            None,
        ),
        pointer: None,
        viewporter: viewporter.clone(),
        viewport: None,
        fractional_scale: None,
        fractional_scale_value: None,
        renderer: EglRenderer::new(
            connection.backend().display_ptr() as *mut c_void,
            config,
        )?,
        running: running.clone(),
        first_configure: true,
        frame_counter: 0,
    };

    for _ in 0..4 {
        let _ = event_queue.dispatch_pending(&mut app);
        let _ = event_queue.flush();
        let _ = event_queue.roundtrip(&mut app);
    }

    let target_output = find_output(&app.output_state, &app.renderer.config.monitor);
    if let Some(ref output) = target_output {
        if let Some(info) = app.output_state.info(output) {
            log::info!(
                "Found output '{}' ({}x{}, wl_output.scale={})",
                info.name.as_deref().unwrap_or("?"),
                info.logical_size.unwrap_or((0, 0)).0,
                info.logical_size.unwrap_or((0, 0)).1,
                info.scale_factor
            );
        }
    } else {
        let available: Vec<String> = app
            .output_state
            .outputs()
            .filter_map(|output| app.output_state.info(&output)?.name.clone())
            .collect();
        log::warn!(
            "Monitor '{}' not found. Available: {:?}",
            app.renderer.config.monitor,
            available
        );
    }

    let surface = compositor.create_surface(&queue_handle);
    let layer = layer_shell.create_layer_surface(
        &queue_handle,
        surface,
        Layer::Background,
        Some("waydeeper"),
        target_output.as_ref(),
    );
    layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer.set_exclusive_zone(-1);

    let viewport = viewporter.as_ref().map(|vp| vp.get_viewport(layer.wl_surface(), &queue_handle, ()));

    let fractional_scale = fractional_scale_manager.as_ref().map(|manager| manager.get_fractional_scale(layer.wl_surface(), &queue_handle, ()));

    layer.commit();

    app.layer = layer;
    app.viewport = viewport;
    app.fractional_scale = fractional_scale;

    let frame_duration = Duration::from_millis(1000 / fps);
    let mut last_time = std::time::Instant::now();
    let preparing_reload: std::sync::Arc<std::sync::Mutex<Option<PendingReload>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    loop {
        if !running.load(Ordering::SeqCst) {
            log::info!("renderer: exiting");
            break;
        }

        // Check for reload: generate assets in background thread while rendering continues
        if reload_state.params.lock().unwrap().is_some() && !reload_state.generating.load(Ordering::SeqCst) {
            reload_state.generating.store(true, Ordering::SeqCst);
            let params = reload_state.params.lock().unwrap().take().unwrap();
            let reload_state_clone = reload_state.clone();

            std::thread::spawn(move || {
                reload_state_clone.push_log("Generating depth map...".to_string());
                let result = generate_reload_assets(&params);
                match result {
                    Ok(reload_result) => {
                        let mut guard = reload_state_clone.result.lock().unwrap();
                        *guard = Some(reload_result);
                        reload_state_clone.push_log("Assets ready, preparing textures...".to_string());
                        reload_state_clone.ready.store(true, Ordering::SeqCst);
                    }
                    Err(e) => {
                        reload_state_clone.push_log(format!("Reload failed: {}", e));
                        reload_state_clone.generating.store(false, Ordering::SeqCst);
                    }
                }
            });
        }

        // Check if reload assets are ready — prepare textures in background (no transition yet)
        if reload_state.ready.swap(false, Ordering::SeqCst) {
            if let Some(result) = reload_state.result.lock().unwrap().take() {
                log::info!("Reload assets ready, preparing textures in background...");
                // Prepare textures and mesh in background thread (image decode is slow)
                let preparing = preparing_reload.clone();
                std::thread::spawn(move || {
                    let wallpaper_result = EglRenderer::prepare_reload_textures(
                        &result.wallpaper_path,
                        &result.depth_path,
                    );
                    match wallpaper_result {
                        Ok((wallpaper, depth)) => {
                            let ply_data = if result.use_inpaint {
                                if let Some(ref ply_path) = result.ply_path {
                                    match EglRenderer::prepare_reload_mesh(ply_path) {
                                        Ok(p) => Some(p),
                                        Err(e) => {
                                            log::warn!("Mesh prepare failed: {}", e);
                                            None
                                        }
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                            let pending = PendingReload {
                                wallpaper,
                                depth,
                                ply_path: result.ply_path,
                                ply_data,
                                strength_x: result.strength_x,
                                strength_y: result.strength_y,
                                smooth_animation: result.smooth_animation,
                                animation_speed: result.animation_speed,
                                fps: result.fps,
                                active_delay_ms: result.active_delay_ms,
                                idle_timeout_ms: result.idle_timeout_ms,
                                invert_depth: result.invert_depth,
                                use_inpaint: result.use_inpaint,
                            };
                            *preparing.lock().unwrap() = Some(pending);
                        }
                        Err(e) => {
                            log::warn!("Texture prepare failed: {}", e);
                        }
                    }
                });
            }
        }

        // Check if background texture preparation is done — capture old wallpaper, upload new, transition
        if let Some(pending) = preparing_reload.lock().unwrap().take() {
            // Capture current frame (old wallpaper) right before the swap
            if let Err(e) = app.renderer.start_transition() {
                log::warn!("Transition start failed: {}", e);
            }
            // Upload new textures (fast GPU operation)
            app.renderer.set_pending_reload(pending);
            if let Err(e) = app.renderer.try_apply_pending_reload() {
                log::warn!("Failed to apply pending reload: {}", e);
            } else {
                reload_state.push_log("Reload complete, wallpaper updated seamlessly".to_string());
                reload_state.mark_reload_complete();
                reload_state.generating.store(false, Ordering::SeqCst);
            }
        }

        let _ = event_queue.dispatch_pending(&mut app);
        let _ = event_queue.flush();

        if let Some(guard) = event_queue.prepare_read() {
            let fd = guard.connection_fd();
            let mut poll_fds = [nix::poll::PollFd::new(
                fd,
                nix::poll::PollFlags::POLLIN,
            )];
            let timeout_ms: u16 = std::cmp::min(500, frame_duration.as_millis() as u16);
            match nix::poll::poll(&mut poll_fds, timeout_ms) {
                Ok(0) => {}
                Ok(_) => {
                    if guard.read().is_ok() {
                        let _ = event_queue.dispatch_pending(&mut app);
                    }
                }
                Err(_) => {}
            }
        } else {
            std::thread::sleep(frame_duration);
        }

        if !app.first_configure {
            let now = std::time::Instant::now();
            let delta = now.duration_since(last_time).as_secs_f64();
            last_time = now;
            app.renderer.update_mouse(delta);
            app.renderer.update_transition(delta);
            app.renderer.update_fade(delta);
            let _ = app.renderer.draw();
            let _ = app.renderer.draw_transition_overlay();
            let _ = app.renderer.draw_fade();
            app.renderer.swap_buffers();
            app.frame_counter += 1;
        }

        std::thread::sleep(frame_duration);
    }

    log::info!("renderer: done ({} frames)", app.frame_counter);
    Ok(())
}

fn generate_reload_assets(params: &ReloadParams) -> Result<ReloadResult> {
    let mut daemon = DepthWallpaperDaemon::new()?;
    daemon.load_configuration();

    let wallpaper = &params.wallpaper_path;
    let model = params.model_path.clone();

    log::info!("Generating depth map for reload: {}...", wallpaper);
    let depth_path = daemon.ensure_depth_map_exists(wallpaper, model.as_deref(), params.regenerate)?;

    let ply_path = if params.use_inpaint {
        match daemon.ensure_ply_exists(
            wallpaper,
            &depth_path,
            &params.inpaint_python,
            params.regenerate,
            params.invert_depth,
        ) {
            Ok(path) => {
                log::info!("Inpaint mesh ready for reload: {}", path);
                Some(path)
            }
            Err(err) => {
                log::warn!(
                    "Inpainting failed during reload, falling back to flat depth mode: {}",
                    err
                );
                None
            }
        }
    } else {
        None
    };

    Ok(ReloadResult {
        wallpaper_path: wallpaper.clone(),
        depth_path,
        ply_path,
        strength_x: params.strength_x,
        strength_y: params.strength_y,
        smooth_animation: params.smooth_animation,
        animation_speed: params.animation_speed,
        fps: params.fps,
        active_delay_ms: params.active_delay_ms,
        idle_timeout_ms: params.idle_timeout_ms,
        invert_depth: params.invert_depth,
        use_inpaint: params.use_inpaint,
    })
}

impl CompositorHandler for App {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    fn frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
    }
    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_output::WlOutput,
    ) {
    }
    fn update_output(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_output::WlOutput,
    ) {
    }
    fn output_destroyed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for App {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        log::info!("Layer surface closed");
        self.running.store(false, Ordering::SeqCst);
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let logical_width = NonZeroU32::new(configure.new_size.0).map_or(1920, NonZeroU32::get);
        let logical_height = NonZeroU32::new(configure.new_size.1).map_or(1080, NonZeroU32::get);

        let scale = self.fractional_scale_value.unwrap_or_else(|| {
            self.output_state
                .outputs()
                .filter_map(|output| self.output_state.info(&output))
                .find(|info| info.scale_factor > 0)
                .map(|info| info.scale_factor as f64)
                .unwrap_or(1.0)
        });

        let physical_width = (logical_width as f64 * scale).round() as u32;
        let physical_height = (logical_height as f64 * scale).round() as u32;

        log::info!(
            "Layer surface configured: {}x{} (scale={:.2}, physical={}x{})",
            logical_width,
            logical_height,
            scale,
            physical_width,
            physical_height
        );

        if let Some(ref viewport) = self.viewport {
            viewport.set_destination(logical_width as i32, logical_height as i32);
        }

        if self.fractional_scale_value.is_some() {
            self.layer.wl_surface().set_buffer_scale(1);
        } else {
            let integer_scale = scale.ceil() as i32;
            self.layer.wl_surface().set_buffer_scale(integer_scale);
        }

        self.renderer.mouse.surface_width = logical_width as f64;
        self.renderer.mouse.surface_height = logical_height as f64;

        if self.first_configure {
            self.first_configure = false;
            let wayland_surface = self.layer.wl_surface().id().as_ptr() as *mut c_void;
            if let Err(error) = self
                .renderer
                .create_surface(wayland_surface, physical_width, physical_height)
            {
                log::error!("EGL surface creation failed: {}", error);
                return;
            }
            if let Err(error) = self.renderer.load_textures() {
                log::error!("Texture loading failed: {}", error);
            }
            if let Some(ply_path) = self.renderer.config.ply_path.clone() {
                if let Err(error) = self.renderer.load_mesh(&ply_path) {
                    log::error!("Mesh loading failed, falling back to flat mode: {}", error);
                    self.renderer.config.ply_path = None;
                }
            }
            self.renderer.start_fade();
        } else {
            self.renderer.resize(physical_width, physical_height);
        }
    }
}

impl SeatHandler for App {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }
    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
    fn new_capability(
        &mut self,
        _: &Connection,
        queue_handle: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer && self.pointer.is_none() {
            self.pointer = self.seat_state.get_pointer(queue_handle, &seat).ok();
        }
    }
    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer && self.pointer.is_some() {
            self.pointer.take().unwrap().release();
        }
    }
    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl PointerHandler for App {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            if &event.surface != self.layer.wl_surface() {
                continue;
            }
            match event.kind {
                PointerEventKind::Enter { .. } => {
                    self.renderer.mouse.mouse_in_window = true;
                    self.renderer.mouse.mouse_active_start = Some(std::time::Instant::now());
                }
                PointerEventKind::Leave { .. } => {
                    self.renderer.mouse.mouse_in_window = false;
                }
                PointerEventKind::Motion { .. } => {
                    let (x, y) = event.position;
                    let surface_width = self.renderer.mouse.surface_width.max(1.0);
                    let surface_height = self.renderer.mouse.surface_height.max(1.0);
                    self.renderer.mouse.mouse_x = (x / surface_width).clamp(0.0, 1.0);
                    self.renderer.mouse.mouse_y = (1.0 - (y / surface_height)).clamp(0.0, 1.0);
                    self.renderer.mouse.last_mouse_time = std::time::Instant::now();
                }
                _ => {}
            }
        }
    }
}

delegate_compositor!(App);
delegate_output!(App);
delegate_seat!(App);
delegate_pointer!(App);
delegate_layer!(App);
delegate_registry!(App);

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

impl Dispatch<wp_viewporter::WpViewporter, ()> for App {
    fn event(
        _: &mut Self,
        _: &wp_viewporter::WpViewporter,
        _: wp_viewporter::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wp_viewport::WpViewport, ()> for App {
    fn event(
        _: &mut Self,
        _: &wp_viewport::WpViewport,
        _: wp_viewport::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1, ()> for App {
    fn event(
        _: &mut Self,
        _: &wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
        _: wp_fractional_scale_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wp_fractional_scale_v1::WpFractionalScaleV1, ()> for App {
    fn event(
        state: &mut Self,
        _: &wp_fractional_scale_v1::WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            let actual_scale = scale as f64 / 120.0;
            state.fractional_scale_value = Some(actual_scale);
            log::info!("wp_fractional_scale_v1: preferred_scale = {:.2}", actual_scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Output probe — lightweight struct used only for enumerating connected outputs
// ---------------------------------------------------------------------------

struct OutputProbe {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
}

impl CompositorHandler for OutputProbe {
    fn scale_factor_changed(&mut self,_:&Connection,_:&QueueHandle<Self>,_:&wl_surface::WlSurface,_:i32) {}
    fn transform_changed(&mut self,_:&Connection,_:&QueueHandle<Self>,_:&wl_surface::WlSurface,_:wl_output::Transform) {}
    fn frame(&mut self,_:&Connection,_:&QueueHandle<Self>,_:&wl_surface::WlSurface,_:u32) {}
    fn surface_enter(&mut self,_:&Connection,_:&QueueHandle<Self>,_:&wl_surface::WlSurface,_:&wl_output::WlOutput) {}
    fn surface_leave(&mut self,_:&Connection,_:&QueueHandle<Self>,_:&wl_surface::WlSurface,_:&wl_output::WlOutput) {}
}
impl OutputHandler for OutputProbe {
    fn output_state(&mut self) -> &mut OutputState { &mut self.output_state }
    fn new_output(&mut self,_:&Connection,_:&QueueHandle<Self>,_:wl_output::WlOutput) {}
    fn update_output(&mut self,_:&Connection,_:&QueueHandle<Self>,_:wl_output::WlOutput) {}
    fn output_destroyed(&mut self,_:&Connection,_:&QueueHandle<Self>,_:wl_output::WlOutput) {}
}
impl SeatHandler for OutputProbe {
    fn seat_state(&mut self) -> &mut SeatState { &mut self.seat_state }
    fn new_seat(&mut self,_:&Connection,_:&QueueHandle<Self>,_:wl_seat::WlSeat) {}
    fn new_capability(&mut self,_:&Connection,_:&QueueHandle<Self>,_:wl_seat::WlSeat,_:Capability) {}
    fn remove_capability(&mut self,_:&Connection,_:&QueueHandle<Self>,_:wl_seat::WlSeat,_:Capability) {}
    fn remove_seat(&mut self,_:&Connection,_:&QueueHandle<Self>,_:wl_seat::WlSeat) {}
}
impl ProvidesRegistryState for OutputProbe {
    fn registry(&mut self) -> &mut RegistryState { &mut self.registry_state }
    registry_handlers![OutputState, SeatState];
}

delegate_compositor!(OutputProbe);
delegate_output!(OutputProbe);
delegate_seat!(OutputProbe);
delegate_registry!(OutputProbe);

/// Return the names of all currently connected Wayland outputs.
pub fn list_connected_outputs() -> Vec<String> {
    let Ok(connection) = Connection::connect_to_env() else { return Vec::new() };
    let Ok((globals, mut event_queue)) = registry_queue_init(&connection) else { return Vec::new() };
    let queue_handle = event_queue.handle();

    let mut probe = OutputProbe {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &queue_handle),
        output_state: OutputState::new(&globals, &queue_handle),
    };

    for _ in 0..4 {
        let _ = event_queue.dispatch_pending(&mut probe);
        let _ = event_queue.flush();
        let _ = event_queue.roundtrip(&mut probe);
    }

    probe.output_state
        .outputs()
        .filter_map(|output| probe.output_state.info(&output)?.name.clone())
        .collect()
}
