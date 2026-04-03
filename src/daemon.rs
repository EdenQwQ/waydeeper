use anyhow::{anyhow, Result};
use serde_json::json;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::cache::{DepthCache, InpaintCache};
use crate::config::{self};
use crate::depth_estimator::DepthEstimator;
use crate::inpaint::{self, InpaintConfig};
use crate::ipc::{DaemonSocket, ReloadParams, ReloadState};
use crate::models;
use crate::renderer;

pub struct DepthWallpaperDaemon {
    pub config: config::Config,
    pub depth_estimator: Option<DepthEstimator>,
    cache_manager: Option<DepthCache>,
    inpaint_cache: Option<InpaintCache>,
}

impl DepthWallpaperDaemon {
    pub fn new() -> Result<Self> {
        let config = config::load_config().unwrap_or_default();
        let config_dir = config::config_dir();
        std::fs::create_dir_all(&config_dir)?;

        Ok(Self {
            config,
            depth_estimator: None,
            cache_manager: None,
            inpaint_cache: None,
        })
    }

    pub fn load_configuration(&mut self) {
        match config::load_config() {
            Ok(config) => self.config = config,
            Err(error) => log::warn!("Failed to load configuration: {}", error),
        }
    }

    fn ensure_cache_initialized(&mut self) -> Result<()> {
        if self.cache_manager.is_none() {
            let cache_dir = self
                .config
                .cache_directory
                .as_ref()
                .map(|path| std::path::Path::new(path.as_str()));
            self.cache_manager = Some(DepthCache::new(cache_dir)?);
        }
        Ok(())
    }

    fn ensure_inpaint_cache_initialized(&mut self) -> Result<()> {
        if self.inpaint_cache.is_none() {
            let cache_dir = self
                .config
                .cache_directory
                .as_ref()
                .map(|path| std::path::Path::new(path.as_str()));
            self.inpaint_cache = Some(InpaintCache::new(cache_dir)?);
        }
        Ok(())
    }

    fn get_model_name_for_cache(model_path: Option<&str>) -> String {
        match model_path {
            Some(path) => {
                let p = std::path::Path::new(path);
                // For directory-format models (model.onnx inside a dir), use the parent dir name
                if p.file_name().is_some_and(|n| n == "model.onnx") {
                    if let Some(parent) = p.parent() {
                        if let Some(name) = parent.file_name().and_then(|n| n.to_str()) {
                            return name.to_string();
                        }
                    }
                }
                p.file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("depth-anything-v3-base")
                    .to_string()
            }
            None => "depth-anything-v3-base".to_string(),
        }
    }

    pub fn ensure_depth_map_exists(
        &mut self,
        image_path: &str,
        model_path: Option<&str>,
        force_regenerate: bool,
    ) -> Result<String> {
        self.ensure_cache_initialized()?;

        let model_name = Self::get_model_name_for_cache(model_path);
        let cache = self.cache_manager.as_ref().unwrap();

        if !force_regenerate {
            let display_path = model_path.unwrap_or("auto");
            if let Ok(Some(_cached)) =
                cache.get_cached_depth(std::path::Path::new(image_path), &model_name, display_path)
            {
                let hash = cache.compute_image_hash(std::path::Path::new(image_path), &model_name)?;
                let depth_path = cache.get_depth_file_path(&hash);
                return Ok(depth_path.to_string_lossy().to_string());
            }
        }

        if self.depth_estimator.is_none() {
            self.depth_estimator = Some(DepthEstimator::new(model_path)?);
        }

        let estimator = self.depth_estimator.as_mut().unwrap();
        let _depth = estimator.estimate_cached(image_path, cache, force_regenerate)?;
        let actual_model_name = estimator.model_name.clone();

        let hash = cache.compute_image_hash(std::path::Path::new(image_path), &actual_model_name)?;
        let depth_path = cache.get_depth_file_path(&hash);

        log::info!("Depth map generated: {}", depth_path.display());
        Ok(depth_path.to_string_lossy().to_string())
    }

    /// Ensure a PLY inpaint mesh exists for the given image + depth map.
    pub fn ensure_ply_exists(
        &mut self,
        image_path: &str,
        depth_path: &str,
        python: &str,
        force_regenerate: bool,
        invert_depth: bool,
    ) -> Result<String> {
        if !models::inpaint_models_present() {
            return Err(anyhow!(
                "3D inpainting model weights not found in {}.\n\
                 Run 'waydeeper download-model inpaint' to download them.",
                models::inpaint_models_dir().display()
            ));
        }

        self.ensure_inpaint_cache_initialized()?;

        let img_p   = Path::new(image_path);
        let depth_p = Path::new(depth_path);

        let inpaint_cfg = InpaintConfig {
            image_path: img_p,
            depth_path: depth_p,
            output_ply: Path::new("/dev/null"),
            models_dir: &models::inpaint_models_dir(),
            python,
            longer_side: 960,
            depth_threshold: 0.04,
            background_thickness: 70,
            context_thickness: 140,
            extrapolation_thickness: 60,
            invert_depth,
        };
        let tag = inpaint_cfg.cache_tag();

        let cache = self.inpaint_cache.as_ref().unwrap();

        if !force_regenerate {
            if let Some(cached_ply) = cache.get_cached_ply(img_p, depth_p, &tag)? {
                return Ok(cached_ply.to_string_lossy().to_string());
            }
        }

        let output_ply = cache.ply_write_path(img_p, depth_p, &tag)?;

        let inpaint_cfg = InpaintConfig {
            image_path: img_p,
            depth_path: depth_p,
            output_ply: &output_ply,
            models_dir: &models::inpaint_models_dir(),
            python,
            longer_side: 960,
            depth_threshold: 0.04,
            background_thickness: 70,
            context_thickness: 140,
            extrapolation_thickness: 60,
            invert_depth,
        };

        println!("Running 3D inpainting...");
        inpaint::run_inpainting(&inpaint_cfg)?;

        Ok(output_ply.to_string_lossy().to_string())
    }

    pub fn pregenerate_depth_map(
        &mut self,
        image_path: &str,
        model_path: Option<&str>,
        force_regenerate: bool,
    ) -> Result<String> {
        if !std::path::Path::new(image_path).exists() {
            return Err(anyhow!("Image not found: {}", image_path));
        }
        self.ensure_depth_map_exists(image_path, model_path, force_regenerate)
    }

    pub fn clear_cache(&mut self) -> Result<()> {
        self.ensure_cache_initialized()?;
        self.cache_manager.as_ref().unwrap().clear_cache()?;
        if let Ok(()) = self.ensure_inpaint_cache_initialized() {
            self.inpaint_cache.as_ref().unwrap().clear_cache()?;
        }
        Ok(())
    }

    pub fn list_cached_wallpapers(&mut self) -> Result<()> {
        self.ensure_cache_initialized()?;

        let items = self.cache_manager.as_ref().unwrap().list_cached()?;
        if items.is_empty() {
            println!("No cached wallpapers found.");
        } else {
            println!("Cached wallpapers:");
            for item in &items {
                println!(
                    "  - {} ({}x{}, model: {})",
                    item.original_path, item.width, item.height, item.model_name
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_daemon(
        &mut self,
        wallpaper_path: &str,
        monitor: &str,
        strength_x: f64,
        strength_y: f64,
        smooth_animation: bool,
        animation_speed: f64,
        fps: u32,
        active_delay_ms: f64,
        idle_timeout_ms: f64,
        model_path: Option<&str>,
        regenerate: bool,
        invert_depth: bool,
        use_inpaint: bool,
        inpaint_python: &str,
    ) -> Result<()> {
        let monitor_id = monitor.to_string();
        let running = Arc::new(AtomicBool::new(true));
        let reload_state = Arc::new(ReloadState::new());

        use std::sync::atomic::AtomicPtr;
        static RUNNING_PTR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

        let running_ptr = Arc::into_raw(running.clone()) as *mut ();
        RUNNING_PTR.store(running_ptr, Ordering::SeqCst);

        extern "C" fn sigterm_handler(_signal: i32) {
            let pointer = RUNNING_PTR.load(Ordering::SeqCst);
            if !pointer.is_null() {
                let running = unsafe { Arc::from_raw(pointer as *const AtomicBool) };
                running.store(false, Ordering::SeqCst);
                let _ = Arc::into_raw(running);
            }
        }

        unsafe {
            nix::sys::signal::sigaction(
                nix::sys::signal::Signal::SIGTERM,
                &nix::sys::signal::SigAction::new(
                    nix::sys::signal::SigHandler::Handler(sigterm_handler),
                    nix::sys::signal::SaFlags::empty(),
                    nix::sys::signal::SigSet::empty(),
                ),
            )
            .ok();
        }

        let effective_config = self
            .config
            .monitors
            .entry(monitor_id.clone())
            .or_default();

        effective_config.wallpaper_path = Some(wallpaper_path.to_string());
        effective_config.strength_x = strength_x;
        effective_config.strength_y = strength_y;
        effective_config.smooth_animation = smooth_animation;
        effective_config.animation_speed = animation_speed;
        effective_config.fps = fps;
        effective_config.active_delay_ms = active_delay_ms;
        effective_config.idle_timeout_ms = idle_timeout_ms;
        if let Some(path) = model_path {
            effective_config.model_path = Some(path.to_string());
        }
        effective_config.invert_depth = invert_depth;
        effective_config.use_inpaint = use_inpaint;
        effective_config.inpaint_python = inpaint_python.to_string();

        let initial_state = DaemonState {
            wallpaper_path: wallpaper_path.to_string(),
            depth_path: String::new(),
            ply_path: None,
            strength_x,
            strength_y,
            smooth_animation,
            animation_speed,
            fps,
            active_delay_ms,
            idle_timeout_ms,
            invert_depth,
            use_inpaint,
            model_path: effective_config.model_path.clone(),
            regenerate,
            inpaint_python: inpaint_python.to_string(),
        };

        self.run_daemon_loop(&monitor_id, &running, &reload_state, initial_state)
    }

    fn run_daemon_loop(
        &mut self,
        monitor_id: &str,
        running: &Arc<AtomicBool>,
        reload_state: &Arc<ReloadState>,
        mut state: DaemonState,
    ) -> Result<()> {
        let monitor_id = monitor_id.to_string();

        // Initial asset generation
        let wallpaper = state.wallpaper_path.clone();
        let model = state.model_path.clone();

        log::info!("Generating depth map for {}...", wallpaper);
        let depth_path = self.ensure_depth_map_exists(&wallpaper, model.as_deref(), state.regenerate)?;
        state.depth_path = depth_path.clone();

        let ply_path = if state.use_inpaint {
            match self.ensure_ply_exists(
                &wallpaper,
                &depth_path,
                &state.inpaint_python,
                state.regenerate,
                state.invert_depth,
            ) {
                Ok(path) => {
                    log::info!("Inpaint mesh ready: {}", path);
                    Some(path)
                }
                Err(err) => {
                    log::warn!(
                        "Inpainting failed, falling back to flat depth mode: {}",
                        err
                    );
                    None
                }
            }
        } else {
            None
        };
        state.ply_path = ply_path.clone();

        // Start IPC socket and renderer
        let mut socket = DaemonSocket::new(&monitor_id)?;
        let running_for_handler = running.clone();
        let monitor_for_handler = monitor_id.clone();
        let reload_for_handler = reload_state.clone();

        let handler: crate::ipc::CommandHandler =
            Arc::new(move |command: &str, params: &serde_json::Value| match command {
                "PING" => Ok(json!({"status": "pong"})),
                "STATUS" => {
                    let generating = reload_for_handler.generating.load(Ordering::SeqCst);
                    let complete = reload_for_handler.is_reload_complete();
                    let logs = reload_for_handler.take_logs();
                    Ok(json!({
                        "monitor": monitor_for_handler,
                        "running": running_for_handler.load(Ordering::SeqCst),
                        "generating": generating,
                        "complete": complete,
                        "logs": logs,
                    }))
                }
                "STOP" => {
                    log::info!("Received STOP command via IPC");
                    running_for_handler.store(false, Ordering::SeqCst);
                    Ok(json!({"stopping": true}))
                }
                "RELOAD" => {
                    log::info!("Received RELOAD command via IPC");
                    let reload_params: Result<ReloadParams> = serde_json::from_value(params.clone())
                        .map_err(|e| anyhow!("Invalid RELOAD params: {}", e));
                    match reload_params {
                        Ok(p) => {
                            reload_for_handler.store_params(p);
                            Ok(json!({"reloaded": true}))
                        }
                        Err(e) => Err(e),
                    }
                }
                _ => Err(anyhow!("Unknown command: {}", command)),
            });

        socket.start(handler)?;

        let render_config = renderer::RendererConfig {
            wallpaper_path: wallpaper.clone(),
            depth_path: depth_path.clone(),
            monitor: monitor_id.to_string(),
            strength_x: state.strength_x,
            strength_y: state.strength_y,
            smooth_animation: state.smooth_animation,
            animation_speed: state.animation_speed,
            fps: state.fps,
            active_delay_ms: state.active_delay_ms,
            idle_timeout_ms: state.idle_timeout_ms,
            invert_depth: state.invert_depth,
            ply_path: ply_path.clone(),
        };

        // Run renderer — it handles reload checking internally
        crate::wayland::run(render_config, running.clone(), reload_state.clone())?;

        if !running.load(Ordering::SeqCst) {
            log::info!("Daemon exiting cleanly");
        }

        Ok(())
    }
}

struct DaemonState {
    wallpaper_path: String,
    depth_path: String,
    ply_path: Option<String>,
    strength_x: f64,
    strength_y: f64,
    smooth_animation: bool,
    animation_speed: f64,
    fps: u32,
    active_delay_ms: f64,
    idle_timeout_ms: f64,
    invert_depth: bool,
    use_inpaint: bool,
    model_path: Option<String>,
    regenerate: bool,
    inpaint_python: String,
}
