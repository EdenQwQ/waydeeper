use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use serde_json::json;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use crate::config::{self, MonitorConfig};
use crate::daemon::DepthWallpaperDaemon;
use crate::ipc::{self, DaemonClient, ReloadParams};
use crate::models;
use crate::wayland;

// ---------------------------------------------------------------------------
// FPS type with validation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
enum Fps {
    Fps30,
    #[default]
    Fps60,
}

impl Fps {
    fn value(&self) -> u32 {
        match self {
            Fps::Fps30 => 30,
            Fps::Fps60 => 60,
        }
    }
}

impl std::str::FromStr for Fps {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "30" => Ok(Fps::Fps30),
            "60" => Ok(Fps::Fps60),
            _ => Err(format!("FPS must be 30 or 60, got '{}'", s)),
        }
    }
}

impl std::fmt::Display for Fps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}


// ---------------------------------------------------------------------------
// Shared animation + inpaint params
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct AnimationParams {
    strength: Option<f64>,
    strength_x: Option<f64>,
    strength_y: Option<f64>,
    smooth_animation: bool,
    no_smooth_animation: bool,
    animation_speed: Option<f64>,
    fps: Option<Fps>,
    active_delay: Option<f64>,
    idle_timeout: Option<f64>,
}

impl AnimationParams {
    fn resolve(&self, config: &MonitorConfig) -> ResolvedAnimationParams {
        let strength_x = self.strength_x
            .or(self.strength)
            .unwrap_or(config.strength_x);
        let strength_y = self.strength_y
            .or(self.strength)
            .unwrap_or(config.strength_y);
        let smooth_animation = if self.no_smooth_animation {
            false
        } else if self.smooth_animation {
            true
        } else {
            config.smooth_animation
        };

        ResolvedAnimationParams {
            strength_x,
            strength_y,
            smooth_animation,
            animation_speed: self.animation_speed.unwrap_or(config.animation_speed),
            fps: self.fps.map(|f| f.value()).unwrap_or(config.fps),
            active_delay: self.active_delay.unwrap_or(config.active_delay_ms),
            idle_timeout: self.idle_timeout.unwrap_or(config.idle_timeout_ms),
        }
    }
}

struct ResolvedAnimationParams {
    strength_x: f64,
    strength_y: f64,
    smooth_animation: bool,
    animation_speed: f64,
    fps: u32,
    active_delay: f64,
    idle_timeout: f64,
}

// ---------------------------------------------------------------------------
// Clap definitions
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "waydeeper", about = "GPU-accelerated depth effect wallpaper for Wayland", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Set wallpaper on a monitor, or reload an existing daemon with updated config.
    ///
    /// If a daemon is already running for the target monitor, it will be reloaded
    /// with the new settings. Otherwise a new daemon is spawned. Omit IMAGE to use
    /// the currently configured wallpaper (useful for regenerating or changing params).
    Set {
        /// Path to the wallpaper image (omit to use the configured image)
        image: Option<String>,
        /// Target monitor name (e.g., eDP-1, HDMI-A-1), or omit to apply to all connected monitors
        #[arg(short, long)]
        monitor: Option<String>,
        /// Depth model name (depth-anything-v3-base, midas-small, depth-pro-q4) or path to .onnx file
        #[arg(long)]
        model: Option<String>,
        /// Enable 3D inpainting for true parallax with occlusion
        #[arg(long)]
        inpaint: bool,
        /// Disable 3D inpainting mode
        #[arg(long)]
        no_inpaint: bool,
        /// Invert depth interpretation (near ↔ far)
        #[arg(long)]
        invert_depth: bool,
        /// Disable depth inversion
        #[arg(long)]
        no_invert_depth: bool,
        /// Parallax strength for both axes (default: 0.02)
        #[arg(short, long)]
        strength: Option<f64>,
        /// Parallax strength on X axis (default: 0.02)
        #[arg(long)]
        strength_x: Option<f64>,
        /// Parallax strength on Y axis (default: 0.02)
        #[arg(long)]
        strength_y: Option<f64>,
        /// Enable smooth easing animation
        #[arg(long)]
        smooth_animation: bool,
        /// Disable smooth easing animation
        #[arg(long)]
        no_smooth_animation: bool,
        /// Animation speed multiplier (default: 0.02)
        #[arg(long)]
        animation_speed: Option<f64>,
        /// Target frame rate: 30 or 60 (default: 60)
        #[arg(long)]
        fps: Option<Fps>,
        /// Delay in ms before animation starts after mouse enters (default: 150)
        #[arg(long)]
        active_delay: Option<f64>,
        /// Idle timeout in ms before animation stops (default: 500)
        #[arg(long)]
        idle_timeout: Option<f64>,
        /// Force regenerate depth map and inpaint mesh
        #[arg(long)]
        regenerate: bool,
    },

    /// Start daemons for configured monitors.
    ///
    /// Skips monitors that are already running. Use `set` to reload a running daemon
    /// with new settings, or `stop` first to force a fresh start.
    Daemon {
        /// Target monitor name, or omit to start all configured monitors
        #[arg(short, long)]
        monitor: Option<String>,
        /// Force regenerate depth map and inpaint mesh
        #[arg(long)]
        regenerate: bool,
        /// Enable verbose logging
        #[arg(short, long)]
        verbose: bool,
    },

    /// Stop wallpaper daemons.
    Stop {
        /// Monitor name to stop, or omit to stop all
        #[arg(short, long)]
        monitor: Option<String>,
    },

    /// List connected Wayland monitors and their wallpaper status.
    ListMonitors,

    /// Pre-generate depth map and optionally inpaint mesh without starting a daemon.
    Pregenerate {
        /// Path to the wallpaper image
        image: String,
        /// Depth model name (depth-anything-v3-base, midas-small, depth-pro-q4) or path
        #[arg(long)]
        model: Option<String>,
        /// Enable 3D inpainting for true parallax with occlusion
        #[arg(long)]
        inpaint: bool,
        /// Invert depth interpretation (near ↔ far)
        #[arg(long)]
        invert_depth: bool,
        /// Force regenerate depth map and inpaint mesh
        #[arg(long)]
        regenerate: bool,
        /// Enable verbose logging
        #[arg(short, long)]
        verbose: bool,
    },

    /// Manage the depth map and inpaint mesh cache.
    Cache {
        /// Clear all cached depth maps and meshes
        #[arg(long, group = "action")]
        clear: bool,
        /// List cached wallpapers
        #[arg(long, group = "action")]
        list: bool,
    },

    /// Download ONNX depth models and 3D inpainting networks.
    DownloadModel {
        /// Model to download: 'depth-anything-v3-base', 'midas-small', 'depth-pro-q4', or 'inpaint'
        model: Option<String>,
    },

    /// Internal: hidden subcommand that runs as the actual daemon process.
    #[command(hide = true)]
    DaemonRun {
        #[arg(short, long)]
        wallpaper: String,
        #[arg(short, long)]
        monitor: String,
        #[arg(long, default_value = "0.02")]
        strength_x: f64,
        #[arg(long, default_value = "0.02")]
        strength_y: f64,
        #[arg(long, default_value_t = true)]
        smooth_animation: bool,
        #[arg(long)]
        no_smooth_animation: bool,
        #[arg(long, default_value = "0.05")]
        animation_speed: f64,
        #[arg(long, default_value = "60")]
        fps: u32,
        #[arg(long, default_value = "150")]
        active_delay: f64,
        #[arg(long, default_value = "5000")]
        idle_timeout: f64,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        regenerate: bool,
        #[arg(long)]
        invert_depth: bool,
        #[arg(long)]
        inpaint: bool,
        #[arg(long, default_value = "python3")]
        inpaint_python: String,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        None => {
            Cli::parse_from(["waydeeper", "--help"]);
            Ok(())
        }
        Some(command) => handle_command(command),
    }
}

fn handle_command(command: Commands) -> Result<()> {
    match command {
        Commands::Set {
            image,
            strength,
            strength_x,
            strength_y,
            monitor,
            smooth_animation,
            no_smooth_animation,
            animation_speed,
            fps,
            active_delay,
            idle_timeout,
            model,
            regenerate,
            invert_depth,
            no_invert_depth,
            inpaint,
            no_inpaint,
        } => cmd_set(
            image.as_deref(),
            AnimationParams {
                strength,
                strength_x,
                strength_y,
                smooth_animation,
                no_smooth_animation,
                animation_speed,
                fps,
                active_delay,
                idle_timeout,
            },
            monitor.as_deref(),
            model.as_deref(),
            regenerate,
            invert_depth,
            no_invert_depth,
            inpaint,
            no_inpaint,
        ),

        Commands::Daemon {
            monitor,
            regenerate,
            verbose,
        } => cmd_daemon(monitor.as_deref(), regenerate, verbose),

        Commands::Stop { monitor } => cmd_stop(monitor.as_deref()),
        Commands::ListMonitors => cmd_list_monitors(),

        Commands::Pregenerate {
            image,
            verbose,
            model,
            regenerate,
            inpaint,
            invert_depth,
        } => cmd_pregenerate(&image, verbose, model.as_deref(), regenerate, inpaint, invert_depth),

        Commands::Cache { clear, list } => handle_cache_command(clear, list),
        Commands::DownloadModel { model } => cmd_download_model(model.as_deref()),

        Commands::DaemonRun {
            wallpaper,
            monitor,
            strength_x,
            strength_y,
            smooth_animation,
            no_smooth_animation,
            animation_speed,
            fps,
            active_delay,
            idle_timeout,
            model,
            regenerate,
            invert_depth,
            inpaint,
            inpaint_python,
        } => cmd_daemon_run(
            &wallpaper,
            &monitor,
            strength_x,
            strength_y,
            smooth_animation,
            no_smooth_animation,
            animation_speed,
            fps,
            active_delay,
            idle_timeout,
            model.as_deref(),
            regenerate,
            invert_depth,
            inpaint,
            &inpaint_python,
        ),
    }
}

fn handle_cache_command(clear: bool, list: bool) -> Result<()> {
    if clear {
        cmd_cache_clear()
    } else if list {
        cmd_cache_list()
    } else {
        println!("Use --clear to clear cache or --list to list cached wallpapers.");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// spawn_daemon: launches a daemon-run subprocess
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn spawn_daemon(
    wallpaper_path: &str,
    monitor: &str,
    params: &ResolvedAnimationParams,
    model_path: Option<&str>,
    regenerate: bool,
    invert_depth: bool,
    use_inpaint: bool,
    inpaint_python: &str,
    verbose: bool,
) -> Result<std::process::Child> {
    let executable = std::env::current_exe()?;

    let mut command = std::process::Command::new(&executable);
    command
        .arg("daemon-run")
        .arg("--wallpaper").arg(wallpaper_path)
        .arg("--monitor").arg(monitor)
        .arg("--strength-x").arg(params.strength_x.to_string())
        .arg("--strength-y").arg(params.strength_y.to_string())
        .arg("--animation-speed").arg(params.animation_speed.to_string())
        .arg("--fps").arg(params.fps.to_string())
        .arg("--active-delay").arg(params.active_delay.to_string())
        .arg("--idle-timeout").arg(params.idle_timeout.to_string());

    if let Some(path) = model_path {
        command.arg("--model").arg(path);
    }
    if regenerate       { command.arg("--regenerate"); }
    if invert_depth     { command.arg("--invert-depth"); }
    if use_inpaint      { command.arg("--inpaint"); }
    if params.smooth_animation {
        command.arg("--smooth-animation");
    } else {
        command.arg("--no-smooth-animation");
    }

    command.arg("--inpaint-python").arg(inpaint_python);

    if verbose {
        command.env("RUST_LOG", "debug");
    } else {
        command.env("RUST_LOG", "warn");
    }

    command.stdin(std::process::Stdio::null());
    command.stdout(std::process::Stdio::inherit());
    command.stderr(std::process::Stdio::inherit());

    Ok(command.spawn()?)
}

fn wait_for_daemon(monitor: &str, timeout_secs: u64, child: &mut std::process::Child) -> bool {
    let delay_ms = 500;
    let max_attempts = (timeout_secs * 1000 / delay_ms) as u32;
    let mut dots = 0;
    
    for _ in 0..max_attempts {
        std::thread::sleep(Duration::from_millis(delay_ms));
        
        match child.try_wait() {
            Ok(Some(status)) => {
                println!();
                println!("Daemon process exited early with status: {}", status);
                return false;
            }
            Ok(None) => {}
            Err(e) => {
                println!();
                println!("Failed to check daemon process status: {}", e);
                return false;
            }
        }
        
        if let Ok(client) = DaemonClient::new(monitor) {
            if client.is_running() {
                println!();
                return true;
            }
        }
        dots = (dots + 1) % 4;
        let spinner = match dots {
            0 => "-",
            1 => "\\",
            2 => "|",
            _ => "/",
        };
        print!("\rWaiting for daemon {}    ", spinner);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();
    false
}

// ---------------------------------------------------------------------------
// send_reload: sends RELOAD IPC to existing daemon
// ---------------------------------------------------------------------------

fn send_reload(
    monitor: &str,
    wallpaper_path: &str,
    params: &ResolvedAnimationParams,
    model_path: Option<&str>,
    regenerate: bool,
    invert_depth: bool,
    use_inpaint: bool,
    inpaint_python: &str,
) -> Result<()> {
    let client = DaemonClient::new(monitor)?;
    let reload_params = ReloadParams {
        wallpaper_path: wallpaper_path.to_string(),
        strength_x: params.strength_x,
        strength_y: params.strength_y,
        smooth_animation: params.smooth_animation,
        animation_speed: params.animation_speed,
        fps: params.fps,
        active_delay_ms: params.active_delay,
        idle_timeout_ms: params.idle_timeout,
        invert_depth,
        use_inpaint,
        model_path: model_path.map(|s| s.to_string()),
        regenerate,
        inpaint_python: inpaint_python.to_string(),
    };
    let response = client.send_command("RELOAD", json!(reload_params), Duration::from_secs(10))?;
    if !response.success {
        return Err(anyhow!("Reload rejected: {:?}", response.error));
    }

    // Always poll STATUS for progress and logs
    let max_attempts = 600; // 5 minutes at 500ms intervals
    for _ in 0..max_attempts {
        std::thread::sleep(Duration::from_millis(500));
        match client.send_command("STATUS", serde_json::Value::Null, Duration::from_secs(2)) {
            Ok(status) => {
                if let Some(logs) = status.result.as_ref().and_then(|r| r.get("logs")) {
                    if let Some(log_array) = logs.as_array() {
                        for log_entry in log_array {
                            if let Some(msg) = log_entry.as_str() {
                                println!("  [daemon] {}", msg);
                            }
                        }
                    }
                }
                let complete = status.result.as_ref()
                    .and_then(|r| r.get("complete"))
                    .and_then(|r| r.as_bool())
                    .unwrap_or(false);
                if complete {
                    return Ok(());
                }
            }
            Err(_) => {
                return Err(anyhow!("Lost connection to daemon during reload"));
            }
        }
    }
    Err(anyhow!("Reload timed out"))
}

// ---------------------------------------------------------------------------
// cmd_set
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cmd_set(
    image: Option<&str>,
    animation_params: AnimationParams,
    monitor: Option<&str>,
    model: Option<&str>,
    regenerate: bool,
    invert_depth: bool,
    no_invert_depth: bool,
    use_inpaint: bool,
    no_inpaint: bool,
) -> Result<()> {
    let image_path_string = match image {
        Some(path) => {
            let image_path = std::fs::canonicalize(path)
                .map_err(|_| anyhow!("Image not found: {}", path))?;
            Some(image_path.to_string_lossy().to_string())
        }
        None => None,
    };

    let model_path = model
        .map(|name| models::get_model_path(name).map(|path| path.to_string_lossy().to_string()))
        .transpose()?;

    // Determine target monitors
    let connected_outputs = wayland::list_connected_outputs();
    let target_monitors: Vec<String> = match monitor {
        Some(name) => vec![name.to_string()],
        None => {
            if connected_outputs.is_empty() {
                return Err(anyhow!("No monitors detected. Specify a monitor with -m."));
            }
            connected_outputs
        }
    };

    let mut reloaded = 0;
    let mut started = 0;
    let mut failed = 0;

    for monitor_id in &target_monitors {
        let saved_config = config::load_config().unwrap_or_default();
        let existing_config = saved_config.monitors.get(monitor_id).cloned();

        // Resolve image path for this monitor
        let img_path = match &image_path_string {
            Some(p) => p.clone(),
            None => {
                match existing_config.as_ref().and_then(|c| c.wallpaper_path.as_ref()) {
                    Some(p) if std::path::Path::new(p).exists() => p.clone(),
                    _ => {
                        println!("Skipping monitor {}: no image specified and no configured wallpaper", monitor_id);
                        failed += 1;
                        continue;
                    }
                }
            }
        };

        // Resolve animation params for this monitor
        let mon_config = existing_config.unwrap_or_default();
        let resolved_params = animation_params.resolve(&mon_config);

        // Resolve model for this monitor
        let eff_model = model_path.as_deref().or(mon_config.model_path.as_deref());

        // Resolve boolean flags: use explicit values if given, otherwise fall back to existing config
        let do_invert_depth = if invert_depth {
            true
        } else if no_invert_depth {
            false
        } else {
            mon_config.invert_depth
        };
        let do_use_inpaint = if use_inpaint {
            true
        } else if no_inpaint {
            false
        } else {
            mon_config.use_inpaint
        };
        let inpaint_python = &mon_config.inpaint_python;

        // Update config with only explicitly provided parameters
        {
            let mut config = config::load_config().unwrap_or_default();
            let monitor_config = config
                .monitors
                .entry(monitor_id.clone())
                .or_default();

            if image.is_some() {
                monitor_config.wallpaper_path = Some(img_path.clone());
            }

            // For animation params: only override if explicitly provided
            if let Some(strength) = animation_params.strength {
                monitor_config.strength_x = strength;
                monitor_config.strength_y = strength;
            }
            if let Some(sx) = animation_params.strength_x {
                monitor_config.strength_x = sx;
            }
            if let Some(sy) = animation_params.strength_y {
                monitor_config.strength_y = sy;
            }

            if animation_params.smooth_animation {
                monitor_config.smooth_animation = true;
            }
            if animation_params.no_smooth_animation {
                monitor_config.smooth_animation = false;
            }
            if let Some(speed) = animation_params.animation_speed {
                monitor_config.animation_speed = speed;
            }
            if let Some(fps_val) = animation_params.fps {
                monitor_config.fps = fps_val.value();
            }
            if let Some(delay) = animation_params.active_delay {
                monitor_config.active_delay_ms = delay;
            }
            if let Some(idle) = animation_params.idle_timeout {
                monitor_config.idle_timeout_ms = idle;
            }
            if let Some(ref path) = model_path {
                monitor_config.model_path = Some(path.clone());
            }
            if invert_depth {
                monitor_config.invert_depth = true;
            }
            if no_invert_depth {
                monitor_config.invert_depth = false;
            }
            if use_inpaint {
                monitor_config.use_inpaint = true;
            }
            if no_inpaint {
                monitor_config.use_inpaint = false;
            }

            config::save_config(&config)?;
        }

        // Reload existing daemon or spawn new one — daemon handles ALL heavy lifting
        if let Ok(client) = DaemonClient::new(monitor_id) {
            if client.is_running() {
                println!("Reloading wallpaper daemon for monitor {}...", monitor_id);
                send_reload(
                    monitor_id,
                    &img_path,
                    &resolved_params,
                    eff_model,
                    regenerate,
                    do_invert_depth,
                    do_use_inpaint,
                    inpaint_python,
                )?;
                println!("Wallpaper daemon reloaded for monitor {}.", monitor_id);
                reloaded += 1;
                continue;
            }
        }

        println!("Starting wallpaper daemon for monitor {}...", monitor_id);

        let mut child = spawn_daemon(
            &img_path,
            monitor_id,
            &resolved_params,
            eff_model,
            regenerate,
            do_invert_depth,
            do_use_inpaint,
            inpaint_python,
            false,
        )?;

        if wait_for_daemon(monitor_id, 180, &mut child) {
            println!("Wallpaper daemon started for monitor {}.", monitor_id);
            started += 1;
        } else {
            println!("Daemon did not become responsive for monitor {}", monitor_id);
            failed += 1;
        }

        std::thread::sleep(Duration::from_millis(200));
    }

    if started  > 0 { println!("\nStarted {} daemon(s).", started); }
    if reloaded > 0 { println!("\nReloaded {} daemon(s).", reloaded); }
    if failed   > 0 { println!("Failed to start {} daemon(s).", failed); }

    if failed > 0 {
        Err(anyhow!("Some daemons failed to start"))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// cmd_daemon
// ---------------------------------------------------------------------------

fn cmd_daemon(monitor: Option<&str>, regenerate: bool, verbose: bool) -> Result<()> {
    let config = config::load_config()?;

    if config.monitors.is_empty() {
        println!("No monitors configured. Use 'waydeeper set <image>' first.");
        return Err(anyhow!("No monitors configured"));
    }

    let target_monitors: Vec<String> = match monitor {
        Some(name) => vec![name.to_string()],
        None => config.monitors.keys().cloned().collect(),
    };

    let connected_outputs = wayland::list_connected_outputs();
    let have_output_list = !connected_outputs.is_empty();
    if have_output_list {
        log::debug!("Connected outputs: {:?}", connected_outputs);
    }

    let mut started = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for monitor_id in &target_monitors {
        if have_output_list && !connected_outputs.contains(monitor_id) {
            if verbose {
                println!(
                    "Skipping monitor {}: not currently connected (available: {}).",
                    monitor_id,
                    connected_outputs.join(", ")
                );
            }
            continue;
        }

        let monitor_config = match config.monitors.get(monitor_id) {
            Some(c) => c,
            None => {
                println!("Monitor {} not configured. Use 'set' command first.", monitor_id);
                failed += 1;
                continue;
            }
        };

        let wallpaper_path = match &monitor_config.wallpaper_path {
            Some(path) if Path::new(path).exists() => path.clone(),
            _ => {
                println!("Skipping monitor {}: invalid wallpaper path", monitor_id);
                failed += 1;
                continue;
            }
        };

        let resolved_params = AnimationParams {
            strength: None,
            strength_x: None,
            strength_y: None,
            smooth_animation: false,
            no_smooth_animation: false,
            animation_speed: None,
            fps: None,
            active_delay: None,
            idle_timeout: None,
        }.resolve(monitor_config);

        let effective_model = monitor_config.model_path.as_deref();

        // Check if daemon is already running — always skip, never reload
        if let Ok(client) = DaemonClient::new(monitor_id) {
            if client.is_running() {
                println!("Daemon for monitor {} is already running. Use 'set' to reload with new config, or 'stop' first.", monitor_id);
                skipped += 1;
                continue;
            }
        }

        if verbose {
            println!("Starting daemon for monitor {}: {}", monitor_id, wallpaper_path);
        }

        match spawn_daemon(
            &wallpaper_path,
            monitor_id,
            &resolved_params,
            effective_model,
            regenerate,
            monitor_config.invert_depth,
            monitor_config.use_inpaint,
            &monitor_config.inpaint_python,
            verbose,
        ) {
            Ok(mut child) => {
                if wait_for_daemon(monitor_id, 180, &mut child) {
                    println!("Started daemon for monitor {}.", monitor_id);
                    started += 1;
                } else {
                    println!("Daemon for monitor {} did not become responsive.", monitor_id);
                    failed += 1;
                }
            }
            Err(error) => {
                println!("Failed to spawn daemon for monitor {}: {}", monitor_id, error);
                failed += 1;
            }
        }

        std::thread::sleep(Duration::from_millis(200));
    }

    if started  > 0 { println!("\nStarted {} daemon(s).", started); }
    if skipped  > 0 { println!("\nSkipped {} daemon(s) already running.", skipped); }
    if failed   > 0 { println!("Failed to start {} daemon(s).", failed); }

    if failed > 0 {
        Err(anyhow!("Some daemons failed to start"))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// cmd_daemon_run (internal)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cmd_daemon_run(
    wallpaper: &str,
    monitor: &str,
    strength_x: f64,
    strength_y: f64,
    smooth_animation: bool,
    no_smooth_animation: bool,
    animation_speed: f64,
    fps: u32,
    active_delay: f64,
    idle_timeout: f64,
    model: Option<&str>,
    regenerate: bool,
    invert_depth: bool,
    use_inpaint: bool,
    inpaint_python: &str,
) -> Result<()> {
    let smooth_animation = if no_smooth_animation { false } else { smooth_animation };
    
    let mut daemon = DepthWallpaperDaemon::new()?;
    daemon.load_configuration();

    daemon.run_daemon(
        wallpaper,
        monitor,
        strength_x,
        strength_y,
        smooth_animation,
        animation_speed,
        fps,
        active_delay,
        idle_timeout,
        model,
        regenerate,
        invert_depth,
        use_inpaint,
        inpaint_python,
    )
}

// ---------------------------------------------------------------------------
// cmd_stop / cmd_list_monitors
// ---------------------------------------------------------------------------

fn cmd_stop(monitor: Option<&str>) -> Result<()> {
    match monitor {
        Some(name) => {
            let client = DaemonClient::new(name)?;
            if client.is_running() {
                println!("Stopping wallpaper daemon for monitor {}...", name);
                if ipc::stop_daemon(name, Duration::from_secs(5))? {
                    println!("Wallpaper daemon stopped for monitor {}.", name);
                    Ok(())
                } else {
                    println!("Failed to stop daemon for monitor {}.", name);
                    Err(anyhow!("Failed to stop daemon"))
                }
            } else {
                println!("No wallpaper daemon is running for monitor {}.", name);
                Ok(())
            }
        }
        None => {
            let running = ipc::list_running_daemons()?;
            if running.is_empty() {
                println!("No wallpaper daemon is running.");
            } else {
                println!("Stopping {} wallpaper daemon(s)...", running.len());
                let results = ipc::stop_all_daemons(Duration::from_secs(5))?;
                let stopped = results.values().filter(|&&success| success).count();
                println!("Stopped {} wallpaper daemon(s).", stopped);
            }
            Ok(())
        }
    }
}

fn cmd_list_monitors() -> Result<()> {
    let config = config::load_config().unwrap_or_default();
    let running = ipc::list_running_daemons().unwrap_or_default();

    if config.monitors.is_empty() {
        println!("No monitors configured.");
        println!("Use 'waydeeper set <image> -m <monitor>' to configure.");
    } else {
        println!("Configured wallpapers:");
        println!("{}", "-".repeat(60));
        for (monitor_id, monitor_config) in &config.monitors {
            let wallpaper = monitor_config.wallpaper_path.as_deref().unwrap_or("Not set");
            let status = if running.contains(&monitor_id.to_string()) { "running" } else { "stopped" };
            let mode = if monitor_config.use_inpaint { "3D inpaint" } else { "flat depth-warp" };

            println!("  Monitor {}:", monitor_id);
            println!("    Status:       {}", status);
            println!("    Wallpaper:    {}", wallpaper);
            println!("    Mode:         {}", mode);
            println!("    Strength X:   {}", monitor_config.strength_x);
            println!("    Strength Y:   {}", monitor_config.strength_y);
            println!("    Smooth anim:  {}", monitor_config.smooth_animation);
            println!("    FPS:          {}", monitor_config.fps);
            println!("    Invert depth: {}", monitor_config.invert_depth);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// cmd_pregenerate
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cmd_pregenerate(
    image: &str,
    _verbose: bool,
    model: Option<&str>,
    regenerate: bool,
    use_inpaint: bool,
    invert_depth: bool,
) -> Result<()> {
    let image_path = std::fs::canonicalize(image)
        .map_err(|_| anyhow!("Image not found: {}", image))?;
    let image_path_string = image_path.to_string_lossy().to_string();

    let model_path = model
        .map(|name| models::get_model_path(name).map(|path| path.to_string_lossy().to_string()))
        .transpose()?;

    let model_display = model.unwrap_or("auto");
    println!(
        "Generating depth map for {} using model '{}'...",
        image_path_string, model_display
    );

    let mut daemon = DepthWallpaperDaemon::new()?;
    daemon.load_configuration();

    let depth_path = match daemon.pregenerate_depth_map(&image_path_string, model_path.as_deref(), regenerate) {
        Ok(p) => {
            if let Some(ref estimator) = daemon.depth_estimator {
                println!("Depth map generated (model: {}): {}", estimator.model_name, p);
            } else {
                println!("Depth map generated: {}", p);
            }
            p
        }
        Err(error) => {
            println!("Failed to generate depth map: {}", error);
            return Err(error);
        }
    };

    if use_inpaint {
        println!("Generating 3D inpaint mesh...");
        match daemon.ensure_ply_exists(
            &image_path_string,
            &depth_path,
            "python3",
            regenerate,
            invert_depth,
        ) {
            Ok(ply) => println!("Inpaint mesh ready: {}", ply),
            Err(e)  => {
                println!("Inpainting failed: {}", e);
                return Err(e);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// cmd_cache_clear / cmd_cache_list
// ---------------------------------------------------------------------------

fn cmd_cache_clear() -> Result<()> {
    let mut daemon = DepthWallpaperDaemon::new()?;
    daemon.load_configuration();
    daemon.clear_cache()?;
    println!("Cache cleared.");
    Ok(())
}

fn cmd_cache_list() -> Result<()> {
    let mut daemon = DepthWallpaperDaemon::new()?;
    daemon.load_configuration();
    daemon.list_cached_wallpapers()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Proxy-aware download helper
// ---------------------------------------------------------------------------

/// Detect HTTP_PROXY/HTTPS_PROXY/ALL_PROXY from environment and build a ureq
/// agent that routes through the proxy.  Respects NO_PROXY for host exclusion.
fn make_proxy_agent(url: &str) -> ureq::Agent {
    use ureq::AgentBuilder;

    // Determine the scheme of the target URL to pick the right proxy env var.
    let lower = url.to_lowercase();
    let proxy_envs: &[&str] = if lower.starts_with("https") {
        &["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]
    } else {
        &["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]
    };

    let proxy_url = proxy_envs
        .iter()
        .find_map(|k| std::env::var(k).ok())
        .filter(|u| !u.is_empty());

    if let Some(pu) = proxy_url {
        // Check NO_PROXY: if the target host matches, skip proxy.
        if let Ok(no_proxy) = std::env::var("NO_PROXY")
            .or_else(|_| std::env::var("no_proxy"))
        {
            if !no_proxy.is_empty() {
                // Extract hostname from target URL.
                let host = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))
                    .and_then(|s| s.split('/').next())
                    .unwrap_or("");
                if no_proxy.split(',').any(|entry| {
                    let entry = entry.trim();
                    if entry == "*" { return true; }
                    host == entry || host.ends_with(&format!(".{}", entry))
                }) {
                    log::debug!("Host {} excluded by NO_PROXY", host);
                    return ureq::agent();
                }
            }
        }

        log::info!("Using proxy: {}", pu);
        match ureq::Proxy::new(&pu) {
            Ok(proxy) => AgentBuilder::new().proxy(proxy).build(),
            Err(e) => {
                log::warn!("Failed to configure proxy {}: {}", pu, e);
                ureq::agent()
            }
        }
    } else {
        ureq::agent()
    }
}

/// Download a file from `url` to `dest` with proxy support and progress bar.
fn download_with_progress(url: &str, dest: &std::path::Path, label: &str) -> Result<()> {
    let agent = make_proxy_agent(url);
    let response = agent.get(url).call()
        .map_err(|e| anyhow!("Failed to download {}: {}", label, e))?;

    let total_size: u64 = response
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(dest)?;
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 65536];

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 { break; }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;

        if total_size > 0 {
            let percent = (downloaded * 100 / total_size).min(100);
            let filled = 30 * percent as usize / 100;
            let bar = "=".repeat(filled) + &"-".repeat(30 - filled);
            let mb  = downloaded as f64 / (1024.0 * 1024.0);
            let tmb = total_size as f64 / (1024.0 * 1024.0);
            print!("\r  [{}] {}% ({:.1}/{:.1} MB)", bar, percent, mb, tmb);
            std::io::stdout().flush()?;
        }
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// cmd_download_model
// ---------------------------------------------------------------------------

fn cmd_download_model(model_name: Option<&str>) -> Result<()> {
    use crate::models;

    let models_dir = config::models_dir();
    std::fs::create_dir_all(&models_dir)?;

    // "inpaint" is a special shortcut
    if model_name == Some("inpaint") {
        return cmd_download_inpaint_models();
    }

    // Download a specific named model (skip if already present)
    if let Some(name) = model_name {
        let model_info = models::get_model(name)?.clone();
        return download_depth_model(&model_info);
    }

    // Interactive flow: depth model first, then ask about inpainting
    let selected_name = models::prompt_model_selection()?;
    let model_info = models::get_model(&selected_name)?.clone();

    if models::depth_model_present(&selected_name) {
        println!("{} already installed, skipping download.", model_info.name);
    } else {
        download_depth_model(&model_info)?;
    }

    // Then ask about inpainting
    if models::inpaint_models_present() {
        println!("\nInpainting models already installed.");
    } else {
        print!("\nDownload 3D inpainting models? (~250 MB) [y/N]: ");
        std::io::Write::flush(&mut std::io::stdout())?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "y" {
            cmd_download_inpaint_models()?;
        }
    }

    Ok(())
}

fn download_depth_model(model_info: &models::ModelInfo) -> Result<()> {
    use crate::models::ModelFormat;

    let models_dir = config::models_dir();
    std::fs::create_dir_all(&models_dir)?;

    if models::depth_model_present(model_info.name) {
        println!("{} already installed.", model_info.name);
        return Ok(());
    }

    let (dest, download_path, temp_file) = match model_info.format {
        ModelFormat::Directory => {
            let model_dir = models_dir.join(model_info.name);
            std::fs::create_dir_all(&model_dir)?;
            let dest = model_dir.join("model.onnx");
            (dest.clone(), dest.clone(), None)
        }
        ModelFormat::Zip => {
            let dest = models_dir.join(format!("{}.onnx", model_info.name));
            let temp = models_dir.join(format!("{}_download.zip", model_info.name));
            (dest.clone(), temp.clone(), Some(temp))
        }
        ModelFormat::Onnx => {
            let dest = models_dir.join(format!("{}.onnx", model_info.name));
            (dest.clone(), dest.clone(), None)
        }
    };

    println!("\nDownloading {} model...", model_info.name);

    download_with_progress(model_info.url, &download_path, model_info.name)?;

    if let ModelFormat::Zip = model_info.format {
        let zip_path = temp_file.as_ref().unwrap();
        if let Some(extracted_name) = model_info.extracted_filename {
            println!("Extracting {}...", extracted_name);

            let zip_file = std::fs::File::open(zip_path)?;
            let mut archive = zip::ZipArchive::new(zip_file)?;

            let mut target_member = None;
            for index in 0..archive.len() {
                let file = archive.by_index(index)?;
                if file.name().ends_with(extracted_name) {
                    target_member = Some(index);
                    break;
                }
            }

            let index = target_member
                .ok_or_else(|| anyhow!("{} not found in zip", extracted_name))?;

            let mut zip_content = archive.by_index(index)?;
            let mut output_file = std::fs::File::create(&dest)?;
            std::io::copy(&mut zip_content, &mut output_file)?;
            std::fs::remove_file(zip_path)?;
        }
    }

    // Download extra files for directory-format models
    for (extra_name, extra_url) in models::MODEL_EXTRA_FILES
        .iter()
        .find(|(name, _)| *name == model_info.name)
        .map(|(_, files)| *files)
        .unwrap_or(&[])
    {
        let extra_dest = models_dir.join(model_info.name).join(extra_name);
        println!("Downloading {}...", extra_name);
        download_with_progress(extra_url, &extra_dest, extra_name)?;
    }

    println!("Model saved to {}", dest.display());
    println!("Done!");
    Ok(())
}

fn cmd_download_inpaint_models() -> Result<()> {
    let inpaint_dir = models::inpaint_models_dir();
    std::fs::create_dir_all(&inpaint_dir)?;

    for (filename, url) in models::INPAINT_URLS {
        let dest = inpaint_dir.join(filename);
        if dest.exists() {
            println!("{} already present, skipping.", filename);
            continue;
        }

        println!("Downloading {}...", filename);
        download_with_progress(url, &dest, filename)?;
        println!("  Saved to {}", dest.display());
    }

    println!("\nAll inpainting models downloaded to {}", inpaint_dir.display());
    println!("Run: waydeeper set <image> --inpaint");
    Ok(())
}
