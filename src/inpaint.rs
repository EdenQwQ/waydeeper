//! 3D inpainting subprocess launcher.
//!
//! Finds `inpaint.py` (bundled at `inpainting/inpaint.py` relative to the
//! binary, or via `WAYDEEPER_INPAINT_SCRIPT` env var), then calls it as a
//! Python subprocess, streaming its stdout/stderr to the logger.

use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Configuration for one inpainting run.
pub struct InpaintConfig<'a> {
    pub image_path: &'a Path,
    pub depth_path: &'a Path,
    pub output_ply: &'a Path,
    pub models_dir: &'a Path,
    pub python:     &'a str,
    /// Max image dimension (pixels).
    pub longer_side: u32,
    /// Disparity threshold for depth-edge detection.
    pub depth_threshold: f32,
    /// Background fill thickness in pixels.
    pub background_thickness: u32,
    /// Context (foreground) thickness in pixels.
    pub context_thickness: u32,
    /// Border extrapolation thickness in pixels.
    pub extrapolation_thickness: u32,
    /// Invert depth map before mesh generation.
    pub invert_depth: bool,
}

impl<'a> InpaintConfig<'a> {
    /// A stable string that uniquely identifies these settings — used as part
    /// of the cache key.
    pub fn cache_tag(&self) -> String {
        format!(
            "v1_ls{}_dt{:.4}_bt{}_ct{}_et{}_inv{}",
            self.longer_side,
            self.depth_threshold,
            self.background_thickness,
            self.context_thickness,
            self.extrapolation_thickness,
            if self.invert_depth { 1 } else { 0 },
        )
    }
}

/// Find the inpainting script.
///
/// Search order:
///   1. `$WAYDEEPER_INPAINT_SCRIPT` env var
///   2. `<exe_dir>/scripts/inpaint.py`
///   3. `<exe_dir>/../share/waydeeper/scripts/inpaint.py` (Nix install)
///   4. `./scripts/inpaint.py` (dev tree fallback)
fn find_inpaint_script() -> Result<PathBuf> {
    // 1. Env var override
    if let Ok(path) = std::env::var("WAYDEEPER_INPAINT_SCRIPT") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
        return Err(anyhow!(
            "WAYDEEPER_INPAINT_SCRIPT is set to '{}' but the file does not exist.",
            path
        ));
    }

    // 2. Relative to the current executable
    if let Ok(exe) = std::env::current_exe() {
        // Same directory as binary
        let candidate = exe
            .parent()
            .unwrap_or(Path::new("/"))
            .join("scripts")
            .join("inpaint.py");
        if candidate.exists() {
            return Ok(candidate);
        }
        // Nix install path: <prefix>/share/waydeeper/scripts/inpaint.py
        let candidate2 = exe
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(Path::new("/"))
            .join("share")
            .join("waydeeper")
            .join("scripts")
            .join("inpaint.py");
        if candidate2.exists() {
            return Ok(candidate2);
        }
    }

    // 3. Dev tree fallback
    let dev = Path::new("scripts/inpaint.py");
    if dev.exists() {
        return Ok(dev.to_path_buf());
    }

    Err(anyhow!(
        "Cannot find scripts/inpaint.py.\n\
         Set WAYDEEPER_INPAINT_SCRIPT=/path/to/inpaint.py"
    ))
}

/// Run the Python inpainting pipeline.
///
/// Streams output lines to `log::info!` / `log::warn!`.
/// Returns `Err` if the process exits non-zero or the PLY is not produced.
pub fn run_inpainting(cfg: &InpaintConfig) -> Result<()> {
    let script = find_inpaint_script()?;

    // Check Python is available
    let python_check = std::process::Command::new(cfg.python)
        .arg("--version")
        .output();
    if python_check.is_err() {
        return Err(anyhow!(
            "Python interpreter '{}' not found.\n\
             Install Python 3 with torch, scipy, networkx and Pillow, then set \
             inpaint_python in your waydeeper config.",
            cfg.python
        ));
    }

    log::info!(
        "Running inpainting: {} {} → {}",
        cfg.python,
        script.display(),
        cfg.output_ply.display()
    );

    let mut cmd = std::process::Command::new(cfg.python);
    cmd.arg(&script)
        .arg("--image")
        .arg(cfg.image_path)
        .arg("--depth")
        .arg(cfg.depth_path)
        .arg("--output")
        .arg(cfg.output_ply)
        .arg("--models-dir")
        .arg(cfg.models_dir)
        .arg("--longer-side")
        .arg(cfg.longer_side.to_string())
        .arg("--depth-threshold")
        .arg(cfg.depth_threshold.to_string())
        .arg("--background-thickness")
        .arg(cfg.background_thickness.to_string())
        .arg("--context-thickness")
        .arg(cfg.context_thickness.to_string())
        .arg("--extrapolation-thickness")
        .arg(cfg.extrapolation_thickness.to_string());

    if cfg.invert_depth {
        cmd.arg("--invert-depth");
    }

    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn()
        .map_err(|e| anyhow!("Failed to spawn '{}': {}", cfg.python, e))?;

    // Stream stdout and stderr in separate threads, log them
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let stdout_thread = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            log::info!("[inpaint] {}", line);
        }
    });
    let stderr_thread = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            log::warn!("[inpaint] {}", line);
        }
    });

    let status = child.wait()
        .map_err(|e| anyhow!("Failed to wait for inpaint process: {}", e))?;

    let _ = stdout_thread.join();
    let _ = stderr_thread.join();

    if !status.success() {
        return Err(anyhow!(
            "Inpainting script exited with status: {}",
            status
        ));
    }

    if !cfg.output_ply.exists() {
        return Err(anyhow!(
            "Inpainting script succeeded but did not produce: {}",
            cfg.output_ply.display()
        ));
    }

    log::info!("Inpainting complete → {}", cfg.output_ply.display());
    Ok(())
}
