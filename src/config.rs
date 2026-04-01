use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    #[serde(default)]
    pub wallpaper_path: Option<String>,
    #[serde(default = "default_strength")]
    pub strength_x: f64,
    #[serde(default = "default_strength")]
    pub strength_y: f64,
    #[serde(default = "default_true")]
    pub smooth_animation: bool,
    #[serde(default = "default_animation_speed")]
    pub animation_speed: f64,
    #[serde(default = "default_fps")]
    pub fps: u32,
    #[serde(default = "default_active_delay")]
    pub active_delay_ms: f64,
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout_ms: f64,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub invert_depth: bool,
    #[serde(default)]
    pub use_inpaint: bool,
    #[serde(default = "default_python")]
    pub inpaint_python: String,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            wallpaper_path: None,
            strength_x: 0.02,
            strength_y: 0.02,
            smooth_animation: true,
            animation_speed: 0.02,
            fps: 60,
            active_delay_ms: 150.0,
            idle_timeout_ms: 500.0,
            model_path: None,
            invert_depth: false,
            use_inpaint: false,
            inpaint_python: "python3".to_string(),
        }
    }
}

fn default_strength() -> f64 { 0.02 }
fn default_true() -> bool { true }
fn default_animation_speed() -> f64 { 0.02 }
fn default_fps() -> u32 { 60 }
fn default_active_delay() -> f64 { 150.0 }
fn default_idle_timeout() -> f64 { 500.0 }
fn default_python() -> String { "python3".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct Config {
    #[serde(default)]
    pub monitors: HashMap<String, MonitorConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_directory: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
}


pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join(".config")
        .join("waydeeper")
}

pub fn config_file() -> PathBuf {
    config_dir().join("config.json")
}

pub fn cache_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join(".cache")
        .join("waydeeper")
}

pub fn models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join(".local")
        .join("share")
        .join("waydeeper")
        .join("models")
}

pub fn load_config() -> Result<Config> {
    let path = config_file();
    if !path.exists() {
        return Ok(Config::default());
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config from {}", path.display()))?;
    let config: Config = serde_json::from_str(&content)
        .with_context(|| "Failed to parse config JSON")?;
    Ok(config)
}

pub fn save_config(config: &Config) -> Result<()> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir)?;

    let clean = Config {
        monitors: config.monitors.clone(),
        cache_directory: config.cache_directory.clone(),
        model_path: config.model_path.clone(),
    };

    let path = config_file();
    let content = serde_json::to_string_pretty(&clean)?;
    std::fs::write(&path, content)
        .with_context(|| format!("Failed to write config to {}", path.display()))?;
    Ok(())
}

pub fn runtime_dir() -> Result<PathBuf> {
    let dir = std::env::temp_dir()
        .join(format!("waydeeper-{}", nix::unistd::getuid()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}
