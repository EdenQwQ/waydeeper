use anyhow::{anyhow, Result};
use std::path::PathBuf;

use crate::config::models_dir;

// ---------------------------------------------------------------------------
// Depth estimation model registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Onnx,
    Zip,
    Directory,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub url: &'static str,
    pub format: ModelFormat,
    pub extracted_filename: Option<&'static str>,
}

pub const AVAILABLE_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "depth-anything-v3-base",
        description: "Balanced quality and speed, good for most use cases (default)",
        url: "https://huggingface.co/onnx-community/depth-anything-v3-base/resolve/main/onnx/model.onnx",
        format: ModelFormat::Directory,
        extracted_filename: None,
    },
    ModelInfo {
        name: "midas-small",
        description: "Lightweight and fast, lower quality",
        url: "https://github.com/rocksdanister/lively-ml-models/releases/download/v1.0.0.0/midas_small.zip",
        format: ModelFormat::Zip,
        extracted_filename: Some("model.onnx"),
    },
    ModelInfo {
        name: "depth-pro-q4",
        description: "Apple's Depth Pro model (4-bit quantized) - high quality, large file size, slow",
        url: "https://huggingface.co/onnx-community/DepthPro-ONNX/resolve/main/onnx/model_q4.onnx",
        format: ModelFormat::Onnx,
        extracted_filename: None,
    },
];

/// Additional files to download for directory-format models.
/// Maps model name → list of (filename, url) pairs.
pub const MODEL_EXTRA_FILES: &[(&str, &[(&str, &str)])] = &[
    ("depth-anything-v3-base", &[
        ("model.onnx_data", "https://huggingface.co/onnx-community/depth-anything-v3-base/resolve/main/onnx/model.onnx_data"),
    ]),
];

// ---------------------------------------------------------------------------
// Inpainting model constants
// ---------------------------------------------------------------------------

pub const INPAINT_MODEL_NAMES: &[&str] = &["edge-model", "depth-model", "color-model"];

pub const INPAINT_URLS: &[(&str, &str)] = &[
    ("edge-model.pth",
     "https://huggingface.co/spaces/Epoching/3D_Photo_Inpainting/resolve/e389e564fd2a55cfa4582be8c8239295d102aebd/checkpoints/edge-model.pth"),
    ("depth-model.pth",
     "https://huggingface.co/spaces/Epoching/3D_Photo_Inpainting/resolve/e389e564fd2a55cfa4582be8c8239295d102aebd/checkpoints/depth-model.pth"),
    ("color-model.pth",
     "https://huggingface.co/spaces/Epoching/3D_Photo_Inpainting/resolve/e389e564fd2a55cfa4582be8c8239295d102aebd/checkpoints/color-model.pth"),
];

/// Return the directory that holds the inpainting .pth weights.
pub fn inpaint_models_dir() -> PathBuf {
    models_dir().join("inpaint")
}

/// Check whether all three inpainting model .pth files are present.
pub fn inpaint_models_present() -> bool {
    let dir = inpaint_models_dir();
    INPAINT_MODEL_NAMES.iter().all(|name| dir.join(format!("{}.pth", name)).exists())
}

// ---------------------------------------------------------------------------
// Model lookup
// ---------------------------------------------------------------------------

/// Look up a known model by name from the registry.
pub fn get_model(name: &str) -> Result<&ModelInfo> {
    AVAILABLE_MODELS
        .iter()
        .find(|model| model.name == name)
        .ok_or_else(|| {
            let available: Vec<&str> = AVAILABLE_MODELS.iter()
                .map(|model| model.name)
                .collect();
            anyhow!(
                "Unknown model '{}'. Available models: {}",
                name,
                available.join(", ")
            )
        })
}

/// Check whether a named depth model is already downloaded.
pub fn depth_model_present(name: &str) -> bool {
    if let Ok(model) = get_model(name) {
        match model.format {
            ModelFormat::Directory => {
                let model_dir = models_dir().join(model.name);
                model_dir.join("model.onnx").exists()
            }
            ModelFormat::Onnx | ModelFormat::Zip => {
                let onnx_path = models_dir().join(format!("{}.onnx", model.name));
                onnx_path.exists()
            }
        }
    } else {
        false
    }
}

/// Get the path to a model's primary ONNX file.
fn model_primary_path(model: &ModelInfo) -> PathBuf {
    match model.format {
        ModelFormat::Directory => models_dir().join(model.name).join("model.onnx"),
        ModelFormat::Onnx | ModelFormat::Zip => models_dir().join(format!("{}.onnx", model.name)),
    }
}

/// Resolve a model name or path to an actual ONNX model file.
///
/// Accepts:
///   - A known model name (e.g. "depth-anything-v3-base") → looks in models dir
///   - An arbitrary name → looks for `name.onnx` or `name/model.onnx`
///   - An absolute or relative path to a .onnx file or directory
///   - If the path is a directory, looks for `model.onnx` inside it
pub fn get_model_path(name_or_path: &str) -> Result<PathBuf> {
    let models_directory = models_dir();

    // Case 1: known model in registry
    if let Ok(model) = get_model(name_or_path) {
        let onnx_path = model_primary_path(model);
        if onnx_path.exists() {
            return Ok(onnx_path);
        }
        return Err(anyhow!(
            "Model '{}' not downloaded yet.\n\
             File not found: {}\n\
             Run: waydeeper download-model {}",
            model.name,
            onnx_path.display(),
            model.name
        ));
    }

    // Case 2: treat as a path (absolute or relative)
    let candidate = PathBuf::from(name_or_path);
    if candidate.exists() {
        if candidate.is_dir() {
            // Directory: look for model.onnx inside
            let model_onnx = candidate.join("model.onnx");
            if model_onnx.exists() {
                return Ok(model_onnx);
            }
            // Any .onnx in the directory
            for e in std::fs::read_dir(&candidate)?.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|ext| ext == "onnx") {
                    return Ok(p);
                }
            }
            return Err(anyhow!(
                "Directory {} exists but contains no .onnx files.",
                candidate.display()
            ));
        }
        return Ok(candidate);
    }

    // Case 3: try in models directory (name as-is)
    let in_models_dir = models_directory.join(name_or_path);
    if in_models_dir.exists() {
        if in_models_dir.is_dir() {
            let model_onnx = in_models_dir.join("model.onnx");
            if model_onnx.exists() {
                return Ok(model_onnx);
            }
            for e in std::fs::read_dir(&in_models_dir)?.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|ext| ext == "onnx") {
                    return Ok(p);
                }
            }
            return Err(anyhow!(
                "Directory {} exists but contains no .onnx files.",
                in_models_dir.display()
            ));
        }
        return Ok(in_models_dir);
    }

    // Case 4: try appending .onnx in models directory
    if !name_or_path.ends_with(".onnx") {
        let with_ext = models_directory.join(format!("{}.onnx", name_or_path));
        if with_ext.exists() {
            return Ok(with_ext);
        }
    }

    Err(anyhow!(
        "Model not found: {}\n\
         Searched:\n  - {} (known model)\n  - {}\n  - {}\n\
         Run 'waydeeper download-model' to download a model, or specify a full path.",
        name_or_path,
        models_directory.join(format!("{}.onnx", name_or_path)).display(),
        PathBuf::from(name_or_path).display(),
        models_directory.join(name_or_path).display(),
    ))
}

/// Auto-detect a model file in the models directory.
/// Prefers depth-anything-v3-base, then midas-small, then falls back to any .onnx file found.
pub fn find_model_file() -> Result<PathBuf> {
    let models_directory = models_dir();

    // Prefer depth-anything-v3-base
    let default_path = models_directory.join("depth-anything-v3-base").join("model.onnx");
    if default_path.exists() {
        return Ok(default_path);
    }

    // Then midas-small
    let midas_path = models_directory.join("midas-small.onnx");
    if midas_path.exists() {
        return Ok(midas_path);
    }

    // Legacy midas
    let legacy_midas = models_directory.join("midas.onnx");
    if legacy_midas.exists() {
        return Ok(legacy_midas);
    }

    if models_directory.exists() {
        // Any .onnx file
        let mut onnx_files: Vec<PathBuf> = std::fs::read_dir(&models_directory)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "onnx"))
            .collect();
        onnx_files.sort();
        if let Some(first) = onnx_files.into_iter().next() {
            log::info!(
                "Default model 'depth-anything-v3-base' not found, using '{}'",
                first.file_stem().unwrap_or_default().to_string_lossy()
            );
            return Ok(first);
        }

        // Directory containing model.onnx
        for e in std::fs::read_dir(&models_directory)?.flatten() {
            let p = e.path();
            if p.is_dir() {
                let model_onnx = p.join("model.onnx");
                if model_onnx.exists() {
                    return Ok(model_onnx);
                }
            }
        }
    }

    // Legacy paths
    for path in &[PathBuf::from("models/model.onnx"), PathBuf::from("models/midas_small/model.onnx")] {
        if path.exists() {
            log::info!("Using legacy model at: {}", path.display());
            return Ok(path.to_path_buf());
        }
    }

    Err(anyhow!(
        "No depth estimation models found.\n\
         Models directory: {}\n\n\
         Run: waydeeper download-model",
        models_directory.display()
    ))
}

// ---------------------------------------------------------------------------
// Interactive model selection (for download-model)
// ---------------------------------------------------------------------------

/// Prompt the user to select a depth estimation model.
/// Shows which models are already installed.
pub fn prompt_model_selection() -> Result<String> {
    println!("Available depth estimation models:");
    println!("{}", "-".repeat(60));

    for (index, model) in AVAILABLE_MODELS.iter().enumerate() {
        let installed = if depth_model_present(model.name) { " (installed)" } else { "" };
        let default = if index == 0 { " (default)" } else { "" };
        println!("  {}. {}{}{}", index + 1, model.name, default, installed);
        println!("     {}", model.description);
    }

    println!("{}", "-".repeat(60));

    loop {
        print!(
            "Select model [1-{}, default: {}]: ",
            AVAILABLE_MODELS.len(),
            AVAILABLE_MODELS[0].name
        );
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            return Ok(AVAILABLE_MODELS[0].name.to_string());
        }

        if let Ok(index) = input.parse::<usize>() {
            if index >= 1 && index <= AVAILABLE_MODELS.len() {
                return Ok(AVAILABLE_MODELS[index - 1].name.to_string());
            } else {
                println!("Please enter a number between 1 and {}", AVAILABLE_MODELS.len());
                continue;
            }
        }

        if AVAILABLE_MODELS.iter().any(|m| m.name == input) {
            return Ok(input.to_string());
        }

        println!(
            "Invalid selection. Please enter a number (1-{}) or model name",
            AVAILABLE_MODELS.len()
        );
    }
}
