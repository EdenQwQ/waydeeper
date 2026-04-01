use anyhow::{Context, Result};
use blake2::{Blake2b, Digest};
use digest::consts::U16;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

type Blake2b128 = Blake2b<U16>;

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub original_path: String,
    pub modification_time: f64,
    pub width: u32,
    pub height: u32,
    pub model_name: String,
}

pub struct DepthCache {
    depth_directory: PathBuf,
    metadata_directory: PathBuf,
}

impl DepthCache {
    pub fn new(custom_directory: Option<&Path>) -> Result<Self> {
        let cache_directory = custom_directory
            .map(|path| path.to_path_buf())
            .unwrap_or_else(crate::config::cache_dir);

        std::fs::create_dir_all(&cache_directory)?;

        let depth_directory = cache_directory.join("depth");
        let metadata_directory = cache_directory.join("metadata");
        std::fs::create_dir_all(&depth_directory)?;
        std::fs::create_dir_all(&metadata_directory)?;

        Ok(Self {
            depth_directory,
            metadata_directory,
        })
    }

    pub fn compute_image_hash(&self, image_path: &Path, model_name: &str) -> Result<String> {
        let mut hasher = Blake2b128::new();
        let mut file = std::fs::File::open(image_path)
            .with_context(|| format!("Failed to open image: {}", image_path.display()))?;
        std::io::copy(&mut file, &mut hasher)?;
        if !model_name.is_empty() {
            hasher.update(model_name.as_bytes());
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn get_depth_file_path(&self, image_hash: &str) -> PathBuf {
        self.depth_directory.join(format!("{}.png", image_hash))
    }

    pub fn get_metadata_file_path(&self, image_hash: &str) -> PathBuf {
        self.metadata_directory.join(format!("{}.json", image_hash))
    }

    pub fn get_cached_depth(
        &self,
        image_path: &Path,
        model_name: &str,
    ) -> Result<Option<Vec<f32>>> {
        if !image_path.exists() {
            return Ok(None);
        }

        let image_hash = self.compute_image_hash(image_path, model_name)?;
        let depth_path = self.get_depth_file_path(&image_hash);
        let metadata_path = self.get_metadata_file_path(&image_hash);

        if !depth_path.exists() || !metadata_path.exists() {
            return Ok(None);
        }

        let metadata_content = std::fs::read_to_string(&metadata_path)?;
        let metadata: CacheMetadata = serde_json::from_str(&metadata_content)?;

        let current_mtime = image_path
            .metadata()?
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        if (metadata.modification_time - current_mtime).abs() > 1.0 {
            log::debug!("Image {:?} modified, cache invalidated", image_path);
            return Ok(None);
        }

        if !model_name.is_empty()
            && !metadata.model_name.is_empty()
            && metadata.model_name != model_name
        {
            log::debug!(
                "Model mismatch for {:?}: {} != {}",
                image_path,
                metadata.model_name,
                model_name
            );
            return Ok(None);
        }

        let depth_map = load_depth_map(&depth_path)?;
        log::debug!("Cache hit for {:?} (model: {})", image_path, model_name);
        Ok(Some(depth_map))
    }

    #[allow(dead_code)]
    pub fn cache_depth(
        &self,
        image_path: &Path,
        depth_data: &[f32],
        width: u32,
        height: u32,
        model_name: &str,
    ) -> Result<()> {
        let image_hash = self.compute_image_hash(image_path, model_name)?;
        let depth_path = self.get_depth_file_path(&image_hash);
        let metadata_path = self.get_metadata_file_path(&image_hash);

        save_depth_map(depth_data, width, height, &depth_path)?;

        let mtime = image_path
            .metadata()?
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let metadata = CacheMetadata {
            original_path: image_path.to_string_lossy().to_string(),
            modification_time: mtime,
            width,
            height,
            model_name: model_name.to_string(),
        };

        let content = serde_json::to_string(&metadata)?;
        std::fs::write(&metadata_path, content)?;

        log::debug!(
            "Cached depth map for {:?} (model: {})",
            image_path,
            model_name
        );
        Ok(())
    }

    pub fn clear_cache(&self) -> Result<()> {
        for entry in std::fs::read_dir(&self.depth_directory)?.flatten() {
            if entry.file_type()?.is_file() {
                std::fs::remove_file(entry.path())?;
            }
        }
        for entry in std::fs::read_dir(&self.metadata_directory)?.flatten() {
            if entry.file_type()?.is_file() {
                std::fs::remove_file(entry.path())?;
            }
        }
        log::info!("Cache cleared");
        Ok(())
    }

    pub fn list_cached(&self) -> Result<Vec<CacheMetadata>> {
        let mut items = Vec::new();
        for entry in std::fs::read_dir(&self.metadata_directory)?.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(metadata) = serde_json::from_str::<CacheMetadata>(&content) {
                        items.push(metadata);
                    }
                }
            }
        }
        Ok(items)
    }
}

#[allow(dead_code)]
pub fn save_depth_map(depth_data: &[f32], width: u32, height: u32, path: &Path) -> Result<()> {
    let image_buffer: Vec<u16> = depth_data
        .iter()
        .map(|&value| (value.clamp(0.0, 1.0) * 65535.0) as u16)
        .collect();

    let image: image::ImageBuffer<image::Luma<u16>, Vec<u16>> =
        image::ImageBuffer::from_raw(width, height, image_buffer)
            .ok_or_else(|| anyhow::anyhow!("Failed to create depth image buffer"))?;

    image.save(path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Inpaint PLY cache
// ---------------------------------------------------------------------------

/// Cache that maps (image, depth_map, config_version) → PLY file on disk.
pub struct InpaintCache {
    ply_directory: PathBuf,
}

impl InpaintCache {
    pub fn new(custom_directory: Option<&Path>) -> Result<Self> {
        let base = custom_directory
            .map(|p| p.to_path_buf())
            .unwrap_or_else(crate::config::cache_dir);
        let ply_directory = base.join("inpaint");
        std::fs::create_dir_all(&ply_directory)?;
        Ok(Self { ply_directory })
    }

    /// Hash = blake2b(image_bytes || depth_file_bytes || config_version_tag).
    pub fn compute_hash(
        &self,
        image_path: &Path,
        depth_path: &Path,
        config_tag: &str,
    ) -> Result<String> {
        let mut hasher = Blake2b128::new();

        let mut f = std::fs::File::open(image_path)
            .with_context(|| format!("open image {}", image_path.display()))?;
        std::io::copy(&mut f, &mut hasher)?;

        let mut f = std::fs::File::open(depth_path)
            .with_context(|| format!("open depth {}", depth_path.display()))?;
        std::io::copy(&mut f, &mut hasher)?;

        hasher.update(config_tag.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn ply_path(&self, hash: &str) -> PathBuf {
        self.ply_directory.join(format!("{}.ply", hash))
    }

    /// Return the cached PLY path if it exists, otherwise None.
    pub fn get_cached_ply(
        &self,
        image_path: &Path,
        depth_path: &Path,
        config_tag: &str,
    ) -> Result<Option<PathBuf>> {
        let hash = self.compute_hash(image_path, depth_path, config_tag)?;
        let path = self.ply_path(&hash);
        if path.exists() {
            log::debug!("Inpaint cache hit: {}", path.display());
            Ok(Some(path))
        } else {
            Ok(None)
        }
    }

    /// Return the path where a new PLY should be written (creating parent dirs).
    pub fn ply_write_path(
        &self,
        image_path: &Path,
        depth_path: &Path,
        config_tag: &str,
    ) -> Result<PathBuf> {
        let hash = self.compute_hash(image_path, depth_path, config_tag)?;
        Ok(self.ply_path(&hash))
    }

    pub fn clear_cache(&self) -> Result<()> {
        for entry in std::fs::read_dir(&self.ply_directory)?.flatten() {
            if entry.file_type()?.is_file() {
                std::fs::remove_file(entry.path())?;
            }
        }
        log::info!("Inpaint cache cleared");
        Ok(())
    }
}

pub fn load_depth_map(path: &Path) -> Result<Vec<f32>> {
    let image = image::open(path)?;
    let gray = image.to_luma16();
    let (width, height) = gray.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for pixel in gray.pixels() {
        data.push(pixel[0] as f32 / 65535.0);
    }

    Ok(data)
}
