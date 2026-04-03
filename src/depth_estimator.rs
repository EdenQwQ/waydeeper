use anyhow::{anyhow, Result};
use std::path::Path;

use crate::cache::DepthCache;
use crate::models::find_model_file;

pub struct DepthEstimator {
    pub model_path: String,
    pub model_name: String,
    model_loaded: bool,
    input_layout: InputLayout,
    model_input_size: (usize, usize),
}

#[derive(Debug, Clone, Copy)]
enum InputLayout {
    ChannelsFirst,
    ChannelsLast,
}

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

impl DepthEstimator {
    pub fn new(model_path: Option<&str>) -> Result<Self> {
        let path = match model_path {
            Some(path) => path.to_string(),
            None => find_model_file()?.to_string_lossy().to_string(),
        };

        let name = Path::new(&path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            model_path: path,
            model_name: name,
            model_loaded: false,
            input_layout: InputLayout::ChannelsFirst,
            model_input_size: (256, 256),
        })
    }

    pub fn ensure_initialized(&mut self) -> Result<()> {
        if self.model_loaded {
            return Ok(());
        }

        log::info!(
            "Loading depth estimation model: {} ({})",
            self.model_name,
            self.model_path
        );

        let dylib_path = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "libonnxruntime.so".to_string());

        if let Ok(builder) = ort::init_from(dylib_path) {
            builder.commit();
        }

        self.model_loaded = true;
        Ok(())
    }

    fn run_inference(&mut self, image_path: &str) -> Result<Vec<f32>> {
        self.ensure_initialized()?;

        let image = image::open(image_path)?;
        let rgb_image = image.to_rgb8();
        let (original_width, original_height) = rgb_image.dimensions();

        let mut session = ort::session::Session::builder()?
            .commit_from_file(&self.model_path)?;

        println!("Estimating depth...");

        // Read model input shape to determine layout and input dimensions
        let mut is_5d = false;
        if let Some(input) = session.inputs().first() {
            if let ort::value::ValueType::Tensor { shape, .. } = input.dtype() {
                log::info!("Model input shape: {:?}", shape);
                let ndim = shape.len();
                if ndim >= 4 {
                    // Find which dimension is 3 (channels)
                    // 4D: [N, C, H, W]  → dim1=3
                    // 4D: [N, H, W, C]  → dim3=3
                    // 5D: [N, V, C, H, W]  → dim2=3  (e.g. depth-anything-v3)
                    // Shape values of -1 indicate dynamic dimensions
                    let shape_positive = |i: usize| -> Option<i64> {
                        shape.get(i).copied().filter(|&v| v > 0)
                    };
                    
                    for i in 0..ndim {
                        if shape.get(i).copied().unwrap_or(0) == 3 {
                            if i == 1 && ndim == 4 {
                                // [N, C, H, W]
                                self.input_layout = InputLayout::ChannelsFirst;
                                self.model_input_size = (
                                    shape_positive(2).unwrap_or(256) as usize,
                                    shape_positive(3).unwrap_or(256) as usize,
                                );
                            } else if i == 3 && ndim == 4 {
                                // [N, H, W, C]
                                self.input_layout = InputLayout::ChannelsLast;
                                self.model_input_size = (
                                    shape_positive(1).unwrap_or(256) as usize,
                                    shape_positive(2).unwrap_or(256) as usize,
                                );
                            } else if i == 2 && ndim >= 5 {
                                // [N, V, C, H, W] — 5D with channels at dim 2
                                self.input_layout = InputLayout::ChannelsFirst;
                                self.model_input_size = (
                                    shape_positive(ndim - 2).unwrap_or(256) as usize,
                                    shape_positive(ndim - 1).unwrap_or(256) as usize,
                                );
                                is_5d = true;
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Now resize image using the model's actual input dimensions
        let (input_height, input_width) = self.model_input_size;
        log::info!("Resizing input to {}x{}", input_width, input_height);
        let resized = image::imageops::resize(
            &rgb_image,
            input_width as u32,
            input_height as u32,
            image::imageops::FilterType::Lanczos3,
        );

        let mut input_data: Vec<f32> = Vec::with_capacity(3 * input_height * input_width);

        match self.input_layout {
            InputLayout::ChannelsFirst => {
                for channel in 0..3 {
                    for y in 0..input_height {
                        for x in 0..input_width {
                            let pixel = resized.get_pixel(x as u32, y as u32);
                            let value = pixel[channel] as f32 / 255.0;
                            let normalized = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
                            input_data.push(normalized);
                        }
                    }
                }
            }
            InputLayout::ChannelsLast => {
                for y in 0..input_height {
                    for x in 0..input_width {
                        let pixel = resized.get_pixel(x as u32, y as u32);
                        for channel in 0..3 {
                            let value = pixel[channel] as f32 / 255.0;
                            let normalized = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
                            input_data.push(normalized);
                        }
                    }
                }
            }
        }

        let input_tensor = if is_5d {
            // 5D input: [1, 1, C, H, W] — single image as a single view
            let input_array = ndarray::Array5::from_shape_vec(
                (1, 1, 3, input_height, input_width),
                input_data,
            )
            .map_err(|error| anyhow!("Failed to create 5D input array: {}", error))?;
            ort::value::Tensor::<f32>::from_array(input_array)?
        } else {
            match self.input_layout {
                InputLayout::ChannelsFirst => {
                    // [N, C, H, W]
                    let input_array = ndarray::Array4::from_shape_vec(
                        (1, 3, input_height, input_width),
                        input_data,
                    )
                    .map_err(|error| anyhow!("Failed to create input array: {}", error))?;
                    ort::value::Tensor::<f32>::from_array(input_array)?
                }
                InputLayout::ChannelsLast => {
                    // [N, H, W, C]
                    let input_array = ndarray::Array4::from_shape_vec(
                        (1, input_height, input_width, 3),
                        input_data,
                    )
                    .map_err(|error| anyhow!("Failed to create input array: {}", error))?;
                    ort::value::Tensor::<f32>::from_array(input_array)?
                }
            }
        };

        let outputs = session.run(ort::inputs![input_tensor])?;

        let output_view: ndarray::ArrayViewD<f32> = outputs[0].try_extract_array()?;
        let output_data: Vec<f32> = output_view.iter().copied().collect();
        
        // Infer actual output dimensions from the data
        let output_shape = output_view.shape();
        log::info!("Model output shape: {:?}", output_shape);
        
        // Most depth models output [N, H, W] or [N, C, H, W]
        // Extract H, W from the last 2 dimensions
        let output_height = output_shape[output_shape.len() - 2];
        let output_width = output_shape[output_shape.len() - 1];
        log::info!("Detected output dimensions: {}x{}", output_width, output_height);

        // Depth models usually output the same spatial dimensions as their input
        let depth = postprocess_output(
            &output_data,
            (output_width, output_height),
            (original_width as usize, original_height as usize),
        )?;

        Ok(depth)
    }

    pub fn estimate_cached(
        &mut self,
        image_path: &str,
        cache: &DepthCache,
        force_regenerate: bool,
    ) -> Result<Vec<f32>> {
        let path = Path::new(image_path);

        if !force_regenerate {
            if let Some(cached) = cache.get_cached_depth(path, &self.model_name, &self.model_path)? {
                return Ok(cached);
            }
        }

        let depth = self.run_inference(image_path)?;

        let image = image::open(image_path)?;
        let width = image.width();
        let height = image.height();

        cache.cache_depth(path, &depth, width, height, &self.model_name, &self.model_path)?;
        Ok(depth)
    }
}

fn postprocess_output(output: &[f32], output_size: (usize, usize), target_size: (usize, usize)) -> Result<Vec<f32>> {
    let mut depth = output.to_vec();

    let mut sorted = depth.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let low_index = (sorted.len() as f32 * 0.01) as usize;
    let high_index = (sorted.len() as f32 * 0.99) as usize;
    let low = sorted.get(low_index).copied().unwrap_or(0.0);
    let high = sorted.get(high_index).copied().unwrap_or(1.0);
    let range = high - low + 1e-8;

    for value in depth.iter_mut() {
        *value = ((*value - low) / range).clamp(0.0, 1.0);
        *value = 1.0 - *value;
    }

    let (input_width, input_height) = output_size;
    let (target_width, target_height) = (target_size.0 as u32, target_size.1 as u32);

    let pixels: Vec<u8> = depth.iter().map(|value| (value * 255.0) as u8).collect();
    
    log::debug!(
        "Creating depth image buffer: {}x{} (expected {} pixels, got {})",
        input_width, input_height, input_width * input_height, pixels.len()
    );
    
    let image = image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(
        input_width as u32,
        input_height as u32,
        pixels,
    )
    .ok_or_else(|| anyhow::anyhow!(
        "Failed to create depth image buffer: dimensions {}x{} don't match data length {}",
        input_width, input_height, depth.len()
    ))?;

    let resized = image::imageops::resize(
        &image,
        target_width,
        target_height,
        image::imageops::FilterType::Lanczos3,
    );

    let blurred = apply_gaussian_blur(&resized, 1);

    Ok(blurred.pixels().map(|pixel| pixel.0[0] as f32 / 255.0).collect())
}

fn apply_gaussian_blur(
    image: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>,
    radius: i32,
) -> image::ImageBuffer<image::Luma<u8>, Vec<u8>> {
    let (width, height) = image.dimensions();
    let sigma = 0.5 + radius as f64 * 0.57;
    let kernel_size = (radius * 2 + 1) as usize;

    let mut kernel = vec![0.0f64; kernel_size];
    let mut sum = 0.0f64;
    for index in 0..kernel_size {
        let x = (index as f64 - radius as f64).powi(2);
        kernel[index] = (-x / (2.0 * sigma * sigma)).exp();
        sum += kernel[index];
    }
    for value in kernel.iter_mut() {
        *value /= sum;
    }

    let mut temp = vec![0.0f64; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0f64;
            for kernel_x in -radius..=radius {
                let source_x = (x as i32 + kernel_x).clamp(0, width as i32 - 1) as u32;
                value += image.get_pixel(source_x, y).0[0] as f64
                    * kernel[(kernel_x + radius) as usize];
            }
            temp[(y * width + x) as usize] = value;
        }
    }

    let mut result = image.clone();
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0f64;
            for kernel_y in -radius..=radius {
                let source_y = (y as i32 + kernel_y).clamp(0, height as i32 - 1) as u32;
                value += temp[(source_y * width + x) as usize] * kernel[(kernel_y + radius) as usize];
            }
            result.put_pixel(x, y, image::Luma([value.clamp(0.0, 255.0) as u8]));
        }
    }

    result
}
