"""Depth estimation using MiDaS ONNX model."""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Default input size (will be auto-detected from model)
default_input_size = (256, 256)
imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default model name
default_model_name = "midas"


def get_models_directory():
    """Get the default models directory."""
    return Path.home() / ".local" / "share" / "waydeeper" / "models"


def list_available_models(models_dir=None):
    """List all available ONNX models in the models directory."""
    if models_dir is None:
        models_dir = get_models_directory()
    
    if not models_dir.exists():
        return []
    
    models = sorted([f for f in models_dir.iterdir() if f.suffix == ".onnx"])
    return models


def prompt_user_for_model_download():
    """Prompt user to download a model when none is available."""
    print("No depth estimation models found.")
    print(f"Models directory: {get_models_directory()}")
    print("")
    print("Please download a model first:")
    print("  waydeeper download-model")
    print("")
    print("Or specify a model path directly:")
    print("  waydeeper set /path/to/wallpaper.jpg --model /path/to/model.onnx")
    sys.exit(1)


class DepthEstimator:
    def __init__(self, model_path=None):
        self.model_path = model_path or self.find_model_file()
        self.model_name = self._get_model_name()
        self.onnx_session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.input_layout = "channels_first"  # or "channels_last"
        self.model_input_size = default_input_size

    def _get_model_name(self):
        """Get the model name from the model path."""
        return Path(self.model_path).stem

    def find_model_file(self):
        """Find the model file to use.
        
        Priority:
        1. midas.onnx in models directory
        2. First available .onnx file in models directory
        3. Legacy paths (for backward compatibility)
        4. Prompt user to download if nothing found
        """
        models_dir = get_models_directory()
        
        # 1. Try default model (midas.onnx)
        default_path = models_dir / f"{default_model_name}.onnx"
        if default_path.exists():
            return str(default_path)
        
        # 2. Try any available ONNX model in the directory
        available_models = list_available_models(models_dir)
        if available_models:
            first_model = available_models[0]
            logger.info(f"Default model '{default_model_name}' not found, using '{first_model.stem}'")
            return str(first_model)
        
        # 3. Legacy paths for backward compatibility
        legacy_paths = [
            Path(__file__).parent.parent / "models" / "model.onnx",
            Path(__file__).parent.parent / "models" / "midas_small" / "model.onnx",
            models_dir / "model.onnx",
            Path("/usr/share/waydeeper/models/model.onnx"),
        ]
        
        for path in legacy_paths:
            if path.exists():
                logger.info(f"Using legacy model at: {path}")
                return str(path)
        
        # 4. No models found - prompt user
        prompt_user_for_model_download()

    def load_model(self):
        if self.onnx_session is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError("onnxruntime is required") from error

        logger.info(f"Loading depth estimation model: {self.model_name} ({self.model_path})")

        available_providers = ort.get_available_providers()
        preferred_providers = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected_providers = [
            provider
            for provider in preferred_providers
            if provider in available_providers
        ]

        self.onnx_session = ort.InferenceSession(
            self.model_path, providers=selected_providers
        )

        # Get input info and detect layout
        input_meta = self.onnx_session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape

        # Detect input layout from shape
        # Typical shapes: [B, 3, H, W] (channels_first) or [B, H, W, 3] (channels_last)
        if len(self.input_shape) >= 4:
            # Check which dimension is likely the channel dim (usually 3 for RGB)
            # Shape is [batch, channels, height, width] or [batch, height, width, channels]
            dim1 = self.input_shape[1] if self.input_shape[1] is not None else 0
            dim3 = self.input_shape[3] if self.input_shape[3] is not None else 0

            if dim1 == 3:
                self.input_layout = "channels_first"
                h_idx, w_idx = 2, 3
                logger.info(f"Detected channels_first layout: {self.input_shape}")
            elif dim3 == 3:
                self.input_layout = "channels_last"
                h_idx, w_idx = 1, 2
                logger.info(f"Detected channels_last layout: {self.input_shape}")
            else:
                # Default assumption
                self.input_layout = "channels_first"
                h_idx, w_idx = 2, 3
                logger.info(f"Using default channels_first layout: {self.input_shape}")

            # Extract model input size
            try:
                h = (
                    int(self.input_shape[h_idx])
                    if self.input_shape[h_idx] is not None
                    else 256
                )
                w = (
                    int(self.input_shape[w_idx])
                    if self.input_shape[w_idx] is not None
                    else 256
                )
                self.model_input_size = (h, w)
            except (ValueError, TypeError):
                self.model_input_size = default_input_size

        self.output_name = self.onnx_session.get_outputs()[0].name

        logger.info(f"Model loaded using providers: {selected_providers}")
        logger.info(f"Input size: {self.model_input_size}, Layout: {self.input_layout}")

    def preprocess_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(self.model_input_size, Image.Resampling.LANCZOS)
        image_array = np.array(image).astype(np.float32)

        image_array = image_array / 255.0
        image_array = (image_array - imagenet_mean) / imagenet_std

        # Adjust layout based on model requirements
        if self.input_layout == "channels_first":
            # [H, W, C] -> [C, H, W]
            image_array = image_array.transpose(2, 0, 1)
        # else: channels_last, keep as [H, W, C]

        image_array = np.expand_dims(image_array, axis=0)

        return image_array.astype(np.float32)

    def postprocess_output(self, model_output, target_size):
        depth = model_output.squeeze()
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]

        percentile_low, percentile_high = np.percentile(depth, [1, 99])
        depth = (depth - percentile_low) / (percentile_high - percentile_low + 1e-8)
        depth = np.clip(depth, 0, 1)

        depth = 1.0 - depth

        depth_uint8 = (depth * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_uint8)
        depth_image = depth_image.resize(target_size, Image.Resampling.LANCZOS)
        depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=2))

        return np.array(depth_image).astype(np.float32) / 255.0

    def estimate(self, image_source):
        self.load_model()

        if isinstance(image_source, str):
            image = Image.open(image_source)
        else:
            image = image_source

        original_size = image.size
        input_tensor = self.preprocess_image(image)

        model_outputs = self.onnx_session.run(
            [self.output_name], {self.input_name: input_tensor}
        )

        depth_map = self.postprocess_output(model_outputs[0], original_size)

        return depth_map


def save_depth_map(depth_map, output_path):
    depth_uint16 = (depth_map * 65535).astype(np.uint16)
    Image.fromarray(depth_uint16).save(output_path)


def load_depth_map(depth_path):
    depth_image = Image.open(depth_path)
    depth_array = np.array(depth_image)

    if depth_array.dtype == np.uint16:
        return depth_array.astype(np.float32) / 65535.0
    elif depth_array.dtype == np.uint8:
        return depth_array.astype(np.float32) / 255.0
    else:
        return depth_array.astype(np.float32)
