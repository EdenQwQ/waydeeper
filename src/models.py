"""Model registry for depth estimation models."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class ModelFormat(Enum):
    ONNX = "onnx"
    ZIP = "zip"


@dataclass
class ModelInfo:
    name: str
    description: str
    url: str
    format: ModelFormat
    extracted_filename: Optional[str] = None
    
    def __post_init__(self):
        if self.format == ModelFormat.ZIP and not self.extracted_filename:
            raise ValueError("ZIP format models must specify extracted_filename")


# Registry of available models
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    "midas": ModelInfo(
        name="midas",
        description="Lightweight and fast, good balance of speed and quality (default)",
        url="https://github.com/rocksdanister/lively-ml-models/releases/download/v1.0.0.0/midas_small.zip",
        format=ModelFormat.ZIP,
        extracted_filename="model.onnx",
    ),
    "depth-pro-q4": ModelInfo(
        name="depth-pro-q4",
        description="Apple's Depth Pro model (4-bit quantized) - good quality, large file size, slow",
        url="https://huggingface.co/onnx-community/DepthPro-ONNX/resolve/main/onnx/model_q4.onnx",
        format=ModelFormat.ONNX,
    ),
}


def get_model(name: str) -> ModelInfo:
    if name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return AVAILABLE_MODELS[name]


def list_models() -> list[ModelInfo]:
    return list(AVAILABLE_MODELS.values())


def get_default_model() -> ModelInfo:
    return AVAILABLE_MODELS["midas"]


def get_model_path(name_or_path: str) -> Path:
    # First check if it's a known model name
    models_directory = Path.home() / ".local" / "share" / "waydeeper" / "models"
    
    if name_or_path in AVAILABLE_MODELS:
        # It's a known model name
        model_path = models_directory / f"{name_or_path}.onnx"
    else:
        # Treat as a path
        model_path = Path(name_or_path).expanduser().resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run 'waydeeper download-model {name_or_path}' to download it."
        )
    
    return model_path


def get_default_model_path() -> Path:
    return get_model_path("midas")
