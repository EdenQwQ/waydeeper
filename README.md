# Waydeeper

Depth effect wallpaper for Wayland.

https://github.com/user-attachments/assets/b5e0ac11-9533-43a7-a0e1-f34e31c7652e

## Features

- **Machine Learning Depth Estimation**: Uses pre-trained models to generate depth maps from any image
- **GPU-Accelerated Rendering**: Uses OpenGL ES 3.0 shaders for smooth parallax effects
- **Lazy Animation**: Only animates when mouse is active, saving resources
- **Configurable Performance**: Choose between 30 or 60 FPS animation
- **Smart Caching**: Depth maps are cached to avoid regeneration
- **Multi-Monitor Support**: Independent wallpapers per monitor

## Requirements

- Wayland compositor with layer-shell support (sway, hyprland, niri, etc.)
- Python 3.11+

## Installation

### Using Nix

#### Run without installing

```bash
nix run github:EdenQwQ/waydeeper
```

#### Install via Flakes

Add to your `flake.nix` inputs:

```nix
{
  inputs.waydeeper.url = "github:EdenQwQ/waydeeper";
}
```

Then add to your system/home packages:

```nix
{ inputs, pkgs, ... }:
{
  environment.systemPackages = [ inputs.waydeeper.packages.${pkgs.system}.default ];
}
```

```nix
{ inputs, pkgs, ... }:
{
  home.packages = [ inputs.waydeeper.packages.${pkgs.system}.default ];
}
```

You can also use the home manager module, which includes a systemd user service:

```nix
# In your Home Manager configuration
{ inputs, ... }:

{
  imports = [ inputs.waydeeper.homeManagerModules.default ];

  services.waydeeper.enable = true;

  # Optional: use a different package
  # services.waydeeper.package = inputs.waydeeper.packages.${pkgs.system}.waydeeper;
}
```

### On Ubuntu/Debian

#### 1. Install system dependencies

```bash
sudo apt install -y \
    python3-pip \
    python3-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    pkg-config \
    libgtk-4-dev \
    libgtk4-layer-shell-dev # Available for Ubuntu 25.04+
```

#### 2. Install Python package

```bash
# Clone the repository
git clone https://github.com/EdenQwQ/waydeeper.git
cd waydeeper

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install waydeeper
pip install .
```

## Post-Installation

### Download Depth Estimation Models

Required for depth map generation:

```bash
waydeeper download-model
```

This will prompt you to select from available models:

- **midas** (default): Lightweight and fast, good balance of speed and quality
- **depth-pro-q4**: Apple's Depth Pro model (4-bit quantized) - good quality, large file size, slow

Models are downloaded to `~/.local/share/waydeeper/models/{model_name}.onnx`.

You can also download a specific model directly:

```bash
waydeeper download-model midas
waydeeper download-model depth-pro-q4
```

## Usage

### Set a wallpaper

```bash
waydeeper set /path/to/wallpaper.jpg
```

### Set wallpaper with custom settings

```bash
waydeeper set /path/to/wallpaper.jpg \
  --monitor eDP-1 \
  --strength 0.05 \
  --no-smoothing-animation \
  --animation-speed 0.1 \
  --fps 30 \
  --active-delay 100 \
  --idle-timeout 300 \
  --invert-depth
```

### Use a specific depth estimation model

```bash
# Use model by name
waydeeper set /path/to/wallpaper.jpg --model depth-pro-q4

# Use custom model path
waydeeper set /path/to/wallpaper.jpg --model /path/to/custom/model.onnx
```

### Invert depth map interpretation

By default, the depth map is interpreted as: white = close, black = far.
Some models (e.g. Depth Pro) produce the opposite: white = far, black = close.
You can invert the interpretation with:

```bash
waydeeper set /path/to/wallpaper.jpg --invert-depth
```

When no model is specified:

1. Uses `midas.onnx` if available
2. Falls back to the first `.onnx` file found in `~/.local/share/waydeeper/models/`
3. Prompts to download a model if none are found

### Force regeneration of depth map

Useful when switching models or if cache is corrupted:

```bash
waydeeper set /path/to/wallpaper.jpg --regenerate
waydeeper pregenerate /path/to/wallpaper.jpg --regenerate
```

### Stop wallpaper

```bash
waydeeper stop # Stops all wallpapers
waydeeper stop --monitor eDP-1 # Stops wallpaper on specific monitor
```

### Pregenerate depth map for an image

Depth map is automatically generated and saved to `~/.cache/waydeeper/depth/`
when setting a wallpaper,
but you can pregenerate it to save time later:

```bash
waydeeper pregenerate /path/to/wallpaper.jpg
```

With specific model:

```bash
waydeeper pregenerate /path/to/wallpaper.jpg --model depth-pro-q4
```

### Manage cached depth maps

```bash
# List cached wallpapers (shows model used for each)
waydeeper cache --list

# Clear all cached depth maps
waydeeper cache --clear
```

### List configured monitors

```bash
waydeeper list-monitors
```

### Start daemon for configured monitors

```bash
waydeeper daemon # Starts on all configured monitors
waydeeper daemon --monitor eDP-1 # Starts on specific monitor
```

### Override settings when starting daemon

```bash
waydeeper daemon --strength 0.05 --fps 30 --model depth-pro-q4
```

## Configuration

Configuration is stored in `~/.config/waydeeper/config.json`.

Available options per monitor:

- `wallpaper_path`: Path to the wallpaper image
- `strength`: Parallax strength (default: 0.02)
- `strength_x`: Parallax strength on X axis (overrides `strength` if set)
- `strength_y`: Parallax strength on Y axis (overrides `strength` if set)
- `smooth_animation`: Smooth animation using easing function (default: true)
- `animation_speed`: Animation speed multiplier (default: 0.02)
- `fps`: Animation frame rate, 30 or 60 (default: 60)
- `active_delay_ms`: Minimum time mouse must be active before animation starts (default: 150ms)
- `idle_timeout_ms`: Time before animation stops after mouse stops (default: 500ms)
- `model_path`: Path to the depth estimation model for this monitor
- `invert_depth`: Invert depth map interpretation (default: false)

### Cache Structure

Depth maps are cached in `~/.cache/waydeeper/` with model-specific keys:

- Different models generate different cache entries for the same image
- Cache includes model name in metadata for tracking

## Acknowledgements

This project is a vibe coding project.
Most of the code is written using [kimi-cli](https://github.com/MoonshotAI/kimi-cli).
As a personal hobby project, it's not production quality and may contain bugs or performance issues.
Issues and pull requests are welcome, but I may not be able to respond to them in a timely manner.

Special thanks to the developers of the [MiDaS model](https://github.com/isl-org/MiDaS)
and the [Depth Pro model](https://github.com/apple/ml-depth-pro)
for providing the depth estimation model used in this project.

Special thanks to [rocksdanister](https://github.com/rocksdanister)
for their work on [lively](https://github.com/rocksdanister/lively).
I created this project because I wanted a similar depth effect wallpaper for Wayland.
Also thanks to them for sharing the [MiDaS model weights in ONNX format](https://github.com/rocksdanister/lively-ml-models/releases).
