# Waydeeper

Depth effect wallpaper for Wayland.

![demo](https://i.imgur.com/2dWA6mU.mp4)

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

### Download the [MiDaS model](https://github.com/isl-org/MiDaS)

Required for depth map generation:

```bash
waydeeper download-model
```

The model is downloaded and extracted to `~/.local/share/waydeeper/models/model.onnx`.
You can use other models by replacing the `model.onnx` file with your ONNX model.
Check out other depth estimation models [here](https://github.com/PINTO0309/PINTO_model_zoo?tab=readme-ov-file#7-depth-estimation-from-monocularstereo-images).

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
  --idle-timeout 300
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

### List configured monitors

```bash
waydeeper list-monitors
```

### Start daemon for configured monitors

```bash
waydeeper daemon # Starts on all configured monitors
waydeeper daemon --monitor eDP-1 # Starts on specific monitor
```

## Configuration

Configuration is stored in `~/.config/waydeeper/config.json`.

Available options:

- `strength`: Parallax strength (default: 0.05)
- `strength-x`: Parallax strength on X axis (overrides `strength` if set)
- `strength-y`: Parallax strength on Y axis (overrides `strength` if set
- `smooth_animation`: Smooth animation using easing function (default: true)
- `
- `fps`: Animation frame rate, 30 or 60 (default: 60)
- `active_delay_ms`: Minimum time mouse must be active before animation starts (default: 150ms)
- `idle_timeout_ms`: Time before animation stops after mouse stops (default: 500ms)

## Acknowledgements

This project is a vibe coding project.
Most of the code is written using [kimi-cli](https://github.com/MoonshotAI/kimi-cli).
As a personal hobby project, it's not production quality and may contain bugs or performance issues.
Issues and pull requests are welcome, but I may not be able to respond to them in a timely manner.

Special thanks to the developers of the [MiDaS model](https://github.com/isl-org/MiDaS)
for providing the depth estimation model used in this project.

Special thanks to [rocksdanister](https://github.com/rocksdanister)
for their work on [lively](https://github.com/rocksdanister/lively).
I created this project because I wanted a similar depth effect wallpaper for Wayland.
Also thanks to them for sharing the [MiDaS model weights in ONNX format](https://github.com/rocksdanister/lively-ml-models/releases).
