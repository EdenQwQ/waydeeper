# waydeeper

Depth effect wallpaper for Wayland — GPU-accelerated parallax from ML depth estimation.

Inspired by [lively wallpaper](https://github.com/rocksdanister/lively) for Windows, waydeeper brings its depth effect to Wayland compositors.

https://github.com/user-attachments/assets/ea7e0999-1a28-4396-b9b1-a24a520aa6fd

## Features

- **ML Depth Estimation**: Generates depth maps from any image using pre-trained ONNX models (Depth Anything V3, MiDaS, Depth Pro)
- **GPU-Accelerated**: OpenGL ES 3.0 shaders render the parallax effect at full resolution
- **3D Mesh Mode**: Optional perspective-projected mesh generated from image + depth map for a stronger parallax effect
- **Fractional HiDPI**: Exact pixel-perfect scaling via `wp_fractional_scale_v1` + `wp_viewporter`
- **Lazy Animation**: Only animates when the mouse is active on the background surface, with configurable delay and idle timeout
- **Smart Caching**: Depth maps cached with blake2b hashing; model-aware cache invalidation
- **Multi-Monitor**: Independent wallpapers per monitor with separate daemon processes
- **Lightweight**: Written in Rust, the running daemon uses minimal CPU and memory.

## Requirements

- Wayland compositor with `wlr-layer-shell` support (niri, sway, Hyprland, river, etc.)
- ONNX Runtime (for depth estimation)

## Installation

### Using Nix (Recommended)

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

Then add to your system or home packages:

```nix
# NixOS configuration
{ inputs, pkgs, ... }:
{
  environment.systemPackages = [ inputs.waydeeper.packages.${pkgs.system}.default ];
}
```

```nix
# Home Manager
{ inputs, pkgs, ... }:
{
  home.packages = [ inputs.waydeeper.packages.${pkgs.system}.default ];
}
```

#### Home Manager module

Includes a systemd user service for auto-start:

```nix
{ inputs, ... }:
{
  imports = [ inputs.waydeeper.homeManagerModules.default ];
  services.waydeeper.enable = true;
}
```

### Building from Source

#### Quick install script (recommended for non-Nix)

```bash
git clone https://github.com/EdenQwQ/waydeeper.git
cd waydeeper
# Installs to ~/.local/bin (user) or /usr/local/bin (root)
# Prompts for depth model download
bash install.sh

# Or force user install:
bash install.sh --user

# Or custom prefix:
bash install.sh --prefix /opt/waydeeper
```

The script builds the binary, checks for missing dependencies, and prompts you to download depth models.

#### Manual build

**1. Install system dependencies**

**Arch Linux:**

```bash
sudo pacman -S --needed \
    base-devel cmake pkg-config rustup \
    wayland wayland-protocols \
    libxkbcommon libglvnd openssl
# onnxruntime (select cpu variant)
sudo pacman -S onnxruntime-cpu
# Set up Rust
rustup default stable
```

**Ubuntu 25.10+ / Debian Trixie+:**

```bash
sudo apt install -y \
    build-essential cmake pkg-config rustc cargo \
    libwayland-dev wayland-protocols \
    libxkbcommon-dev libegl-dev libglvnd-dev \
    libssl-dev libonnxruntime-dev
```

**2. Build**

```bash
git clone https://github.com/EdenQwQ/waydeeper.git
cd waydeeper
cargo build --release
```

The binary will be at `target/release/waydeeper`.

**3. Install manually**

```bash
sudo cp target/release/waydeeper /usr/local/bin/
# Or with cargo install
cargo install --path .
```

**4. Configure ONNX Runtime path**

The `ort` crate loads `libonnxruntime.so` at runtime. Set the path before running:

```bash
# Arch Linux
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

# Ubuntu / Debian
export ORT_DYLIB_PATH=/usr/lib/x86_64-linux-gnu/libonnxruntime.so

# Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
```

## Post-Installation

### Download Depth Estimation Models

Required for depth map generation:

```bash
waydeeper download-model
```

This prompts you to select from available models:

| Model                              | Description                                            |
| ---------------------------------- | ------------------------------------------------------ |
| `depth-anything-v3-base` (default) | Balanced quality and speed, good for most use cases    |
| `midas-small`                      | Lightweight and fast, lower quality                    |
| `depth-pro-q4`                     | Apple Depth Pro (4-bit quantized) — high quality, slow |

Models are stored in `~/.local/share/waydeeper/models/`.

Downloads respect `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY`
environment variables.

## Usage

### Set a wallpaper

```bash
# Basic usage (applies to all connected monitors)
waydeeper set /path/to/wallpaper.jpg

# On a specific monitor
waydeeper set /path/to/wallpaper.jpg -m eDP-1

# Omit image to use the configured wallpaper (useful for changing params or regenerating)
waydeeper set -m eDP-1 --strength 0.05

# With custom settings
waydeeper set /path/to/wallpaper.jpg \
  -m eDP-1 \
  --strength 0.05 \
  --smooth-animation \
  --animation-speed 0.02 \
  --fps 60 \
  --active-delay 150 \
  --idle-timeout 1000 \
  --invert-depth
```

If a daemon is already running for the target monitor, it will be reloaded with
the new settings in the background (no visible interruption). Otherwise a new
daemon is spawned.

### Use a specific depth model

```bash
waydeeper set /path/to/wallpaper.jpg --model depth-anything-v3-base
waydeeper set /path/to/wallpaper.jpg --model midas-small
waydeeper set /path/to/wallpaper.jpg --model depth-pro-q4
waydeeper set /path/to/wallpaper.jpg --model /path/to/custom/model.onnx
```

### 3D mesh mode

Generates a 3D perspective mesh from the image and depth map for a stronger
parallax effect than the default flat depth-warp. The mesh is produced in pure
Rust — no Python, no extra ML models — and cached after the first run.

```bash
# Enable 3D mesh mode
waydeeper set /path/to/wallpaper.jpg --3d

# Adjust parallax strength
waydeeper set /path/to/wallpaper.jpg --3d --strength-x 0.05 --strength-y 0.02

# Regenerate depth map and mesh
waydeeper set /path/to/wallpaper.jpg --3d --regenerate

# Pregenerate mesh without starting daemon
waydeeper pregenerate /path/to/wallpaper.jpg --3d

# Disable and go back to flat mode
waydeeper set --no-3d
```

### Start daemon for all configured monitors

```bash
# Start all configured monitors (skips already running)
waydeeper daemon

# Start specific monitor
waydeeper daemon -m eDP-1

# Force regenerate depth maps and meshes
waydeeper daemon --regenerate

# Verbose output
waydeeper daemon --verbose
```

### Stop wallpaper

```bash
# Stop all
waydeeper stop

# Stop specific monitor
waydeeper stop -m eDP-1
```

### Other commands

```bash
# List monitors and their status
waydeeper list-monitors

# Pregenerate depth map (saves time later)
waydeeper pregenerate /path/to/wallpaper.jpg

# Pregenerate depth map + 3D mesh
waydeeper pregenerate /path/to/wallpaper.jpg --3d

# Download depth estimation models
waydeeper download-model depth-anything-v3-base

# Manage cache
waydeeper cache --list
waydeeper cache --clear
```

## Configuration

Stored in `~/.config/waydeeper/config.json`:

```json
{
  "monitors": {
    "eDP-1": {
      "wallpaper_path": "/path/to/wallpaper.jpg",
      "strength_x": 0.05,
      "strength_y": 0.05,
      "smooth_animation": true,
      "animation_speed": 0.05,
      "fps": 60,
      "active_delay_ms": 150,
      "idle_timeout_ms": 5000,
      "model_path": "~/.local/share/waydeeper/models/depth-anything-v3-base/model.onnx",
      "invert_depth": false,
      "use_3d": false
    }
  }
}
```

| Option             | Default   | Description                                      |
| ------------------ | --------- | ------------------------------------------------ |
| `strength_x`       | 0.02      | Parallax strength on X axis                      |
| `strength_y`       | 0.02      | Parallax strength on Y axis                      |
| `smooth_animation` | true      | Smooth easing animation                          |
| `animation_speed`  | 0.05      | Animation speed multiplier                       |
| `fps`              | 60        | Frame rate (30 or 60)                            |
| `active_delay_ms`  | 150       | Delay before animation starts after mouse enters |
| `idle_timeout_ms`  | 5000      | Time before animation stops after mouse is idle  |
| `model_path`       | —         | Path to ONNX model                               |
| `invert_depth`     | false     | Invert depth interpretation                      |
| `use_3d`           | false     | Enable 3D perspective mesh mode                  |

### Cache

Depth maps are cached in `~/.cache/waydeeper/depth/` with model-specific keys.
3D mesh PLY files are cached in `~/.cache/waydeeper/mesh/` keyed by image
content, depth map, and mesh parameters. The same image with different depth
models or mesh settings produces separate cache entries.

## Architecture

```
src/
  main.rs            - Entry point
  cli.rs             - CLI with subprocess-based daemon spawning
                       set: config update + IPC reload (no asset generation)
                       daemon: spawn new daemons, skip running
  config.rs          - JSON config management
  models.rs          - Model registry and download URLs
  cache.rs           - Blake2b-hashed depth map + 3D mesh cache
  ipc.rs             - Unix domain socket IPC with reload state tracking
  depth_estimator.rs - ONNX inference + Lanczos resize + Gaussian blur
  daemon.rs          - DepthWallpaperDaemon with background reload state machine
  mesh_gen.rs        - Pure-Rust 3D mesh generation (image + depth → PLY)
  mesh.rs            - Binary/ASCII PLY parser with UV + FoV metadata
  math.rs            - Perspective/translation 4×4 matrix helpers
  renderer.rs        - EGL context, GLSL shaders, flat + mesh modes
                       reload_textures() and reload_mesh() for in-place swap
  wayland.rs         - smithay-client-toolkit layer-shell + fractional scaling
                       Background reload thread, in-place texture swap
  egl_bridge.c       - C FFI for EGL/Wayland bridge
```

## Acknowledgements

This is a vibe coding project.
Most of the code is written using [kimi-cli](https://github.com/MoonshotAI/kimi-cli).
As a personal hobby project, it's not production quality and may contain bugs or performance issues.
Issues and pull requests are welcome, but I may not be able to respond to them in a timely manner.

Special thanks to:

- [lively wallpaper](https://github.com/rocksdanister/lively) — this project is inspired by its depth effect wallpaper feature. waydeeper started as a Wayland implementation of that Windows app's depth wallpaper functionality
- [Depth Anything V3](https://github.com/ByteDance-Seed/depth-anything-3), [MiDaS](https://github.com/isl-org/MiDaS), and [Depth Pro](https://github.com/apple/ml-depth-pro) for depth estimation
- [rocksdanister](https://github.com/rocksdanister) for [ONNX model weights](https://github.com/rocksdanister/lively-ml-models/releases)
- [awww](https://github.com/BC100Dev/awww) for Wayland wallpaper daemon reference
- [smithay-client-toolkit](https://github.com/Smithay/client-toolkit) for the Wayland client library
