# AGENTS.md - waydeeper-rust

## What This Is

A Rust rewrite of `waydeeper` (Python, at `../waydeeper/`), a GPU-accelerated depth effect wallpaper daemon for Wayland compositors. Uses ML-based monocular depth estimation (ONNX) to create a parallax effect where wallpaper layers shift as the mouse moves. An optional `--inpaint` mode uses 3D-photo-inpainting to generate a 3D mesh with correct occlusion.

## Current Status — Fully Working

All CLI commands and the full rendering pipeline are functional. Tested on niri compositor with both integer and fractional HiDPI scaling.

**Working:**
- Full rendering pipeline: Wayland layer-shell + EGL + OpenGL ES 3.0
- GPU-accelerated parallax depth effect (GLSL ES 300 shaders, ported from Python)
- 3D inpainting mode: two-pass rendering (flat background + 3D mesh) with UV-texture sampling from full-res wallpaper
- Fractional HiDPI scaling via `wp_fractional_scale_v1` + `wp_viewporter`
- Multi-monitor support (independent daemon subprocess per output)
- All CLI commands: `set`, `daemon`, `stop`, `list-monitors`, `pregenerate`, `cache --list/--clear`, `download-model`, `download-model inpaint`
- ONNX depth estimation via `ort` crate (load-dynamic, system libonnxruntime)
- Depth map caching with blake2b hashing, model-aware cache keys
- Inpaint PLY mesh caching with blake2b hashing (image + depth + config)
- Unix domain socket IPC (PING/STATUS/STOP/RELOAD)
- Subprocess-based daemon spawning (parent waits for wallpaper ready, then exits)
- Signal handling (SIGTERM via `nix::sys::signal::sigaction`)
- Smooth animation with configurable delay/idle timers
- Proxy support (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/NO_PROXY) for all downloads
- Nix flake build (`nix build`, `nix develop`)
- Zero compiler warnings

## Architecture

```
src/
  main.rs            - Entry point, dispatches to cli::run()
  cli.rs             - Clap CLI. spawn_daemon() forks subprocess with "daemon-run" command
                       Proxy-aware download helper (HTTP_PROXY/HTTPS_PROXY/NO_PROXY)
                       wait_for_daemon() with spinner animation, 180s timeout
  config.rs          - JSON config (~/.config/waydeeper/config.json)
  models.rs          - Model registry (midas, depth-pro-q4), download URLs
                       inpaint_models_dir(), inpaint_models_present()
  cache.rs           - DepthCache with blake2b hashing, 16-bit PNG depth map I/O
                       InpaintCache for PLY mesh caching
  ipc.rs             - DaemonSocket (server) / DaemonClient (client) over Unix sockets
  depth_estimator.rs - ort crate ONNX wrapper, Lanczos3 resize, Gaussian blur (PIL-compatible)
  daemon.rs          - DepthWallpaperDaemon: depth/inpaint → IPC → renderer (in order)
  inpaint.rs         - Python subprocess launcher (stdout/stderr streaming)
  mesh.rs            - Binary/ASCII PLY parser with UV coords, image_aspect, fov_y_deg
                       Detects old (16-byte, vertex color only) vs new (24-byte, with UVs) format
  math.rs            - perspective() and translation() 4×4 column-major matrix helpers
  renderer.rs        - Dual-mode renderer (flat depth-warp + mesh perspective)
                       Two-pass draw: flat background quad + mesh with back-face culling
                       Mesh shader samples full-res wallpaper via UV coords (not baked vertex colors)
  wayland.rs         - smithay-client-toolkit: layer-shell, pointer tracking, fractional scale
                       OutputProbe for list_connected_outputs() (monitor availability check)
  egl_bridge.c       - ~100 lines C: EGL init from wl_display, window surface via wl_egl_window
build.rs             - Compiles egl_bridge.c, links libEGL + libwayland-egl
scripts/
  inpaint.py         - Full 3D inpainting pipeline:
                       Bilateral smoothing, edge/depth/color ML inpainting, vectorised PLY builder
                       Binary PLY output with per-vertex UV texture coordinates
  networks.py        - Neural network architectures (MIT, from 3d-photo-inpainting)
```

## Key Design Decisions

1. **Subprocess spawning**: `cmd_set`/`cmd_daemon` spawn the binary as a subprocess with the hidden `daemon-run` subcommand. Parent waits for IPC responsiveness (max 180s) then exits. Each monitor gets its own subprocess. Daemon inherits stdout/stderr for progress reporting.

2. **Daemon startup sequence**: Depth estimation → inpainting → IPC socket binding → renderer start. IPC only becomes available after the wallpaper is actually rendering, so "Started daemon" means the wallpaper is visible.

3. **EGL bridge (C)**: Small C file bridges EGL to Wayland because khronos-egl's Rust type system doesn't expose native `wl_display*`/`wl_surface*` types. Uses `wl_egl_window_create` for the EGL window surface.

4. **Fractional scaling**: Binds `wp_fractional_scale_v1` (staging) + `wp_viewporter` (stable) from `wayland-protocols` with `staging` feature. Gets exact scale (e.g., 192 = 1.6× in 1/120th units). Creates EGL surface at physical pixels. Sets `set_buffer_scale(1)` and `viewport.set_destination(logical_w, logical_h)`.

5. **Texture orientation**: Images flipped vertically via `image::imageops::flip_vertical_in_place` (matches Python's `np.flipud`). Combined with standard OpenGL UVs (v=0 at bottom).

6. **Mouse y inversion**: `mouse_y = 1.0 - (y / height)` matching Python's `normalized_y = 1.0 - (y_position / window_height)`.

7. **Depth postprocessing**: Matches Python exactly — percentile normalization → uint8 → `image::imageops::resize(Lanczos3)` → Gaussian blur with PIL's sigma formula (`0.5 + radius × 0.57`).

8. **ONNX**: `ort` crate with `load-dynamic` feature. `ORT_DYLIB_PATH` set in flake.nix to nixpkgs' onnxruntime.

9. **IPC socket binding**: Socket bound AFTER depth/inpainting completes, so IPC responsiveness means wallpaper is ready. Socket bound once in `DaemonSocket::start()`, already-bound `UnixListener` moved into accept thread.

10. **Signal handling**: Static `AtomicPtr` passes `running` Arc to `extern "C"` signal handler. Renderer loop checks the flag each frame.

11. **Two rendering modes**:
    - **Flat mode** (default): single-pass UV-warp fragment shader on a fullscreen quad. The shader samples the wallpaper texture with parallax offsets based on depth. Uses mipmap trilinear filtering (`LINEAR_MIPMAP_LINEAR`) for clean downsampling.
    - **Mesh mode** (`--inpaint`): two-pass rendering. Pass 1 draws a static flat background quad (no parallax) to fill holes from back-face culling. Pass 2 draws the 3D mesh on top with `CULL_FACE` enabled. When UVs are present in the PLY, the mesh fragment shader samples the full-resolution wallpaper texture via `texture(wallpaper_texture, v_uv)`, not the baked vertex colors. Both axes use `-travel` for camera translation so objects follow the mouse on both X and Y.

12. **Camera/travel formula (mesh mode)**:
    ```
    travel_x = mesh_near_z * 0.015 * (strength_x / 0.02)
    travel_y = mesh_near_z * 0.015 * (strength_y / 0.02)
    tx = -(mouse_x - 0.5) * 2.0 * travel_x
    ty = -(mouse_y - 0.5) * 2.0 * travel_y
    ```
    Near pixels shift ≤3% of half-width at default strength. Both axes negated so the scene follows the cursor (camera moves opposite to mouse). Separate `strength_x` and `strength_y` allow independent horizontal/vertical parallax.

13. **Cover FoV (mesh mode)**: The renderer computes a cover FoV to fill the screen regardless of image aspect vs screen aspect. If the image is narrower than the screen, `fov_y` is reduced (zoomed in) until both axes are covered. Formula:
    ```
    x_half_tan = image_aspect * tan(fov_y_intrinsic / 2)
    fov_y_for_x = 2 * atan(x_half_tan / screen_aspect)
    fov_y_cover = min(fov_y_intrinsic, fov_y_for_x)
    ```

14. **Depth mapping (inpaint)**: `depth = 5^normalised` → range [1.0, 5.0], ratio 5×. This keeps the near/far ratio constant regardless of image content, preventing extreme parallax stretching. Border falloff: `1 + 0.005 * offset_px` (max 1.3× at 60px).

15. **Mesh generation**: `inpaint.py` resize round-trip uses log space: `log5(depth) → uint16 → bilinear resize → 5^(normalised)`. PLY binary format: 24-byte vertices (3×f32 + 4×u8 + 2×f32) with `image_aspect` and `fov_y_deg` in header comments.

16. **Proxy support**: `make_proxy_agent()` in `cli.rs` detects `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`/`NO_PROXY` environment variables and configures a `ureq` proxy agent. Used by all download commands.

17. **Monitor availability**: `cmd_daemon` calls `wayland::list_connected_outputs()` before spawning to skip configured monitors that are not currently connected. `OutputProbe` is a lightweight Wayland client that only enumerates outputs.

18. **CLI parameter consistency**: Both `set` and `daemon` commands expose the same animation parameters (`--strength-x`, `--strength-y`, `--smooth-animation`, etc.). `daemon` CLI flags override saved config values. All parameters have help text explaining their purpose.

19. **Logging**: Main process defaults to `warn` level. Daemon subprocess receives `RUST_LOG=warn` by default (only progress messages via println/eprintln), or `RUST_LOG=debug` with `--verbose`. The `-v` flag enables detailed logging for debugging.

## Dependencies (Cargo.toml)

| Crate | Version | Purpose |
|-------|---------|---------|
| smithay-client-toolkit | 0.19 | Wayland layer-shell, output/seat/input |
| wayland-client | 0.31 (system) | Raw pointer access for EGL bridge |
| wayland-protocols | 0.32 (staging) | wp_viewporter, wp_fractional_scale_v1 |
| glow | 0.14 | OpenGL function loading via eglGetProcAddress |
| image | 0.25 | Image I/O, Lanczos resize, Gaussian blur |
| ort | 2.0.0-rc.12 | ONNX inference (load-dynamic) |
| clap | 4 | CLI parsing |
| nix | 0.29 | Unix signals, process management |
| ureq | 2 | HTTP downloads with proxy support |
| bytemuck | 1 | Safe byte casting for GPU buffers |
| cc | 1 (build) | Compiles egl_bridge.c |

## Build & Run

```bash
cd /home/eden/Repos/waydeeper-rust
nix develop          # Enter dev shell with all deps
cargo check          # Verify compilation
nix build            # Build nix package
./result/bin/waydeeper set ~/Pictures/image.jpg -m eDP-1 --verbose
./result/bin/waydeeper set ~/Pictures/image.jpg --inpaint   # 3D inpainting
./result/bin/waydeeper daemon      # Start all configured
./result/bin/waydeeper stop        # Stop all
```

## Model Architecture

Two separate depth-related models are used:
- **ONNX depth model** (MiDaS / Depth Pro): generates the initial depth map from the full wallpaper image. Always required.
- **`depth-model.pth`**: a neural network from 3D-photo-inpainting used during mesh generation to fill depth values in synthesised occlusion regions. Only needed for `--inpaint` mode.

The `edge-model.pth` and `color-model.pth` are also needed for inpainting — they predict edge patterns and fill colour in synthesised regions respectively.

## Known Minor Issues

- Depth estimation is CPU-only (ONNX). GPU-accelerated inference would require CUDA/DirectML provider setup.
- Occlusion regions are always 0 for most images (ML inpainting networks load but the near/far separator logic needs tuning). The flat mesh (no inpainting holes) is produced instead and works fine visually.
- Fish-net/stretching artefacts at high `--strength` values in inpaint mode — accepted as inherent limitation of depth-based parallax.

## Original Python Reference

Python at `../waydeeper/` uses GTK4 + Gtk4LayerShell + GtkGLArea. Key files for porting:
- `renderer.py` — shaders (vertex + fragment), texture upload, draw loop
- `depth_estimator.py` — ONNX inference, postprocess pipeline
- `daemon.py` — orchestration, IPC, multi-monitor subprocess spawning

## Relevant Files

```
waydeeper-rust/
├── scripts/
│   ├── inpaint.py              ← Full Python inpainting pipeline
│   └── networks.py             ← Neural network architectures (MIT, from 3d-photo-inpainting)
├── src/
│   ├── main.rs                 ← Module declarations
│   ├── cli.rs                  ← CLI flags, proxy-aware download, monitor check, wait spinner
│   ├── config.rs               ← Config with use_inpaint
│   ├── cache.rs                ← DepthCache + InpaintCache
│   ├── models.rs               ← inpaint_models_dir(), inpaint_models_present()
│   ├── daemon.rs               ← ensure_ply_exists(), run_daemon() with IPC-after-work
│   ├── inpaint.rs              ← Python subprocess launcher
│   ├── mesh.rs                 ← PLY parser (old 16-byte + new 24-byte format with UVs)
│   ├── math.rs                 ← perspective(), translation() matrix helpers
│   ├── renderer.rs             ← Dual-mode renderer, two-pass mesh draw, cover FoV, mipmaps
│   └── wayland.rs              ← OutputProbe for monitor enumeration
├── flake.nix                   ← inpaintPythonEnv, wrapProgram PATH prepend
├── Cargo.toml                  ← Dependencies
├── README.md                   ← User-facing documentation
└── AGENTS.md                   ← This file
```
