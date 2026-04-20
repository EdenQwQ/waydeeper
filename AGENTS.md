# AGENTS.md - waydeeper

## What This Is

`waydeeper` is a GPU-accelerated depth effect wallpaper daemon for Wayland compositors. Uses ML-based monocular depth estimation (ONNX) to create a parallax effect where wallpaper layers shift as the mouse moves. An optional `--3d` mode generates a perspective-projected mesh from the image + depth map for a stronger parallax effect.

## Current Status — Fully Working

All CLI commands and the full rendering pipeline are functional. Tested on niri compositor with both integer and fractional HiDPI scaling.

**Working:**
- Full rendering pipeline: Wayland layer-shell + EGL + OpenGL ES 3.0
- GPU-accelerated parallax depth effect (GLSL ES 300 shaders)
- 3D mesh mode: two-pass rendering (flat background + 3D mesh) with UV-texture sampling from full-res wallpaper
- Pure-Rust mesh generation in `mesh_gen.rs` (no Python, no ML inpainting)
- Fractional HiDPI scaling via `wp_fractional_scale_v1` + `wp_viewporter`
- Multi-monitor support (independent daemon subprocess per output)
- All CLI commands: `set`, `daemon`, `stop`, `list-monitors`, `pregenerate`, `cache --list/--clear`, `download-model`
- ONNX depth estimation via `ort` crate (load-dynamic, system libonnxruntime)
- Depth map caching with blake2b hashing, model-aware cache keys
- 3D mesh PLY caching with blake2b hashing (image + depth + config)
- Unix domain socket IPC (PING/STATUS/STOP/RELOAD)
- Subprocess-based daemon spawning (parent waits for wallpaper ready, then exits)
- Background reload: daemon regenerates assets in a background thread while the renderer continues, then swaps textures/mesh in-place
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
                       send_reload() polls STATUS for progress during background reload
  config.rs          - JSON config (~/.config/waydeeper/config.json)
                       `use_3d` (with `use_inpaint` serde alias for backward compat)
  models.rs          - Model registry (depth-anything-v3-base, midas-small, depth-pro-q4)
  cache.rs           - DepthCache with blake2b hashing, 16-bit PNG depth map I/O
                       MeshCache for 3D PLY caching
  ipc.rs             - DaemonSocket (server) / DaemonClient (client) over Unix sockets
                       ReloadState for tracking background reload progress
  depth_estimator.rs - ort crate ONNX wrapper, Lanczos3 resize, Gaussian blur
  daemon.rs          - DepthWallpaperDaemon: depth/mesh → IPC → renderer (in order)
                       run_daemon_loop() handles reload state machine
  mesh_gen.rs        - Pure-Rust mesh generator: image + depth → binary PLY
                       Padded grid with border extrapolation (depth × (1 + 0.002·offset))
                       2-triangle quads, CCW winding, per-vertex UVs into original image
  mesh.rs            - Binary/ASCII PLY parser with UV coords, image_aspect, fov_y_deg
  math.rs            - perspective() and translation() 4×4 column-major matrix helpers
  renderer.rs        - Dual-mode renderer (flat depth-warp + mesh perspective)
                       Two-pass draw: flat background quad + mesh with back-face culling
                       Mesh shader samples full-res wallpaper via UV coords (not baked vertex colors)
                       Fragment shader discards out-of-UV-range fragments
                       reload_textures() and reload_mesh() for in-place swap during reload
  wayland.rs         - smithay-client-toolkit: layer-shell, pointer tracking, fractional scale
                       OutputProbe for list_connected_outputs() (monitor availability check)
                       Reload asset generation in background thread, in-place texture swap
  egl_bridge.c       - ~100 lines C: EGL init from wl_display, window surface via wl_egl_window
build.rs             - Compiles egl_bridge.c, links libEGL + libwayland-egl
```

## Key Design Decisions

1. **Subprocess spawning**: `cmd_set`/`cmd_daemon` spawn the binary as a subprocess with the hidden `daemon-run` subcommand. Parent waits for IPC responsiveness (max 180s) then exits. Each monitor gets its own subprocess. Daemon inherits stdout/stderr for progress reporting.

2. **`set` command — config + IPC only**: The `set` command does NOT generate assets. It only updates the config file and either sends a RELOAD IPC to a running daemon or spawns a new one. All heavy lifting (depth estimation, mesh generation, rendering) is done by the daemon process. The `image` argument is optional — omit to use the configured wallpaper (useful for regenerating or changing params). The `-m/--monitor` flag is optional — defaults to all connected monitors.

3. **`daemon` command — starts new, skips running**: The `daemon` subcommand always starts new daemons for configured monitors, skipping any that are already running. Use `set` to reload a running daemon with new settings, or `stop` first to force a fresh start. Has `--regenerate` and `--verbose` flags.

4. **Daemon startup sequence**: Depth estimation → mesh generation (if `--3d`) → IPC socket binding → renderer start. IPC only becomes available after the wallpaper is actually rendering, so "Started daemon" means the wallpaper is visible.

5. **Background reload**: When `set` sends a RELOAD IPC to a running daemon, the daemon generates new assets (depth map, optionally 3D mesh) in a background thread while the renderer continues displaying the current wallpaper. Once assets are ready, textures and mesh are swapped in-place with no visible interruption. The CLI polls STATUS for progress logs during this time.

6. **EGL bridge (C)**: Small C file bridges EGL to Wayland because khronos-egl's Rust type system doesn't expose native `wl_display*`/`wl_surface*` types. Uses `wl_egl_window_create` for the EGL window surface.

7. **Fractional scaling**: Binds `wp_fractional_scale_v1` (staging) + `wp_viewporter` (stable) from `wayland-protocols` with `staging` feature. Gets exact scale (e.g., 192 = 1.6× in 1/120th units). Creates EGL surface at physical pixels. Sets `set_buffer_scale(1)` and `viewport.set_destination(logical_w, logical_h)`.

8. **Texture orientation**: Images flipped vertically via `image::imageops::flip_vertical_in_place`. Combined with standard OpenGL UVs (v=0 at bottom).

9. **Mouse y inversion**: `mouse_y = 1.0 - (y / height)`.

10. **Depth postprocessing**: Percentile normalization → **invert** (1.0 - x) → uint8 → `image::imageops::resize(Lanczos3)` → Gaussian blur with PIL's sigma formula (`0.5 + radius × 0.57`). The inversion means saved depth maps have 0=near (dark), 1=far (bright).

11. **ONNX**: `ort` crate with `load-dynamic` feature. `ORT_DYLIB_PATH` set in flake.nix to nixpkgs' onnxruntime.

12. **Signal handling**: Static `AtomicPtr` passes `running` Arc to `extern "C"` signal handler. Renderer loop checks the flag each frame.

13. **Two rendering modes**:
    - **Flat mode** (default): single-pass UV-warp fragment shader on a fullscreen quad. The shader samples the wallpaper texture with parallax offsets based on depth. Uses mipmap trilinear filtering (`LINEAR_MIPMAP_LINEAR`) for clean downsampling.
    - **Mesh mode** (`--3d`): two-pass rendering. Pass 1 draws a static flat background quad (no parallax) to fill holes from back-face culling. Pass 2 draws the 3D mesh on top with `CULL_FACE` enabled. When UVs are present in the PLY, the mesh fragment shader samples the full-resolution wallpaper texture via `texture(wallpaper_texture, v_uv)`, not the baked vertex colors. Both axes use `-travel` for camera translation so objects follow the mouse on both X and Y.

14. **Camera/travel formula (mesh mode)**:
    ```
    travel_x = mesh_near_z * 0.015 * (strength_x / 0.02)
    travel_y = mesh_near_z * 0.015 * (strength_y / 0.02)
    tx = -(mouse_x - 0.5) * 2.0 * travel_x
    ty = -(mouse_y - 0.5) * 2.0 * travel_y
    ```
    Near pixels shift ≤3% of half-width at default strength. Both axes negated so the scene follows the cursor (camera moves opposite to mouse). Separate `strength_x` and `strength_y` allow independent horizontal/vertical parallax.

15. **Cover FoV (mesh mode)**: The renderer computes a cover FoV to fill the screen regardless of image aspect vs screen aspect. If the image is narrower than the screen, `fov_y` is reduced (zoomed in) until both axes are covered. Formula:
    ```
    x_half_tan = image_aspect * tan(fov_y_intrinsic / 2)
    fov_y_for_x = 2 * atan(x_half_tan / screen_aspect)
    fov_y_cover = min(fov_y_intrinsic, fov_y_for_x)
    ```

16. **Depth mapping (mesh)**: `depth = 5^normalised` → range [1.0, 5.0], ratio 5×. This keeps the near/far ratio constant regardless of image content, preventing extreme parallax stretching. Border falloff: `1 + 0.002 * offset_px` (max 1.12× at 60px).

17. **Mesh generation** (`mesh_gen.rs`):
    - Loads image + 16-bit depth PNG, resizes both to `longer_side` (default 960) using Lanczos/Triangle.
    - Normalises depth to [0,1], applies 5^norm to get power-curve depth ∈ [1, 5].
    - Pads the grid by `extrapolation_thickness` (default 60 px) on all sides; edge pixels are copied outward with gradual depth falloff (`1 + 0.002 × offset`).
    - Reprojects each padded pixel into OpenGL camera space: `x = (c + 0.5 - pcx) * d / focal`, `y = -(r + 0.5 - pcy) * d / focal`, `z = -d`, with `focal = max(H, W)`.
    - Emits 2 CCW triangles per quad (`tl, bl, br` and `tl, br, tr`) — front-facing when viewed from +Z looking at -Z.
    - Writes a binary little-endian PLY with 24-byte vertices (xyz f32, rgba u8, uv f32) and 13-byte faces. Header carries `comment fov_y_deg` and `comment image_aspect`.

18. **Proxy support**: `make_proxy_agent()` in `cli.rs` detects `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`/`NO_PROXY` environment variables and configures a `ureq` proxy agent. Used by all download commands.

19. **Monitor availability**: `cmd_daemon` calls `wayland::list_connected_outputs()` before spawning to skip configured monitors that are not currently connected. `OutputProbe` is a lightweight Wayland client that only enumerates outputs.

20. **CLI parameter consistency**: Both `set` and `daemon` commands expose the same animation parameters (`--strength-x`, `--strength-y`, `--smooth-animation`, etc.). `daemon` CLI flags override saved config values. All parameters have help text explaining their purpose.

21. **Logging**: Main process defaults to `warn` level. Daemon subprocess receives `RUST_LOG=warn` by default (only progress messages via println), or `RUST_LOG=debug` with `--verbose`. The `-v` flag enables detailed logging for debugging.

22. **Depth convention & parallax direction**:
    - Saved depth PNGs: 0=near (dark), 1=far (bright) — inverted from standard due to `1.0 - x` in depth_estimator.rs
    - Flat shader: uses `(1.0 - depth)` for parallax amount → near pixels shift MORE, far pixels shift LESS
    - Mesh mode: perspective projection automatically gives correct parallax (near shifts more at constant camera speed)
    - Effect: **near objects follow the mouse, far objects avoid the mouse** (parallax follows gaze direction)

23. **Backward-compat config**: `MonitorConfig::use_3d` is deserialised with `#[serde(alias = "use_inpaint")]` so old configs keep working.

## Dependencies (Cargo.toml)

| Crate | Version | Purpose |
|-------|---------|---------|
| smithay-client-toolkit | 0.19 | Wayland layer-shell, output/seat/input |
| wayland-client | 0.31 (system) | Raw pointer access for EGL bridge |
| wayland-protocols | 0.32 (staging) | wp_viewporter, wp_fractional_scale_v1 |
| glow | 0.14 | OpenGL function loading via eglGetProcAddress |
| image | 0.25 | Image I/O, Lanczos resize, Gaussian blur |
| ort | 2.0.0-rc.12 | ONNX inference (load-dynamic) |
| ndarray | 0.17 | N-dimensional arrays (used directly, also re-exported by ort) |
| clap | 4 | CLI parsing |
| nix | 0.29 | Unix signals, process management, poll |
| ureq | 2 | HTTP downloads with proxy support |
| bytemuck | 1 | Safe byte casting for GPU buffers |
| cc | 1 (build) | Compiles egl_bridge.c |

## Build & Run

```bash
cd /home/eden/Repos/waydeeper
nix develop          # Enter dev shell with all deps
cargo check          # Verify compilation
nix build            # Build nix package
./result/bin/waydeeper set ~/Pictures/image.jpg -m eDP-1
./result/bin/waydeeper set ~/Pictures/image.jpg --3d   # 3D perspective mesh
./result/bin/waydeeper daemon      # Start all configured
./result/bin/waydeeper stop        # Stop all
```

## Models

Only ONNX depth models are required:
- **depth-anything-v3-base** (default), **midas-small**, **depth-pro-q4**

The previous `--inpaint` flow (Python + 3D-photo-inpainting `.pth` weights) has been removed entirely; 3D mesh generation is now pure Rust.

## Relevant Files

```
waydeeper/
├── src/
│   ├── main.rs                 ← Module declarations
│   ├── cli.rs                  ← CLI flags, proxy-aware download, monitor check, wait spinner
│   │                           ← set: config update + IPC reload (no asset generation)
│   │                           ← daemon: spawn new daemons, skip running
│   ├── config.rs               ← Config with use_3d (use_inpaint alias)
│   ├── cache.rs                ← DepthCache + MeshCache
│   ├── models.rs               ← Depth model registry
│   ├── daemon.rs               ← ensure_ply_exists(), run_daemon() with IPC-after-work
│   │                           ← run_daemon_loop(): background reload state machine
│   ├── mesh_gen.rs             ← Pure-Rust mesh generator (image + depth → PLY)
│   ├── mesh.rs                 ← PLY parser (24-byte format with UVs)
│   ├── math.rs                 ← perspective(), translation() matrix helpers
│   ├── renderer.rs             ← Dual-mode renderer, two-pass mesh draw, cover FoV, mipmaps
│   │                           ← reload_textures(), reload_mesh() for in-place swap
│   └── wayland.rs              ← OutputProbe for monitor enumeration
│                               ← Background reload thread, in-place texture swap
├── flake.nix                   ← Nix build (no Python deps anymore)
├── Cargo.toml                  ← Dependencies
├── README.md                   ← User-facing documentation
└── AGENTS.md                   ← This file
```
