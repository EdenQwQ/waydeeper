# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`waydeeper` is a GPU-accelerated depth-effect wallpaper daemon for Wayland compositors. It uses ONNX monocular depth estimation to create a parallax effect where wallpaper layers shift as the mouse moves. An optional `--3d` mode generates a perspective-projected mesh from the image + depth map for a stronger parallax effect. It's a single Rust binary (`waydeeper`) with a small C file bridging EGL to Wayland.

This project also maintains `AGENTS.md` at the repo root with detailed design-decision notes (numbered list of ~23 items covering shader math, camera formulas, depth conventions, etc.) — read it for the "why" behind non-obvious code. Keep both files in sync when architecture changes.

## Build & Development Commands

```bash
nix develop                    # enter dev shell with all deps (wayland, onnxruntime, EGL, etc.)
cargo check                    # fast compile check
cargo build --release          # release build → target/release/waydeeper
nix build                      # build via Nix flake → ./result/bin/waydeeper
cargo clippy                   # lint (clippy + rustfmt are in the dev shell)
cargo fmt
```

There is no automated test suite (no `tests/`, no `#[test]` modules) — verification is done by running the daemon against a real Wayland compositor (see below).

Running/verifying changes requires a live Wayland session with `wlr-layer-shell` (niri, sway, Hyprland, river, etc.) and `ORT_DYLIB_PATH` pointing at `libonnxruntime.so`. `nix develop` sets this up; outside Nix, export it manually (see README).

```bash
./result/bin/waydeeper set ~/Pictures/image.jpg -m eDP-1
./result/bin/waydeeper set ~/Pictures/image.jpg --3d
./result/bin/waydeeper daemon
./result/bin/waydeeper stop
./result/bin/waydeeper list-monitors
```

## Architecture

Module layout (`src/`), in the order data flows through them:

```
cli.rs            Clap CLI. Subcommands: set, daemon, stop, list-monitors, pregenerate,
                   cache, download-model. Hidden `daemon-run` subcommand is what actually
                   gets spawned as a subprocess.
config.rs          JSON config at ~/.config/waydeeper/config.json (per-monitor settings)
models.rs           Depth model registry (depth-anything-v3-base, midas-small, depth-pro-q4)
depth_estimator.rs  ort (ONNX) wrapper: inference, Lanczos3 resize, Gaussian blur
cache.rs            DepthCache (16-bit PNG) + MeshCache (PLY), blake2b-keyed
mesh_gen.rs         Pure-Rust mesh generator: image + depth → binary PLY (no Python/ML inpainting)
mesh.rs             Binary/ASCII PLY parser (UV coords, image_aspect, fov_y_deg in header)
math.rs             perspective()/translation() 4x4 column-major matrix helpers
daemon.rs           DepthWallpaperDaemon: orchestrates depth → mesh → IPC → renderer,
                    plus the background-reload state machine
ipc.rs              Unix domain socket IPC (PING/STATUS/STOP/RELOAD), ReloadState
wayland.rs          smithay-client-toolkit: layer-shell surface, pointer tracking,
                    fractional scaling (wp_fractional_scale_v1 + wp_viewporter),
                    background reload thread + in-place texture swap
renderer.rs         EGL/GL ES 3.0: dual-mode rendering (flat depth-warp vs. 3D mesh),
                    two-pass mesh draw (flat background + culled mesh), cover-FoV math
egl_bridge.c        ~100 lines of C bridging EGL init to a raw wl_display/wl_egl_window
                    (khronos-egl's Rust types don't expose these directly)
build.rs            Compiles egl_bridge.c, links libEGL + libwayland-egl
```

### Process model

- `waydeeper set`/`waydeeper daemon` are thin CLI commands that **spawn the binary itself** as a subprocess with the hidden `daemon-run` command — each monitor gets its own daemon subprocess. The parent waits for the daemon to become IPC-responsive (up to 180s) then exits.
- `set` only updates config and talks IPC to a running daemon (RELOAD) or spawns a new one — it never does asset generation itself. All depth estimation / mesh generation happens inside the daemon process.
- Daemon startup order: depth estimation → mesh generation (if `--3d`) → IPC socket bind → renderer start. The IPC socket only becomes available once the wallpaper is actually rendering.
- **Background reload**: on RELOAD, the daemon regenerates depth/mesh in a background thread while the old wallpaper keeps rendering, then swaps textures/mesh in-place (`reload_textures()`/`reload_mesh()`) with no visible interruption. The CLI polls STATUS to relay progress.

### Rendering modes

- **Flat mode** (default): single-pass fullscreen-quad fragment shader does UV-warp parallax based on the depth map directly; mipmap trilinear filtering.
- **Mesh mode** (`--3d`): two-pass — a static flat background quad first (fills holes left by back-face culling), then the perspective-projected 3D mesh on top with `CULL_FACE`. The mesh shader samples the full-res wallpaper texture via UV coordinates baked into the PLY, not vertex colors.

### Conventions worth knowing before touching depth/parallax code

- Saved depth PNGs are **inverted**: 0 = near (dark), 1 = far (bright) — because of `1.0 - x` in `depth_estimator.rs`. Downstream shader/mesh code must match this convention.
- Effect direction: near objects follow the mouse, far objects avoid it.
- Mesh depth mapping uses `depth = 5^normalized` (range [1, 5]) to keep the near/far ratio constant regardless of image content.
- `MonitorConfig::use_3d` deserializes with `#[serde(alias = "use_inpaint")]` for backward compatibility with older configs — don't remove without a migration plan.

For the full rationale behind these (camera/travel formulas, cover-FoV derivation, mesh padding/extrapolation, proxy handling, signal handling via `AtomicPtr`, etc.), see the numbered "Key Design Decisions" section in `AGENTS.md`.
