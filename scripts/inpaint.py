#!/usr/bin/env python3
"""
waydeeper 3D inpainting pipeline.

Reads an RGB image + pre-computed depth map, runs the 3D Photo Inpainting
pipeline (edge/depth/color inpainting networks), and writes a binary PLY mesh.

Usage:
    python inpaint.py --image <path> --depth <path> --output <path.ply>
                      --models-dir <dir>
                      [--longer-side 960] [--depth-threshold 0.04]
                      [--background-thickness 70] [--context-thickness 140]
                      [--extrapolation-thickness 60]

PLY format:
    comment fov_y_deg <float>    — vertical FoV the renderer must use
    element vertex N  →  x y z r g b alpha  (all binary float32 / uchar)
    element face M    →  3 i0 i1 i2         (binary, int32 indices)

Coordinate convention (OpenGL right-hand, Y-up, camera looks down -Z):
    +X = right,  +Y = up,  -Z = into scene
    The origin is the camera. Geometry at depth d maps to z = -d.
    Y is negated relative to image rows (row 0 = top = +Y).

The camera must be placed at Z=0 with FoV = fov_y_deg (read from PLY comment)
to reproduce the original image exactly.

alpha encodes layer type:
    1=original, 2=inpaint pass1, 3=inpaint pass2, 5=border extrapolation
"""

import argparse
import sys
import os
import math
import struct
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="3D inpainting: image+depth → PLY mesh")
    p.add_argument("--image",  required=True)
    p.add_argument("--depth",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--longer-side",             type=int,   default=960)
    p.add_argument("--depth-threshold",         type=float, default=0.04)
    p.add_argument("--background-thickness",    type=int,   default=70)
    p.add_argument("--context-thickness",       type=int,   default=140)
    p.add_argument("--extrapolation-thickness", type=int,   default=60)
    p.add_argument("--sparse-iter",             type=int,   default=5)
    p.add_argument("--ext-edge-threshold",      type=float, default=0.002)
    p.add_argument("--largest-size",            type=int,   default=512)
    return p.parse_args()

# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------

def import_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("ERROR: PyTorch not installed.", file=sys.stderr)
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu",
              file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Bilateral filtering (depth pre-smoothing)
# ---------------------------------------------------------------------------

def sparse_bilateral_filtering(depth, image, config, num_iter=5):
    from scipy.ndimage import uniform_filter

    filter_sizes    = config.get("filter_size",    [7, 7, 5, 5, 5])
    depth_threshold = config.get("depth_threshold", 0.04)

    disp = 1.0 / (depth + 1e-8)
    dmin, dmax = disp.min(), disp.max()
    smooth = (disp - dmin) / (dmax - dmin + 1e-8)

    for i in range(num_iter):
        fsize = filter_sizes[min(i, len(filter_sizes) - 1)]
        grad_h = np.abs(np.diff(smooth, axis=0, prepend=smooth[:1]))
        grad_w = np.abs(np.diff(smooth, axis=1, prepend=smooth[:, :1]))
        mask   = (np.sqrt(grad_h**2 + grad_w**2) < depth_threshold)
        w  = uniform_filter(mask.astype(np.float32), size=fsize)
        ws = uniform_filter(smooth * mask,            size=fsize)
        smooth = np.where(mask, ws / (w + 1e-8), smooth)

    smooth = smooth * (dmax - dmin + 1e-8) + dmin
    depth_out = 1.0 / (smooth + 1e-8)
    depth_out *= depth.mean() / (depth_out.mean() + 1e-8)
    return depth_out.astype(np.float32)

# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def compute_disparity(depth):
    return 1.0 / (depth + 1e-8)

def find_depth_edges(depth, threshold):
    """Boolean mask — True where depth changes steeply."""
    grad_h = np.abs(np.diff(depth, axis=0, append=depth[-1:]))
    grad_w = np.abs(np.diff(depth, axis=1, append=depth[:, -1:]))
    return (grad_h > threshold) | (grad_w > threshold)

# ---------------------------------------------------------------------------
# LDI mesh (padded canvas of per-pixel depth + color)
# ---------------------------------------------------------------------------

class LDIMesh:
    def __init__(self, rgb, depth, focal, pad):
        """
        focal : pixels, scalar — used for reprojection (fx = fy = focal)
        pad   : extrapolation border in pixels
        """
        H, W = depth.shape
        self.H, self.W = H, W
        self.pad   = pad
        self.focal = float(focal)
        self.cx    = W / 2.0
        self.cy    = H / 2.0

        pH, pW = H + 2*pad, W + 2*pad
        self.pH, self.pW = pH, pW
        # principal point shifts with padding
        self.pcx = self.cx + pad
        self.pcy = self.cy + pad

        self.prgb        = np.zeros((pH, pW, 3), dtype=np.uint8)
        self.pdepth      = np.zeros((pH, pW),    dtype=np.float32)
        self.pvalid      = np.zeros((pH, pW),    dtype=bool)
        self.player_type = np.zeros((pH, pW),    dtype=np.uint8)

        self.prgb[pad:pad+H, pad:pad+W]   = rgb
        self.pdepth[pad:pad+H, pad:pad+W] = depth
        self.pvalid[pad:pad+H, pad:pad+W] = True
        self.player_type[pad:pad+H, pad:pad+W] = 1

        # Extra inpainted pixels (second layer)
        self._inp_rows  = []
        self._inp_cols  = []
        self._inp_rgb   = []
        self._inp_depth = []
        self._inp_type  = []

    def add_inpainted(self, rows, cols, rgb, depth, layer_type=2):
        self._inp_rows.extend(rows.tolist())
        self._inp_cols.extend(cols.tolist())
        self._inp_rgb.append(rgb[rows, cols] if hasattr(rgb, '__getitem__') else rgb)
        self._inp_depth.extend(depth.tolist() if hasattr(depth, 'tolist') else depth)
        self._inp_type.extend([layer_type] * len(rows))

    def fov_y_deg(self):
        """Vertical FoV in degrees for the renderer's perspective camera."""
        return math.degrees(2.0 * math.atan(self.H / (2.0 * self.focal)))

# ---------------------------------------------------------------------------
# Border extrapolation (vectorised)
# ---------------------------------------------------------------------------

def extrapolate_border(mesh, thickness):
    pH, pW = mesh.pdepth.shape
    pad = mesh.pad
    if pad <= 0:
        return

    # Top: copy first valid row upward.
    # Falloff of 0.005/px means the outermost border pixel (60px away) has
    # depth = edge_depth × 1.3  — gentle enough not to distort the mesh.
    src_r = pad
    for r in range(pad - 1, -1, -1):
        falloff = 1.0 + 0.005 * (pad - r)
        mesh.prgb[r, pad:pW-pad]   = mesh.prgb[src_r, pad:pW-pad]
        mesh.pdepth[r, pad:pW-pad] = mesh.pdepth[src_r, pad:pW-pad] * falloff
        mesh.pvalid[r, pad:pW-pad] = True
        mesh.player_type[r, pad:pW-pad] = 5

    # Bottom
    src_r = pH - pad - 1
    for r in range(pH - pad, pH):
        falloff = 1.0 + 0.005 * (r - src_r)
        mesh.prgb[r, pad:pW-pad]   = mesh.prgb[src_r, pad:pW-pad]
        mesh.pdepth[r, pad:pW-pad] = mesh.pdepth[src_r, pad:pW-pad] * falloff
        mesh.pvalid[r, pad:pW-pad] = True
        mesh.player_type[r, pad:pW-pad] = 5

    # Left
    src_c = pad
    for c in range(pad - 1, -1, -1):
        falloff = 1.0 + 0.005 * (pad - c)
        mesh.prgb[:, c]   = mesh.prgb[:, src_c]
        mesh.pdepth[:, c] = mesh.pdepth[:, src_c] * falloff
        mesh.pvalid[:, c] = True
        mesh.player_type[:, c] = 5

    # Right
    src_c = pW - pad - 1
    for c in range(pW - pad, pW):
        falloff = 1.0 + 0.005 * (c - src_c)
        mesh.prgb[:, c]   = mesh.prgb[:, src_c]
        mesh.pdepth[:, c] = mesh.pdepth[:, src_c] * falloff
        mesh.pvalid[:, c] = True
        mesh.player_type[:, c] = 5

# ---------------------------------------------------------------------------
# Occlusion region detection (vectorised, no Python loops)
# ---------------------------------------------------------------------------

def find_occlusion_regions(depth, edge_mask, bg_thickness, ctx_thickness):
    """
    Classify depth-edge pixels as near (foreground) or far (background/hole).
    Returns context_map, mask_map, near_map, far_map  (all bool H×W).
    """
    from scipy.ndimage import binary_dilation

    H, W = depth.shape
    near_map = np.zeros((H, W), dtype=bool)
    far_map  = np.zeros((H, W), dtype=bool)

    # Depth convention: smaller value = closer (foreground), larger = farther (background).
    # At a depth edge, the foreground (near) pixel has SMALLER depth than its neighbour.
    # Horizontal: compare each pixel with its right neighbour
    dl = depth[:, :-1]
    dr = depth[:, 1:]
    eh = edge_mask[:, :-1] | edge_mask[:, 1:]
    # Left pixel is foreground (closer) when dl < dr
    near_map[:, :-1] |= eh & (dl < dr - 1e-3)
    far_map[:, 1:]   |= eh & (dl < dr - 1e-3)
    # Right pixel is foreground (closer) when dr < dl
    near_map[:, 1:]  |= eh & (dr < dl - 1e-3)
    far_map[:, :-1]  |= eh & (dr < dl - 1e-3)

    # Vertical: compare each pixel with its bottom neighbour
    dt = depth[:-1, :]
    db = depth[1:, :]
    ev = edge_mask[:-1, :] | edge_mask[1:, :]
    near_map[:-1, :] |= ev & (dt < db - 1e-3)
    far_map[1:, :]   |= ev & (dt < db - 1e-3)
    near_map[1:, :]  |= ev & (db < dt - 1e-3)
    far_map[:-1, :]  |= ev & (db < dt - 1e-3)

    struct = np.ones((3, 3), dtype=bool)

    # Build context (foreground side): dilate near_map outward
    context_map = binary_dilation(near_map, structure=struct,
                                  iterations=max(1, ctx_thickness // 3))

    # Build the inpaint mask (background hole):
    # 1. Grow near_map slightly to get a "separator" zone
    # 2. Dilate far_map outward (into background)
    # 3. Subtract context to avoid painting over foreground
    # We give mask a head-start by pushing far_map 3px away from near_map first
    near_bloated = binary_dilation(near_map, structure=struct, iterations=3)
    far_seed     = far_map & ~near_bloated  # far pixels not immediately adjacent to fg
    if not far_seed.any():
        # Fallback: just use far pixels directly if no separation possible
        far_seed = far_map
    mask_map = binary_dilation(far_seed, structure=struct,
                               iterations=max(1, bg_thickness // 3))
    mask_map &= ~context_map

    print(f"Occlusion: {near_map.sum()} near-edge px, "
          f"{mask_map.sum()} hole px ({mask_map.mean()*100:.1f}%)", flush=True)
    return context_map, mask_map, near_map, far_map

# ---------------------------------------------------------------------------
# Load inpainting networks
# ---------------------------------------------------------------------------

def load_networks(models_dir, device):
    script_dir = Path(__file__).parent

    # networks.py is bundled alongside this script
    networks_path = script_dir / "networks.py"

    if not networks_path.exists():
        print("ERROR: Cannot find networks.py (should be in the same directory as inpaint.py).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading networks from {networks_path}", flush=True)
    sys.path.insert(0, str(networks_path.parent))

    # networks.py imports matplotlib at module level for debug plots never called
    # in inference. Stub it out if absent so the import succeeds.
    import types as _types
    if "matplotlib" not in sys.modules:
        _mpl  = _types.ModuleType("matplotlib")
        _plt  = _types.ModuleType("matplotlib.pyplot")
        _mpl.__dict__["pyplot"] = _plt
        sys.modules["matplotlib"]         = _mpl
        sys.modules["matplotlib.pyplot"]  = _plt

    import networks  # noqa: PLC0415

    torch = import_torch()

    def load_pth(cls, filename, **kwargs):
        path = Path(models_dir) / filename
        if not path.exists():
            print(f"ERROR: Model not found: {path}", file=sys.stderr)
            print("Run: waydeeper download-model inpaint", file=sys.stderr)
            sys.exit(1)
        net = cls(**kwargs)
        net.load_state_dict(torch.load(str(path), map_location=device))
        net.to(device).eval()
        return net

    print("Loading edge model...",  flush=True)
    edge_model  = load_pth(networks.Inpaint_Edge_Net,  "edge-model.pth")
    print("Loading depth model...", flush=True)
    depth_model = load_pth(networks.Inpaint_Depth_Net, "depth-model.pth")
    print("Loading color model...", flush=True)
    color_model = load_pth(networks.Inpaint_Color_Net, "color-model.pth")
    return edge_model, depth_model, color_model

# ---------------------------------------------------------------------------
# ML inpainting pass
# ---------------------------------------------------------------------------

def run_inpaint_pass(rgb, depth, edge_mask, context_map, mask_map,
                     edge_model, depth_model, color_model,
                     ext_edge_threshold, largest_size, device, layer_type=2):
    torch = import_torch()
    import torch.nn.functional as F

    H, W = depth.shape
    rgb_f = rgb.astype(np.float32) / 255.0

    def t(arr):
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    mask_t    = t(mask_map)
    context_t = t(context_map)
    edge_t    = t(edge_mask.astype(np.float32))
    rgb_t     = torch.from_numpy(rgb_f.transpose(2,0,1)).unsqueeze(0).to(device)

    disp      = 1.0 / (depth + 1e-8)
    disp_norm = disp / (disp.max() + 1e-8)
    disp_t    = t(disp_norm)

    depth_mean = float(depth[context_map].mean()) if context_map.any() else 1.0
    log_depth  = np.log(depth + 1e-8) - math.log(depth_mean + 1e-8)
    depth_t    = t(log_depth)

    with torch.no_grad():
        # 1. Edge inpainting
        edge_out = edge_model.forward_3P(mask_t, context_t, rgb_t, disp_t, edge_t,
                                          cuda=device)
        edge_pred = (edge_out.squeeze().cpu().numpy() > ext_edge_threshold).astype(np.float32)
        edge_combo_t = torch.from_numpy(
            np.maximum(edge_mask.astype(np.float32), edge_pred)
        ).unsqueeze(0).unsqueeze(0).to(device)

        # 2. Depth inpainting
        depth_out = depth_model.forward_3P(mask_t, context_t, depth_t, edge_combo_t,
                                            cuda=device)
        depth_pred = np.exp(depth_out.squeeze().cpu().numpy() + math.log(depth_mean + 1e-8))

        # 3. Color inpainting (downscale if large)
        scale = min(1.0, largest_size / math.sqrt(H * W))
        Hn = max(1, int(H * scale)); Wn = max(1, int(W * scale))
        if scale < 1.0:
            rgb_s  = F.interpolate(rgb_t,     (Hn, Wn), mode='bilinear',  align_corners=False)
            edge_s = F.interpolate(edge_combo_t, (Hn, Wn), mode='nearest')
            ctx_s  = F.interpolate(context_t, (Hn, Wn), mode='nearest')
            msk_s  = F.interpolate(mask_t,    (Hn, Wn), mode='nearest')
        else:
            rgb_s, edge_s, ctx_s, msk_s = rgb_t, edge_combo_t, context_t, mask_t

        color_out = color_model.forward_3P(msk_s, ctx_s, rgb_s, edge_s, cuda=device)
        if scale < 1.0:
            color_out = F.interpolate(color_out, (H, W), mode='bicubic', align_corners=False)

        color_pred = (color_out.squeeze().cpu().numpy().transpose(1,2,0) * 255
                      ).clip(0, 255).astype(np.uint8)

    return color_pred, depth_pred.astype(np.float32), mask_map.copy()

# ---------------------------------------------------------------------------
# Vectorised PLY mesh builder
# ---------------------------------------------------------------------------

def build_and_write_ply(mesh, output_path):
    """
    Build vertex/face arrays entirely with numpy (no Python pixel loops),
    then write a binary PLY for fast I/O and compact size.
    """
    pH, pW = mesh.pdepth.shape
    focal  = mesh.focal
    pcx    = mesh.pcx
    pcy    = mesh.pcy

    # --- Vectorised reprojection of the padded grid ---
    # rows[r,c] = r, cols[r,c] = c
    rows = np.arange(pH, dtype=np.float32)[:, None] * np.ones((1, pW), dtype=np.float32)
    cols = np.ones((pH, 1), dtype=np.float32) * np.arange(pW, dtype=np.float32)[None, :]
    d    = mesh.pdepth  # (pH, pW)

    # OpenGL convention: x=right, y=up (negate row), z=-depth
    x_all =  (cols + 0.5 - pcx) * d / focal
    y_all = -(rows + 0.5 - pcy) * d / focal
    z_all = -d

    valid  = mesh.pvalid & (d > 0)

    # Assign sequential indices to valid pixels
    vertex_index = np.full((pH, pW), -1, dtype=np.int32)
    valid_rc = np.argwhere(valid)           # (N, 2)
    n_valid  = len(valid_rc)
    vertex_index[valid_rc[:, 0], valid_rc[:, 1]] = np.arange(n_valid, dtype=np.int32)

    # Collect position and color arrays
    vr = valid_rc[:, 0]
    vc = valid_rc[:, 1]
    pos_arr = np.stack([x_all[vr, vc], y_all[vr, vc], z_all[vr, vc]], axis=1).astype(np.float32)
    col_arr = mesh.prgb[vr, vc].astype(np.uint8)                   # (N, 3)
    typ_arr = mesh.player_type[vr, vc].astype(np.uint8)[:, None]   # (N, 1)
    col_arr = np.concatenate([col_arr, typ_arr], axis=1)            # (N, 4)

    # UV coordinates into the original (unpadded) image, normalized [0,1].
    # Border/inpaint pixels outside the original image are clamped to [0,1].
    # v is flipped: the wallpaper texture is uploaded with flip_vertical_in_place,
    # so GL v=0 is the image bottom (original last row) and v=1 is the image top
    # (original row 0).  We must invert: v_gl = 1 - v_image.
    pad = mesh.pad
    H, W = mesh.H, mesh.W
    u_arr = np.clip((vc - pad) / W,       0.0, 1.0).astype(np.float32)
    v_arr = np.clip(1.0 - (vr - pad) / H, 0.0, 1.0).astype(np.float32)
    uv_arr = np.stack([u_arr, v_arr], axis=1).astype(np.float32)    # (N, 2)

    # --- Vectorised quad→triangle face generation ---
    # For every quad (r,c),(r+1,c),(r,c+1),(r+1,c+1):
    r0 = np.arange(pH - 1, dtype=np.int32)[:, None] * np.ones((1, pW-1), dtype=np.int32)
    c0 = np.ones((pH-1, 1), dtype=np.int32) * np.arange(pW - 1, dtype=np.int32)[None, :]
    r1, c1 = r0 + 1, c0 + 1

    i00 = vertex_index[r0, c0].ravel()
    i10 = vertex_index[r1, c0].ravel()
    i01 = vertex_index[r0, c1].ravel()
    i11 = vertex_index[r1, c1].ravel()

    # Lower-left triangle: (00, 10, 01)
    valid_lo = (i00 >= 0) & (i10 >= 0) & (i01 >= 0)
    tri_lo   = np.stack([i00[valid_lo], i10[valid_lo], i01[valid_lo]], axis=1)

    # Upper-right triangle: (10, 11, 01)
    valid_hi = (i10 >= 0) & (i11 >= 0) & (i01 >= 0)
    tri_hi   = np.stack([i10[valid_hi], i11[valid_hi], i01[valid_hi]], axis=1)

    faces = np.concatenate([tri_lo, tri_hi], axis=0).astype(np.int32)

    print(f"PLY: {n_valid} vertices, {len(faces)} faces", flush=True)

    # --- Write binary PLY ---
    fov_y = mesh.fov_y_deg()
    image_aspect = mesh.W / mesh.H
    _write_binary_ply(output_path, pos_arr, col_arr, uv_arr, faces, fov_y, image_aspect)


def _write_binary_ply(output_path, positions, colors, uvs, faces, fov_y_deg, image_aspect):
    """Write a little-endian binary PLY with UV texture coordinates.

    Vertex layout (24 bytes): x y z  r g b alpha  u v
        x,y,z      : float32  (3D camera-space position)
        r,g,b,alpha: uint8    (baked color + layer type tag)
        u,v        : float32  (normalized image UVs [0,1] into the original image)
    """
    n_vert = len(positions)
    n_face = len(faces)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment fov_y_deg {fov_y_deg:.6f}\n"
        f"comment image_aspect {image_aspect:.6f}\n"
        f"element vertex {n_vert}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        "property float texture_u\n"
        "property float texture_v\n"
        f"element face {n_face}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))

        # Vertices: 3×float32 + 4×uint8 + 2×float32 = 24 bytes each
        vert_dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'),  ('g', 'u1'),  ('b', 'u1'), ('a', 'u1'),
            ('u', '<f4'), ('v', '<f4'),
        ])
        verts = np.empty(n_vert, dtype=vert_dtype)
        verts['x'] = positions[:, 0]
        verts['y'] = positions[:, 1]
        verts['z'] = positions[:, 2]
        verts['r'] = colors[:, 0]
        verts['g'] = colors[:, 1]
        verts['b'] = colors[:, 2]
        verts['a'] = colors[:, 3]
        verts['u'] = uvs[:, 0]
        verts['v'] = uvs[:, 1]
        f.write(verts.tobytes())

        # Faces: 1×uint8 (count=3) + 3×int32 = 13 bytes each
        face_dtype = np.dtype([('n', 'u1'), ('i0', '<i4'), ('i1', '<i4'), ('i2', '<i4')])
        face_arr = np.empty(n_face, dtype=face_dtype)
        face_arr['n']  = 3
        face_arr['i0'] = faces[:, 0]
        face_arr['i1'] = faces[:, 1]
        face_arr['i2'] = faces[:, 2]
        f.write(face_arr.tobytes())

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    from PIL import Image

    # ---- Load image ----
    print(f"Loading image: {args.image}", flush=True)
    img_pil  = Image.open(args.image).convert("RGB")
    rgb_orig = np.array(img_pil, dtype=np.uint8)

    # ---- Load depth map ----
    print(f"Loading depth map: {args.depth}", flush=True)
    depth_pil = Image.open(args.depth)
    depth_arr = np.array(depth_pil, dtype=np.float32)
    # Depth PNG: 1=near, 0=far.  Normalise to [0,1] then invert → 0=near, 1=far.
    if depth_arr.max() > 1.0:
        depth_arr /= depth_arr.max()
    depth_arr = 1.0 - depth_arr
    # Power-curve remap: depth = 5^normalised  →  [1.0, 5.0], ratio exactly 5×.
    # Near pixels (normalised≈0) → depth≈1.0  (z≈−1 in OpenGL space)
    # Far  pixels (normalised≈1) → depth≈5.0  (z≈−5 in OpenGL space)
    # Keeps near/far ratio at 5× regardless of image content, preventing the
    # 100× ratio that caused extreme parallax stretching at near-field pixels.
    depth_arr = np.power(5.0, depth_arr).astype(np.float32)  # [1.0, 5.0]

    H_orig, W_orig = rgb_orig.shape[:2]

    # ---- Resize to target longer side ----
    longer = max(H_orig, W_orig)
    if longer > args.longer_side:
        scale = args.longer_side / longer
        new_w = int(W_orig * scale)
        new_h = int(H_orig * scale)
        img_pil  = img_pil.resize((new_w, new_h), Image.LANCZOS)
        rgb      = np.array(img_pil, dtype=np.uint8)
        # Round-trip in log space so the power curve is preserved across resize.
        # log5(depth) maps [1.0, 5.0] → [0, 1]; store as uint16 then restore.
        depth_log = (np.log(depth_arr) / math.log(5.0) * 65535).clip(0, 65535).astype(np.uint16)
        depth_pil_r = Image.fromarray(depth_log).resize((new_w, new_h), Image.BILINEAR)
        depth = np.power(5.0, np.array(depth_pil_r, dtype=np.float32) / 65535.0).astype(np.float32)
    else:
        rgb   = rgb_orig
        depth = depth_arr

    H, W = rgb.shape[:2]
    print(f"Working at {W}×{H}", flush=True)

    # ---- Focal length ----
    # Use the larger dimension so the FoV is not too wide.
    focal = float(max(H, W))

    # ---- Bilateral depth smoothing ----
    print("Smoothing depth map...", flush=True)
    cfg = {"filter_size": [7,7,5,5,5], "depth_threshold": args.depth_threshold}
    depth_smooth = sparse_bilateral_filtering(depth, rgb, cfg, num_iter=args.sparse_iter)

    # ---- Build LDI mesh ----
    pad = args.extrapolation_thickness
    ldi = LDIMesh(rgb, depth_smooth, focal, pad)

    # ---- Extrapolate borders ----
    print("Extrapolating borders...", flush=True)
    extrapolate_border(ldi, pad)

    # ---- Depth edges ----
    edge_mask = find_depth_edges(depth_smooth, args.depth_threshold)

    # ---- ML models ----
    torch  = import_torch()
    device = torch.device("cpu")
    print("Loading inpainting models...", flush=True)
    edge_model, depth_model, color_model = load_networks(args.models_dir, device)

    # ---- Occlusion regions ----
    print("Finding occlusion regions...", flush=True)
    context_map, mask_map, near_map, _ = find_occlusion_regions(
        depth_smooth, edge_mask, args.background_thickness, args.context_thickness)

    if mask_map.any():
        # ---- Inpainting pass 1 ----
        print("Running inpainting pass 1...", flush=True)
        inp_rgb, inp_depth, inp_mask = run_inpaint_pass(
            rgb, depth_smooth, edge_mask, context_map, mask_map,
            edge_model, depth_model, color_model,
            args.ext_edge_threshold, args.largest_size, device, layer_type=2)

        rows, cols = np.where(inp_mask)
        if len(rows):
            ldi.add_inpainted(rows, cols,
                              inp_rgb, inp_depth[rows, cols], layer_type=2)

        # ---- Inpainting pass 2 ----
        edge2  = find_depth_edges(inp_depth, args.depth_threshold) & ~edge_mask
        ctx2, mask2, _, _ = find_occlusion_regions(
            inp_depth, edge2,
            args.background_thickness, args.context_thickness // 2)
        mask2 &= ~inp_mask & ~context_map

        if mask2.any():
            print("Running inpainting pass 2...", flush=True)
            inp_rgb2, inp_depth2, inp_mask2 = run_inpaint_pass(
                rgb, inp_depth, edge2, ctx2, mask2,
                edge_model, depth_model, color_model,
                args.ext_edge_threshold, args.largest_size, device, layer_type=3)
            rows2, cols2 = np.where(inp_mask2)
            if len(rows2):
                ldi.add_inpainted(rows2, cols2,
                                  inp_rgb2, inp_depth2[rows2, cols2], layer_type=3)
    else:
        print("No occlusion regions found, writing flat mesh.", flush=True)

    # ---- Write PLY ----
    print(f"Writing PLY to {args.output}...", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    build_and_write_ply(ldi, args.output)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
