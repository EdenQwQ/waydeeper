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
    p.add_argument("--invert-depth", action="store_true",
                   help="Do not invert depth map (swap near/far in mesh)")
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
    """Boolean mask — True where disparity (1/depth) changes steeply.

    Uses disparity comparison (like 3d-photo-inpainting) instead of raw depth,
    because disparity is proportional to scene depth difference and properly
    weights near vs far discontinuities.

    Applies morphological cleanup to smooth saw-like staircase artifacts
    at diagonal depth boundaries caused by the 4-connected pixel grid.
    """
    from scipy.ndimage import binary_opening, binary_closing

    disp = 1.0 / (depth + 1e-8)
    grad_h = np.abs(np.diff(disp, axis=0, append=disp[-1:]))
    grad_w = np.abs(np.diff(disp, axis=1, append=disp[:, -1:]))
    edges = (grad_h > threshold) | (grad_w > threshold)

    # Morphological closing: fills 1-pixel gaps along diagonal boundaries,
    # turning staircase patterns into smoother continuous edges.
    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)
    edges = binary_closing(edges, structure=struct, iterations=1)

    # Morphological opening: removes isolated single-pixel edge noise.
    edges = binary_opening(edges, structure=struct, iterations=1)

    return edges

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

    def merge_inpainted(self):
        """Merge inpainted pixels from internal lists into the padded grid.

        Must be called before build_and_write_ply() so that ML-inpainted
        content actually appears in the output mesh.  Without this, the
        inpainting output is silently discarded.
        """
        if not self._inp_rows:
            return
        pad = self.pad
        # Concatenate RGB from all inpainting passes
        if self._inp_rgb:
            rgb_all = np.concatenate(self._inp_rgb, axis=0)
        else:
            rgb_all = np.zeros((0, 3), dtype=np.uint8)
        for i, (r, c) in enumerate(zip(self._inp_rows, self._inp_cols)):
            pr, pc = r + pad, c + pad
            if 0 <= pr < self.pH and 0 <= pc < self.pW:
                self.pdepth[pr, pc] = self._inp_depth[i]
                if i < len(rgb_all):
                    self.prgb[pr, pc] = rgb_all[i]
                self.pvalid[pr, pc] = True
                if i < len(self._inp_type):
                    self.player_type[pr, pc] = self._inp_type[i]

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

    # Border extrapolation: copy edge pixels outward with gradual depth increase.
    # Use a SMALLER falloff to prevent extreme Z values.
    # At 60px: falloff = 1 + 0.002*60 = 1.12 → far depth 5.0 becomes 5.6 (z=-5.6)
    # This is more conservative than the original 1.3× multiplier.
    src_r = pad
    for r in range(pad - 1, -1, -1):
        falloff = 1.0 + 0.002 * (pad - r)  # Reduced from 0.005 to 0.002
        mesh.prgb[r, pad:pW-pad]   = mesh.prgb[src_r, pad:pW-pad]
        mesh.pdepth[r, pad:pW-pad] = mesh.pdepth[src_r, pad:pW-pad] * falloff
        mesh.pvalid[r, pad:pW-pad] = True
        mesh.player_type[r, pad:pW-pad] = 5

    # Bottom
    src_r = pH - pad - 1
    for r in range(pH - pad, pH):
        falloff = 1.0 + 0.002 * (r - src_r)
        mesh.prgb[r, pad:pW-pad]   = mesh.prgb[src_r, pad:pW-pad]
        mesh.pdepth[r, pad:pW-pad] = mesh.pdepth[src_r, pad:pW-pad] * falloff
        mesh.pvalid[r, pad:pW-pad] = True
        mesh.player_type[r, pad:pW-pad] = 5

    # Left
    src_c = pad
    for c in range(pad - 1, -1, -1):
        falloff = 1.0 + 0.002 * (pad - c)
        mesh.prgb[:, c]   = mesh.prgb[:, src_c]
        mesh.pdepth[:, c] = mesh.pdepth[:, src_c] * falloff
        mesh.pvalid[:, c] = True
        mesh.player_type[:, c] = 5

    # Right
    src_c = pW - pad - 1
    for c in range(pW - pad, pW):
        falloff = 1.0 + 0.002 * (c - src_c)
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

    Uses disparity (1/depth) for near/far classification, matching
    3d-photo-inpainting's tear_edges approach where the pixel with
    smaller abs(z) (closer to camera) is the near/foreground side.
    """
    from scipy.ndimage import binary_dilation

    H, W = depth.shape
    near_map = np.zeros((H, W), dtype=bool)
    far_map  = np.zeros((H, W), dtype=bool)

    # Use disparity for comparison — disparity is proportional to scene depth
    # and gives appropriate weight to near vs far discontinuities.
    disp = 1.0 / (depth + 1e-8)

    # Horizontal neighbors: classify left vs right
    dl = disp[:, :-1]
    dr = disp[:, 1:]
    depth_l = depth[:, :-1]
    depth_r = depth[:, 1:]
    eh = edge_mask[:, :-1] | edge_mask[:, 1:]

    # Near (foreground) = smaller depth = closer to camera = larger disparity
    left_is_near = eh & (depth_l < depth_r)
    near_map[:, :-1] |= left_is_near
    far_map[:, 1:]   |= left_is_near
    right_is_near = eh & (depth_r < depth_l)
    near_map[:, 1:]  |= right_is_near
    far_map[:, :-1]  |= right_is_near

    # Vertical neighbors: classify top vs bottom
    dt = disp[:-1, :]
    db = disp[1:, :]
    depth_t = depth[:-1, :]
    depth_b = depth[1:, :]
    ev = edge_mask[:-1, :] | edge_mask[1:, :]

    top_is_near = ev & (depth_t < depth_b)
    near_map[:-1, :] |= top_is_near
    far_map[1:, :]   |= top_is_near
    bottom_is_near = ev & (depth_b < depth_t)
    near_map[1:, :]  |= bottom_is_near
    far_map[:-1, :]  |= bottom_is_near

    struct = np.ones((3, 3), dtype=bool)

    # Build context (known/good region): dilate the foreground (near) side.
    # This provides reference pixels for the inpainting networks.
    context_map = binary_dilation(near_map, structure=struct,
                                  iterations=max(1, ctx_thickness // 3))

    # Build the inpaint mask (background hole behind foreground objects):
    # Following 3d-photo-inpainting's approach:
    # 1. Create a separator by dilating near_map slightly
    # 2. Remove separator zone from far_map to get far_seed
    #    (pixels clearly on the background side, not right at the boundary)
    # 3. Dilate far_seed outward (into background) to fill the hole region
    # 4. Exclude context to avoid painting over foreground
    near_sep = binary_dilation(near_map, structure=struct, iterations=2)
    far_seed = far_map & ~near_sep
    if not far_seed.any():
        # Fallback: if separator eats all far pixels, use far directly
        far_seed = far_map.copy()
    if not far_seed.any():
        # Last resort: use edge_mask itself as seed (both sides)
        far_seed = edge_mask.copy()

    # Grow mask into the background region
    dilated_far = binary_dilation(far_seed, structure=struct,
                                  iterations=max(1, bg_thickness // 3))
    # Ensure mask doesn't overlap context (foreground)
    mask_map = dilated_far & ~context_map

    # If mask is still empty but we have edges, create a minimal mask
    # by growing from edge pixels directly
    if not mask_map.any() and edge_mask.any():
        mask_map = binary_dilation(edge_mask, structure=struct, iterations=3)
        mask_map &= ~context_map

    print(f"Occlusion: {near_map.sum()} near-edge px, {far_map.sum()} far-edge px, "
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
# Graph-based mesh builder (replicates 3d-photo-inpainting approach)
# ---------------------------------------------------------------------------

def build_and_write_ply(mesh, output_path, depth_threshold=0.04):
    """
    Build a graph-based mesh with depth-aware edge tearing, then write binary PLY.
    
    Steps (following 3d-photo-inpainting/mesh.py):
    1. Create connectivity graph for all valid pixels
    2. Tear edges at depth discontinuities (disparity diff > threshold)
    3. Remove dangling edges (degree < 2)
    4. Filter small isolated components (< min_nodes)
    5. Generate triangles from graph connectivity
    """
    pH, pW = mesh.pdepth.shape
    focal  = mesh.focal
    pcx    = mesh.pcx
    pcy    = mesh.pcy

    # --- Vectorised reprojection of the padded grid ---
    rows = np.arange(pH, dtype=np.float32)[:, None] * np.ones((1, pW), dtype=np.float32)
    cols = np.ones((pH, 1), dtype=np.float32) * np.arange(pW, dtype=np.float32)[None, :]
    d    = mesh.pdepth  # (pH, pW)

    # OpenGL convention: x=right, y=up (negate row), z=-depth
    x_all =  (cols + 0.5 - pcx) * d / focal
    y_all = -(rows + 0.5 - pcy) * d / focal
    z_all = -d

    valid  = mesh.pvalid & (d > 0)

    # --- Build connectivity graph with disparity-based edge tearing ---
    # Following 3d-photo-inpainting's approach: tear edges where the disparity
    # (1/depth) difference exceeds a threshold. Disparity properly weights
    # near vs far depth discontinuities — a 0.1m jump at 1m is much more
    # significant than at 50m, and disparity captures this.
    print("Building mesh graph with disparity-based edges...", flush=True)

    # Compute disparity for edge tearing decisions
    disp = 1.0 / (d + 1e-8)

    # Each pixel is a node: (r, c) → node_id
    node_map = np.full((pH, pW), -1, dtype=np.int32)
    valid_rc = np.argwhere(valid)
    n_nodes = len(valid_rc)
    for i, (r, c) in enumerate(valid_rc):
        node_map[r, c] = i

    # Edge tearing disabled — all pixels stay connected.
    # No holes, no abrupt disappearing of pixels.
    disp_threshold = float('inf')

    edges = []
    disp_diffs = []

    for r, c in valid_rc:
        node_id = node_map[r, c]
        dp_cur = disp[r, c]

        # Right neighbor
        if c + 1 < pW and node_map[r, c + 1] >= 0:
            dp_nb = disp[r, c + 1]
            diff = abs(dp_cur - dp_nb)
            disp_diffs.append(diff)
            if diff < disp_threshold:
                edges.append((node_id, node_map[r, c + 1]))

        # Down neighbor
        if r + 1 < pH and node_map[r + 1, c] >= 0:
            dp_nb = disp[r + 1, c]
            diff = abs(dp_cur - dp_nb)
            disp_diffs.append(diff)
            if diff < disp_threshold:
                edges.append((node_id, node_map[r + 1, c]))
    
    if disp_diffs:
        disp_diffs_arr = np.array(disp_diffs)
        print(f"Initial graph: {n_nodes} nodes, {len(edges)} edges", flush=True)
        print(f"Disparity diff stats: min={disp_diffs_arr.min():.4f}, "
              f"median={np.median(disp_diffs_arr):.4f}, "
              f"95th={np.percentile(disp_diffs_arr, 95):.4f}, "
              f"max={disp_diffs_arr.max():.4f} (threshold={disp_threshold})", flush=True)
    
    # Build adjacency list
    adjacency = [set() for _ in range(n_nodes)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    
    # --- Remove dangling edges (nodes with degree < 2) ---
    # This is less aggressive than 3d-photo-inpainting's approach.
    # We only remove truly isolated dangles, not entire chains.
    print("Removing dangling edges...", flush=True)
    removed = True
    iterations = 0
    while removed and iterations < 10:  # Limit iterations to 10 (not 100)
        removed = False
        iterations += 1
        for node in range(n_nodes):
            if len(adjacency[node]) == 1:
                neighbor = list(adjacency[node])[0]
                adjacency[node].clear()
                adjacency[neighbor].discard(node)
                removed = True
    
    print(f"Removed dangling edges in {iterations} iterations", flush=True)
    
    # --- Filter small isolated components ---
    print("Filtering small components...", flush=True)
    visited = [False] * n_nodes
    components = []
    
    def bfs(start):
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return component
    
    for node in range(n_nodes):
        if not visited[node] and adjacency[node]:
            component = bfs(node)
            components.append(component)
    
    # More lenient component filtering: keep components with ≥100 nodes (not 200)
    # or any component that's ≥10% of the largest component size
    min_component_size = 100
    if components:
        largest_size = max(len(c) for c in components)
        min_size_threshold = max(min_component_size, int(largest_size * 0.1))
        large_components = [c for c in components if len(c) >= min_size_threshold]
        
        if not large_components:
            # Fallback: keep largest component
            large_components = [max(components, key=len)]
    else:
        print("WARNING: No components found, generating flat mesh", flush=True)
        large_components = []
    
    # Keep only nodes in large components
    valid_nodes = set()
    for component in large_components:
        valid_nodes.update(component)
    
    # Rebuild adjacency with only valid nodes
    for node in range(n_nodes):
        if node not in valid_nodes:
            adjacency[node].clear()
        else:
            adjacency[node] = {n for n in adjacency[node] if n in valid_nodes}
    
    print(f"After filtering: {len(large_components)} components, {len(valid_nodes)} nodes", flush=True)
    
    # --- Assign vertex indices (only for nodes in valid components) ---
    node_to_vertex = {}
    vertex_to_node = []
    for node in sorted(valid_nodes):
        node_to_vertex[node] = len(vertex_to_node)
        vertex_to_node.append(node)
    
    n_vertices = len(vertex_to_node)
    
    # --- Collect vertex data ---
    vr = valid_rc[vertex_to_node, 0]
    vc = valid_rc[vertex_to_node, 1]
    pos_arr = np.stack([x_all[vr, vc], y_all[vr, vc], z_all[vr, vc]], axis=1).astype(np.float32)
    col_arr = mesh.prgb[vr, vc].astype(np.uint8)
    typ_arr = mesh.player_type[vr, vc].astype(np.uint8)[:, None]
    col_arr = np.concatenate([col_arr, typ_arr], axis=1)
    
    pad = mesh.pad
    H, W = mesh.H, mesh.W
    u_arr = np.clip((vc - pad) / W, 0.0, 1.0).astype(np.float32)
    v_arr = np.clip(1.0 - (vr - pad) / H, 0.0, 1.0).astype(np.float32)
    uv_arr = np.stack([u_arr, v_arr], axis=1).astype(np.float32)
    
    # --- Generate triangles from graph (4-way subdivision per node) ---
    print("Generating triangles from graph...", flush=True)
    faces = []
    
    for node in valid_nodes:
        if node not in node_to_vertex:
            continue
        
        vid = node_to_vertex[node]
        r, c = valid_rc[node]
        
        # Find neighbors in 4 directions by checking adjacency list
        neighbors = {}
        # Up
        if r > 0:
            up_node = node_map[r - 1, c]
            if up_node >= 0 and up_node in adjacency[node]:
                neighbors['up'] = node_to_vertex.get(up_node)
        # Right
        if c < pW - 1:
            right_node = node_map[r, c + 1]
            if right_node >= 0 and right_node in adjacency[node]:
                neighbors['right'] = node_to_vertex.get(right_node)
        # Down
        if r < pH - 1:
            down_node = node_map[r + 1, c]
            if down_node >= 0 and down_node in adjacency[node]:
                neighbors['down'] = node_to_vertex.get(down_node)
        # Left
        if c > 0:
            left_node = node_map[r, c - 1]
            if left_node >= 0 and left_node in adjacency[node]:
                neighbors['left'] = node_to_vertex.get(left_node)
        
        # Create triangles: (center, neighbor1, neighbor2) for adjacent pairs
        # Triangle winding must be counter-clockwise (CCW) when viewed from the front
        # (camera looking down -Z). With Y up, CCW means: center → right → up, etc.
        order = ['up', 'right', 'down', 'left']
        present = [d for d in order if d in neighbors]
        
        for i in range(len(present)):
            n1 = neighbors[present[i]]
            n2 = neighbors[present[(i + 1) % len(present)]]
            if n1 is not None and n2 is not None:
                # Reverse winding order for CCW: (center, n2, n1) instead of (center, n1, n2)
                faces.append([vid, n2, n1])
    
    faces = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    
    print(f"Final mesh: {n_vertices} vertices, {len(faces)} faces", flush=True)
    
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
    
    # Depth PNG convention from waydeeper depth estimator:
    #   0.0 (dark) = near, 1.0 (bright) = far
    # The depth estimator already inverts after percentile normalization,
    # so the saved PNG has the correct convention for our use.
    
    # Normalise to [0,1]
    if depth_arr.max() > 1.0:
        depth_arr /= depth_arr.max()
    
    # The --invert-depth flag is for special cases where the user has a non-standard
    # depth map. By default we use the PNG as-is.
    if args.invert_depth:
        depth_arr = 1.0 - depth_arr
    
    # Power-curve remap: depth_final = 5^normalised → range [1.0, 5.0], ratio 5×.
    # PNG convention: normalised=0 (near) → depth=1.0 → z=-1.0 (close to camera)
    #                 normalised=1 (far)  → depth=5.0 → z=-5.0 (far from camera)
    # This 5× ratio prevents extreme parallax stretching at depth discontinuities.
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
    # Merge inpainted pixels into the grid before building the mesh.
    # Without this, the ML inpainting output is stored in internal lists
    # but never appears in the PLY output.
    ldi.merge_inpainted()
    print(f"Writing PLY to {args.output}...", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    build_and_write_ply(ldi, args.output, depth_threshold=args.depth_threshold)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
