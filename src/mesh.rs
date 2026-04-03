//! Binary PLY parser for 3D inpainting mesh output.
//!
//! Reads the binary little-endian PLY files produced by `inpainting/inpaint.py`.
//!
//! New vertex layout (24 bytes): x:f32 y:f32 z:f32  r:u8 g:u8 b:u8 alpha:u8  u:f32 v:f32
//! Old vertex layout (16 bytes): x:f32 y:f32 z:f32  r:u8 g:u8 b:u8 alpha:u8         (no UV)
//! Face layout       (13 bytes): count:u8(=3)  i0:i32 i1:i32 i2:i32
//!
//! Header comments:
//!   `comment fov_y_deg <float>`     — vertical FoV for the renderer's perspective camera
//!   `comment image_aspect <float>`  — W/H of the source image (for cover-fill on wide screens)

use anyhow::{anyhow, Context, Result};
use std::io::Read;
use std::path::Path;

/// A loaded 3D mesh ready for GPU upload.
pub struct Mesh {
    /// XYZ positions, interleaved: [x0,y0,z0, x1,y1,z1, ...]
    pub positions: Vec<f32>,
    /// RGBA colors (0-255), interleaved: [r0,g0,b0,a0, ...]
    pub colors: Vec<u8>,
    /// UV texture coordinates into the original image, interleaved: [u0,v0, u1,v1, ...]
    /// Empty for old PLY files without UV data (has_uvs == false).
    pub uvs: Vec<f32>,
    /// Whether this PLY contains per-vertex UV coordinates.
    pub has_uvs: bool,
    /// Triangle indices (i32 as written by numpy, cast to u32 for GL).
    pub indices: Vec<u32>,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Number of triangles.
    pub triangle_count: u32,
    /// AABB: [min_x, max_x, min_y, max_y, min_z, max_z]
    pub aabb: [f32; 6],
    /// Vertical FoV in degrees baked into the mesh coordinate system.
    /// The renderer MUST use this FoV (not the user-configured one) to show
    /// the original image undistorted.
    pub fov_y_deg: f32,
    /// Aspect ratio (W/H) of the source image used to build the mesh.
    /// Used by the renderer to compute the cover FoV when the screen aspect
    /// differs from the image aspect.
    pub image_aspect: f32,
}

impl Mesh {
    pub fn load_ply(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Cannot open PLY: {}", path.display()))?;

        let mut raw = Vec::new();
        file.read_to_end(&mut raw)?;

        // --- Parse header (ASCII, ends with "end_header\n") ---
        let header_end = find_header_end(&raw)
            .ok_or_else(|| anyhow!("PLY: no end_header found"))?;

        let header_str = std::str::from_utf8(&raw[..header_end])
            .with_context(|| "PLY header is not valid UTF-8")?;

        let mut vertex_count: usize = 0;
        let mut face_count:   usize = 0;
        let mut fov_y_deg:    f32   = 60.0; // safe fallback
        let mut image_aspect: f32   = 1.0;  // safe fallback (square)
        let mut is_binary    = false;
        // Detect UV presence by looking for the texture_u property in the header.
        let mut has_uvs      = false;

        for line in header_str.lines() {
            let line = line.trim();
            if line.starts_with("format binary") {
                is_binary = true;
            } else if let Some(rest) = line.strip_prefix("comment fov_y_deg") {
                if let Ok(v) = rest.trim().parse::<f32>() {
                    fov_y_deg = v;
                }
            } else if let Some(rest) = line.strip_prefix("comment image_aspect") {
                if let Ok(v) = rest.trim().parse::<f32>() {
                    image_aspect = v;
                }
            } else if line == "property float texture_u" {
                has_uvs = true;
            } else if line.starts_with("element vertex") {
                vertex_count = line.split_whitespace().nth(2)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            } else if line.starts_with("element face") {
                face_count = line.split_whitespace().nth(2)
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
            }
        }

        if vertex_count == 0 {
            return Err(anyhow!("PLY: vertex count is 0"));
        }

        let data = &raw[header_end..];

        if is_binary {
            Self::parse_binary(data, vertex_count, face_count, fov_y_deg, image_aspect, has_uvs)
        } else {
            Self::parse_ascii(data, vertex_count, face_count, fov_y_deg, image_aspect, has_uvs)
        }
    }

    // -----------------------------------------------------------------------
    // Binary parser (primary path)
    // -----------------------------------------------------------------------

    fn parse_binary(
        data: &[u8],
        vertex_count: usize,
        face_count: usize,
        fov_y_deg: f32,
        image_aspect: f32,
        has_uvs: bool,
    ) -> Result<Self> {
        // Old: 3×f32 + 4×u8 = 16 bytes.  New: 3×f32 + 4×u8 + 2×f32 = 24 bytes.
        let vert_stride: usize = if has_uvs { 24 } else { 16 };
        // Each face: 1×u8 + 3×i32 = 13 bytes
        const FACE_STRIDE: usize = 13;

        let needed = vertex_count * vert_stride + face_count * FACE_STRIDE;
        if data.len() < needed {
            return Err(anyhow!(
                "PLY binary data too short: have {} bytes, need {} (vert_stride={})",
                data.len(), needed, vert_stride
            ));
        }

        let mut positions = Vec::with_capacity(vertex_count * 3);
        let mut colors    = Vec::with_capacity(vertex_count * 4);
        let mut uvs       = if has_uvs { Vec::with_capacity(vertex_count * 2) } else { Vec::new() };
        let mut min_x = f32::MAX; let mut max_x = f32::MIN;
        let mut min_y = f32::MAX; let mut max_y = f32::MIN;
        let mut min_z = f32::MAX; let mut max_z = f32::MIN;

        let verts_data = &data[..vertex_count * vert_stride];
        for chunk in verts_data.chunks_exact(vert_stride) {
            let x = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
            let r = chunk[12];
            let g = chunk[13];
            let b = chunk[14];
            let a = chunk[15];

            positions.push(x); positions.push(y); positions.push(z);
            colors.push(r); colors.push(g); colors.push(b); colors.push(a);

            if has_uvs {
                let u = f32::from_le_bytes(chunk[16..20].try_into().unwrap());
                let v = f32::from_le_bytes(chunk[20..24].try_into().unwrap());
                uvs.push(u); uvs.push(v);
            }

            if x < min_x {
                min_x = x;
            } else if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            } else if y > max_y {
                max_y = y;
            }
            if z < min_z {
                min_z = z;
            } else if z > max_z {
                max_z = z;
            }
        }

        let mut indices = Vec::with_capacity(face_count * 3);
        let faces_data = &data[vertex_count * vert_stride..];
        for chunk in faces_data.chunks_exact(FACE_STRIDE) {
            // chunk[0] = count (should be 3), then 3×i32
            let i0 = i32::from_le_bytes(chunk[1..5].try_into().unwrap()) as u32;
            let i1 = i32::from_le_bytes(chunk[5..9].try_into().unwrap()) as u32;
            let i2 = i32::from_le_bytes(chunk[9..13].try_into().unwrap()) as u32;
            indices.push(i0);
            indices.push(i1);
            indices.push(i2);
        }

        let tri_count = (indices.len() / 3) as u32;
        log::info!(
            "Loaded PLY: {} vertices, {} triangles, Z [{:.3}, {:.3}], FoV_y {:.2}°, image_aspect {:.4}, uvs={}",
            vertex_count, tri_count, min_z, max_z, fov_y_deg, image_aspect, has_uvs
        );

        Ok(Mesh {
            positions,
            colors,
            uvs,
            has_uvs,
            indices,
            vertex_count: vertex_count as u32,
            triangle_count: tri_count,
            aabb: [min_x, max_x, min_y, max_y, min_z, max_z],
            fov_y_deg,
            image_aspect,
        })
    }

    // -----------------------------------------------------------------------
    // ASCII fallback parser
    // -----------------------------------------------------------------------

    fn parse_ascii(
        data: &[u8],
        vertex_count: usize,
        face_count: usize,
        fov_y_deg: f32,
        image_aspect: f32,
        has_uvs: bool,
    ) -> Result<Self> {
        let text = std::str::from_utf8(data)
            .with_context(|| "PLY ASCII data is not valid UTF-8")?;
        let mut lines = text.lines();

        let mut positions = Vec::with_capacity(vertex_count * 3);
        let mut colors    = Vec::with_capacity(vertex_count * 4);
        let mut uvs       = if has_uvs { Vec::with_capacity(vertex_count * 2) } else { Vec::new() };
        let mut min_x = f32::MAX; let mut max_x = f32::MIN;
        let mut min_y = f32::MAX; let mut max_y = f32::MIN;
        let mut min_z = f32::MAX; let mut max_z = f32::MIN;

        for i in 0..vertex_count {
            let line = lines.next()
                .ok_or_else(|| anyhow!("PLY ASCII: missing vertex line {}", i))?;
            let mut p = line.split_ascii_whitespace();
            let x: f32 = p.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let y: f32 = p.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let z: f32 = p.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let r: u8  = p.next().and_then(|s| s.parse().ok()).unwrap_or(128);
            let g: u8  = p.next().and_then(|s| s.parse().ok()).unwrap_or(128);
            let b: u8  = p.next().and_then(|s| s.parse().ok()).unwrap_or(128);
            let a: u8  = p.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            positions.push(x); positions.push(y); positions.push(z);
            colors.push(r); colors.push(g); colors.push(b); colors.push(a);
            if has_uvs {
                let u: f32 = p.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let v: f32 = p.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                uvs.push(u); uvs.push(v);
            }
            if x < min_x {
                min_x = x;
            } else if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            } else if y > max_y {
                max_y = y;
            }
            if z < min_z {
                min_z = z;
            } else if z > max_z {
                max_z = z;
            }
        }

        let mut indices = Vec::with_capacity(face_count * 3);
        for i in 0..face_count {
            let line = lines.next()
                .ok_or_else(|| anyhow!("PLY ASCII: missing face line {}", i))?;
            let mut p = line.split_ascii_whitespace();
            let count: usize = p.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            if count == 3 {
                for _ in 0..3 {
                    indices.push(p.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0));
                }
            }
        }

        let tri_count = (indices.len() / 3) as u32;
        log::info!(
            "Loaded PLY (ASCII): {} vertices, {} triangles, Z [{:.3}, {:.3}], FoV_y {:.2}°, image_aspect {:.4}, uvs={}",
            vertex_count, tri_count, min_z, max_z, fov_y_deg, image_aspect, has_uvs
        );

        Ok(Mesh {
            positions, colors, uvs, has_uvs, indices,
            vertex_count: vertex_count as u32,
            triangle_count: tri_count,
            aabb: [min_x, max_x, min_y, max_y, min_z, max_z],
            fov_y_deg,
            image_aspect,
        })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Z-depth span of the mesh.
    pub fn depth_range(&self) -> f32 {
        (self.aabb[5] - self.aabb[4]).abs()
    }

    /// Larger of the X and Y half-extents (used to scale camera travel).
    pub fn xy_half_extent(&self) -> f32 {
        let xs = (self.aabb[1] - self.aabb[0]).abs();
        let ys = (self.aabb[3] - self.aabb[2]).abs();
        xs.max(ys) * 0.5
    }
}

/// Find the byte offset of the first byte *after* the "end_header\n" line.
fn find_header_end(data: &[u8]) -> Option<usize> {
    let marker = b"end_header\n";
    data.windows(marker.len())
        .position(|w| w == marker)
        .map(|pos| pos + marker.len())
}
