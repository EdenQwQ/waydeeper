//! Minimal 4×4 matrix helpers for 3D mesh rendering.
//!
//! All matrices are column-major (matching OpenGL convention) stored as
//! `[f32; 16]`.  Indices: column c, row r → index c*4 + r.

/// Build an OpenGL-style perspective projection matrix.
///
/// * `fov_y` — vertical field of view in **radians**
/// * `aspect` — viewport width / height
/// * `near`, `far` — near/far clip distances (positive)
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect, 0.0,  0.0,                       0.0,
        0.0,        f,    0.0,                       0.0,
        0.0,        0.0,  (far + near) * nf,        -1.0,
        0.0,        0.0,  2.0 * far * near * nf,     0.0,
    ]
}

/// Build a pure-translation matrix T(tx, ty, tz).
pub fn translation(tx: f32, ty: f32, tz: f32) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        tx,  ty,  tz,  1.0,
    ]
}

/// Multiply two 4×4 column-major matrices: result = a × b.
#[allow(dead_code)]
pub fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}
