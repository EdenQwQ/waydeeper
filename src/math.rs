/// Build an OpenGL-style perspective projection matrix (column-major).
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

/// Build a pure-translation matrix T(tx, ty, tz) (column-major).
pub fn translation(tx: f32, ty: f32, tz: f32) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        tx,  ty,  tz,  1.0,
    ]
}
