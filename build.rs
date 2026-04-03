fn main() {
    cc::Build::new()
        .file("src/egl_bridge.c")
        .opt_level(2)
        .compile("egl_bridge");

    // Link against EGL and wayland-egl
    println!("cargo:rustc-link-lib=EGL");
    println!("cargo:rustc-link-lib=wayland-egl");
    println!("cargo:rerun-if-changed=src/egl_bridge.c");
}
