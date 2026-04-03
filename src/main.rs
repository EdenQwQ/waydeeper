mod cli;
mod config;
mod daemon;
mod depth_estimator;
mod cache;
mod inpaint;
mod ipc;
mod math;
mod mesh;
mod models;
mod renderer;
mod wayland;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn")
    ).init();
    cli::run()
}
