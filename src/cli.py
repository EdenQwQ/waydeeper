"""Unified command-line interface for waydeeper."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.ipc import (
    DaemonClient,
    list_running_daemons,
    get_daemon_info,
    stop_daemon,
    stop_all_daemons,
)
from src.models import get_model, list_models, get_default_model, ModelFormat

configuration_directory = Path.home() / ".config" / "waydeeper"
configuration_file = configuration_directory / "config.json"


def load_configuration():
    if configuration_file.exists():
        with open(configuration_file, "r") as file:
            return json.load(file)
    return {"version": 1, "monitors": {}}


def save_configuration(configuration):
    configuration_directory.mkdir(parents=True, exist_ok=True)

    clean_configuration = {
        "version": configuration.get("version", 1),
        "monitors": configuration.get("monitors", {}),
    }

    if "cache_directory" in configuration:
        clean_configuration["cache_directory"] = configuration["cache_directory"]
    if "model_path" in configuration and configuration["model_path"] is not None:
        clean_configuration["model_path"] = configuration["model_path"]

    with open(configuration_file, "w") as file:
        json.dump(clean_configuration, file, indent=2)


def start_daemon(
    image,
    monitor,
    strength=None,
    strength_x=None,
    strength_y=None,
    smooth_animation=True,
    animation_speed=0.02,
    fps=60,
    active_delay_ms=150.0,
    idle_timeout_ms=500.0,
    verbose=False,
    model_path=None,
    regenerate=False,
    invert_depth=False,
):
    """Start a daemon process and wait for it to signal readiness.

    Returns:
        tuple: (success, error_message) where error_message is None on success
    """
    command = [
        sys.executable,
        "-m",
        "src",
        "_daemon",
        "-w",
        image,
        "-m",
        monitor,
        "--animation-speed",
        str(animation_speed),
        "--fps",
        str(fps),
        "--active-delay",
        str(active_delay_ms),
        "--idle-timeout",
        str(idle_timeout_ms),
    ]

    if strength is not None:
        command.extend(["-s", str(strength)])
    if strength_x is not None:
        command.extend(["--strength-x", str(strength_x)])
    if strength_y is not None:
        command.extend(["--strength-y", str(strength_y)])
    if smooth_animation:
        command.append("--smooth-animation")
    else:
        command.append("--no-smooth-animation")
    if verbose:
        command.append("-v")
    if model_path:
        command.extend(["--model", model_path])
    if regenerate:
        command.append("--regenerate")
    if invert_depth:
        command.append("--invert-depth")

    if verbose:
        print(f"Starting daemon: {' '.join(command)}")

    # Pass current environment to inherit PYTHONPATH and LD_PRELOAD from Nix wrapper
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    if verbose:
        print("Starting daemon...", flush=True)
        # In verbose mode, let output go to terminal directly
        process = subprocess.Popen(
            command,
            stdout=None,
            stderr=None,
            start_new_session=True,
            env=env,
        )
        # Wait for socket to appear
        for _ in range(30):  # 6 seconds timeout
            time.sleep(0.2)
            client = DaemonClient(monitor)
            if client.is_running():
                return True, None
        return False, "Daemon did not start (socket not found)"

    # Non-verbose mode
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        text=True,
        bufsize=1,
    )

    # Wait for socket to appear
    for _ in range(30):  # 6 seconds timeout
        time.sleep(0.2)
        client = DaemonClient(monitor)
        if client.is_running():
            return True, None

    # Process exited - collect error output
    try:
        error_output, _ = process.communicate(timeout=1)
    except:
        error_output = ""

    return False, f"Daemon exited with code {process.poll()}: {error_output[-500:]}"


def command_set(arguments):
    image_path = os.path.abspath(arguments.image)

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return 1

    monitor = arguments.monitor

    from src.daemon import DepthWallpaperDaemon
    from src.models import get_model_path

    daemon = DepthWallpaperDaemon()
    daemon.load_configuration()

    # Handle model selection
    model_path = None
    if arguments.model:
        try:
            model_path = str(get_model_path(arguments.model))
            model_name = arguments.model
        except FileNotFoundError as error:
            print(f"Error: {error}")
            return 1
    else:
        # Let depth_estimator auto-detect the model
        model_name = None  # Will be determined by depth_estimator

    if model_name:
        print(
            f"Generating depth map for {image_path} using model '{model_name}'...",
            flush=True,
        )
    else:
        print(f"Generating depth map for {image_path}...", flush=True)

    try:
        depth_path = daemon.pregenerate_depth_map(
            image_path, model_path=model_path, force_regenerate=arguments.regenerate
        )
        # Get the actual model name used
        if daemon.depth_estimator:
            actual_model = daemon.depth_estimator.model_name
            print(f"Depth map ready (model: {actual_model}): {depth_path}", flush=True)
        else:
            print(f"Depth map ready: {depth_path}", flush=True)
    except Exception as error:
        print(f"Failed to generate depth map: {error}")
        return 1

    configuration = load_configuration()
    if "monitors" not in configuration:
        configuration["monitors"] = {}

    monitor_configuration = configuration["monitors"].get(monitor, {})
    monitor_configuration["wallpaper_path"] = image_path

    if arguments.strength is not None:
        monitor_configuration["strength"] = arguments.strength
        monitor_configuration["strength_x"] = arguments.strength
        monitor_configuration["strength_y"] = arguments.strength
    else:
        # Reset to defaults when not specified
        default_strength = 0.02
        monitor_configuration["strength"] = default_strength
        if arguments.strength_x is not None:
            monitor_configuration["strength_x"] = arguments.strength_x
        else:
            monitor_configuration["strength_x"] = default_strength
        if arguments.strength_y is not None:
            monitor_configuration["strength_y"] = arguments.strength_y
        else:
            monitor_configuration["strength_y"] = default_strength
    if arguments.smooth_animation:
        monitor_configuration["smooth_animation"] = True
    if arguments.no_smooth_animation:
        monitor_configuration["smooth_animation"] = False
    if arguments.animation_speed is not None:
        monitor_configuration["animation_speed"] = arguments.animation_speed
    if arguments.fps is not None:
        monitor_configuration["fps"] = arguments.fps
    if arguments.active_delay is not None:
        monitor_configuration["active_delay_ms"] = arguments.active_delay
    if arguments.idle_timeout is not None:
        monitor_configuration["idle_timeout_ms"] = arguments.idle_timeout
    # Store model path if specified
    if model_path:
        monitor_configuration["model_path"] = model_path
    if arguments.invert_depth:
        monitor_configuration["invert_depth"] = True

    configuration["monitors"][monitor] = monitor_configuration
    save_configuration(configuration)

    # Check if daemon is already running via IPC
    client = DaemonClient(monitor)
    if client.is_running():
        info = get_daemon_info(monitor)
        print(f"Stopping existing daemon for monitor {monitor}...")
        stop_daemon(monitor)
        time.sleep(0.3)

    print(f"Starting wallpaper daemon for monitor {monitor}...", flush=True)

    smooth = not arguments.no_smooth_animation
    if arguments.smooth_animation:
        smooth = True

    # Resolve model path for daemon
    model_path = None
    if arguments.model:
        try:
            from src.models import get_model_path

            model_path = str(get_model_path(arguments.model))
        except FileNotFoundError:
            pass  # Will be handled by daemon

    success, error = start_daemon(
        image=image_path,
        monitor=monitor,
        strength=arguments.strength,
        strength_x=arguments.strength_x,
        strength_y=arguments.strength_y,
        smooth_animation=smooth,
        animation_speed=arguments.animation_speed or 0.02,
        fps=arguments.fps or 60,
        active_delay_ms=arguments.active_delay or 150.0,
        idle_timeout_ms=arguments.idle_timeout or 500.0,
        verbose=arguments.verbose,
        model_path=model_path,
        invert_depth=arguments.invert_depth,
    )

    if success:
        print(f"Wallpaper daemon started for monitor {monitor}.", flush=True)
        return 0
    else:
        print(f"Failed to start daemon: {error}", flush=True)
        return 1


def command_stop(arguments):
    if arguments.monitor is not None:
        monitor = arguments.monitor
        client = DaemonClient(monitor)
        if client.is_running():
            print(f"Stopping wallpaper daemon for monitor {monitor}...")
            if stop_daemon(monitor):
                print(f"Wallpaper daemon stopped for monitor {monitor}.")
                return 0
            else:
                print(f"Failed to stop daemon for monitor {monitor}.")
                return 1
        else:
            print(f"No wallpaper daemon is running for monitor {monitor}.")
            return 0
    else:
        running = list_running_daemons()
        if running:
            print(f"Stopping {len(running)} wallpaper daemon(s)...")
            results = stop_all_daemons()
            stopped = sum(1 for success in results.values() if success)
            print(f"Stopped {stopped} wallpaper daemon(s).")
            return 0
        else:
            print("No wallpaper daemon is running.")
            return 0


def command_list_monitors(arguments):
    configuration = load_configuration()
    configured = configuration.get("monitors", {})
    running = list_running_daemons()

    if configured:
        print("Configured wallpapers:")
        print("-" * 60)
        for monitor_id, monitor_config in configured.items():
            wallpaper = monitor_config.get("wallpaper_path", "Not set")
            status = "running" if monitor_id in running else "stopped"
            strength_x = monitor_config.get(
                "strength_x", monitor_config.get("strength", 0.02)
            )
            strength_y = monitor_config.get(
                "strength_y", monitor_config.get("strength", 0.02)
            )
            smooth = monitor_config.get("smooth_animation", True)
            fps = monitor_config.get("fps", 60)
            active_delay = monitor_config.get("active_delay_ms", 150.0)
            idle_timeout = monitor_config.get("idle_timeout_ms", 500.0)
            invert_depth = monitor_config.get("invert_depth", False)

            print(f"  Monitor {monitor_id}:")
            print(f"    Status: {status}")
            print(f"    Wallpaper: {wallpaper}")
            print(f"    Strength: X={strength_x}, Y={strength_y}")
            print(f"    Smooth animation: {smooth}")
            print(f"    FPS: {fps}")
            print(f"    Active delay: {active_delay}ms")
            print(f"    Idle timeout: {idle_timeout}ms")
            print(f"    invert depth: {invert_depth}")
    else:
        print("No monitors configured.")
        print("Use 'waydeeper set <image> -m <monitor>' to configure.")

    return 0


def command_pregenerate(arguments):
    image_path = os.path.abspath(arguments.image)

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return 1

    from src.daemon import DepthWallpaperDaemon
    from src.models import get_model_path

    daemon = DepthWallpaperDaemon()
    daemon.load_configuration()

    # Handle model selection
    model_path = None
    if arguments.model:
        try:
            model_path = str(get_model_path(arguments.model))
            model_name = arguments.model
        except FileNotFoundError as error:
            print(f"Error: {error}")
            return 1
    else:
        # Let depth_estimator auto-detect the model
        model_name = None  # Will be determined by depth_estimator

    if model_name:
        print(f"Generating depth map for {image_path} using model '{model_name}'...")
    else:
        print(f"Generating depth map for {image_path}...")

    try:
        depth_path = daemon.pregenerate_depth_map(
            image_path, model_path=model_path, force_regenerate=arguments.regenerate
        )
        # Get the actual model name used
        if daemon.depth_estimator:
            actual_model = daemon.depth_estimator.model_name
            print(f"Depth map generated using model '{actual_model}': {depth_path}")
        else:
            print(f"Depth map generated: {depth_path}")
        return 0
    except Exception as error:
        print(f"Failed to generate depth map: {error}")
        return 1


def command_cache(arguments):
    from src.daemon import DepthWallpaperDaemon

    daemon = DepthWallpaperDaemon()
    daemon.load_configuration()

    if arguments.clear:
        daemon.clear_cache()
        print("Cache cleared.")
        return 0
    elif arguments.list:
        daemon.list_cached_wallpapers()
        return 0
    else:
        print("Use --clear to clear cache or --list to list cached wallpapers.")
        return 1


def prompt_model_selection() -> str:
    """Prompt user to select a model from available options.

    Returns:
        The selected model name
    """
    models = list_models()
    default_model = get_default_model()

    print("Available depth estimation models:")
    print("-" * 60)

    for idx, model in enumerate(models, 1):
        marker = " (default)" if model.name == default_model.name else ""
        print(f"  {idx}. {model.name}{marker}")
        print(f"     {model.description}")

    print("-" * 60)

    while True:
        try:
            choice = input(
                f"Select model [1-{len(models)}, default: {default_model.name}]: "
            ).strip()

            # Empty input means default
            if not choice:
                return default_model.name

            # Check if input is a number
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(models):
                    return models[idx - 1].name
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
                    continue

            # Check if input is a model name
            if choice in [m.name for m in models]:
                return choice

            print(
                f"Invalid selection. Please enter a number (1-{len(models)}) or model name"
            )

        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)


def command_download_models(arguments):
    import urllib.request
    import zipfile

    models_directory = Path.home() / ".local" / "share" / "waydeeper" / "models"
    models_directory.mkdir(parents=True, exist_ok=True)

    # Determine which model to download
    if arguments.model:
        try:
            model_info = get_model(arguments.model)
        except KeyError as e:
            print(f"Error: {e}")
            return 1
    else:
        # Interactive selection
        model_name = prompt_model_selection()
        model_info = get_model(model_name)

    download_url = model_info.url
    model_file_path = models_directory / f"{model_info.name}.onnx"

    print(f"\nDownloading {model_info.name} model...")
    print(f"URL: {download_url}")

    proxy_handler = None
    http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")

    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
            print(f"Using HTTP proxy: {http_proxy}")
        if https_proxy:
            proxies["https"] = https_proxy
            print(f"Using HTTPS proxy: {https_proxy}")
        proxy_handler = urllib.request.ProxyHandler(proxies)

    def download_with_progress(url: str, dest_path: Path) -> None:
        """Download file with a simple progress bar."""
        if proxy_handler:
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 8192

            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = min(downloaded * 100 // total_size, 100)
                        bar_length = 30
                        filled = int(bar_length * percent / 100)
                        bar = "=" * filled + "-" * (bar_length - filled)
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        sys.stdout.write(
                            f"\r[{bar}] {percent}% ({mb:.1f}/{total_mb:.1f} MB)"
                        )
                        sys.stdout.flush()

            sys.stdout.write("\n")
            sys.stdout.flush()

    try:
        if model_info.format == ModelFormat.ZIP:
            # Download ZIP and extract
            temp_file = models_directory / f"{model_info.name}_download.zip"
            download_with_progress(download_url, temp_file)
            print(f"Downloaded to {temp_file}")

            print(f"Extracting {model_info.extracted_filename}...")
            with zipfile.ZipFile(temp_file, "r") as zip_file:
                # Find the target file in the zip
                target_member = None
                for member in zip_file.namelist():
                    if member.endswith(model_info.extracted_filename):
                        target_member = member
                        break

                if target_member is None:
                    raise ValueError(
                        f"{model_info.extracted_filename} not found in the downloaded zip file"
                    )

                # Extract to model.onnx
                with (
                    zip_file.open(target_member) as source,
                    open(model_file_path, "wb") as target,
                ):
                    target.write(source.read())

            temp_file.unlink()
        else:
            # Direct ONNX download
            download_with_progress(download_url, model_file_path)

        print(f"Model saved to {model_file_path}")
        print("Done!")
        return 0

    except Exception as error:
        print(f"\nFailed to download model: {error}")
        print("\nYou can manually download the model from:")
        print(f"  {download_url}")
        print(f"\nSave it to: {model_file_path}")
        return 1


def command_daemon(arguments):
    configuration = load_configuration()
    monitors_configuration = configuration.get("monitors", {})

    if not monitors_configuration:
        print("No monitors configured. Use 'waydeeper set <image>' first.")
        return 1

    if arguments.monitor is not None:
        target_monitors = [arguments.monitor]
    else:
        target_monitors = list(monitors_configuration.keys())

    started = 0
    failed = 0

    for monitor in target_monitors:
        if monitor not in monitors_configuration:
            print(f"Monitor {monitor} not configured. Use 'set' command first.")
            failed += 1
            continue

        monitor_config = monitors_configuration[monitor]
        wallpaper_path = monitor_config.get("wallpaper_path")

        if not wallpaper_path or not os.path.exists(wallpaper_path):
            print(f"Skipping monitor {monitor}: invalid wallpaper path")
            failed += 1
            continue

        # Check if daemon already running via IPC
        client = DaemonClient(monitor)
        if client.is_running():
            if arguments.verbose:
                print(f"Stopping existing daemon for monitor {monitor}...")
            stop_daemon(monitor)
            time.sleep(0.3)

        strength = monitor_config.get("strength")
        strength_x = monitor_config.get("strength_x", strength)
        strength_y = monitor_config.get("strength_y", strength)
        smooth_animation = monitor_config.get("smooth_animation", True)
        animation_speed = monitor_config.get("animation_speed", 0.02)
        fps = monitor_config.get("fps", 60)
        active_delay_ms = monitor_config.get("active_delay_ms", 150.0)
        idle_timeout_ms = monitor_config.get("idle_timeout_ms", 500.0)

        if arguments.strength is not None:
            strength_x = arguments.strength
            strength_y = arguments.strength
        if arguments.strength_x is not None:
            strength_x = arguments.strength_x
        if arguments.strength_y is not None:
            strength_y = arguments.strength_y
        if arguments.smooth_animation:
            smooth_animation = True
        if arguments.no_smooth_animation:
            smooth_animation = False
        if arguments.animation_speed is not None:
            animation_speed = arguments.animation_speed
        if arguments.fps is not None:
            fps = arguments.fps
        if arguments.active_delay is not None:
            active_delay_ms = arguments.active_delay
        if arguments.idle_timeout is not None:
            idle_timeout_ms = arguments.idle_timeout

        # Get model from configuration
        model_path = monitor_config.get("model_path")

        # Override with command line argument if provided
        if arguments.model:
            try:
                from src.models import get_model_path

                model_path = str(get_model_path(arguments.model))
            except FileNotFoundError as error:
                print(f"Warning: {error}")

        invert_depth = monitor_config.get("invert_depth", False)
        if arguments.invert_depth:
            invert_depth = True

        if arguments.verbose:
            print(f"Starting daemon for monitor {monitor}: {wallpaper_path}")

        success, error = start_daemon(
            image=wallpaper_path,
            monitor=monitor,
            strength=None,
            strength_x=strength_x,
            strength_y=strength_y,
            smooth_animation=smooth_animation,
            animation_speed=animation_speed,
            fps=fps,
            active_delay_ms=active_delay_ms,
            idle_timeout_ms=idle_timeout_ms,
            verbose=arguments.verbose,
            model_path=model_path,
            regenerate=arguments.regenerate,
            invert_depth=invert_depth,
        )

        if success:
            print(f"Started daemon for monitor {monitor}.")
            started += 1
        else:
            print(f"Failed to start daemon for monitor {monitor}: {error}")
            failed += 1

        time.sleep(0.5)

    if started > 0:
        print(f"\nStarted {started} daemon(s).")
    if failed > 0:
        print(f"Failed to start {failed} daemon(s).")

    return 0 if failed == 0 else 1


def create_argument_parser():
    parser = argparse.ArgumentParser(
        prog="waydeeper",
        description="GPU-accelerated depth effect wallpaper for Wayland",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  waydeeper set /path/to/wallpaper.jpg
  waydeeper set /path/to/wallpaper.jpg -m eDP-1
  waydeeper set /path/to/wallpaper.jpg -s 0.03 --strength-y 0.05
  waydeeper set /path/to/wallpaper.jpg --model depth-pro-q4
  waydeeper set /path/to/wallpaper.jpg --model /path/to/custom/model.onnx
  waydeeper set /path/to/wallpaper.jpg --regenerate
  waydeeper daemon
  waydeeper daemon -m eDP-1
  waydeeper stop
  waydeeper stop -m HDMI-A-1
  waydeeper list-monitors
  waydeeper pregenerate /path/to/wallpaper.jpg
  waydeeper pregenerate /path/to/wallpaper.jpg --model depth-pro-q4
  waydeeper cache --list
  waydeeper download-model              # Interactive model selection
  waydeeper download-model midas        # Download specific model
  waydeeper download-model depth-pro-q4
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    set_parser = subparsers.add_parser("set", help="Set a wallpaper with depth effect")
    set_parser.add_argument("image", help="Path to the wallpaper image")
    set_parser.add_argument(
        "-s", "--strength", type=float, default=None, help="Parallax strength"
    )
    set_parser.add_argument(
        "--strength-x", type=float, default=None, help="Parallax strength for X axis"
    )
    set_parser.add_argument(
        "--strength-y", type=float, default=None, help="Parallax strength for Y axis"
    )
    set_parser.add_argument(
        "-m", "--monitor", type=str, default="0", help="Monitor name or index"
    )
    set_parser.add_argument(
        "--smooth-animation", action="store_true", help="Enable smooth animation"
    )
    set_parser.add_argument(
        "--no-smooth-animation", action="store_true", help="Disable smooth animation"
    )
    set_parser.add_argument(
        "--animation-speed", type=float, default=None, help="Animation speed (0.0-1.0)"
    )
    set_parser.add_argument(
        "--fps", type=int, default=None, choices=[30, 60], help="Frame rate"
    )
    set_parser.add_argument(
        "--active-delay", type=float, default=None, help="Active delay in ms"
    )
    set_parser.add_argument(
        "--idle-timeout", type=float, default=None, help="Idle timeout in ms"
    )
    set_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for depth estimation (name or path, default: midas)",
    )
    set_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of depth map even if cached",
    )
    set_parser.add_argument(
        "--invert-depth",
        action="store_true",
        help="Invert depth map interpretation (white=far, black=near)",
    )
    set_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    set_parser.set_defaults(func=command_set)

    daemon_parser = subparsers.add_parser(
        "daemon", help="Start daemon(s) with existing configuration"
    )
    daemon_parser.add_argument(
        "-s", "--strength", type=float, default=None, help="Parallax strength"
    )
    daemon_parser.add_argument(
        "--strength-x", type=float, default=None, help="Parallax strength for X axis"
    )
    daemon_parser.add_argument(
        "--strength-y", type=float, default=None, help="Parallax strength for Y axis"
    )
    daemon_parser.add_argument(
        "-m", "--monitor", type=str, default=None, help="Monitor name or index"
    )
    daemon_parser.add_argument(
        "--smooth-animation", action="store_true", help="Enable smooth animation"
    )
    daemon_parser.add_argument(
        "--no-smooth-animation", action="store_true", help="Disable smooth animation"
    )
    daemon_parser.add_argument(
        "--animation-speed", type=float, default=None, help="Animation speed (0.0-1.0)"
    )
    daemon_parser.add_argument(
        "--fps", type=int, default=None, choices=[30, 60], help="Frame rate"
    )
    daemon_parser.add_argument(
        "--active-delay", type=float, default=None, help="Active delay in ms"
    )
    daemon_parser.add_argument(
        "--idle-timeout", type=float, default=None, help="Idle timeout in ms"
    )
    daemon_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for depth estimation (name or path, default: midas)",
    )
    daemon_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of depth map even if cached",
    )
    daemon_parser.add_argument(
        "--invert-depth",
        action="store_true",
        help="Invert depth map interpretation (white=far, black=near)",
    )
    daemon_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    daemon_parser.set_defaults(func=command_daemon)

    stop_parser = subparsers.add_parser("stop", help="Stop the wallpaper daemon(s)")
    stop_parser.add_argument(
        "-m", "--monitor", type=str, default=None, help="Stop specific monitor only"
    )
    stop_parser.set_defaults(func=command_stop)

    list_parser = subparsers.add_parser(
        "list-monitors", help="List all monitors and their status"
    )
    list_parser.set_defaults(func=command_list_monitors)

    pregen_parser = subparsers.add_parser(
        "pregenerate", help="Pre-generate depth map for an image"
    )
    pregen_parser.add_argument("image", help="Path to the image")
    pregen_parser.add_argument("-v", "--verbose", action="store_true")
    pregen_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for depth estimation (name or path, default: midas)",
    )
    pregen_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of depth map even if cached",
    )
    pregen_parser.set_defaults(func=command_pregenerate)

    cache_parser = subparsers.add_parser("cache", help="Manage depth map cache")
    cache_group = cache_parser.add_mutually_exclusive_group(required=True)
    cache_group.add_argument("--clear", action="store_true", help="Clear the cache")
    cache_group.add_argument(
        "--list", action="store_true", help="List cached wallpapers"
    )
    cache_parser.set_defaults(func=command_cache)

    download_parser = subparsers.add_parser(
        "download-model", help="Download a depth estimation model"
    )
    download_parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model to download (default: prompt for selection)",
    )
    download_parser.set_defaults(func=command_download_models)

    return parser


def main():
    parser = create_argument_parser()
    arguments = parser.parse_args()

    if not arguments.command:
        parser.print_help()
        return 1

    return arguments.func(arguments)


if __name__ == "__main__":
    sys.exit(main())
