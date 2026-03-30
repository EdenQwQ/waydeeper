"""Daemon that manages depth map generation and wallpaper rendering."""

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from src.depth_estimator import DepthEstimator
from src.cache import DepthCache
from src.ipc import DaemonSocket

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfiguration:
    wallpaper_path: str | None = None
    strength: float = 0.02
    strength_x: float = 0.02
    strength_y: float = 0.02
    smooth_animation: bool = True
    animation_speed: float = 0.02
    fps: int = 60
    active_delay_ms: float = 150.0
    idle_timeout_ms: float = 500.0
    model_path: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DaemonConfiguration:
    wallpaper_path: str | None = None
    strength: float = 0.02
    strength_x: float = 0.02
    strength_y: float = 0.02
    smooth_animation: bool = True
    animation_speed: float = 0.02
    fps: int = 60
    active_delay_ms: float = 150.0
    idle_timeout_ms: float = 500.0
    monitor: str = "0"
    monitors: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_directory: str = field(
        default_factory=lambda: os.path.expanduser("~/.cache/waydeeper")
    )
    version: int = 1

    @classmethod
    def from_json_file(cls, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

        saved_fields = {"monitors", "cache_directory", "version"}
        filtered_data = {
            key: value for key, value in data.items() if key in saved_fields
        }

        return cls(**filtered_data)

    def to_json_file(self, file_path):
        data = {
            "version": self.version,
            "monitors": self.monitors,
        }

        default_cache = os.path.expanduser("~/.cache/waydeeper")
        if self.cache_directory != default_cache:
            data["cache_directory"] = self.cache_directory

        with open(file_path, "w") as file:
            json.dump(data, file, indent=2)

    def get_monitor_config(self, monitor_id):
        key = str(monitor_id)
        if key in self.monitors:
            return MonitorConfiguration.from_dict(self.monitors[key])

        return MonitorConfiguration(
            wallpaper_path=self.wallpaper_path,
            strength=self.strength,
            strength_x=self.strength_x,
            strength_y=self.strength_y,
            smooth_animation=self.smooth_animation,
            animation_speed=self.animation_speed,
            fps=self.fps,
            active_delay_ms=self.active_delay_ms,
            idle_timeout_ms=self.idle_timeout_ms,
            model_path=None,
        )

    def set_monitor_config(self, monitor_id, config):
        self.monitors[str(monitor_id)] = config.to_dict()


class DepthWallpaperDaemon:
    configuration_directory = Path.home() / ".config" / "waydeeper"
    configuration_file_path = configuration_directory / "config.json"

    def __init__(self):
        self.configuration = DaemonConfiguration()
        self.cache_manager = None
        self.depth_estimator = None
        self.is_running = False
        self.current_monitor = "0"
        self.stop_event = threading.Event()
        self.ipc_socket = None
        self.start_time = None
        self.force_regenerate = False
        self.configuration_directory.mkdir(parents=True, exist_ok=True)

    def load_configuration(self):
        if self.configuration_file_path.exists():
            self.configuration = DaemonConfiguration.from_json_file(
                str(self.configuration_file_path)
            )
            logger.info(f"Loaded configuration from {self.configuration_file_path}")

    def save_configuration(self):
        self.configuration.to_json_file(str(self.configuration_file_path))
        logger.info(f"Saved configuration to {self.configuration_file_path}")

    def compute_image_hash(self, image_path):
        hasher = hashlib.blake2b(digest_size=16)
        with open(image_path, "rb") as file:
            while chunk := file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_model_name_for_cache(self, model_path=None):
        """Get the model name for cache key generation."""
        if model_path:
            return Path(model_path).stem
        return "midas"  # default

    def ensure_depth_map_exists(self, image_path, model_path=None, force_regenerate=False):
        if self.cache_manager is None:
            self.cache_manager = DepthCache(self.configuration.cache_directory)

        model_name = self.get_model_name_for_cache(model_path)

        if not force_regenerate:
            cached_depth = self.cache_manager.get_cached_depth(image_path, model_name)
            if cached_depth is not None:
                # Include model in hash for consistency
                hasher = hashlib.blake2b(digest_size=16)
                with open(image_path, "rb") as file:
                    while chunk := file.read(8192):
                        hasher.update(chunk)
                hasher.update(model_name.encode("utf-8"))
                image_hash = hasher.hexdigest()
                depth_file_path = self.cache_manager.get_depth_file_path(image_hash)
                return str(depth_file_path)

        if self.depth_estimator is None:
            self.depth_estimator = DepthEstimator(model_path)

        # Get the actual model name being used
        actual_model_name = self.depth_estimator.model_name
        logger.info(f"Generating depth map for: {image_path} (model: {actual_model_name})")

        depth_map = self.depth_estimator.estimate(image_path)
        self.cache_manager.cache_depth(image_path, depth_map, actual_model_name)

        # Compute hash with model name for the output path
        hasher = hashlib.blake2b(digest_size=16)
        with open(image_path, "rb") as file:
            while chunk := file.read(8192):
                hasher.update(chunk)
        hasher.update(actual_model_name.encode("utf-8"))
        image_hash = hasher.hexdigest()
        depth_file_path = self.cache_manager.get_depth_file_path(image_hash)

        logger.info(f"Depth map generated: {depth_file_path}")
        return str(depth_file_path)

    def handle_ipc_command(self, command: str, params: dict) -> Any:
        """Handle IPC commands from clients."""
        monitor_config = self.configuration.get_monitor_config(self.current_monitor)

        if command == "PING":
            return {"status": "pong"}

        elif command == "STATUS":
            return {
                "monitor": self.current_monitor,
                "running": self.is_running,
                "wallpaper": monitor_config.wallpaper_path,
                "strength_x": monitor_config.strength_x,
                "strength_y": monitor_config.strength_y,
                "fps": monitor_config.fps,
                "smooth_animation": monitor_config.smooth_animation,
                "animation_speed": monitor_config.animation_speed,
                "uptime": time.time() - self.start_time if self.start_time else 0,
            }

        elif command == "STOP":
            logger.info("Received STOP command via IPC")

            # Schedule shutdown - give time for response to be sent
            def delayed_stop():
                time.sleep(0.5)  # Give client time to receive response
                os.kill(os.getpid(), signal.SIGTERM)

            threading.Thread(target=delayed_stop, daemon=True).start()
            return {"stopping": True}

        elif command == "RELOAD":
            logger.info("Received RELOAD command via IPC")
            self.load_configuration()
            return {"reloaded": True}

        else:
            return {"error": f"Unknown command: {command}"}

    def start_ipc(self):
        """Start IPC socket server."""
        self.ipc_socket = DaemonSocket(self.current_monitor, self.handle_ipc_command)
        self.ipc_socket.start()

    def stop_ipc(self):
        """Stop IPC socket server."""
        if self.ipc_socket:
            self.ipc_socket.stop()
            self.ipc_socket = None

    def start(
        self,
        wallpaper_path=None,
        strength=None,
        strength_x=None,
        strength_y=None,
        monitor=None,
        smooth_animation=None,
        animation_speed=None,
        fps=None,
        active_delay_ms=None,
        idle_timeout_ms=None,
        model_path=None,
        regenerate=False,
        ready_callback=None,
    ):
        if monitor is not None:
            self.configuration.monitor = str(monitor)
        self.current_monitor = self.configuration.monitor
        monitor_id = self.current_monitor

        self.start_time = time.time()

        # Start IPC socket
        self.start_ipc()

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        monitor_config = self.configuration.get_monitor_config(monitor_id)

        if wallpaper_path is None and monitor_config.wallpaper_path is not None:
            wallpaper_path = monitor_config.wallpaper_path
            logger.info(
                f"Using configured wallpaper for monitor {monitor_id}: {wallpaper_path}"
            )

        if wallpaper_path:
            monitor_config.wallpaper_path = wallpaper_path
        if strength is not None:
            monitor_config.strength = strength
            monitor_config.strength_x = strength
            monitor_config.strength_y = strength
        if strength_x is not None:
            monitor_config.strength_x = strength_x
        if strength_y is not None:
            monitor_config.strength_y = strength_y
        if smooth_animation is not None:
            monitor_config.smooth_animation = smooth_animation
        if animation_speed is not None:
            monitor_config.animation_speed = animation_speed
        if fps is not None:
            monitor_config.fps = fps
        if active_delay_ms is not None:
            monitor_config.active_delay_ms = active_delay_ms
        if idle_timeout_ms is not None:
            monitor_config.idle_timeout_ms = idle_timeout_ms
        if model_path is not None:
            monitor_config.model_path = model_path

        self.configuration.set_monitor_config(monitor_id, monitor_config)
        self.force_regenerate = regenerate

        self.configuration.wallpaper_path = monitor_config.wallpaper_path
        self.configuration.strength = monitor_config.strength
        self.configuration.strength_x = monitor_config.strength_x
        self.configuration.strength_y = monitor_config.strength_y
        self.configuration.smooth_animation = monitor_config.smooth_animation
        self.configuration.animation_speed = monitor_config.animation_speed
        self.configuration.fps = monitor_config.fps
        self.configuration.active_delay_ms = monitor_config.active_delay_ms
        self.configuration.idle_timeout_ms = monitor_config.idle_timeout_ms

        logger.info(
            f"Config for monitor {monitor_id}: "
            f"wallpaper={monitor_config.wallpaper_path}, "
            f"strength_x={monitor_config.strength_x}, "
            f"strength_y={monitor_config.strength_y}, "
            f"smooth={monitor_config.smooth_animation}, "
            f"fps={monitor_config.fps}"
        )

        if not monitor_config.wallpaper_path:
            self.stop_ipc()
            raise ValueError("Wallpaper path is required")

        if not os.path.exists(monitor_config.wallpaper_path):
            self.stop_ipc()
            raise FileNotFoundError(
                f"Wallpaper not found: {monitor_config.wallpaper_path}"
            )

        self.save_configuration()

        depth_path = self.ensure_depth_map_exists(
            monitor_config.wallpaper_path,
            model_path=monitor_config.model_path,
            force_regenerate=self.force_regenerate
        )

        logger.info("Starting wallpaper renderer...")
        self.is_running = True

        try:
            from src.renderer import run_wallpaper

            logger.info(
                f"Calling renderer with: "
                f"strength_x={monitor_config.strength_x}, "
                f"strength_y={monitor_config.strength_y}"
            )

            # GTK must run in main thread, so we run renderer here
            # IPC runs in background thread and will signal us to stop via SIGTERM
            run_wallpaper(
                monitor_config.wallpaper_path,
                depth_path,
                monitor_id,
                monitor_config.strength_x,
                monitor_config.strength_y,
                monitor_config.smooth_animation,
                monitor_config.animation_speed,
                monitor_config.fps,
                monitor_config.active_delay_ms,
                monitor_config.idle_timeout_ms,
                ready_callback,
            )

        except Exception:
            logger.exception("Renderer failed")
            self.stop_ipc()
            raise

    def stop(self):
        self.is_running = False
        self.stop_event.set()
        self.stop_ipc()
        logger.info("Daemon stopped")

    def pregenerate_depth_map(self, image_path, model_path=None, force_regenerate=False):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        return self.ensure_depth_map_exists(image_path, model_path, force_regenerate)

    def clear_cache(self):
        if self.cache_manager is None:
            self.cache_manager = DepthCache(self.configuration.cache_directory)
        self.cache_manager.clear_cache()

    def list_cached_wallpapers(self):
        if self.cache_manager is None:
            self.cache_manager = DepthCache(self.configuration.cache_directory)

        cached_items = self.cache_manager.list_cached()
        if not cached_items:
            print("No cached wallpapers found.")
            return

        print("Cached wallpapers:")
        for item in cached_items:
            model_name = item.get('model_name', 'unknown')
            print(f"  - {item['original_path']} ({item['width']}x{item['height']}, model: {model_name})")


def create_argument_parser():
    parser = argparse.ArgumentParser(
        prog="waydeeper-daemon",
        description="Waydeeper daemon for Wayland (internal use)",
    )

    parser.add_argument(
        "-w", "--wallpaper", required=True, help="Path to the wallpaper image"
    )
    parser.add_argument(
        "-s", "--strength", type=float, default=None, help="Parallax strength"
    )
    parser.add_argument("--strength-x", type=float, help="Parallax strength for X axis")
    parser.add_argument("--strength-y", type=float, help="Parallax strength for Y axis")
    parser.add_argument(
        "-m", "--monitor", type=str, default="0", help="Monitor name or index"
    )
    parser.add_argument(
        "--smooth-animation",
        action="store_true",
        default=True,
        help="Enable smooth animation",
    )
    parser.add_argument(
        "--no-smooth-animation", action="store_true", help="Disable smooth animation"
    )
    parser.add_argument(
        "--animation-speed", type=float, default=0.02, help="Animation speed"
    )
    parser.add_argument(
        "--fps", type=int, default=60, choices=[30, 60], help="Frame rate"
    )
    parser.add_argument(
        "--active-delay", type=float, default=150.0, help="Active delay in ms"
    )
    parser.add_argument(
        "--idle-timeout", type=float, default=500.0, help="Idle timeout in ms"
    )
    parser.add_argument("--model-path", help="Path to MiDaS ONNX model")
    parser.add_argument(
        "--cache-dir",
        default=os.path.expanduser("~/.cache/waydeeper"),
        help="Cache directory",
    )
    parser.add_argument(
        "--regenerate", action="store_true", help="Force regeneration of depth map"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--ready-signal", action="store_true", help="Output READY signal when started"
    )

    return parser


def main():
    parser = create_argument_parser()
    arguments = parser.parse_args()

    log_level = logging.DEBUG if arguments.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    daemon = DepthWallpaperDaemon()
    daemon.load_configuration()

    if arguments.cache_dir:
        daemon.configuration.cache_directory = arguments.cache_dir

    try:
        smooth_animation = arguments.smooth_animation
        if arguments.no_smooth_animation:
            smooth_animation = False

        def ready_callback():
            if arguments.ready_signal:
                import sys

                print("READY", flush=True)
                sys.stdout.flush()

        daemon.start(
            wallpaper_path=arguments.wallpaper,
            strength=arguments.strength,
            strength_x=arguments.strength_x,
            strength_y=arguments.strength_y,
            monitor=arguments.monitor,
            smooth_animation=smooth_animation,
            animation_speed=arguments.animation_speed,
            fps=arguments.fps,
            active_delay_ms=arguments.active_delay,
            idle_timeout_ms=arguments.idle_timeout,
            model_path=arguments.model_path,
            regenerate=arguments.regenerate,
            ready_callback=ready_callback,
        )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        daemon.stop()
        return 0
    except Exception:
        logger.exception("Daemon failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
