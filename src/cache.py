"""Caching system for depth maps."""

import json
import hashlib
import logging
from pathlib import Path


from src.depth_estimator import load_depth_map, save_depth_map

logger = logging.getLogger(__name__)

cache_directory = Path.home() / ".cache" / "waydeeper"


class DepthCache:
    def __init__(self, custom_directory=None):
        self.cache_directory = (
            Path(custom_directory) if custom_directory else cache_directory
        )
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.depth_directory = self.cache_directory / "depth"
        self.metadata_directory = self.cache_directory / "metadata"
        self.depth_directory.mkdir(exist_ok=True)
        self.metadata_directory.mkdir(exist_ok=True)

    def compute_image_hash(self, image_path):
        hasher = hashlib.blake2b(digest_size=16)
        with open(image_path, "rb") as file:
            while chunk := file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_depth_file_path(self, image_hash):
        return self.depth_directory / f"{image_hash}.png"

    def get_metadata_file_path(self, image_hash):
        return self.metadata_directory / f"{image_hash}.json"

    def get_cached_depth(self, image_path):
        image_path = Path(image_path)
        if not image_path.exists():
            return None

        image_hash = self.compute_image_hash(image_path)
        depth_path = self.get_depth_file_path(image_hash)
        metadata_path = self.get_metadata_file_path(image_hash)

        if not depth_path.exists() or not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as file:
                metadata = json.load(file)

            current_modification_time = image_path.stat().st_mtime
            if metadata.get("modification_time") != current_modification_time:
                logger.debug(f"Image {image_path} modified, cache invalidated")
                return None

            depth_map = load_depth_map(str(depth_path))
            logger.debug(f"Cache hit for {image_path}")
            return depth_map

        except Exception as error:
            logger.warning(f"Failed to load cached depth for {image_path}: {error}")
            return None

    def cache_depth(self, image_path, depth_map):
        image_path = Path(image_path)
        image_hash = self.compute_image_hash(image_path)
        depth_path = self.get_depth_file_path(image_hash)
        metadata_path = self.get_metadata_file_path(image_hash)

        try:
            save_depth_map(depth_map, str(depth_path))

            metadata = {
                "original_path": str(image_path),
                "modification_time": image_path.stat().st_mtime,
                "width": depth_map.shape[1],
                "height": depth_map.shape[0],
            }

            with open(metadata_path, "w") as file:
                json.dump(metadata, file)

            logger.debug(f"Cached depth map for {image_path}")

        except Exception as error:
            logger.warning(f"Failed to cache depth for {image_path}: {error}")

    def clear_cache(self):
        for file_path in self.depth_directory.iterdir():
            if file_path.is_file():
                file_path.unlink()

        for file_path in self.metadata_directory.iterdir():
            if file_path.is_file():
                file_path.unlink()

        logger.info("Cache cleared")

    def list_cached(self):
        cached_items = []

        for metadata_file in self.metadata_directory.iterdir():
            if metadata_file.suffix == ".json":
                try:
                    with open(metadata_file, "r") as file:
                        cached_items.append(json.load(file))
                except Exception:
                    pass

        return cached_items

    def get_cache_size_bytes(self):
        total_size = 0

        for file_path in self.depth_directory.iterdir():
            if file_path.is_file():
                total_size += file_path.stat().st_size

        for file_path in self.metadata_directory.iterdir():
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size
