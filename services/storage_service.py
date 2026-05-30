from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import Config


class StorageService:
    def __init__(self, config: Config):
        self.config = config
        self.base_dir = Path(config.output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stop_event = threading.Event()

    def save_uploaded_image(self, request_id: str, upload_file: FileStorage) -> Path:
        request_dir = self.base_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        original_name = secure_filename(upload_file.filename or "upload")
        if not original_name:
            original_name = "upload"
        source_path = request_dir / f"source_{original_name}"
        upload_file.save(source_path)
        return source_path

    def save_mask(self, request_id: str, idx: int, mask: np.ndarray) -> tuple[str, str]:
        request_dir = self.base_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask_name = f"mask_{idx:03d}.png"
        mask_path = request_dir / mask_name
        Image.fromarray(mask_u8, mode="L").save(mask_path)

        rel_path = f"{request_id}/{mask_name}"
        return rel_path, self.build_url(rel_path)

    def save_overlay(self, request_id: str, overlay_bgr: np.ndarray) -> tuple[str, str]:
        request_dir = self.base_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        overlay_name = "overlay.png"
        overlay_path = request_dir / overlay_name
        overlay_rgb = overlay_bgr[:, :, ::-1]
        Image.fromarray(overlay_rgb).save(overlay_path)

        rel_path = f"{request_id}/{overlay_name}"
        return rel_path, self.build_url(rel_path)

    def build_url(self, rel_path: str) -> str:
        return f"{self.config.output_url_prefix.rstrip('/')}/{rel_path}"

    def start_cleanup_thread(self) -> None:
        thread = threading.Thread(target=self._cleanup_worker, daemon=True, name="output-cleanup-worker")
        thread.start()

    def stop_cleanup_thread(self) -> None:
        self._cleanup_stop_event.set()

    def cleanup_once(self) -> int:
        now = time.time()
        threshold = now - (self.config.output_retention_hours * 3600)
        removed = 0

        for child in self.base_dir.iterdir():
            try:
                if not child.is_dir():
                    continue
                if child.stat().st_mtime >= threshold:
                    continue
                for sub in child.glob("**/*"):
                    if sub.is_file():
                        sub.unlink(missing_ok=True)
                for subdir in sorted([p for p in child.glob("**/*") if p.is_dir()], reverse=True):
                    subdir.rmdir()
                child.rmdir()
                removed += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                logging.warning("Cleanup failed for %s: %s", child, exc)
        return removed

    def _cleanup_worker(self) -> None:
        interval = max(60, self.config.cleanup_interval_seconds)
        while not self._cleanup_stop_event.wait(interval):
            removed = self.cleanup_once()
            if removed:
                logging.info("Removed %d expired output request directories.", removed)
