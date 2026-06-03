from __future__ import annotations

import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from api.parsers import validate_upload_filename
from config import Config
from errors import APIError
from services.storage_service import StorageService


class UploadService:
    def __init__(self, config: Config, storage_service: StorageService):
        self.config = config
        self.storage_service = storage_service

    def save_and_validate_upload(self, upload, request_id: str) -> tuple[Path, int, int, float]:
        validate_upload_filename(upload.filename or "", self.config)
        decode_start = time.perf_counter()
        source_path = self.storage_service.save_uploaded_image(request_id, upload)
        try:
            with Image.open(source_path) as image:
                width, height = image.size
        except UnidentifiedImageError as exc:
            raise APIError("INVALID_IMAGE", "Uploaded file is not a valid image.", 400) from exc

        actual_pixels = width * height
        if actual_pixels > self.config.max_image_pixels:
            raise APIError(
                "IMAGE_TOO_LARGE",
                "Image pixel count exceeds limit.",
                400,
                {"max_pixels": self.config.max_image_pixels, "actual_pixels": actual_pixels},
            )

        decode_ms = (time.perf_counter() - decode_start) * 1000
        return source_path, width, height, decode_ms
