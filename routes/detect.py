from __future__ import annotations

import time

from flask import Blueprint, Flask, g, jsonify, request

from errors import APIError
from middlewares.auth import require_api_key
from middlewares.rate_limit import apply_rate_limit
from services.app_services import AppServices


def register_detect_routes(app: Flask, services: AppServices) -> None:
    config = services.config
    blueprint = Blueprint("detect", __name__)

    @blueprint.post("/v1/detect")
    @require_api_key(config.api_key)
    @apply_rate_limit(services.rate_limiter)
    def detect():
        result = services.detect_use_case.run_sync_request(
            upload=request.files.get("image"),
            form_data=request.form,
            request_id=g.request_id,
        )
        g.prompt = None
        return jsonify(result.payload), result.status_code

    @blueprint.post("/v1/detect/batch")
    @require_api_key(config.api_key)
    @apply_rate_limit(services.rate_limiter)
    def detect_batch():
        uploads = request.files.getlist("images")
        if not uploads:
            raise APIError("MISSING_IMAGE", "images is required.", 400)
        if len(uploads) > config.max_batch_images:
            raise APIError("BATCH_TOO_LARGE", f"Maximum {config.max_batch_images} images per batch.", 400)

        use_case = services.detect_use_case
        params = use_case.parse_params(request.form)
        g.prompt = None

        batch_start = time.perf_counter()
        items = []
        for idx, upload in enumerate(uploads):
            item_request_id = f"{g.request_id}_{idx:03d}"
            try:
                result = use_case.run_batch_item(upload, params, item_request_id)
                items.append(
                    {
                        "index": idx,
                        "filename": upload.filename,
                        "status": "ok",
                        "result": result,
                    }
                )
            except APIError as err:
                items.append(
                    {
                        "index": idx,
                        "filename": upload.filename,
                        "status": "error",
                        "error": {"code": err.code, "message": err.message, "details": err.details},
                    }
                )

        total_ms = (time.perf_counter() - batch_start) * 1000
        return (
            jsonify(
                {
                    "request_id": g.request_id,
                    "batch_size": len(uploads),
                    "timing_ms": {"total": round(total_ms, 2)},
                    "items": items,
                }
            ),
            200,
        )

    app.register_blueprint(blueprint)
