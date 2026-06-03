from __future__ import annotations

import logging
import os
import urllib
from pathlib import Path

from flask import Flask

from api.error_handlers import register_error_handlers
from api.parsers import parse_bool
from api.request_hooks import RequestIDFilter, register_request_hooks
from config import Config
from errors import APIError
from routes.detect import register_detect_routes
from routes.jobs import register_job_routes
from routes.segment import register_segment_routes
from routes.system import register_system_routes
from services.app_services import build_app_services
from services.infra.webhook_utils import post_webhook


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(request_id)s] %(message)s",
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIDFilter())


def _initialize_runtime(app: Flask) -> None:
    services = app.extensions["vision_services"]
    config: Config = services["config"]

    def post_webhook_with_config(url: str, body: dict, secret: str | None = None) -> None:
        post_webhook(
            url=url,
            body=body,
            secret=secret,
            timeout_seconds=config.webhook_timeout_seconds,
        )

    def job_completion_hook(record) -> None:
        if not record.webhook_url:
            return
        payload = {
            "job_id": record.job_id,
            "status": record.status,
            "result": record.result,
            "error": record.error,
            "updated_at": record.updated_at,
        }
        secret = None
        if record.payload:
            secret = record.payload.get("webhook_secret")
        services["webhook_retry_service"].submit(record.job_id, record.webhook_url, payload, secret=secret)

    def unified_job_handler(payload: dict):
        task = (payload or {}).get("task", "segment")
        if task == "detect":
            return services["detection_pipeline"].job_handler(payload)
        if task == "segment":
            return services["pipeline"].job_handler(payload)
        raise APIError("INVALID_TASK", f"Unsupported task: {task}", 400)

    with app.app_context():
        try:
            if not config.skip_model_load:
                services["segmentation_service"].load()
                services["detection_service"].load()
                sample_image = "image (2).jpg"
                if Path(sample_image).exists():
                    services["segmentation_service"].warmup(sample_image_path=sample_image)
                    services["detection_service"].warmup(sample_image_path=sample_image)
            services["storage_service"].start_cleanup_thread()
            services["webhook_retry_service"].start(post_webhook_with_config)
            services["job_service"].start(unified_job_handler, completion_hook=job_completion_hook)
            logging.info("Segmentation + detection services initialized.")
        except Exception as exc:
            logging.exception("Failed to initialize model services: %s", exc)


def create_app(
    sam3_backend=None,
    detection_backend=None,
    detection_backends=None,
    yolo_seg_backend=None,
    segmentation_service=None,
    detection_service=None,
    storage_service=None,
) -> Flask:
    config = Config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_bytes

    _configure_logging()
    built = build_app_services(
        config=config,
        sam3_backend=sam3_backend,
        detection_backend=detection_backend,
        detection_backends=detection_backends,
        yolo_seg_backend=yolo_seg_backend,
        segmentation_service=segmentation_service,
        detection_service=detection_service,
        storage_service=storage_service,
    )

    app.extensions["vision_services"] = {
        "config": built.config,
        **built.extension_map(),
    }

    register_request_hooks(app, built.metrics_service)
    register_error_handlers(app)
    register_system_routes(app, built)
    register_segment_routes(app, built)
    register_detect_routes(app, built)
    register_job_routes(app, built)
    _initialize_runtime(app)
    return app


if parse_bool(os.getenv("SAM3_AUTOSTART_APP"), True):
    app = create_app()
else:
    app = Flask(__name__)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
