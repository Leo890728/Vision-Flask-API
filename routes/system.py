from __future__ import annotations

from flask import Blueprint, Flask, jsonify, send_from_directory

from api.openapi import openapi_spec
from middlewares.auth import require_api_key
from services.app_services import AppServices
from services.infra.metrics_exporter import render_metrics_text


def _model_catalog_response(services: AppServices) -> dict:
    runtime_by_name: dict[str, dict] = {}

    detection_meta = services.detection_service.metadata()
    runtime_by_name[detection_meta["name"]] = detection_meta

    segmentation_meta = services.segmentation_service.metadata()
    for backend_meta in segmentation_meta["backends"].values():
        runtime_by_name[backend_meta["name"]] = backend_meta

    models = []
    for name, model_cfg in services.config.models.items():
        runtime = runtime_by_name.get(name)
        models.append(
            {
                "name": model_cfg.name,
                "task": model_cfg.task,
                "model_path": model_cfg.model_path,
                "default_conf": model_cfg.default_conf,
                "half": model_cfg.half,
                "device": model_cfg.device,
                "ready": bool(runtime["ready"]) if runtime else False,
                "busy": bool(runtime["busy"]) if runtime else False,
                "last_error": runtime["last_error"] if runtime else None,
                "active": runtime is not None,
                "default": (
                    name == services.config.detect_default_model
                    if model_cfg.task == "detect"
                    else name == services.config.segment_default_model
                ),
            }
        )

    return {
        "defaults": {
            "detect": services.config.detect_default_model,
            "segment": services.config.segment_default_model,
        },
        "models": models,
    }


def register_system_routes(app: Flask, services: AppServices) -> None:
    config = services.config
    blueprint = Blueprint("system", __name__)

    @blueprint.get("/openapi.json")
    def openapi_json():
        return jsonify(openapi_spec(config)), 200

    @blueprint.get("/metrics")
    @require_api_key(config.api_key)
    def metrics():
        payload = render_metrics_text(
            metrics_service=services.metrics_service,
            job_service=services.job_service,
            webhook_retry_service=services.webhook_retry_service,
            cache_service=services.cache_service,
        )
        return payload, 200, {"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}

    @blueprint.get("/docs")
    def swagger_docs():
        html = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SAM3 API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true
      });
    </script>
  </body>
</html>"""
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    @blueprint.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"}), 200

    @blueprint.get("/readyz")
    def readyz():
        if services.segmentation_service.is_ready:
            return jsonify({"status": "ready"}), 200
        return jsonify({"status": "not_ready", "reason": services.segmentation_service.last_error}), 503

    @blueprint.get("/v1/models")
    @require_api_key(config.api_key)
    def model_metadata():
        return jsonify(_model_catalog_response(services)), 200

    @blueprint.get(f"{config.output_url_prefix}/<path:filename>")
    def serve_output_file(filename: str):
        return send_from_directory(config.output_dir, filename)

    app.register_blueprint(blueprint)
