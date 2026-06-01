from __future__ import annotations

from flask import Blueprint, Flask, jsonify, send_from_directory

from api.openapi import openapi_spec
from middlewares.auth import require_api_key
from services.app_services import AppServices
from services.metrics_exporter import render_metrics_text


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
        if services.sam3_service.is_ready:
            return jsonify({"status": "ready"}), 200
        return jsonify({"status": "not_ready", "reason": services.sam3_service.last_error}), 503

    @blueprint.get("/v1/models")
    @require_api_key(config.api_key)
    def model_metadata():
        return jsonify(services.sam3_service.metadata()), 200

    @blueprint.get(f"{config.output_url_prefix}/<path:filename>")
    def serve_output_file(filename: str):
        return send_from_directory(config.output_dir, filename)

    app.register_blueprint(blueprint)

