from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, g, has_request_context, request

from services.infra.metrics_service import MetricsService


class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(g, "request_id", "-") if has_request_context() else "-"
        return True


def register_request_hooks(app: Flask, metrics_service: MetricsService) -> None:
    @app.before_request
    def setup_request_context():
        g.request_id = str(uuid.uuid4())
        g.request_start = time.perf_counter()
        g.prompt = None
        metrics_service.inc_request()

    @app.after_request
    def log_request(response):
        elapsed_ms = (time.perf_counter() - g.request_start) * 1000
        logging.info(
            "method=%s path=%s status=%s latency_ms=%.1f prompt=%s",
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
            getattr(g, "prompt", None),
        )
        if response.status_code >= 400:
            metrics_service.inc_error()
        response.headers["X-Request-ID"] = g.request_id
        return response

