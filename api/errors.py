from __future__ import annotations

import logging

from flask import Flask, g, jsonify
from werkzeug.exceptions import HTTPException

from errors import APIError


def error_response(code: str, message: str, status_code: int, details: dict | None = None):
    payload = {
        "code": code,
        "message": message,
        "details": details or {},
        "request_id": getattr(g, "request_id", None),
    }
    return jsonify(payload), status_code


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(err: APIError):
        return error_response(err.code, err.message, err.status_code, err.details)

    @app.errorhandler(HTTPException)
    def handle_http_exception(err: HTTPException):
        return error_response("HTTP_ERROR", err.description, err.code or 500)

    @app.errorhandler(Exception)
    def handle_unexpected_error(err: Exception):
        logging.exception("Unhandled error: %s", err)
        return error_response("INTERNAL_ERROR", "Internal server error.", 500)

    @app.errorhandler(413)
    def handle_file_too_large(_err):
        return error_response("FILE_TOO_LARGE", "Uploaded file exceeds size limit.", 413)

