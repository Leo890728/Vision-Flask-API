from __future__ import annotations

from functools import wraps

from flask import Request, request

from errors import APIError


def _api_key_from_header(req: Request) -> str:
    return req.headers.get("X-API-Key", "")


def require_api_key(expected_api_key: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _api_key_from_header(request) != expected_api_key:
                raise APIError("UNAUTHORIZED", "Invalid or missing API key.", 401)
            return func(*args, **kwargs)

        return wrapper

    return decorator
