from __future__ import annotations

import base64
import hashlib
import hmac
import json
import urllib.request


def build_webhook_signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def post_webhook(url: str, body: dict, secret: str | None = None, timeout_seconds: float = 5.0) -> None:
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=body_bytes, method="POST")
    req.add_header("Content-Type", "application/json")
    if secret:
        req.add_header("X-Webhook-Signature", build_webhook_signature(secret, body_bytes))
    with urllib.request.urlopen(req, timeout=timeout_seconds):
        return

