from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class UseCaseResult:
    payload: dict[str, Any]
    status_code: int
    prompt: str | None = None


@dataclass
class RequestContext:
    request_id: str
    source_path: Path
    image_sha256: str
    decode_ms: float
    params: Any
    cache_key: str | None = None
