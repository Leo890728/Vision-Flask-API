from __future__ import annotations

from typing import Any, Mapping

from api.request_models import DetectParams
from services.contexts import RequestContext
from services.use_case import BaseSyncUseCase


class DetectUseCase(BaseSyncUseCase):
    task = "detect"

    def parse_params(self, form_data: Mapping[str, Any]) -> DetectParams:
        return DetectParams.from_form(form_data, self.config)

    def _build_cache_key(self, params: DetectParams, image_sha256: str) -> str:
        return self.pipeline.build_cache_key(
            image_sha256=image_sha256,
            conf=params.conf,
            overlay=params.overlay,
            classes=params.classes,
        )

    def _is_model_busy(self, ctx: RequestContext) -> bool:
        return self.model_service.is_busy

    def _build_job_payload(self, ctx: RequestContext) -> dict[str, Any]:
        params: DetectParams = ctx.params
        return {
            "task": "detect",
            "request_id": ctx.request_id,
            "source_path": str(ctx.source_path),
            "conf": params.conf,
            "overlay": params.overlay,
            "classes": params.classes,
            "decode_ms": ctx.decode_ms,
        }

    def _run_pipeline(self, ctx: RequestContext) -> dict[str, Any]:
        params: DetectParams = ctx.params
        return self.pipeline.detect_from_saved(
            source_path=str(ctx.source_path),
            request_id=ctx.request_id,
            conf=params.conf,
            overlay=params.overlay,
            classes=params.classes,
            decode_ms=ctx.decode_ms,
        )
