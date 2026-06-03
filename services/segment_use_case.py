from __future__ import annotations

from typing import Any, Mapping

from api.request_models import SegmentParams
from services.contexts import RequestContext
from services.use_case import BaseSyncUseCase


class SegmentUseCase(BaseSyncUseCase):
    task = "segment"

    def parse_params(self, form_data: Mapping[str, Any]) -> SegmentParams:
        return SegmentParams.from_form(form_data, self.config, require_input=True)

    def _build_cache_key(self, params: SegmentParams, image_sha256: str) -> str:
        return self.pipeline.build_cache_key(
            image_sha256=image_sha256,
            prompt_inputs=params.to_prompt_inputs(),
            conf=params.conf,
            overlay=params.overlay,
            output_formats=params.output_formats,
            segment_model=params.segment_model,
            classes=params.classes,
        )

    def _is_model_busy(self, ctx: RequestContext) -> bool:
        return self.model_service.is_busy_for(ctx.params.segment_model)

    def _build_job_payload(self, ctx: RequestContext) -> dict[str, Any]:
        params: SegmentParams = ctx.params
        return {
            "task": "segment",
            "segment_model": params.segment_model,
            "request_id": ctx.request_id,
            "source_path": str(ctx.source_path),
            "prompt_inputs": params.to_prompt_inputs(),
            "classes": params.classes,
            "conf": params.conf,
            "overlay": params.overlay,
            "output_formats": sorted(params.output_formats),
            "decode_ms": ctx.decode_ms,
        }

    def _run_pipeline(self, ctx: RequestContext) -> dict[str, Any]:
        params: SegmentParams = ctx.params
        return self.pipeline.segment_from_saved(
            source_path=str(ctx.source_path),
            request_id=ctx.request_id,
            prompt_inputs=params.to_prompt_inputs(),
            conf=params.conf,
            overlay=params.overlay,
            output_formats=params.output_formats,
            decode_ms=ctx.decode_ms,
            segment_model=params.segment_model,
            classes=params.classes,
        )

    def _result_prompt(self, ctx: RequestContext) -> str | None:
        return ctx.params.prompt
