from __future__ import annotations

from typing import Any, Mapping

from werkzeug.datastructures import FileStorage

from config import Config
from errors import APIError
from services.cache_service import CacheService
from services.contexts import RequestContext, UseCaseResult
from services.job_service import JobService
from services.metrics_service import MetricsService
from services.storage_service import StorageService
from services.upload_service import UploadService


class BaseSyncUseCase:
    """Shared sync/auto-queue flow for the detect and segment endpoints.

    Subclasses fill in the task-specific hooks (param parsing, cache key,
    busy check, job payload, pipeline call). The base owns the request flow:
    cache lookup -> auto-queue -> run now, plus the equivalent batch item path.
    """

    task: str = ""

    def __init__(
        self,
        config: Config,
        pipeline: Any,
        upload_service: UploadService,
        storage_service: StorageService,
        cache_service: CacheService,
        job_service: JobService,
        metrics_service: MetricsService,
        model_service: Any,
    ):
        self.config = config
        self.pipeline = pipeline
        self.upload_service = upload_service
        self.storage_service = storage_service
        self.cache_service = cache_service
        self.job_service = job_service
        self.metrics_service = metrics_service
        self.model_service = model_service

    def run_sync_request(
        self, upload: FileStorage | None, form_data: Mapping[str, Any], request_id: str
    ) -> UseCaseResult:
        if upload is None:
            raise APIError("MISSING_IMAGE", "image is required.", 400)
        params = self.parse_params(form_data)
        ctx = self._build_context(upload, params, request_id)
        return self._get_cached(ctx) or self._maybe_queue(ctx) or self._run_now(ctx)

    def run_batch_item(self, upload: FileStorage, params: Any, request_id: str) -> dict[str, Any]:
        """Cache-aware synchronous run for a single batch image (no auto-queue)."""
        ctx = self._build_context(upload, params, request_id)
        cached = self._get_cached(ctx)
        if cached is not None:
            return cached.payload
        return self._run_now(ctx).payload

    def _build_context(self, upload: FileStorage, params: Any, request_id: str) -> RequestContext:
        source_path, _width, _height, decode_ms = self.upload_service.save_and_validate_upload(upload, request_id)
        image_sha256 = self.storage_service.file_sha256(source_path)
        return RequestContext(
            request_id=request_id,
            source_path=source_path,
            image_sha256=image_sha256,
            decode_ms=decode_ms,
            params=params,
            cache_key=self._build_cache_key(params, image_sha256),
        )

    def _get_cached(self, ctx: RequestContext) -> UseCaseResult | None:
        if ctx.cache_key is None:
            return None
        cached = self.cache_service.get(ctx.cache_key)
        if cached is None:
            return None
        payload = dict(cached)
        payload["request_id"] = ctx.request_id
        payload["cached"] = True
        payload["cache_key"] = ctx.cache_key
        return UseCaseResult(payload=payload, status_code=200, prompt=self._result_prompt(ctx))

    def _maybe_queue(self, ctx: RequestContext) -> UseCaseResult | None:
        if not self.config.enable_auto_queue or not self._is_model_busy(ctx):
            return None
        if self.job_service.stats()["queue_size"] >= self.config.auto_queue_max_size:
            raise APIError("QUEUE_FULL", "Server is overloaded, queue is full.", 503)

        job_id = self.job_service.submit(self._build_job_payload(ctx))
        self.metrics_service.inc_auto_queued()
        self.metrics_service.inc_jobs_created()
        return UseCaseResult(
            payload={
                "request_id": ctx.request_id,
                "status": "queued",
                "mode": "auto_queued",
                "job_id": job_id,
                "status_url": f"/v1/jobs/{job_id}",
            },
            status_code=202,
            prompt=self._result_prompt(ctx),
        )

    def _run_now(self, ctx: RequestContext) -> UseCaseResult:
        payload = self._run_pipeline(ctx)
        payload["cache_key"] = ctx.cache_key
        if ctx.cache_key is not None:
            self.cache_service.set(ctx.cache_key, payload)
        return UseCaseResult(payload=payload, status_code=200, prompt=self._result_prompt(ctx))

    # --- task-specific hooks -------------------------------------------------
    def parse_params(self, form_data: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def _build_cache_key(self, params: Any, image_sha256: str) -> str | None:
        raise NotImplementedError

    def _is_model_busy(self, ctx: RequestContext) -> bool:
        raise NotImplementedError

    def _build_job_payload(self, ctx: RequestContext) -> dict[str, Any]:
        raise NotImplementedError

    def _run_pipeline(self, ctx: RequestContext) -> dict[str, Any]:
        raise NotImplementedError

    def _result_prompt(self, ctx: RequestContext) -> str | None:
        return None
