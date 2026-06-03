from __future__ import annotations

from dataclasses import dataclass

from config import Config
from middlewares.rate_limit import InMemoryRateLimiter
from services.infra.cache_service import CacheService
from services.usecases.detect_use_case import DetectUseCase
from services.backends.detection_service import DetectionService
from services.usecases.detection_pipeline import DetectionPipeline
from services.infra.job_service import JobService
from services.infra.metrics_service import MetricsService
from services.infra.model_pool import ModelPool
from services.usecases.segment_use_case import SegmentUseCase
from services.usecases.segmentation_pipeline import SegmentationPipeline
from services.backends.segmentation_service import SegmentationService
from services.infra.storage_service import StorageService
from services.infra.upload_service import UploadService
from services.infra.webhook_retry_service import WebhookRetryService


@dataclass
class AppServices:
    config: Config
    segmentation_service: SegmentationService
    detection_service: DetectionService
    storage_service: StorageService
    upload_service: UploadService
    job_service: JobService
    cache_service: CacheService
    metrics_service: MetricsService
    webhook_retry_service: WebhookRetryService
    rate_limiter: InMemoryRateLimiter
    pipeline: SegmentationPipeline
    detection_pipeline: DetectionPipeline
    segment_use_case: SegmentUseCase
    detect_use_case: DetectUseCase

    def extension_map(self) -> dict:
        return {
            "segmentation_service": self.segmentation_service,
            "detection_service": self.detection_service,
            "storage_service": self.storage_service,
            "upload_service": self.upload_service,
            "job_service": self.job_service,
            "cache_service": self.cache_service,
            "metrics_service": self.metrics_service,
            "webhook_retry_service": self.webhook_retry_service,
            "pipeline": self.pipeline,
            "detection_pipeline": self.detection_pipeline,
            "segment_use_case": self.segment_use_case,
            "detect_use_case": self.detect_use_case,
        }


def build_app_services(
    config: Config,
    sam3_backend=None,
    detection_backend=None,
    detection_backends: dict | None = None,
    yolo_seg_backend=None,
    segmentation_service: SegmentationService | None = None,
    detection_service: DetectionService | None = None,
    storage_service: StorageService | None = None,
) -> AppServices:
    pool = ModelPool(max_loaded=config.max_loaded_models)
    segmentation_service = segmentation_service or SegmentationService(
        config=config,
        sam3_backend=sam3_backend,
        yolo_seg_backend=yolo_seg_backend,
        pool=pool,
    )
    if detection_service is None:
        if detection_backends is not None:
            detection_service = DetectionService(config, backends=detection_backends, pool=pool)
        elif detection_backend is not None:
            detection_service = DetectionService(config, backends={detection_backend.name: detection_backend}, pool=pool)
        else:
            detection_service = DetectionService(config, pool=pool)
    storage_service = storage_service or StorageService(config)
    upload_service = UploadService(config=config, storage_service=storage_service)
    rate_limiter = InMemoryRateLimiter(config.rate_limit_per_minute)
    job_service = JobService(
        db_path=config.job_db_path,
        worker_count=config.job_worker_count,
        retention_hours=config.job_retention_hours,
    )
    cache_service = CacheService(ttl_seconds=config.cache_ttl_seconds)
    metrics_service = MetricsService()
    webhook_retry_service = WebhookRetryService(
        max_retries=config.webhook_max_retries,
        base_delay_seconds=config.webhook_retry_base_seconds,
    )
    pipeline = SegmentationPipeline(
        config=config,
        segmentation_service=segmentation_service,
        storage_service=storage_service,
        metrics_service=metrics_service,
        upload_service=upload_service,
    )
    detection_pipeline = DetectionPipeline(
        config=config,
        detection_service=detection_service,
        storage_service=storage_service,
        metrics_service=metrics_service,
        upload_service=upload_service,
    )
    segment_use_case = SegmentUseCase(
        config=config,
        pipeline=pipeline,
        upload_service=upload_service,
        storage_service=storage_service,
        cache_service=cache_service,
        job_service=job_service,
        metrics_service=metrics_service,
        model_service=segmentation_service,
    )
    detect_use_case = DetectUseCase(
        config=config,
        pipeline=detection_pipeline,
        upload_service=upload_service,
        storage_service=storage_service,
        cache_service=cache_service,
        job_service=job_service,
        metrics_service=metrics_service,
        model_service=detection_service,
    )
    return AppServices(
        config=config,
        segmentation_service=segmentation_service,
        detection_service=detection_service,
        storage_service=storage_service,
        upload_service=upload_service,
        job_service=job_service,
        cache_service=cache_service,
        metrics_service=metrics_service,
        webhook_retry_service=webhook_retry_service,
        rate_limiter=rate_limiter,
        pipeline=pipeline,
        detection_pipeline=detection_pipeline,
        segment_use_case=segment_use_case,
        detect_use_case=detect_use_case,
    )
