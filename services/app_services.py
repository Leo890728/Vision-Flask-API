from __future__ import annotations

from dataclasses import dataclass

from config import Config
from middlewares.rate_limit import InMemoryRateLimiter
from services.cache_service import CacheService
from services.job_service import JobService
from services.metrics_service import MetricsService
from services.sam3_service import SAM3Service
from services.segmentation_pipeline import SegmentationPipeline
from services.storage_service import StorageService
from services.webhook_retry_service import WebhookRetryService


@dataclass
class AppServices:
    config: Config
    sam3_service: SAM3Service
    storage_service: StorageService
    job_service: JobService
    cache_service: CacheService
    metrics_service: MetricsService
    webhook_retry_service: WebhookRetryService
    rate_limiter: InMemoryRateLimiter
    pipeline: SegmentationPipeline

    def extension_map(self) -> dict:
        return {
            "sam3_service": self.sam3_service,
            "storage_service": self.storage_service,
            "job_service": self.job_service,
            "cache_service": self.cache_service,
            "metrics_service": self.metrics_service,
            "webhook_retry_service": self.webhook_retry_service,
            "pipeline": self.pipeline,
        }


def build_app_services(
    config: Config,
    sam3_service: SAM3Service | None = None,
    storage_service: StorageService | None = None,
) -> AppServices:
    sam3_service = sam3_service or SAM3Service(config)
    storage_service = storage_service or StorageService(config)
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
        sam3_service=sam3_service,
        storage_service=storage_service,
        metrics_service=metrics_service,
    )
    return AppServices(
        config=config,
        sam3_service=sam3_service,
        storage_service=storage_service,
        job_service=job_service,
        cache_service=cache_service,
        metrics_service=metrics_service,
        webhook_retry_service=webhook_retry_service,
        rate_limiter=rate_limiter,
        pipeline=pipeline,
    )

