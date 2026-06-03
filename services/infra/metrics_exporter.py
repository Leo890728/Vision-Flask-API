from __future__ import annotations

from services.infra.cache_service import CacheService
from services.infra.job_service import JobService
from services.infra.metrics_service import MetricsService
from services.infra.webhook_retry_service import WebhookRetryService

try:
    import torch
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None


def render_metrics_text(
    metrics_service: MetricsService,
    job_service: JobService,
    webhook_retry_service: WebhookRetryService,
    cache_service: CacheService,
) -> str:
    snap = metrics_service.snapshot()
    job_stats = job_service.stats()
    webhook_stats = webhook_retry_service.stats()
    cache_size = cache_service.size()

    gpu_mem_used = 0
    gpu_mem_total = 0
    if torch is not None and torch.cuda.is_available():
        try:
            gpu_mem_used = int(torch.cuda.memory_allocated())
            gpu_mem_total = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            gpu_mem_used = 0
            gpu_mem_total = 0

    lines = [
        f"sam3_requests_total {snap['requests_total']}",
        f"sam3_errors_total {snap['errors_total']}",
        f"sam3_error_rate {snap['error_rate']}",
        f"sam3_qps {snap['qps']}",
        f"sam3_inference_count {snap['inference_count']}",
        f"sam3_inference_latency_sum_ms {snap['inference_latency_sum_ms']}",
        f"sam3_inference_latency_avg_ms {snap['inference_latency_avg_ms']}",
        f"sam3_auto_queued_total {snap['auto_queued_total']}",
        f"sam3_jobs_created_total {snap['jobs_created_total']}",
        f"sam3_job_queue_size {job_stats['queue_size']}",
        f"sam3_job_queued_jobs {job_stats['queued_jobs']}",
        f"sam3_job_running_jobs {job_stats['running_jobs']}",
        f"sam3_job_canceling_jobs {job_stats['canceling_jobs']}",
        f"sam3_webhook_pending {webhook_stats['pending']}",
        f"sam3_webhook_delivered_total {webhook_stats['delivered_total']}",
        f"sam3_webhook_retried_total {webhook_stats['retried_total']}",
        f"sam3_webhook_failed_total {webhook_stats['failed_total']}",
        f"sam3_cache_hits_total {cache_service.hits}",
        f"sam3_cache_misses_total {cache_service.misses}",
        f"sam3_cache_entries {cache_size}",
        f"sam3_gpu_memory_bytes {gpu_mem_used}",
        f"sam3_gpu_total_memory_bytes {gpu_mem_total}",
    ]
    return "\n".join(lines) + "\n"

