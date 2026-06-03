from __future__ import annotations

from services.infra.cache_service import CacheService
from services.infra.job_service import JobService
from services.infra.metrics_service import MetricsService
from services.infra.webhook_retry_service import WebhookRetryService

try:
    import torch
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None


def _format_labels(labels: dict[str, str]) -> str:
    """Render a Prometheus label set, escaping per the exposition format."""
    if not labels:
        return ""
    parts = []
    for key, value in labels.items():
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{key}="{escaped}"')
    return "{" + ",".join(parts) + "}"


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
        f"requests_total {snap['requests_total']}",
        f"errors_total {snap['errors_total']}",
        f"error_rate {snap['error_rate']}",
        f"qps {snap['qps']}",
    ]

    # Per task/model inference series (aggregate via sum() in PromQL).
    for entry in snap["inference"]:
        labels = _format_labels({"task": entry["task"], "model": entry["model"]})
        lines.append(f"inference_count{labels} {entry['count']}")
        lines.append(f"inference_latency_sum_ms{labels} {entry['latency_sum_ms']}")
        lines.append(f"inference_latency_avg_ms{labels} {entry['latency_avg_ms']}")

    lines += [
        f"auto_queued_total {snap['auto_queued_total']}",
        f"jobs_created_total {snap['jobs_created_total']}",
        f"job_queue_size {job_stats['queue_size']}",
        f"job_queued_jobs {job_stats['queued_jobs']}",
        f"job_running_jobs {job_stats['running_jobs']}",
        f"job_canceling_jobs {job_stats['canceling_jobs']}",
        f"webhook_pending {webhook_stats['pending']}",
        f"webhook_delivered_total {webhook_stats['delivered_total']}",
        f"webhook_retried_total {webhook_stats['retried_total']}",
        f"webhook_failed_total {webhook_stats['failed_total']}",
        f"cache_hits_total {cache_service.hits}",
        f"cache_misses_total {cache_service.misses}",
        f"cache_entries {cache_size}",
        f"gpu_memory_bytes {gpu_mem_used}",
        f"gpu_total_memory_bytes {gpu_mem_total}",
    ]
    return "\n".join(lines) + "\n"

