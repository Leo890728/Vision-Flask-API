# Project Documentation

## Purpose
- Provide a Flask API for segmentation and detection.
- Segmentation supports SAM3 and YOLO segmentation backends.
- Detection uses YOLO bounding-box detection.
- Includes uploads, generated output files, cache, metrics, async jobs, retryable webhooks, and export.

## Architecture
- `app.py`: Flask app factory, route registration, runtime startup.
- `config.py`: environment-driven settings.
- `errors.py`: `APIError` domain exception.
- `api/error_handlers.py`: Flask error handlers and JSON error responses.
- `api/request_models.py`: Pydantic request parameter models (single source of parameter parsing).
- `api/parsers.py`: shared parsing primitives (`parse_bool`, `parse_task`, `validate_upload_filename`).
- `routes/`: HTTP boundary and response formatting.
- `services/app_services.py`: dependency composition (composition root).

Service layers (`services/`):
- `backends/`: model inference backends.
  - `model_backend.py`: shared backend base (lock, inference counter, `is_ready`/`is_busy`) and `resolve_yolo_class_ids`.
  - `segmentation_service.py`: task-level segmentation service with private SAM3 and YOLO-seg backends.
  - `detection_service.py`: task-level detection service.
- `usecases/`: application orchestration.
  - `use_case.py`: base sync/auto-queue workflow shared by detect and segment.
  - `segment_use_case.py` / `detect_use_case.py`: single-image workflows (cache, auto-queue, run now).
  - `segmentation_pipeline.py` / `detection_pipeline.py`: inference timing, output/overlay rendering, response shaping.
  - `contexts.py`: `RequestContext` and `UseCaseResult` DTOs.
- `infra/`: cross-cutting infrastructure.
  - `upload_service.py`: upload save, filename validation, image decode, and pixel limit.
  - `job_service.py`: SQLite-backed job queue.
  - `cache_service.py`: in-memory TTL cache.
  - `metrics_service.py` / `metrics_exporter.py`: runtime counters, latency metrics, Prometheus export.
  - `storage_service.py`: output file persistence and cleanup.
  - `webhook_retry_service.py` / `webhook_utils.py`: background webhook retry worker and delivery.

## Runtime Configuration
Per-model settings live in `models.yaml` (see Model Catalog below), not in env.
Important environment variables:
- `API_KEY`
- `DETECT_DEFAULT_MODEL` (selects a default detection model by name from `models.yaml`)
- `SEGMENT_DEFAULT_MODEL` (selects a default segmentation model by name from `models.yaml`)
- `MODELS_CONFIG_PATH` (override the model catalog path; default `models.yaml`)
- `MAX_UPLOAD_MB`
- `MAX_IMAGE_PIXELS`
- `ENABLE_AUTO_QUEUE`
- `AUTO_QUEUE_MAX_SIZE`
- `JOB_DB_PATH`
- `JOB_WORKER_COUNT`
- `CACHE_TTL_SECONDS`
- `WEBHOOK_MAX_RETRIES`
- `WEBHOOK_RETRY_BASE_SECONDS`

## Model Catalog
Defined in `models.yaml` (the single source of truth, loaded by `config.py` into
`Config.models` as `ModelConfig` entries). Each entry's `name`/`task` also drive
the metrics `model` label and `/v1/models` metadata. Defaults:
- `sam3` (segment): `models/sam3.1_multiplex.pt`
- `yolo26n` (detect): `models/yolo26n.pt`
- `yolo11n` (detect): `models/yolo11n.pt`
- `yolo_seg` (segment): `models/yolo26n-seg.pt`

Add a model by appending an entry (task + model_path, optional default_conf/half/device)
and wiring a backend; no code/env changes are needed for its settings.

## Key Flows
- `/v1/segment`: synchronous segmentation; can auto-queue when selected segment backend is busy.
- `/v1/detect`: synchronous detection; can auto-queue when detection backend is busy.
- `/v1/jobs`: explicit async job submission for segment or detect.
- `/v1/jobs/{job_id}/retry`: retry failed or canceled jobs.
- `/v1/jobs/{job_id}/export`: export output files and `result.json`.

## Tests
```powershell
uv run python -m unittest discover -s tests -p "test_*.py" -v
```
