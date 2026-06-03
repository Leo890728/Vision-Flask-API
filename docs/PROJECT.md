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
Important environment variables:
- `API_KEY`
- `SEGMENT_DEFAULT_MODEL`
- `SAM3_MODEL_PATH`
- `SAM3_DEFAULT_CONF`
- `YOLO_MODEL_PATH`
- `YOLO_DEFAULT_CONF`
- `YOLO_SEG_MODEL_PATH`
- `YOLO_SEG_DEFAULT_CONF`
- `MAX_UPLOAD_MB`
- `MAX_IMAGE_PIXELS`
- `ENABLE_AUTO_QUEUE`
- `AUTO_QUEUE_MAX_SIZE`
- `JOB_DB_PATH`
- `JOB_WORKER_COUNT`
- `CACHE_TTL_SECONDS`
- `WEBHOOK_MAX_RETRIES`
- `WEBHOOK_RETRY_BASE_SECONDS`

## Model Defaults
- SAM3 segmentation: `models/sam3.1_multiplex.pt`
- YOLO detection: `models/yolo11n.pt`
- YOLO segmentation: `models/yolo11n-seg.pt`

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
