# Project Documentation

## Purpose
- Provide a Flask API for segmentation and detection.
- Segmentation supports SAM3 and YOLO segmentation backends.
- Detection uses YOLO bounding-box detection.
- Includes uploads, generated output files, cache, metrics, async jobs, retryable webhooks, and export.

## Architecture
- `app.py`: Flask app factory, route registration, runtime startup.
- `config.py`: environment-driven settings.
- `api/request_models.py`: Pydantic request parameter models.
- `routes/`: HTTP boundary and response formatting.
- `services/app_services.py`: dependency composition.
- `services/segmentation_service.py`: task-level segmentation service with private SAM3 and YOLO-seg backends.
- `services/detection_service.py`: task-level detection service.
- `services/segmentation_pipeline.py`: segmentation inference timing, output rendering, and response shaping.
- `services/detection_pipeline.py`: detection inference timing, overlay rendering, and response shaping.
- `services/segment_use_case.py`: single-image segment workflow, including cache and auto-queue.
- `services/detect_use_case.py`: single-image detect workflow, including cache and auto-queue.
- `services/upload_service.py`: upload save, filename validation, image decode, and pixel limit.
- `services/job_service.py`: SQLite-backed job queue.
- `services/cache_service.py`: in-memory TTL cache.
- `services/metrics_service.py`: runtime counters and latency metrics.
- `services/storage_service.py`: output file persistence and cleanup.
- `services/webhook_retry_service.py`: background webhook retry worker.

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
