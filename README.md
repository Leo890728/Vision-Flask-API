# Vision Flask API

Flask API for image segmentation and object detection.

Segmentation supports two backends:
- `sam3`: prompt, point, or box guided segmentation
- `yolo_seg`: YOLO segmentation with optional class filtering

Detection uses YOLO bounding-box detection.

## Documents

- [API Docs](docs/API.md)
- [Project Docs](docs/PROJECT.md)

## Quick Start

1. Put model weights under `models/`.
   - SAM3 default: `models/sam3.1_multiplex.pt`
   - YOLO detect default: `models/yolo11n.pt`
   - YOLO segment default: `models/yolo11n-seg.pt`
2. Set environment variables (copy from `.env.example`).
3. Run:

```powershell
uv run python app.py
```

Server default: `http://127.0.0.1:5000`

## Endpoints

- `GET /docs` (Swagger UI)
- `GET /openapi.json` (OpenAPI spec)
- `GET /healthz`
- `GET /readyz`
- `GET /v1/models` (requires `X-API-Key`)
- `GET /metrics` (requires `X-API-Key`)
- `POST /v1/detect` (requires `X-API-Key`)
- `POST /v1/detect/batch` (requires `X-API-Key`)
- `POST /v1/segment` (requires `X-API-Key`)
- `POST /v1/segment/batch` (requires `X-API-Key`)
- `POST /v1/jobs` (requires `X-API-Key`)
- `GET /v1/jobs/{job_id}` (requires `X-API-Key`)
- `DELETE /v1/jobs/{job_id}` (requires `X-API-Key`)
- `POST /v1/jobs/{job_id}/retry` (requires `X-API-Key`)
- `GET /v1/jobs/{job_id}/export` (requires `X-API-Key`)

`POST /v1/segment` supports:
- `segment_model=sam3` for text prompt, point prompt (`points`, `point_labels`), or box prompt (`boxes`)
- `segment_model=yolo_seg` for YOLO segmentation with optional `classes`
- output formats: `mask_png`, `rle`, `polygon`, `alpha_matte`

`POST /v1/detect` supports (bbox only):
- class filter (`classes`, JSON list of class IDs or names)
- overlay: `none` or `bbox`
- returns `bbox`, `score`, `class_id`, `class_name`

When the model is busy and `ENABLE_AUTO_QUEUE=true`, `/v1/segment` can return `202` and enqueue a job automatically.
`/v1/detect` also supports auto-queue.

## Example Request

```powershell
curl -X POST "http://127.0.0.1:5000/v1/segment" ^
  -H "X-API-Key: change-me" ^
  -F "image=@image (2).jpg" ^
  -F "segment_model=sam3" ^
  -F "prompt=a person" ^
  -F "output_formats=[\"mask_png\",\"rle\",\"polygon\"]" ^
  -F "conf=0.25" ^
  -F "overlay=both"
```

YOLO segmentation:

```powershell
curl -X POST "http://127.0.0.1:5000/v1/segment" ^
  -H "X-API-Key: change-me" ^
  -F "image=@image (2).jpg" ^
  -F "segment_model=yolo_seg" ^
  -F "classes=[\"person\"]" ^
  -F "overlay=mask"
```

## Visual Prompt Examples

Point prompts:

```powershell
curl -X POST "http://127.0.0.1:5000/v1/segment" ^
  -H "X-API-Key: change-me" ^
  -F "image=@image (2).jpg" ^
  -F "points=[[1200,900],[1300,950]]" ^
  -F "point_labels=[1,0]"
```

Box prompts:

```powershell
curl -X POST "http://127.0.0.1:5000/v1/segment" ^
  -H "X-API-Key: change-me" ^
  -F "image=@image (2).jpg" ^
  -F "boxes=[[1000,700,1600,1500]]"
```

## Async Job with Webhook

```powershell
curl -X POST "http://127.0.0.1:5000/v1/jobs" ^
  -H "X-API-Key: change-me" ^
  -F "image=@image (2).jpg" ^
  -F "task=segment" ^
  -F "segment_model=sam3" ^
  -F "prompt=person" ^
  -F "webhook_url=https://example.com/callback" ^
  -F "webhook_secret=change-this"
```

Async detect job:

```powershell
curl -X POST "http://127.0.0.1:5000/v1/jobs" ^
  -H "X-API-Key: change-me" ^
  -F "image=@image (2).jpg" ^
  -F "task=detect" ^
  -F "classes=[\"person\"]" ^
  -F "overlay=bbox"
```

Webhook delivery uses background retries with exponential backoff:
- `WEBHOOK_MAX_RETRIES` (default `3`)
- `WEBHOOK_RETRY_BASE_SECONDS` (default `1.0`)

## Retry / Export

Retry a failed/canceled job:

```powershell
curl -X POST "http://127.0.0.1:5000/v1/jobs/{job_id}/retry" ^
  -H "X-API-Key: change-me"
```

Export completed job outputs as zip:

```powershell
curl -L "http://127.0.0.1:5000/v1/jobs/{job_id}/export" ^
  -H "X-API-Key: change-me" ^
  -o "{job_id}.zip"
```

## Run Tests

```powershell
uv run python -m unittest discover -s tests -p "test_*.py" -v
```
