# API Documentation

## Base URL
- `http://127.0.0.1:5000`

## Authentication
- Header: `X-API-Key: <your_key>`

## Error Format
```json
{
  "code": "ERROR_CODE",
  "message": "error message",
  "details": {},
  "request_id": "..."
}
```

## System Endpoints

### `GET /healthz`
Liveness check.

### `GET /readyz`
Readiness check for the segmentation model service.

### `GET /v1/models`
Returns runtime metadata for both models:
- `sam3`
- `yolo`

### `GET /metrics`
Prometheus text metrics (requires API key).

## Detection (YOLO, bbox only)

### `POST /v1/detect`
Single-image object detection.

`multipart/form-data` fields:
- `image` (required)
- `classes` (optional, JSON list of class ids or class names, e.g. `[0, "person"]`)
- `conf` (optional)
- `overlay` (optional, `none` or `bbox`)

Response:
- `200` success
- `202` auto-queued (when busy and auto queue enabled)

Each detection includes:
- `bbox`
- `score`
- `class_id`
- `class_name`

### `POST /v1/detect/batch`
Batch detection for multiple images.

`multipart/form-data` fields:
- `images` (required, multiple files)
- same optional fields as `/v1/detect`

## Segmentation (SAM3)

### `POST /v1/segment`
Single-image segmentation by prompt, points, or boxes.

`multipart/form-data` fields:
- `image` (required)
- `prompt` (optional, but required if no points/boxes)
- `points` (optional, JSON: `[[x,y], ...]`)
- `point_labels` (optional, JSON: `[1,0,...]`)
- `boxes` (optional, JSON: `[x1,y1,x2,y2]` or `[[...], ...]`)
- `conf` (optional)
- `overlay` (optional, `none|bbox|mask|both`)
- `output_formats` (optional, JSON list or CSV)
  - `mask_png`, `rle`, `polygon`, `alpha_matte`

### `POST /v1/segment/batch`
Batch segmentation for multiple images.

## Async Jobs (Unified)

### `POST /v1/jobs`
Submit async work for both tasks.

`multipart/form-data` fields:
- `image` (required)
- `task` (optional, `segment` or `detect`, default `segment`)
- segment-related fields: same as `/v1/segment`
- detect-related fields: `classes`, `conf`, `overlay`
- `webhook_url` (optional)
- `webhook_secret` (optional)

### `GET /v1/jobs/{job_id}`
Get job status and result.

Statuses:
- `queued`, `running`, `canceling`, `done`, `failed`, `canceled`

### `DELETE /v1/jobs/{job_id}`
Cancel a queued/running job.

### `POST /v1/jobs/{job_id}/retry`
Retry failed or canceled jobs.

### `GET /v1/jobs/{job_id}/export`
Export generated files and `result.json` as zip.

## Webhook
When a job is completed/failed/canceled, server posts to `webhook_url` if provided.

Signature header (if `webhook_secret` is provided):
- `X-Webhook-Signature: <base64(hmac_sha256(secret, body))>`
