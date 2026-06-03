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
Readiness check for the default segmentation backend.

### `GET /v1/models`
Returns runtime metadata:
- `segmentation.default_model`
- `segmentation.backends.sam3`
- `segmentation.backends.yolo_seg`
- `detection`

### `GET /metrics`
Prometheus text metrics. Requires API key.

## Detection

### `POST /v1/detect`
Single-image YOLO object detection.

`multipart/form-data` fields:
- `image` required
- `classes` optional JSON list of class ids or names, e.g. `[0, "person"]`
- `conf` optional, defaults to `YOLO_DEFAULT_CONF`
- `overlay` optional, `none` or `bbox`

Each detection includes:
- `bbox`
- `score`
- `class_id`
- `class_name`

### `POST /v1/detect/batch`
Batch detection for multiple images.

Fields:
- `images` required, multiple files
- same optional fields as `/v1/detect`

## Segmentation

### `POST /v1/segment`
Single-image segmentation.

Common fields:
- `image` required
- `segment_model` optional, `sam3` or `yolo_seg`, default `SEGMENT_DEFAULT_MODEL`
- `conf` optional
- `overlay` optional, `none|bbox|mask|both`
- `output_formats` optional JSON list or CSV: `mask_png`, `rle`, `polygon`, `alpha_matte`

For `segment_model=sam3`:
- `prompt` optional, required if no points/boxes
- `points` optional JSON `[[x,y], ...]`
- `point_labels` optional JSON `[1,0,...]`
- `boxes` optional JSON `[x1,y1,x2,y2]` or `[[...], ...]`

For `segment_model=yolo_seg`:
- `classes` optional JSON list of class ids or names
- `prompt` can be used as a single class-name filter when `classes` is omitted
- `points` and `boxes` are not supported

### `POST /v1/segment/batch`
Batch segmentation for multiple images. Uses the same segmentation fields as `/v1/segment`.

## Async Jobs

### `POST /v1/jobs`
Submit async work for either task.

Fields:
- `image` required
- `task` optional, `segment` or `detect`, default `segment`
- segment fields: same as `/v1/segment`
- detect fields: same as `/v1/detect`
- `webhook_url` optional
- `webhook_secret` optional

### `GET /v1/jobs/{job_id}`
Get job status and result.

Statuses:
- `queued`
- `running`
- `canceling`
- `done`
- `failed`
- `canceled`

### `DELETE /v1/jobs/{job_id}`
Cancel a queued/running job.

### `POST /v1/jobs/{job_id}/retry`
Retry failed or canceled jobs.

### `GET /v1/jobs/{job_id}/export`
Export generated files and `result.json` as zip.

## Webhook
When a job is completed, failed, or canceled, the server posts to `webhook_url` if provided.

Signature header, when `webhook_secret` is provided:
- `X-Webhook-Signature: <base64(hmac_sha256(secret, body))>`
