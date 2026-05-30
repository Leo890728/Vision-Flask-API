# SAM3 Flask API

Flask API for SAM3 semantic segmentation with text prompts.

## Documents

- [API Docs](docs/API.md)
- [Project Docs](docs/PROJECT.md)

## Quick Start

1. Put model weights at `models/sam3.1_multiplex.pt`.
2. Set environment variables (copy from `.env.example`).
3. Run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Server default: `http://127.0.0.1:5000`

## Endpoints

- `GET /docs` (Swagger UI)
- `GET /openapi.json` (OpenAPI spec)
- `GET /healthz`
- `GET /readyz`
- `GET /v1/models` (requires `X-API-Key`)
- `GET /metrics` (requires `X-API-Key`)
- `POST /v1/segment` (requires `X-API-Key`)
- `POST /v1/segment/batch` (requires `X-API-Key`)
- `POST /v1/jobs` (requires `X-API-Key`)
- `GET /v1/jobs/{job_id}` (requires `X-API-Key`)
- `DELETE /v1/jobs/{job_id}` (requires `X-API-Key`)

`POST /v1/segment` supports:
- text prompt
- point prompt (`points`, `point_labels`)
- box prompt (`boxes`)
- output formats: `mask_png`, `rle`, `polygon`, `alpha_matte`

When the model is busy and `ENABLE_AUTO_QUEUE=true`, `/v1/segment` can return `202` and enqueue a job automatically.

## Example Request

```powershell
curl -X POST "http://127.0.0.1:5000/v1/segment" ^
  -H "X-API-Key: change-me" ^
  -F "image=@image (2).jpg" ^
  -F "prompt=a person" ^
  -F "output_formats=[\"mask_png\",\"rle\",\"polygon\"]" ^
  -F "conf=0.25" ^
  -F "return_overlay=true"
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
  -F "prompt=person" ^
  -F "webhook_url=https://example.com/callback" ^
  -F "webhook_secret=change-this"
```

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```
