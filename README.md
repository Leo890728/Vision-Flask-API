# SAM3 Flask API

Flask API for SAM3 semantic segmentation with text prompts.

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
- `POST /v1/segment` (requires `X-API-Key`)

## Example Request

```powershell
curl -X POST "http://127.0.0.1:5000/v1/segment" ^
  -H "X-API-Key: change-me" ^
  -F "image=@image (2).jpg" ^
  -F "prompt=a person" ^
  -F "conf=0.25" ^
  -F "return_overlay=true"
```

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```
