# API 文件

## Base URL
- 本機：`http://127.0.0.1:5000`

## 認證
- Header：`X-API-Key: <your_key>`

## 共用回應欄位
- 成功：`request_id`（多數端點）
- 失敗：
```json
{
  "code": "ERROR_CODE",
  "message": "error message",
  "details": {},
  "request_id": "..."
}
```

## 端點

### `GET /healthz`
- 程式存活檢查

### `GET /readyz`
- 模型就緒檢查

### `GET /v1/models`
- 回傳模型設定與狀態

### `GET /metrics`
- Prometheus text 指標（需 API Key）
- 例：`sam3_requests_total`, `sam3_error_rate`, `sam3_job_queue_size`

### `POST /v1/segment`
- 單圖分割（文字/點/框提示）
- `multipart/form-data`：
  - `image` (required)
  - `prompt` (optional, 與 points/boxes 至少一個)
  - `points` (optional, JSON: `[[x,y], ...]`)
  - `point_labels` (optional, JSON: `[1,0,...]`)
  - `boxes` (optional, JSON: `[x1,y1,x2,y2]` 或 `[[...], ...]`)
  - `conf` (optional)
  - `return_overlay` (optional)
  - `output_formats` (optional, JSON list 或 csv)
    - 可選：`mask_png`, `rle`, `polygon`, `alpha_matte`
- 回傳：
  - `200`：同步完成
  - `202`：自動轉排隊（`mode=auto_queued`, `job_id`）

### `POST /v1/segment/batch`
- 多圖同步分割
- 欄位：
  - `images` (required, 多檔)
  - 其餘提示與輸出欄位同 `/v1/segment`

### `POST /v1/jobs`
- 建立非同步任務
- 欄位：
  - `image` (required)
  - 提示/輸出欄位同 `/v1/segment`
  - `webhook_url` (optional)
  - `webhook_secret` (optional, 會簽 `X-Webhook-Signature`)

### `GET /v1/jobs/{job_id}`
- 查詢任務狀態
- 狀態：`queued`, `running`, `canceling`, `done`, `failed`, `canceled`

### `DELETE /v1/jobs/{job_id}`
- 取消任務

### `POST /v1/jobs/{job_id}/retry`
- 重送失敗/已取消任務（建立新 job）
- 僅允許原任務狀態為：`failed` 或 `canceled`
- 回傳 `202`：
```json
{
  "job_id": "new-job-id",
  "status": "queued",
  "retry_of": "old-job-id",
  "status_url": "/v1/jobs/new-job-id"
}
```

### `GET /v1/jobs/{job_id}/export`
- 匯出完成任務的輸出檔與 `result.json`（zip）
- 只有 `done` 狀態可匯出，否則 `409`
- 成功回應：`application/zip`

## Webhook
- 任務完成後 callback `POST webhook_url`
- 背景重試機制：指數退避（`WEBHOOK_MAX_RETRIES`, `WEBHOOK_RETRY_BASE_SECONDS`）
- Body：
```json
{
  "job_id": "...",
  "status": "done|failed|canceled",
  "result": {},
  "error": {},
  "updated_at": 0
}
```
- Header（有設定 secret 時）：
  - `X-Webhook-Signature: <base64(hmac_sha256(secret, body))>`
