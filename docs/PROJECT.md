# 專案說明

## 專案目標
- 提供基於 Ultralytics SAM3 的分割 API（Flask）
- 支援文字提示、點/框提示、批次分割、非同步任務
- 內建快取、排隊、metrics、webhook 與任務持久化

## 目錄重點
- `app.py`：主 API 與路由
- `config.py`：環境設定
- `services/sam3_service.py`：模型推論
- `services/job_service.py`：SQLite 持久化任務佇列
- `services/cache_service.py`：結果快取
- `services/metrics_service.py`：指標統計
- `services/storage_service.py`：輸出檔案與清理
- `tests/test_api.py`：API 測試

## 執行方式
1. 放置模型：`models/sam3.1_multiplex.pt`
2. 設定環境變數：參考 `.env.example`
3. 啟動：
```powershell
.\.venv\Scripts\python.exe app.py
```

## 核心行為
- `/v1/segment` 同步推論；模型忙碌時可自動轉成 job（202）
- `/v1/jobs` 系列為非同步模式
- 任務資料寫入 `JOB_DB_PATH`（SQLite），重啟可保留狀態
- 輸出檔預設在 `OUTPUT_DIR`

## 重要環境變數
- `API_KEY`
- `SAM3_MODEL_PATH`
- `MAX_UPLOAD_MB`
- `MAX_IMAGE_PIXELS`
- `ENABLE_AUTO_QUEUE`
- `AUTO_QUEUE_MAX_SIZE`
- `JOB_DB_PATH`
- `JOB_WORKER_COUNT`
- `CACHE_TTL_SECONDS`

## 測試
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## 文件
- API 規格：`docs/API.md`
- 快速使用：`README.md`

