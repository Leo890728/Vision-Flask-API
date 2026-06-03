from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path

from flask import Blueprint, Flask, g, jsonify, request, send_file

from api.parsers import parse_task
from api.request_models import DetectParams, SegmentParams
from errors import APIError
from middlewares.auth import require_api_key
from middlewares.rate_limit import apply_rate_limit
from services.app_services import AppServices


def register_job_routes(app: Flask, services: AppServices) -> None:
    config = services.config
    blueprint = Blueprint("jobs", __name__)

    @blueprint.post("/v1/jobs")
    @require_api_key(config.api_key)
    @apply_rate_limit(services.rate_limiter)
    def create_job():
        upload = request.files.get("image")
        if upload is None:
            raise APIError("MISSING_IMAGE", "image is required.", 400)
        task = parse_task(request.form.get("task"))
        webhook_url = (request.form.get("webhook_url") or "").strip() or None
        webhook_secret = (request.form.get("webhook_secret") or "").strip() or None
        if task == "segment":
            params = SegmentParams.from_form(request.form, config, require_input=True)
            g.prompt = params.prompt
            source_path, _w, _h, decode_ms = services.pipeline.save_and_validate_upload(upload, g.request_id)
            job_payload = {
                "task": "segment",
                "segment_model": params.segment_model,
                "request_id": g.request_id,
                "source_path": str(source_path),
                "prompt_inputs": params.to_prompt_inputs(),
                "classes": params.classes,
                "conf": params.conf,
                "overlay": params.overlay,
                "output_formats": sorted(params.output_formats),
                "decode_ms": decode_ms,
                "webhook_secret": webhook_secret,
            }
        else:
            params = DetectParams.from_form(request.form, config)
            g.prompt = None
            source_path, _w, _h, decode_ms = services.detection_pipeline.save_and_validate_upload(upload, g.request_id)
            job_payload = {
                "task": "detect",
                "request_id": g.request_id,
                "source_path": str(source_path),
                "conf": params.conf,
                "overlay": params.overlay,
                "classes": params.classes,
                "decode_ms": decode_ms,
                "webhook_secret": webhook_secret,
            }
        if services.job_service.stats()["queue_size"] >= config.auto_queue_max_size:
            raise APIError("QUEUE_FULL", "Server queue is full.", 503)
        job_id = services.job_service.submit(job_payload, webhook_url=webhook_url)
        services.metrics_service.inc_jobs_created()
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "task": task,
                    "status_url": f"/v1/jobs/{job_id}",
                    "webhook_url": webhook_url,
                }
            ),
            202,
        )

    @blueprint.get("/v1/jobs/<job_id>")
    @require_api_key(config.api_key)
    def get_job(job_id: str):
        record = services.job_service.get(job_id)
        if record is None:
            raise APIError("JOB_NOT_FOUND", "Job not found.", 404)

        payload = {
            "job_id": record.job_id,
            "status": record.status,
            "task": (record.payload or {}).get("task", "segment"),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        if record.result is not None:
            payload["result"] = record.result
        if record.error is not None:
            payload["error"] = record.error
        return jsonify(payload), 200

    @blueprint.delete("/v1/jobs/<job_id>")
    @require_api_key(config.api_key)
    def cancel_job(job_id: str):
        record = services.job_service.cancel(job_id)
        if record is None:
            raise APIError("JOB_NOT_FOUND", "Job not found.", 404)
        return (
            jsonify(
                {
                    "job_id": record.job_id,
                    "status": record.status,
                    "cancel_requested": record.cancel_requested,
                }
            ),
            200,
        )

    @blueprint.post("/v1/jobs/<job_id>/retry")
    @require_api_key(config.api_key)
    def retry_job(job_id: str):
        record = services.job_service.get(job_id)
        if record is None:
            raise APIError("JOB_NOT_FOUND", "Job not found.", 404)

        if record.status not in {"failed", "canceled"}:
            raise APIError(
                "INVALID_JOB_STATE",
                "Retry is only allowed for failed or canceled jobs.",
                409,
                {"status": record.status},
            )

        if not record.payload or not record.payload.get("source_path"):
            raise APIError("INVALID_JOB_PAYLOAD", "Job payload is incomplete and cannot be retried.", 409)

        source_path = Path(record.payload["source_path"])
        if not source_path.exists():
            raise APIError("SOURCE_NOT_FOUND", "Source image for retry no longer exists.", 404)

        retry_payload = json.loads(json.dumps(record.payload, ensure_ascii=False))
        retry_payload["request_id"] = str(uuid.uuid4())
        retry_payload["retry_of"] = record.job_id

        if services.job_service.stats()["queue_size"] >= config.auto_queue_max_size:
            raise APIError("QUEUE_FULL", "Server queue is full.", 503)

        retry_job_id = services.job_service.submit(retry_payload, webhook_url=record.webhook_url)
        services.metrics_service.inc_jobs_created()
        return (
            jsonify(
                {
                    "job_id": retry_job_id,
                    "status": "queued",
                    "retry_of": record.job_id,
                    "status_url": f"/v1/jobs/{retry_job_id}",
                }
            ),
            202,
        )

    @blueprint.get("/v1/jobs/<job_id>/export")
    @require_api_key(config.api_key)
    def export_job(job_id: str):
        record = services.job_service.get(job_id)
        if record is None:
            raise APIError("JOB_NOT_FOUND", "Job not found.", 404)
        if record.status != "done" or record.result is None:
            raise APIError("JOB_NOT_READY", "Job result is not ready for export.", 409, {"status": record.status})

        request_id = record.result.get("request_id") or record.payload.get("request_id")
        if not request_id:
            raise APIError("EXPORT_NOT_FOUND", "Cannot resolve output directory for this job.", 404)

        output_dir = Path(config.output_dir) / request_id
        if not output_dir.exists() or not output_dir.is_dir():
            raise APIError("EXPORT_NOT_FOUND", "Output directory not found.", 404)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            files = sorted([path for path in output_dir.rglob("*") if path.is_file()])
            if not files:
                raise APIError("EXPORT_NOT_FOUND", "No files available to export.", 404)
            for path in files:
                arcname = f"{request_id}/{path.relative_to(output_dir).as_posix()}"
                zf.write(path, arcname=arcname)
            zf.writestr(
                f"{request_id}/result.json",
                json.dumps(
                    {
                        "job_id": record.job_id,
                        "status": record.status,
                        "created_at": record.created_at,
                        "updated_at": record.updated_at,
                        "result": record.result,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{job_id}.zip",
        )

    app.register_blueprint(blueprint)
