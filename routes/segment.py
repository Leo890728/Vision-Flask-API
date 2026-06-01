from __future__ import annotations

import time

from flask import Blueprint, Flask, g, jsonify, request

from api.parsers import parse_output_formats, parse_overlay, parse_prompt_inputs, validate_conf
from errors import APIError
from middlewares.auth import require_api_key
from middlewares.rate_limit import apply_rate_limit
from services.app_services import AppServices


def register_segment_routes(app: Flask, services: AppServices) -> None:
    config = services.config
    blueprint = Blueprint("segment", __name__)

    @blueprint.post("/v1/segment")
    @require_api_key(config.api_key)
    @apply_rate_limit(services.rate_limiter)
    def segment():
        upload = request.files.get("image")
        if upload is None:
            raise APIError("MISSING_IMAGE", "image is required.", 400)

        prompt_inputs = parse_prompt_inputs(request.form, require_input=True)
        output_formats = parse_output_formats(request.form)
        overlay = parse_overlay(request.form.get("overlay"))
        conf = validate_conf(request.form.get("conf"), config)
        g.prompt = prompt_inputs["prompt"]

        source_path, _w, _h, decode_ms = services.pipeline.save_and_validate_upload(upload, g.request_id)
        image_sha256 = services.storage_service.file_sha256(source_path)
        cache_key = services.pipeline.build_cache_key(image_sha256, prompt_inputs, conf, overlay, output_formats)

        cached = services.cache_service.get(cache_key)
        if cached is not None:
            payload = dict(cached)
            payload["request_id"] = g.request_id
            payload["cached"] = True
            payload["cache_key"] = cache_key
            return jsonify(payload), 200

        if config.enable_auto_queue and services.sam3_service.is_busy:
            qsize = services.job_service.stats()["queue_size"]
            if qsize >= config.auto_queue_max_size:
                raise APIError("QUEUE_FULL", "Server is overloaded, queue is full.", 503)
            job_payload = {
                "request_id": g.request_id,
                "source_path": str(source_path),
                "prompt_inputs": prompt_inputs,
                "conf": conf,
                "overlay": overlay,
                "output_formats": sorted(output_formats),
                "decode_ms": decode_ms,
            }
            job_id = services.job_service.submit(job_payload)
            services.metrics_service.inc_auto_queued()
            services.metrics_service.inc_jobs_created()
            return (
                jsonify(
                    {
                        "request_id": g.request_id,
                        "status": "queued",
                        "mode": "auto_queued",
                        "job_id": job_id,
                        "status_url": f"/v1/jobs/{job_id}",
                    }
                ),
                202,
            )

        payload = services.pipeline.segment_from_saved(
            source_path=str(source_path),
            request_id=g.request_id,
            prompt_inputs=prompt_inputs,
            conf=conf,
            overlay=overlay,
            output_formats=output_formats,
            decode_ms=decode_ms,
        )
        payload["cache_key"] = cache_key
        services.cache_service.set(cache_key, payload)
        return jsonify(payload), 200

    @blueprint.post("/v1/segment/batch")
    @require_api_key(config.api_key)
    @apply_rate_limit(services.rate_limiter)
    def segment_batch():
        uploads = request.files.getlist("images")
        if not uploads:
            raise APIError("MISSING_IMAGE", "images is required.", 400)
        if len(uploads) > config.max_batch_images:
            raise APIError("BATCH_TOO_LARGE", f"Maximum {config.max_batch_images} images per batch.", 400)

        prompt_inputs = parse_prompt_inputs(request.form, require_input=True)
        output_formats = parse_output_formats(request.form)
        overlay = parse_overlay(request.form.get("overlay"))
        conf = validate_conf(request.form.get("conf"), config)
        g.prompt = prompt_inputs["prompt"]

        batch_start = time.perf_counter()
        items = []
        for idx, upload in enumerate(uploads):
            item_request_id = f"{g.request_id}_{idx:03d}"
            try:
                source_path, _w, _h, decode_ms = services.pipeline.save_and_validate_upload(upload, item_request_id)
                image_sha256 = services.storage_service.file_sha256(source_path)
                cache_key = services.pipeline.build_cache_key(image_sha256, prompt_inputs, conf, overlay, output_formats)
                cached = services.cache_service.get(cache_key)

                if cached is not None:
                    result = dict(cached)
                    result["request_id"] = item_request_id
                    result["cached"] = True
                    result["cache_key"] = cache_key
                else:
                    result = services.pipeline.segment_from_saved(
                        source_path=str(source_path),
                        request_id=item_request_id,
                        prompt_inputs=prompt_inputs,
                        conf=conf,
                        overlay=overlay,
                        output_formats=output_formats,
                        decode_ms=decode_ms,
                    )
                    result["cache_key"] = cache_key
                    services.cache_service.set(cache_key, result)

                items.append(
                    {
                        "index": idx,
                        "filename": upload.filename,
                        "status": "ok",
                        "result": result,
                    }
                )
            except APIError as err:
                items.append(
                    {
                        "index": idx,
                        "filename": upload.filename,
                        "status": "error",
                        "error": {"code": err.code, "message": err.message, "details": err.details},
                    }
                )

        total_ms = (time.perf_counter() - batch_start) * 1000
        return (
            jsonify(
                {
                    "request_id": g.request_id,
                    "batch_size": len(uploads),
                    "timing_ms": {"total": round(total_ms, 2)},
                    "items": items,
                }
            ),
            200,
        )

    app.register_blueprint(blueprint)
