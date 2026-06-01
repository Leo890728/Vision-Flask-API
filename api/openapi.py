from __future__ import annotations

from config import Config


def openapi_spec(config: Config) -> dict:
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "SAM3 Flask API",
            "version": "1.0.0",
            "description": "SAM3 semantic segmentation API with text prompt input.",
        },
        "servers": [{"url": "/"}],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
            },
            "schemas": {
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"},
                        "request_id": {"type": "string", "nullable": True},
                    },
                    "required": ["code", "message", "details", "request_id"],
                }
            },
        },
        "paths": {
            "/healthz": {
                "get": {
                    "summary": "Liveness check",
                    "responses": {"200": {"description": "Service is alive"}},
                }
            },
            "/readyz": {
                "get": {
                    "summary": "Readiness check",
                    "responses": {
                        "200": {"description": "Model is ready"},
                        "503": {"description": "Model is not ready"},
                    },
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "Get model metadata",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {
                        "200": {"description": "Model metadata"},
                        "401": {"description": "Unauthorized"},
                    },
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Prometheus-style metrics",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {"200": {"description": "Metrics text"}},
                }
            },
            "/v1/segment": {
                "post": {
                    "summary": "Segment single image by text or visual prompts",
                    "security": [{"ApiKeyAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["image"],
                                    "properties": {
                                        "image": {"type": "string", "format": "binary"},
                                        "prompt": {"type": "string"},
                                        "points": {"type": "string", "description": "JSON: [[x,y], ...]"},
                                        "point_labels": {"type": "string", "description": "JSON: [1,0,...]"},
                                        "boxes": {"type": "string", "description": "JSON: [x1,y1,x2,y2] or [[...],...]"},
                                        "output_formats": {
                                            "type": "string",
                                            "description": "JSON list or csv. Options: mask_png,rle,polygon,alpha_matte",
                                        },
                                        "conf": {"type": "number", "default": config.model_default_conf},
                                        "overlay": {
                                            "type": "string",
                                            "enum": ["none", "bbox", "mask", "both"],
                                            "default": "none",
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Segmentation result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "request_id": {"type": "string"},
                                            "prompt": {"type": "string"},
                                            "image_meta": {
                                                "type": "object",
                                                "properties": {
                                                    "width": {"type": "integer"},
                                                    "height": {"type": "integer"},
                                                },
                                            },
                                            "detections": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "id": {"type": "integer"},
                                                        "score": {"type": "number"},
                                                        "bbox": {
                                                            "type": "array",
                                                            "items": {"type": "number"},
                                                            "minItems": 4,
                                                            "maxItems": 4,
                                                        },
                                                        "mask_url": {"type": "string"},
                                                    },
                                                },
                                            },
                                            "overlay_url": {"type": "string", "nullable": True},
                                            "timing_ms": {
                                                "type": "object",
                                                "properties": {
                                                    "decode": {"type": "number"},
                                                    "infer": {"type": "number"},
                                                    "postprocess": {"type": "number"},
                                                    "total": {"type": "number"},
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad request"},
                        "401": {"description": "Unauthorized"},
                        "202": {"description": "Auto queued as async job"},
                        "429": {"description": "Rate limit exceeded"},
                        "503": {"description": "Queue full or model not ready"},
                        "500": {"description": "Inference failed"},
                    },
                }
            },
            "/v1/segment/batch": {
                "post": {
                    "summary": "Batch segment multiple images in one request",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {"200": {"description": "Batch segmentation result"}},
                }
            },
            "/v1/jobs": {
                "post": {
                    "summary": "Submit async segmentation job",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {"202": {"description": "Job accepted"}},
                }
            },
            "/v1/jobs/{job_id}": {
                "get": {
                    "summary": "Get async job status/result",
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Job status"}},
                },
                "delete": {
                    "summary": "Cancel queued/running async job",
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Job canceled"}},
                }
            },
            "/v1/jobs/{job_id}/retry": {
                "post": {
                    "summary": "Retry a failed/canceled async job",
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "202": {"description": "Retry job accepted"},
                        "404": {"description": "Job not found"},
                        "409": {"description": "Job state does not allow retry"},
                    },
                }
            },
            "/v1/jobs/{job_id}/export": {
                "get": {
                    "summary": "Export job outputs and result JSON as zip",
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {"description": "Zip file download"},
                        "404": {"description": "Job/output not found"},
                        "409": {"description": "Job not completed yet"},
                    },
                }
            },
            f"{config.output_url_prefix}/{{filename}}": {
                "get": {
                    "summary": "Get generated output file",
                    "parameters": [
                        {
                            "name": "filename",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "File content"}},
                }
            },
        },
    }
