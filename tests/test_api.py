from __future__ import annotations

import io
import os
import tempfile
import time
import unittest

import numpy as np
from PIL import Image

os.environ["SAM3_AUTOSTART_APP"] = "0"

from app import create_app


class FakeSAM3Service:
    def __init__(self):
        self.is_ready = True
        self.last_error = None

    def metadata(self):
        return {
            "model_path": "models/sam3.1_multiplex.pt",
            "default_conf": 0.25,
            "half": True,
            "device": "auto",
            "ready": True,
            "last_error": None,
        }

    def segment(
        self,
        image_path: str,
        prompt: str | None = None,
        conf: float | None = None,
        points=None,
        point_labels=None,
        boxes=None,
    ):
        _ = image_path, prompt, conf, points, point_labels, boxes
        masks = np.zeros((1, 16, 16), dtype=np.uint8)
        masks[0, 2:10, 3:12] = 1
        overlay = np.zeros((16, 16, 3), dtype=np.uint8)
        return {
            "width": 16,
            "height": 16,
            "masks": masks,
            "boxes": [[3.0, 2.0, 12.0, 10.0]],
            "scores": [0.99],
            "overlay_bgr": overlay,
        }


def _make_image_bytes(size=(16, 16), fmt="PNG") -> io.BytesIO:
    image = Image.new("RGB", size=size, color=(255, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf


class APITestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["API_KEY"] = "test-key"
        os.environ["OUTPUT_DIR"] = os.path.join(self._tmp.name, "outputs")
        os.environ["SAM3_SKIP_MODEL_LOAD"] = "1"
        os.environ["RATE_LIMIT_PER_MINUTE"] = "60"
        app = create_app(sam3_service=FakeSAM3Service())
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def test_healthz(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_openapi_json(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        spec = response.get_json()
        self.assertEqual(spec["openapi"], "3.0.3")
        self.assertIn("/v1/segment", spec["paths"])
        self.assertIn("ApiKeyAuth", spec["components"]["securitySchemes"])

    def test_docs(self):
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("swagger-ui", text)
        self.assertIn("/openapi.json", text)

    def test_readyz(self):
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ready")

    def test_models_requires_api_key(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["code"], "UNAUTHORIZED")

    def test_segment_missing_prompt(self):
        data = {
            "image": (_make_image_bytes(), "sample.png"),
        }
        response = self.client.post(
            "/v1/segment",
            data=data,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "MISSING_PROMPT")

    def test_segment_success_and_mask_url_accessible(self):
        data = {
            "image": (_make_image_bytes(), "sample.png"),
            "prompt": "a person",
            "return_overlay": "true",
        }
        response = self.client.post(
            "/v1/segment",
            data=data,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)

        payload = response.get_json()
        self.assertEqual(payload["prompt"], "a person")
        self.assertEqual(len(payload["detections"]), 1)
        mask_url = payload["detections"][0]["mask_url"]
        self.assertTrue(mask_url.startswith("/outputs/"))

        mask_response = self.client.get(mask_url)
        self.assertEqual(mask_response.status_code, 200)
        self.assertEqual(mask_response.content_type, "image/png")
        self.assertIsNotNone(payload["overlay_url"])

    def test_segment_supports_point_prompt(self):
        data = {
            "image": (_make_image_bytes(), "sample.png"),
            "points": "[[10,10],[12,12]]",
            "point_labels": "[1,0]",
        }
        response = self.client.post(
            "/v1/segment",
            data=data,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["detections"]), 1)

    def test_batch_segment(self):
        data = {
            "images": [
                (_make_image_bytes(), "sample1.png"),
                (_make_image_bytes(), "sample2.png"),
            ],
            "prompt": "person",
        }
        response = self.client.post(
            "/v1/segment/batch",
            data=data,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["batch_size"], 2)
        self.assertEqual(payload["items"][0]["status"], "ok")

    def test_async_job_flow(self):
        create = self.client.post(
            "/v1/jobs",
            data={"image": (_make_image_bytes(), "sample.png"), "prompt": "person"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(create.status_code, 202)
        job_id = create.get_json()["job_id"]

        # Poll briefly for completion from background worker.
        status_payload = None
        for _ in range(20):
            status = self.client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "test-key"})
            self.assertEqual(status.status_code, 200)
            status_payload = status.get_json()
            if status_payload["status"] == "done":
                break
            time.sleep(0.01)

        self.assertIsNotNone(status_payload)
        self.assertIn(status_payload["status"], {"done", "running", "queued"})
        if status_payload["status"] == "done":
            self.assertIn("result", status_payload)


class RateLimitTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["API_KEY"] = "test-key"
        os.environ["OUTPUT_DIR"] = os.path.join(self._tmp.name, "outputs")
        os.environ["SAM3_SKIP_MODEL_LOAD"] = "1"
        os.environ["RATE_LIMIT_PER_MINUTE"] = "1"
        app = create_app(sam3_service=FakeSAM3Service())
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def test_rate_limit_exceeded(self):
        data1 = {"image": (_make_image_bytes(), "sample.png"), "prompt": "p1"}
        r1 = self.client.post(
            "/v1/segment",
            data=data1,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r1.status_code, 200)

        data2 = {"image": (_make_image_bytes(), "sample.png"), "prompt": "p2"}
        r2 = self.client.post(
            "/v1/segment",
            data=data2,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r2.status_code, 429)
        self.assertEqual(r2.get_json()["code"], "RATE_LIMIT_EXCEEDED")


if __name__ == "__main__":
    unittest.main()
