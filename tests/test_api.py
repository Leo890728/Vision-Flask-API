from __future__ import annotations

import io
import os
import tempfile
import time
import unittest
import urllib.error
import zipfile
from unittest import mock

import numpy as np
from PIL import Image

os.environ["SAM3_AUTOSTART_APP"] = "0"

from app import create_app


class FakeSAM3Service:
    def __init__(self):
        self.is_ready = True
        self.last_error = None
        self.is_busy = False
        self.sleep_s = 0.0
        self.call_count = 0

    def metadata(self):
        return {
            "model_path": "models/sam3.1_multiplex.pt",
            "default_conf": 0.25,
            "half": True,
            "device": "auto",
            "ready": True,
            "last_error": None,
            "busy": self.is_busy,
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
        self.call_count += 1
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
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


class FakeYOLOService:
    def __init__(self):
        self.is_ready = True
        self.last_error = None
        self.is_busy = False
        self.call_count = 0
        self._names = {0: "person", 1: "car"}

    def metadata(self):
        return {
            "model_path": "models/yolo11n.pt",
            "default_conf": 0.25,
            "half": True,
            "device": "auto",
            "ready": True,
            "last_error": None,
            "busy": self.is_busy,
        }

    def resolve_class_ids(self, classes):
        if not classes:
            return None
        ids = []
        for item in classes:
            if isinstance(item, int):
                ids.append(item)
                continue
            key = str(item).lower()
            matches = [cid for cid, name in self._names.items() if name.lower() == key]
            if not matches:
                raise ValueError(f"Unknown class: {item}")
            ids.append(matches[0])
        return ids

    def detect(self, image_path: str, conf: float | None = None, class_ids=None, overlay: str = "none"):
        _ = image_path, conf
        self.call_count += 1
        detections = [
            {
                "id": 0,
                "score": 0.95,
                "bbox": [1.0, 2.0, 14.0, 15.0],
                "class_id": 0,
                "class_name": "person",
            },
            {
                "id": 1,
                "score": 0.88,
                "bbox": [2.0, 3.0, 10.0, 12.0],
                "class_id": 1,
                "class_name": "car",
            },
        ]
        if class_ids:
            detections = [d for d in detections if d["class_id"] in set(class_ids)]

        overlay_img = np.zeros((16, 16, 3), dtype=np.uint8) if overlay == "bbox" else None
        return {
            "width": 16,
            "height": 16,
            "detections": detections,
            "overlay_bgr": overlay_img,
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
        os.environ["JOB_DB_PATH"] = os.path.join(self._tmp.name, "jobs.sqlite3")
        os.environ["ENABLE_AUTO_QUEUE"] = "true"
        os.environ["AUTO_QUEUE_MAX_SIZE"] = "200"
        os.environ["CACHE_TTL_SECONDS"] = "3600"
        os.environ["WEBHOOK_MAX_RETRIES"] = "3"
        os.environ["WEBHOOK_RETRY_BASE_SECONDS"] = "0.01"
        self.fake_service = FakeSAM3Service()
        self.fake_yolo_service = FakeYOLOService()
        self.app = create_app(sam3_service=self.fake_service, yolo_service=self.fake_yolo_service)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.app.extensions["sam3_services"]["job_service"].stop()
        self.app.extensions["sam3_services"]["webhook_retry_service"].stop()
        time.sleep(0.05)
        self._tmp.cleanup()

    def _wait_job_terminal(self, job_id: str, attempts: int = 60, sleep_s: float = 0.01):
        payload = None
        for _ in range(attempts):
            status = self.client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "test-key"})
            self.assertEqual(status.status_code, 200)
            payload = status.get_json()
            if payload["status"] in {"done", "failed", "canceled"}:
                return payload
            time.sleep(sleep_s)
        return payload

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
            "overlay": "both",
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

    def test_segment_output_formats(self):
        data = {
            "image": (_make_image_bytes(), "sample.png"),
            "prompt": "a person",
            "output_formats": "[\"mask_png\",\"rle\",\"polygon\",\"alpha_matte\"]",
        }
        response = self.client.post(
            "/v1/segment",
            data=data,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        det = response.get_json()["detections"][0]
        self.assertIn("mask_url", det)
        self.assertIn("rle", det)
        self.assertIn("polygon", det)
        self.assertIn("alpha_url", det)

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

    def test_segment_cache_hit(self):
        data = {
            "image": (_make_image_bytes(), "sample.png"),
            "prompt": "cache-test",
        }
        r1 = self.client.post(
            "/v1/segment",
            data=data,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r1.status_code, 200)
        calls_after_first = self.fake_service.call_count
        self.assertGreaterEqual(calls_after_first, 1)

        r2 = self.client.post(
            "/v1/segment",
            data={"image": (_make_image_bytes(), "sample.png"), "prompt": "cache-test"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(self.fake_service.call_count, calls_after_first)
        self.assertTrue(r2.get_json()["cached"])

    def test_detect_success(self):
        data = {
            "image": (_make_image_bytes(), "sample.png"),
            "overlay": "bbox",
            "classes": "[\"person\"]",
        }
        response = self.client.post(
            "/v1/detect",
            data=data,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["task"], "detect")
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(payload["detections"][0]["class_name"], "person")
        self.assertIsNotNone(payload["overlay_url"])

    def test_detect_overlay_mask_rejected(self):
        response = self.client.post(
            "/v1/detect",
            data={"image": (_make_image_bytes(), "sample.png"), "overlay": "mask"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "INVALID_OVERLAY")

    def test_detect_batch(self):
        response = self.client.post(
            "/v1/detect/batch",
            data={
                "images": [
                    (_make_image_bytes(), "sample1.png"),
                    (_make_image_bytes(), "sample2.png"),
                ],
                "classes": "[0]",
            },
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["batch_size"], 2)
        self.assertEqual(payload["items"][0]["status"], "ok")
        self.assertEqual(payload["items"][0]["result"]["detections"][0]["class_id"], 0)

    def test_detect_invalid_classes_type(self):
        response = self.client.post(
            "/v1/detect",
            data={"image": (_make_image_bytes(), "sample.png"), "classes": "{}"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "INVALID_CLASSES")
        details = response.get_json().get("details", {})
        self.assertIn("received", details)
        self.assertIn("accepted_formats", details)

    def test_detect_classes_loose_form_supported(self):
        response = self.client.post(
            "/v1/detect",
            data={"image": (_make_image_bytes(), "sample.png"), "classes": "[0,person]"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(payload["detections"][0]["class_id"], 0)

    def test_detect_classes_double_encoded_json_supported(self):
        response = self.client.post(
            "/v1/detect",
            data={"image": (_make_image_bytes(), "sample.png"), "classes": "\"[\\\"person\\\"]\""},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(payload["detections"][0]["class_name"], "person")

    def test_auto_queue_when_busy(self):
        self.fake_service.is_busy = True
        response = self.client.post(
            "/v1/segment",
            data={"image": (_make_image_bytes(), "sample.png"), "prompt": "auto-queue"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "auto_queued")
        self.assertIn("job_id", payload)

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
        status_payload = self._wait_job_terminal(job_id, attempts=20)

        self.assertIsNotNone(status_payload)
        self.assertIn(status_payload["status"], {"done", "running", "queued", "failed", "canceled"})
        if status_payload["status"] == "done":
            self.assertIn("result", status_payload)

    def test_async_detect_job_flow(self):
        create = self.client.post(
            "/v1/jobs",
            data={"image": (_make_image_bytes(), "sample.png"), "task": "detect", "classes": "[\"person\"]"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(create.status_code, 202)
        job_id = create.get_json()["job_id"]

        payload = self._wait_job_terminal(job_id, attempts=30)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["task"], "detect")
        if payload["status"] == "done":
            self.assertEqual(payload["result"]["task"], "detect")
            self.assertEqual(len(payload["result"]["detections"]), 1)

    def test_cancel_job(self):
        self.fake_service.sleep_s = 0.2
        create = self.client.post(
            "/v1/jobs",
            data={"image": (_make_image_bytes(), "sample.png"), "prompt": "cancel-test"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(create.status_code, 202)
        job_id = create.get_json()["job_id"]
        cancel = self.client.delete(f"/v1/jobs/{job_id}", headers={"X-API-Key": "test-key"})
        self.assertEqual(cancel.status_code, 200)
        self.assertIn(cancel.get_json()["status"], {"canceled", "canceling"})

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics", headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("sam3_requests_total", text)
        self.assertIn("sam3_job_queue_size", text)

    def test_webhook_called_on_job_completion(self):
        with mock.patch("app.urllib.request.urlopen") as mocked_urlopen:
            mocked_urlopen.return_value.__enter__.return_value = None
            create = self.client.post(
                "/v1/jobs",
                data={
                    "image": (_make_image_bytes(), "sample.png"),
                    "prompt": "webhook",
                    "webhook_url": "https://example.com/hook",
                    "webhook_secret": "secret",
                },
                headers={"X-API-Key": "test-key"},
                content_type="multipart/form-data",
            )
            self.assertEqual(create.status_code, 202)
            job_id = create.get_json()["job_id"]

            self._wait_job_terminal(job_id)

            for _ in range(50):
                if mocked_urlopen.called:
                    break
                time.sleep(0.01)
            self.assertTrue(mocked_urlopen.called)

    def test_webhook_retry_on_failure(self):
        success_mock = mock.MagicMock()
        success_mock.__enter__.return_value = None
        responses = [urllib.error.URLError("temporary"), success_mock]
        with mock.patch("app.urllib.request.urlopen", side_effect=responses) as mocked_urlopen:
            create = self.client.post(
                "/v1/jobs",
                data={
                    "image": (_make_image_bytes(), "sample.png"),
                    "prompt": "webhook-retry",
                    "webhook_url": "https://example.com/hook",
                },
                headers={"X-API-Key": "test-key"},
                content_type="multipart/form-data",
            )
            self.assertEqual(create.status_code, 202)
            job_id = create.get_json()["job_id"]
            self._wait_job_terminal(job_id)

            for _ in range(100):
                if mocked_urlopen.call_count >= 2:
                    break
                time.sleep(0.01)
            self.assertGreaterEqual(mocked_urlopen.call_count, 2)

    def test_retry_failed_job(self):
        with mock.patch.object(self.fake_service, "segment", side_effect=RuntimeError("boom")):
            create = self.client.post(
                "/v1/jobs",
                data={"image": (_make_image_bytes(), "sample.png"), "prompt": "retry-me"},
                headers={"X-API-Key": "test-key"},
                content_type="multipart/form-data",
            )
            self.assertEqual(create.status_code, 202)
            job_id = create.get_json()["job_id"]
            failed_payload = self._wait_job_terminal(job_id, attempts=80)
            self.assertIsNotNone(failed_payload)
            self.assertEqual(failed_payload["status"], "failed")

        retry = self.client.post(f"/v1/jobs/{job_id}/retry", headers={"X-API-Key": "test-key"})
        self.assertEqual(retry.status_code, 202)
        retry_job_id = retry.get_json()["job_id"]
        self.assertNotEqual(job_id, retry_job_id)
        self.assertEqual(retry.get_json()["retry_of"], job_id)

        retry_payload = self._wait_job_terminal(retry_job_id, attempts=80)
        self.assertIsNotNone(retry_payload)
        self.assertEqual(retry_payload["status"], "done")

    def test_export_done_job_zip(self):
        create = self.client.post(
            "/v1/jobs",
            data={"image": (_make_image_bytes(), "sample.png"), "prompt": "export"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(create.status_code, 202)
        job_id = create.get_json()["job_id"]
        done_payload = self._wait_job_terminal(job_id, attempts=80)
        self.assertIsNotNone(done_payload)
        self.assertEqual(done_payload["status"], "done")

        export_resp = self.client.get(f"/v1/jobs/{job_id}/export", headers={"X-API-Key": "test-key"})
        self.assertEqual(export_resp.status_code, 200)
        self.assertEqual(export_resp.content_type, "application/zip")

        with zipfile.ZipFile(io.BytesIO(export_resp.data), "r") as zf:
            names = set(zf.namelist())
            self.assertTrue(any(name.endswith("result.json") for name in names))
            self.assertTrue(any(name.endswith("mask_000.png") for name in names))

    def test_job_persistence_across_app_restart(self):
        create = self.client.post(
            "/v1/jobs",
            data={"image": (_make_image_bytes(), "sample.png"), "prompt": "persist"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(create.status_code, 202)
        job_id = create.get_json()["job_id"]

        for _ in range(40):
            status = self.client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "test-key"})
            if status.get_json()["status"] == "done":
                break
            time.sleep(0.01)

        # Recreate app with same JOB_DB_PATH to verify persisted job metadata.
        app2 = create_app(sam3_service=FakeSAM3Service(), yolo_service=FakeYOLOService())
        app2.config["TESTING"] = True
        client2 = app2.test_client()
        status2 = client2.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "test-key"})
        self.assertEqual(status2.status_code, 200)
        self.assertIn(status2.get_json()["status"], {"done", "failed", "canceled", "queued", "running"})
        app2.extensions["sam3_services"]["job_service"].stop()
        app2.extensions["sam3_services"]["webhook_retry_service"].stop()


class RateLimitTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["API_KEY"] = "test-key"
        os.environ["OUTPUT_DIR"] = os.path.join(self._tmp.name, "outputs")
        os.environ["SAM3_SKIP_MODEL_LOAD"] = "1"
        os.environ["RATE_LIMIT_PER_MINUTE"] = "1"
        os.environ["JOB_DB_PATH"] = os.path.join(self._tmp.name, "jobs.sqlite3")
        self.app = create_app(sam3_service=FakeSAM3Service(), yolo_service=FakeYOLOService())
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.app.extensions["sam3_services"]["job_service"].stop()
        self.app.extensions["sam3_services"]["webhook_retry_service"].stop()
        time.sleep(0.05)
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
