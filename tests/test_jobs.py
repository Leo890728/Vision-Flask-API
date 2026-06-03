from __future__ import annotations

import io
import time
import zipfile
from unittest import mock

from tests.helpers import BaseAPITestCase, FakeSAM3Backend, FakeDetectionBackend, FakeYOLOSegBackend, make_image_bytes

from app import create_app


class JobsTest(BaseAPITestCase):
    def test_async_job_flow(self):
        create = self.client.post(
            "/v1/jobs",
            data={"image": (make_image_bytes(), "sample.png"), "prompt": "person"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(create.status_code, 202)
        job_id = create.get_json()["job_id"]

        status_payload = self._wait_job_terminal(job_id, attempts=20)

        self.assertIsNotNone(status_payload)
        self.assertIn(status_payload["status"], {"done", "running", "queued", "failed", "canceled"})
        if status_payload["status"] == "done":
            self.assertIn("result", status_payload)

    def test_async_detect_job_flow(self):
        create = self.client.post(
            "/v1/jobs",
            data={"image": (make_image_bytes(), "sample.png"), "task": "detect", "classes": "[\"person\"]"},
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

    def test_async_detect_job_preserves_detect_model(self):
        create = self.client.post(
            "/v1/jobs",
            data={"image": (make_image_bytes(), "sample.png"), "task": "detect", "detect_model": "yolo11n"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(create.status_code, 202)
        self.assertEqual(create.get_json()["model"], "yolo11n")
        job_id = create.get_json()["job_id"]

        payload = self._wait_job_terminal(job_id, attempts=30)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["task"], "detect")
        if payload["status"] == "done":
            self.assertEqual(payload["result"]["detect_model"], "yolo11n")
            self.assertEqual(self.fake_detection_backend_b.call_count, 1)

    def test_cancel_job(self):
        self.fake_service.sleep_s = 0.2
        create = self.client.post(
            "/v1/jobs",
            data={"image": (make_image_bytes(), "sample.png"), "prompt": "cancel-test"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(create.status_code, 202)
        job_id = create.get_json()["job_id"]
        cancel = self.client.delete(f"/v1/jobs/{job_id}", headers={"X-API-Key": "test-key"})
        self.assertEqual(cancel.status_code, 200)
        self.assertIn(cancel.get_json()["status"], {"canceled", "canceling"})

    def test_retry_failed_job(self):
        with mock.patch.object(self.fake_service, "segment", side_effect=RuntimeError("boom")):
            create = self.client.post(
                "/v1/jobs",
                data={"image": (make_image_bytes(), "sample.png"), "prompt": "retry-me"},
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
            data={"image": (make_image_bytes(), "sample.png"), "prompt": "export"},
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
            data={"image": (make_image_bytes(), "sample.png"), "prompt": "persist"},
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

        app2 = create_app(
            sam3_backend=FakeSAM3Backend(),
            detection_backend=FakeDetectionBackend(),
            yolo_seg_backend=FakeYOLOSegBackend(),
        )
        app2.config["TESTING"] = True
        client2 = app2.test_client()
        status2 = client2.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "test-key"})
        self.assertEqual(status2.status_code, 200)
        self.assertIn(status2.get_json()["status"], {"done", "failed", "canceled", "queued", "running"})
        app2.extensions["vision_services"]["job_service"].stop()
        app2.extensions["vision_services"]["webhook_retry_service"].stop()
