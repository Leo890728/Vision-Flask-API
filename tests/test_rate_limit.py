from __future__ import annotations

import os
import tempfile
import time
import unittest

from tests.helpers import FakeSAM3Backend, FakeDetectionBackend, FakeYOLOSegBackend, make_image_bytes

from app import create_app


class RateLimitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["API_KEY"] = "test-key"
        os.environ["OUTPUT_DIR"] = os.path.join(self._tmp.name, "outputs")
        os.environ["SAM3_SKIP_MODEL_LOAD"] = "1"
        os.environ["RATE_LIMIT_PER_MINUTE"] = "1"
        os.environ["JOB_DB_PATH"] = os.path.join(self._tmp.name, "jobs.sqlite3")
        self.app = create_app(
            sam3_backend=FakeSAM3Backend(),
            detection_backend=FakeDetectionBackend(),
            yolo_seg_backend=FakeYOLOSegBackend(),
        )
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.app.extensions["vision_services"]["job_service"].stop()
        self.app.extensions["vision_services"]["webhook_retry_service"].stop()
        time.sleep(0.05)
        self._tmp.cleanup()

    def test_rate_limit_exceeded(self):
        data1 = {"image": (make_image_bytes(), "sample.png"), "prompt": "p1"}
        r1 = self.client.post(
            "/v1/segment",
            data=data1,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r1.status_code, 200)

        data2 = {"image": (make_image_bytes(), "sample.png"), "prompt": "p2"}
        r2 = self.client.post(
            "/v1/segment",
            data=data2,
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r2.status_code, 429)
        self.assertEqual(r2.get_json()["code"], "RATE_LIMIT_EXCEEDED")
