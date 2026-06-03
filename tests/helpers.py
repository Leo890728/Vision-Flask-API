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


class FakeSAM3Backend:
    def __init__(self):
        self.is_ready = True
        self.last_error = None
        self.is_busy = False
        self.sleep_s = 0.0
        self.call_count = 0
        self.name = "sam3"
        self.task = "segment"

    def metadata(self):
        return {
            "name": "sam3",
            "task": "segment",
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
        overlay: str = "none",
    ):
        _ = image_path, prompt, conf, points, point_labels, boxes, overlay
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


class FakeDetectionBackend:
    def __init__(self):
        self.is_ready = True
        self.last_error = None
        self.is_busy = False
        self.call_count = 0
        self._names = {0: "person", 1: "car"}
        self.name = "yolo26n"
        self.task = "detect"

    def metadata(self):
        return {
            "name": "yolo26n",
            "task": "detect",
            "model_path": "models/yolo26n.pt",
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


class FakeYOLOSegBackend:
    def __init__(self):
        self.is_ready = True
        self.last_error = None
        self.is_busy = False
        self.call_count = 0
        self._names = {0: "person", 1: "car"}
        self.name = "yolo_seg"
        self.task = "segment"

    def metadata(self):
        return {
            "name": "yolo_seg",
            "task": "segment",
            "model_path": "models/yolo11n-seg.pt",
            "default_conf": 0.25,
            "half": True,
            "device": "auto",
            "ready": True,
            "last_error": None,
            "busy": self.is_busy,
        }

    def segment(self, image_path: str, conf: float | None = None, classes=None, overlay: str = "none"):
        _ = image_path, conf
        self.call_count += 1
        mask = np.zeros((1, 16, 16), dtype=np.uint8)
        mask[0, 4:13, 5:14] = 1
        overlay_img = np.zeros((16, 16, 3), dtype=np.uint8) if overlay != "none" else None
        return {
            "width": 16,
            "height": 16,
            "masks": mask,
            "boxes": [[5.0, 4.0, 14.0, 13.0]],
            "scores": [0.91],
            "overlay_bgr": overlay_img,
        }


def make_image_bytes(size=(16, 16), fmt="PNG") -> io.BytesIO:
    image = Image.new("RGB", size=size, color=(255, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf


class BaseAPITestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["API_KEY"] = "test-key"
        os.environ["DETECT_DEFAULT_MODEL"] = "yolo26n"
        os.environ["SEGMENT_DEFAULT_MODEL"] = "sam3"
        os.environ["OUTPUT_DIR"] = os.path.join(self._tmp.name, "outputs")
        os.environ["SAM3_SKIP_MODEL_LOAD"] = "1"
        os.environ["RATE_LIMIT_PER_MINUTE"] = "60"
        os.environ["JOB_DB_PATH"] = os.path.join(self._tmp.name, "jobs.sqlite3")
        os.environ["ENABLE_AUTO_QUEUE"] = "true"
        os.environ["AUTO_QUEUE_MAX_SIZE"] = "200"
        os.environ["CACHE_TTL_SECONDS"] = "3600"
        os.environ["WEBHOOK_MAX_RETRIES"] = "3"
        os.environ["WEBHOOK_RETRY_BASE_SECONDS"] = "0.01"
        self.fake_service = FakeSAM3Backend()
        self.fake_detection_backend = FakeDetectionBackend()
        self.fake_yolo_seg_backend = FakeYOLOSegBackend()
        self.app = create_app(
            sam3_backend=self.fake_service,
            detection_backend=self.fake_detection_backend,
            yolo_seg_backend=self.fake_yolo_seg_backend,
        )
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.app.extensions["vision_services"]["job_service"].stop()
        self.app.extensions["vision_services"]["webhook_retry_service"].stop()
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
