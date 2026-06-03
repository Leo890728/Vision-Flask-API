from __future__ import annotations

from tests.helpers import BaseAPITestCase, make_image_bytes


class SegmentTest(BaseAPITestCase):
    def test_segment_missing_prompt(self):
        data = {
            "image": (make_image_bytes(), "sample.png"),
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
            "image": (make_image_bytes(), "sample.png"),
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

    def test_segment_non_default_sam_model_routes_to_selected_backend(self):
        response = self.client.post(
            "/v1/segment",
            data={
                "image": (make_image_bytes(), "sample.png"),
                "segment_model": "sam3.1",
                "prompt": "a person",
            },
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["segment_model"], "sam3.1")
        self.assertEqual(self.fake_service.call_count, 0)
        self.assertEqual(self.fake_sam31_backend.call_count, 1)

    def test_segment_output_formats(self):
        data = {
            "image": (make_image_bytes(), "sample.png"),
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
            "image": (make_image_bytes(), "sample.png"),
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

    def test_segment_yolo_seg_without_prompt(self):
        response = self.client.post(
            "/v1/segment",
            data={
                "image": (make_image_bytes(), "sample.png"),
                "segment_model": "yolo_seg",
                "classes": "[\"person\"]",
                "overlay": "mask",
            },
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["segment_model"], "yolo_seg")
        self.assertEqual(payload["classes"], ["person"])
        self.assertEqual(len(payload["detections"]), 1)
        self.assertIsNotNone(payload["overlay_url"])
        self.assertEqual(self.fake_yolo_seg_backend.call_count, 1)

    def test_segment_cache_hit(self):
        data = {
            "image": (make_image_bytes(), "sample.png"),
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
            data={"image": (make_image_bytes(), "sample.png"), "prompt": "cache-test"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(self.fake_service.call_count, calls_after_first)
        self.assertTrue(r2.get_json()["cached"])

    def test_auto_queue_when_busy(self):
        self.fake_service.is_busy = True
        response = self.client.post(
            "/v1/segment",
            data={"image": (make_image_bytes(), "sample.png"), "prompt": "auto-queue"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "auto_queued")
        self.assertIn("job_id", payload)
        self._wait_job_terminal(payload["job_id"], attempts=80)

    def test_batch_segment(self):
        data = {
            "images": [
                (make_image_bytes(), "sample1.png"),
                (make_image_bytes(), "sample2.png"),
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
