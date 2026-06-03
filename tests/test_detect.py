from __future__ import annotations

from tests.helpers import BaseAPITestCase, make_image_bytes


class DetectTest(BaseAPITestCase):
    def test_detect_success(self):
        data = {
            "image": (make_image_bytes(), "sample.png"),
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
            data={"image": (make_image_bytes(), "sample.png"), "overlay": "mask"},
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
                    (make_image_bytes(), "sample1.png"),
                    (make_image_bytes(), "sample2.png"),
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
            data={"image": (make_image_bytes(), "sample.png"), "classes": "{}"},
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
            data={"image": (make_image_bytes(), "sample.png"), "classes": "[0,person]"},
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
            data={"image": (make_image_bytes(), "sample.png"), "classes": "\"[\\\"person\\\"]\""},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(payload["detections"][0]["class_name"], "person")
