from __future__ import annotations

from tests.helpers import BaseAPITestCase, make_image_bytes


class SystemTest(BaseAPITestCase):
    def test_healthz(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_readyz(self):
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ready")

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

    def test_models_requires_api_key(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["code"], "UNAUTHORIZED")

    def test_models_response_uses_uniform_model_entries(self):
        response = self.client.get("/v1/models", headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["defaults"], {"detect": "yolo26n", "segment": "sam3"})

        by_name = {model["name"]: model for model in payload["models"]}
        self.assertIn("sam3", by_name)
        self.assertIn("yolo26n", by_name)
        self.assertIn("yolo11n", by_name)
        self.assertIn("yolo_seg", by_name)

        expected_keys = {
            "name",
            "task",
            "model_path",
            "default_conf",
            "half",
            "device",
            "ready",
            "busy",
            "last_error",
            "active",
            "default",
        }
        for model in payload["models"]:
            self.assertEqual(set(model), expected_keys)

        self.assertTrue(by_name["sam3"]["active"])
        self.assertTrue(by_name["sam3"]["default"])
        self.assertTrue(by_name["yolo26n"]["active"])
        self.assertTrue(by_name["yolo26n"]["default"])
        self.assertTrue(by_name["yolo11n"]["active"])
        self.assertTrue(by_name["yolo11n"]["ready"])

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics", headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("requests_total", text)
        self.assertIn("job_queue_size", text)

    def test_metrics_inference_labeled_by_task_and_model(self):
        self.client.post(
            "/v1/detect",
            data={"image": (make_image_bytes(), "d.png")},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )
        self.client.post(
            "/v1/segment",
            data={"image": (make_image_bytes(), "s.png"), "prompt": "object"},
            headers={"X-API-Key": "test-key"},
            content_type="multipart/form-data",
        )

        text = self.client.get("/metrics", headers={"X-API-Key": "test-key"}).get_data(as_text=True)
        # Detection and segmentation are tracked as distinct labeled series,
        # not merged into a single "sam3" inference bucket.
        self.assertIn('inference_count{task="detect",model="yolo26n"} 1', text)
        self.assertIn('inference_count{task="segment",model="sam3"} 1', text)
        self.assertIn('inference_latency_avg_ms{task="segment",model="sam3"}', text)
