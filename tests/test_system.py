from __future__ import annotations

from tests.helpers import BaseAPITestCase


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

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics", headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("sam3_requests_total", text)
        self.assertIn("sam3_job_queue_size", text)
