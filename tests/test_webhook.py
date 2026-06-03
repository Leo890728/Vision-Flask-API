from __future__ import annotations

import time
import urllib.error
from unittest import mock

from tests.helpers import BaseAPITestCase, make_image_bytes


class WebhookTest(BaseAPITestCase):
    def test_webhook_called_on_job_completion(self):
        with mock.patch("app.urllib.request.urlopen") as mocked_urlopen:
            mocked_urlopen.return_value.__enter__.return_value = None
            create = self.client.post(
                "/v1/jobs",
                data={
                    "image": (make_image_bytes(), "sample.png"),
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
                    "image": (make_image_bytes(), "sample.png"),
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
