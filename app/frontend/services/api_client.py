# app/frontend/services/api_client.py

import requests

class VisionOpsApiClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def predict(self, filename: str, content: bytes, content_type: str) -> dict:
        files = {"file": (filename, content, content_type)}
        r = requests.post(f"{self.base_url}/predict", files=files, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def predict_file(self, filename: str, content: bytes, content_type: str):
        files = {"file": (filename, content, content_type)}
        r = requests.post(f"{self.base_url}/predict_file", files=files, timeout=self.timeout)
        r.raise_for_status()
        # metadata comes in headers
        pred_class = r.headers.get("X-Predicted-Class", "unknown")
        conf = float(r.headers.get("X-Confidence", "0.0"))
        return pred_class, conf, r.content
