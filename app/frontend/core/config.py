# app/frontend/core/config.py

import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    api_url: str = os.getenv("API_URL", "http://localhost:8002")

settings = Settings()
