# app/backend/core/config.py

import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "classifier.pt")
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    device: str = os.getenv("DEVICE", "cpu")  # production: cpu-only

settings = Settings()
