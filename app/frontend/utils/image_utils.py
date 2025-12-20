# app/frontend/utils/image_utils.py

import base64
from PIL import Image
import io

def load_pil_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def decode_base64_png(b64_str: str) -> bytes:
    return base64.b64decode(b64_str)
