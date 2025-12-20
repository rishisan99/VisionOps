# app/backend/schemas.py

from pydantic import BaseModel

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    gradcam_base64: str
