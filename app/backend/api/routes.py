# app/backend/api/routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response

from schemas import PredictResponse
from services.inference import InferenceService
from utils.image_ops import (
    read_image_from_bytes,
    cam_overlay_to_base64,
    cam_overlay_to_png_bytes,
)

router = APIRouter()
svc = InferenceService()

@router.get("/health")
def health():
    return {"status": "ok", "device": "cpu"}

@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    contents = await file.read()
    try:
        rgb = read_image_from_bytes(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")

    pred_class, confidence, cam = svc.predict_with_gradcam(rgb)
    gradcam_b64 = cam_overlay_to_base64(rgb, cam)

    return PredictResponse(
        predicted_class=pred_class,
        confidence=confidence,
        gradcam_base64=gradcam_b64
    )

@router.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """
    Returns Grad-CAM overlay as PNG bytes.
    Also includes prediction metadata in headers.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    contents = await file.read()
    try:
        rgb = read_image_from_bytes(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")

    pred_class, confidence, cam = svc.predict_with_gradcam(rgb)
    png_bytes = cam_overlay_to_png_bytes(rgb, cam)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Predicted-Class": pred_class,
            "X-Confidence": f"{confidence:.6f}",
        }
    )
