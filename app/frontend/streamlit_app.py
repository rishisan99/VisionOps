# app/frontend/streamlit_app.py

import streamlit as st

from core.config import settings
from services.api_client import VisionOpsApiClient
from ui.layout import set_page, header, sidebar
from utils.image_utils import load_pil_image

def main():
    set_page()
    header()
    sidebar(settings.api_url)

    client = VisionOpsApiClient(settings.api_url)

    try:
        health = client.health()
        st.success(f"Backend healthy ✅ (device: {health.get('device', 'unknown')})")
    except Exception as e:
        st.error(f"Backend not reachable: {e}")
        st.stop()

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    img_bytes = uploaded.read()
    st.image(load_pil_image(img_bytes), caption="Input Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            pred_class, confidence, heatmap_png = client.predict_file(
                uploaded.name, img_bytes, uploaded.type
            )

        st.success(f"Prediction: **{pred_class}**  |  Confidence: **{confidence:.3f}**")
        st.image(heatmap_png, caption="Grad-CAM Explanation", use_column_width=True)

if __name__ == "__main__":
    main()
