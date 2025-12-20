# app/frontend/ui/layout.py

import streamlit as st

def set_page():
    st.set_page_config(page_title="VisionOps Plant Disease Demo", layout="centered")

def header():
    st.title("🌿 VisionOps — Tomato Leaf Disease Classifier")
    st.write("Upload a tomato leaf image. You’ll get prediction + confidence + Grad-CAM explanation.")

def sidebar(api_url: str):
    st.sidebar.header("Settings")
    st.sidebar.code(f"API_URL = {api_url}", language="text")
