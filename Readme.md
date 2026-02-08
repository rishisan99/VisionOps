# VisionOps: End-to-End MLOps Vision System

VisionOps is an end-to-end computer vision project for tomato leaf disease classification.  
It combines self-supervised learning, supervised fine-tuning, Grad-CAM explainability, and containerized deployment with FastAPI + Streamlit.

---

## Key Features

- Self-supervised representation learning (SimCLR-style) with ResNet-18.
- Two-phase supervised fine-tuning for stable performance.
- Grad-CAM explainability for visual model interpretation.
- MLOps workflow with DVC and MLflow.
- Dockerized backend and frontend for consistent local deployment.

---

## Model Performance (Final Run)

| Metric | Value |
| --- | --- |
| Test Accuracy | **85.5%** |
| Macro F1 | **0.856** |
| Precision | **0.859** |
| Recall | **0.855** |

Dataset:
- PlantVillage (Tomato subset)
- Classes: Tomato Healthy, Early Blight, Late Blight, Septoria Leaf Spot

---

## System Flow

```text
Data (_source)
-> ETL (Ingestion -> Transformation -> Splitting)
-> Self-Supervised Learning (ResNet-18)
-> Supervised Fine-Tuning
-> Explainability (Grad-CAM)
-> FastAPI Inference API
-> Streamlit UI
```

---

## Project Structure

```text
src/
  components/        # training, explainability, model logic
  pipeline/          # stage orchestration
  entity/            # config and artifact dataclasses
  logging/
  exception/

app/
  backend/           # FastAPI inference service
  frontend/          # Streamlit user interface

Artifacts/
  <timestamp>/       # run outputs (metrics, checkpoints, reports)
```

---

## App Preview

### Home
![VisionOps Home](screenshots/Screenshot%20Home.png)

### Selected Image
![VisionOps Selected Image](screenshots/Screenshot%20Selected%20Image.png)

### Prediction Result
![VisionOps Prediction](screenshots/Screenshot%20Predict.png)

---

## Run Locally (Docker)

```bash
docker compose up -d --build
```

Default local endpoints:
- Frontend: [http://localhost:8502](http://localhost:8502)
- Backend Health: [http://localhost:8002/health](http://localhost:8002/health)
- Backend Docs: [http://localhost:8002/docs](http://localhost:8002/docs)

To stop:

```bash
docker compose down
```

---

## Inference Output

For each uploaded image, the app returns:
- Predicted class
- Confidence score
- Grad-CAM heatmap overlay

---

## MLOps Highlights

- Data and artifact traceability with DVC.
- Experiment tracking with MLflow.
- Reproducible, stage-based training pipeline.
- Containerized inference stack for deployment readiness.
