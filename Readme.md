# VisionOps – End-to-End MLOps Vision System

VisionOps is a **production-ready computer vision system** that detects tomato leaf diseases using a **self-supervised + supervised deep learning pipeline**, with full **MLOps lifecycle management** and a **Dockerized inference application**.

---

## 🚀 Features

-   **Self-Supervised Representation Learning**
    -   Contrastive learning (SimCLR-style)
    -   ResNet-18 backbone
-   **Supervised Fine-Tuning**
    -   Two-phase training (head → partial backbone unfreeze)
-   **Explainability**
    -   Grad-CAM heatmaps for model interpretability
-   **MLOps Tooling**
    -   DVC for data versioning
    -   MLflow for experiment tracking & artifacts
    -   GitHub for version control
-   **Production Deployment**
    -   FastAPI backend
    -   Streamlit frontend
    -   Docker & docker-compose

---

## 🧠 Model Performance (Final Run)

| Metric        | Value     |
| ------------- | --------- |
| Test Accuracy | **85.5%** |
| Macro F1      | **0.856** |
| Precision     | **0.859** |
| Recall        | **0.855** |

Dataset:

-   PlantVillage (Tomato subset)
-   Classes: Healthy, Early Blight, Late Blight, Septoria Leaf Spot

---

## 🏗️ Architecture Overview

```

Data (_source)
↓
ETL (Ingestion → Transformation → Splitting)
↓
Self-Supervised Learning (ResNet-18)
↓
Supervised Fine-Tuning
↓
Explainability (Grad-CAM)
↓
FastAPI Inference API
↓
Streamlit UI

```

---

## 📂 Project Structure (Simplified)

```

src/
├── components/        # Training, inference, explainability logic
├── pipeline/          # Pipeline orchestration
├── entity/            # Config & artifact schemas
├── logging/
├── exception/
app/
├── backend/           # FastAPI inference service
├── frontend/          # Streamlit UI
Artifacts/
├── <timestamp>/       # Versioned runs (DVC + MLflow)

```

---

## 🖥️ Run Locally (Docker)

```bash
docker compose up --build
```

-   Backend: [http://localhost:8000](http://localhost:8000)
-   API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
-   Frontend: [http://localhost:8501](http://localhost:8501)

---

## 🔍 Inference Output

-   Predicted class
-   Confidence score
-   Grad-CAM heatmap highlighting decision regions

---

## 🧪 MLOps Highlights

-   **Data ↔ Model Traceability** via DVC
-   **Experiment tracking** via MLflow
-   **Reproducible pipelines**
-   **Stateless, containerized inference**

---
