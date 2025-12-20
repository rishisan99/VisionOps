# app/backend/app.py

from fastapi import FastAPI
from api.routes import router as api_router
from middlewares.request_logging import RequestIdLoggingMiddleware

def create_app() -> FastAPI:
    app = FastAPI(title="VisionOps Inference API", version="1.0")
    app.add_middleware(RequestIdLoggingMiddleware)
    app.include_router(api_router)
    return app

app = create_app()
