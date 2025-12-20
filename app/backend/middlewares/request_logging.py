# app/backend/middlewares/request_logging.py

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIdLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        start = time.time()

        response = await call_next(request)

        duration_ms = (time.time() - start) * 1000
        response.headers["X-Request-Id"] = request_id

        # Simple structured-ish log line (works everywhere)
        print(
            f'[REQ] id={request_id} method={request.method} path={request.url.path} '
            f'status={response.status_code} duration_ms={duration_ms:.2f}'
        )
        return response
