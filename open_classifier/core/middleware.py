"""Middleware for FastAPI application."""

import time
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog

from .config import settings
from .logging import request_logger
from .exceptions import RateLimitError

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Log request
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            # Reset the stream so it can be read again
            request._body = body
        
        request_logger.log_request(
            method=request.method,
            url=request.url,
            body=body.decode() if body else None
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        request_logger.log_response(
            status_code=response.status_code,
            response_time=process_time
        )
        
        # Add response time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = structlog.get_logger("metrics")
        self.request_count = 0
        self.total_response_time = 0.0
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        self.request_count += 1
        
        response = await call_next(request)
        
        response_time = time.time() - start_time
        self.total_response_time += response_time
        
        # Log metrics
        avg_response_time = self.total_response_time / self.request_count
        self.logger.info(
            "request_metrics",
            request_count=self.request_count,
            avg_response_time=avg_response_time,
            current_response_time=response_time,
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code
        )
        
        return response

def setup_middleware(app):
    """Set up all middleware for the FastAPI app."""
    from fastapi.middleware.cors import CORSMiddleware
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    if settings.METRICS_ENABLED:
        app.add_middleware(MetricsMiddleware)
    
    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    return app 