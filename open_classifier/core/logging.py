import logging
import sys
from typing import Optional
from datetime import datetime
import json
import structlog
from .config import settings

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Set up and configure a logger with structured logging."""
    if level is None:
        level = settings.LOG_LEVEL

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if settings.DEBUG else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(settings.LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)

class RequestLogger:
    """Middleware for logging requests and responses."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_request(self, method: str, url: str, body: str = None):
        """Log incoming request."""
        log_data = {
            "event": "request_received",
            "method": method,
            "url": str(url),
            "timestamp": datetime.utcnow().isoformat()
        }
        if body:
            log_data["body_length"] = len(body)
        
        self.logger.info(json.dumps(log_data))
    
    def log_response(self, status_code: int, response_time: float):
        """Log response details."""
        log_data = {
            "event": "response_sent",
            "status_code": status_code,
            "response_time_ms": round(response_time * 1000, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(json.dumps(log_data))

# Global logger instances
logger = setup_logger("open-classifier")
struct_logger = get_logger("open-classifier")
request_logger = RequestLogger(logger)