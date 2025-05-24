"""Custom exceptions for the Open Classifier."""

from typing import Optional, Dict, Any

class OpenClassifierBaseException(Exception):
    """Base exception for all Open Classifier errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ClassificationError(OpenClassifierBaseException):
    """Exception raised when classification fails."""
    pass

class ValidationError(OpenClassifierBaseException):
    """Exception raised when input validation fails."""
    pass

class AuthenticationError(OpenClassifierBaseException):
    """Exception raised when authentication fails."""
    pass

class ConfigurationError(OpenClassifierBaseException):
    """Exception raised when configuration is invalid."""
    pass

class ModelLoadError(OpenClassifierBaseException):
    """Exception raised when model loading fails."""
    pass

class RateLimitError(OpenClassifierBaseException):
    """Exception raised when rate limit is exceeded."""
    pass

class TimeoutError(OpenClassifierBaseException):
    """Exception raised when operations timeout."""
    pass 