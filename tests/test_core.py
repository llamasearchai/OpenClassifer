"""
Unit tests for core modules.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request, HTTPException
from fastapi.testclient import TestClient
import time

from open_classifier.core.config import Settings, settings
from open_classifier.core.logging import logger, struct_logger, RequestLogger
from open_classifier.core.exceptions import (
    OpenClassifierBaseException,
    ClassificationError,
    ValidationError,
    AuthenticationError,
    ConfigurationError,
    ModelLoadError,
    RateLimitError,
    TimeoutError
)
from open_classifier.core.security import APIKeyValidator, hash_text, generate_request_id
from open_classifier.core.middleware import RequestLoggingMiddleware, MetricsMiddleware


class TestSettings:
    """Test configuration settings."""
    
    @pytest.mark.unit
    def test_default_settings(self):
        """Test default configuration values."""
        test_settings = Settings(OPENAI_API_KEY="test-key")
        
        assert test_settings.API_HOST == "0.0.0.0"
        assert test_settings.API_PORT == 8000
        assert test_settings.DEBUG is False
        assert test_settings.API_PREFIX == "/api/v1"
        assert test_settings.OPENAI_MODEL == "gpt-4"
        assert test_settings.OPENAI_TEMPERATURE == 0.1
        assert test_settings.CONFIDENCE_THRESHOLD == 0.5
        assert test_settings.MAX_CONCURRENT_REQUESTS == 10
        assert test_settings.RATE_LIMIT_REQUESTS == 100
        assert test_settings.LOG_LEVEL == "INFO"
    
    @pytest.mark.unit
    def test_class_labels_parsing(self):
        """Test class labels parsing from string."""
        test_settings = Settings(
            OPENAI_API_KEY="test-key",
            CLASS_LABELS="positive,negative,neutral"
        )
        
        assert test_settings.CLASS_LABELS == ["positive", "negative", "neutral"]
    
    @pytest.mark.unit
    def test_log_level_validation(self):
        """Test log level validation."""
        with pytest.raises(ValueError):
            Settings(OPENAI_API_KEY="test-key", LOG_LEVEL="INVALID")
    
    @pytest.mark.unit
    def test_environment_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {"API_PORT": "9000", "DEBUG": "true"}):
            test_settings = Settings(OPENAI_API_KEY="test-key")
            assert test_settings.API_PORT == 9000
            assert test_settings.DEBUG is True


class TestLogging:
    """Test logging functionality."""
    
    @pytest.mark.unit
    def test_logger_exists(self):
        """Test that logger is properly configured."""
        assert logger is not None
        assert struct_logger is not None
    
    @pytest.mark.unit
    def test_request_logger(self):
        """Test request logger functionality."""
        request_logger = RequestLogger()
        
        # Mock request
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url = "http://test.com/api/v1/classify"
        mock_request.headers = {"Authorization": "Bearer test-key"}
        
        # Test log request
        with patch.object(struct_logger, 'info') as mock_log:
            request_logger.log_request(mock_request, "test-request-id")
            mock_log.assert_called_once()
    
    @pytest.mark.unit
    def test_request_logger_response(self):
        """Test request logger response logging."""
        request_logger = RequestLogger()
        
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(struct_logger, 'info') as mock_log:
            request_logger.log_response(mock_response, "test-request-id", 0.5)
            mock_log.assert_called_once()


class TestExceptions:
    """Test custom exceptions."""
    
    @pytest.mark.unit
    def test_base_exception(self):
        """Test base exception."""
        exc = OpenClassifierBaseException("Test error", {"key": "value"})
        
        assert exc.message == "Test error"
        assert exc.details == {"key": "value"}
        assert str(exc) == "Test error"
    
    @pytest.mark.unit
    def test_classification_error(self):
        """Test classification error."""
        exc = ClassificationError("Classification failed", {"model": "dspy"})
        
        assert exc.message == "Classification failed"
        assert exc.details == {"model": "dspy"}
        assert isinstance(exc, OpenClassifierBaseException)
    
    @pytest.mark.unit
    def test_validation_error(self):
        """Test validation error."""
        exc = ValidationError("Invalid input", {"field": "text"})
        
        assert exc.message == "Invalid input"
        assert exc.details == {"field": "text"}
    
    @pytest.mark.unit
    def test_authentication_error(self):
        """Test authentication error."""
        exc = AuthenticationError("Invalid API key")
        
        assert exc.message == "Invalid API key"
        assert exc.details == {}
    
    @pytest.mark.unit
    def test_configuration_error(self):
        """Test configuration error."""
        exc = ConfigurationError("Missing config", {"setting": "OPENAI_API_KEY"})
        
        assert exc.message == "Missing config"
        assert exc.details == {"setting": "OPENAI_API_KEY"}
    
    @pytest.mark.unit
    def test_model_load_error(self):
        """Test model load error."""
        exc = ModelLoadError("Failed to load model", {"model": "dspy"})
        
        assert exc.message == "Failed to load model"
        assert exc.details == {"model": "dspy"}
    
    @pytest.mark.unit
    def test_rate_limit_error(self):
        """Test rate limit error."""
        exc = RateLimitError("Rate limit exceeded", {"limit": 100})
        
        assert exc.message == "Rate limit exceeded"
        assert exc.details == {"limit": 100}
    
    @pytest.mark.unit
    def test_timeout_error(self):
        """Test timeout error."""
        exc = TimeoutError("Request timeout", {"timeout": 30})
        
        assert exc.message == "Request timeout"
        assert exc.details == {"timeout": 30}


class TestSecurity:
    """Test security functionality."""
    
    @pytest.mark.unit
    def test_api_key_validator_init(self):
        """Test API key validator initialization."""
        validator = APIKeyValidator()
        assert validator is not None
    
    @pytest.mark.unit
    async def test_valid_api_key(self):
        """Test valid API key validation."""
        validator = APIKeyValidator()
        
        # Mock request with valid key
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer demo-key"}
        
        # Should not raise exception
        await validator.validate_api_key(mock_request)
    
    @pytest.mark.unit
    async def test_invalid_api_key(self):
        """Test invalid API key validation."""
        validator = APIKeyValidator()
        
        # Mock request with invalid key
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer invalid-key"}
        
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_api_key(mock_request)
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.unit
    async def test_missing_api_key(self):
        """Test missing API key validation."""
        validator = APIKeyValidator()
        
        # Mock request without authorization header
        mock_request = Mock()
        mock_request.headers = {}
        
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_api_key(mock_request)
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.unit
    async def test_malformed_authorization_header(self):
        """Test malformed authorization header."""
        validator = APIKeyValidator()
        
        # Mock request with malformed header
        mock_request = Mock()
        mock_request.headers = {"Authorization": "InvalidFormat"}
        
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_api_key(mock_request)
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.unit
    def test_hash_text(self):
        """Test text hashing."""
        text = "test text"
        hashed = hash_text(text)
        
        assert hashed is not None
        assert len(hashed) == 64  # SHA-256 hex digest length
        assert hashed == hash_text(text)  # Consistent hashing
    
    @pytest.mark.unit
    def test_generate_request_id(self):
        """Test request ID generation."""
        request_id = generate_request_id()
        
        assert request_id is not None
        assert len(request_id) > 0
        assert generate_request_id() != request_id  # Should be unique


class TestMiddleware:
    """Test middleware functionality."""
    
    @pytest.mark.unit
    async def test_request_logging_middleware(self):
        """Test request logging middleware."""
        middleware = RequestLoggingMiddleware()
        
        # Mock request and call_next
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url = "http://test.com/api/v1/classify"
        mock_request.headers = {"Authorization": "Bearer test-key"}
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        async def mock_call_next(request):
            return mock_response
        
        with patch('open_classifier.core.middleware.generate_request_id') as mock_gen_id:
            mock_gen_id.return_value = "test-request-id"
            
            with patch.object(middleware.request_logger, 'log_request') as mock_log_req:
                with patch.object(middleware.request_logger, 'log_response') as mock_log_resp:
                    response = await middleware.dispatch(mock_request, mock_call_next)
                    
                    assert response == mock_response
                    mock_log_req.assert_called_once()
                    mock_log_resp.assert_called_once()
    
    @pytest.mark.unit
    async def test_metrics_middleware(self):
        """Test metrics middleware."""
        middleware = MetricsMiddleware()
        
        # Mock request and call_next
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url = Mock()
        mock_request.url.path = "/api/v1/classify"
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        async def mock_call_next(request):
            return mock_response
        
        with patch.object(middleware.monitor, 'record_request') as mock_record:
            response = await middleware.dispatch(mock_request, mock_call_next)
            
            assert response == mock_response
            mock_record.assert_called_once()
    
    @pytest.mark.unit
    async def test_metrics_middleware_error(self):
        """Test metrics middleware with error."""
        middleware = MetricsMiddleware()
        
        # Mock request and call_next that raises exception
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url = Mock()
        mock_request.url.path = "/api/v1/classify"
        
        async def mock_call_next(request):
            raise Exception("Test error")
        
        with patch.object(middleware.monitor, 'record_request') as mock_record:
            with pytest.raises(Exception):
                await middleware.dispatch(mock_request, mock_call_next)
            
            mock_record.assert_called_once()


class TestIntegration:
    """Integration tests for core modules."""
    
    @pytest.mark.integration
    def test_settings_with_logging(self):
        """Test settings integration with logging."""
        # Test that settings can be used to configure logging
        test_settings = Settings(
            OPENAI_API_KEY="test-key",
            LOG_LEVEL="DEBUG"
        )
        
        assert test_settings.LOG_LEVEL == "DEBUG"
    
    @pytest.mark.integration
    def test_exception_with_logging(self):
        """Test exception handling with logging."""
        with patch.object(logger, 'error') as mock_log:
            try:
                raise ClassificationError("Test error", {"model": "test"})
            except ClassificationError as e:
                logger.error("Classification failed", error=str(e))
                mock_log.assert_called_once()
    
    @pytest.mark.integration
    async def test_security_with_middleware(self):
        """Test security integration with middleware."""
        validator = APIKeyValidator()
        
        # Mock request with valid key
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer demo-key"}
        
        # Should validate successfully
        await validator.validate_api_key(mock_request)
        
        # Test with middleware context
        middleware = RequestLoggingMiddleware()
        assert middleware is not None 