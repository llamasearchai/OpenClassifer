import os
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List
import logging

class Settings(BaseSettings):
    # API configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = Field(default=False)
    API_PREFIX: str = "/api/v1"
    
    # OpenAI configuration
    OPENAI_API_KEY: str = Field(...)
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_TEMPERATURE: float = 0.1
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TIMEOUT: int = 30
    
    # DSPy configuration
    DSPY_VERBOSE: bool = False
    DSPY_CACHE_DIR: str = "./dspy_cache"
    
    # LangChain configuration
    LANGCHAIN_TRACING: bool = False
    LANGCHAIN_CACHE_DIR: str = "./langchain_cache"
    
    # Classifier settings
    MODEL_CACHE_DIR: str = "./model_cache"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CLASS_LABELS: List[str] = ["positive", "negative", "neutral"]
    ENSEMBLE_ENABLED: bool = True
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 60
    CACHE_TTL: int = 3600
    
    # Security settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring settings
    METRICS_ENABLED: bool = True
    HEALTH_CHECK_INTERVAL: int = 30
    
    # Jina AI settings
    JINA_API_KEY: Optional[str] = None
    JINA_MODEL: str = "jina-embeddings-v2-base-en"
    
    # Agent settings
    AGENT_ENABLED: bool = True
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_MEMORY_SIZE: int = 1000

    @validator('CLASS_LABELS', pre=True)
    def parse_class_labels(cls, v):
        if isinstance(v, str):
            return [label.strip() for label in v.split(',')]
        return v

    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()