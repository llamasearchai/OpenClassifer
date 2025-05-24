from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum

class ClassificationMode(str, Enum):
    """Available classification modes."""
    DSPY_ONLY = "dspy_only"
    LANGCHAIN_ONLY = "langchain_only"
    ENSEMBLE = "ensemble"
    AGENT = "agent"

class ClassificationRequest(BaseModel):
    """Request model for text classification."""
    text: str = Field(..., description="Text to classify", min_length=1, max_length=10000)
    mode: Optional[ClassificationMode] = Field(None, description="Classification mode to use")
    include_embeddings: bool = Field(False, description="Include text embeddings in response")
    include_alternatives: bool = Field(False, description="Include alternative predictions")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class BatchClassificationRequest(BaseModel):
    """Request model for batch text classification."""
    texts: List[str] = Field(..., description="List of texts to classify", min_items=1, max_items=100)
    mode: Optional[ClassificationMode] = Field(None, description="Classification mode to use")
    max_concurrent: Optional[int] = Field(None, description="Maximum concurrent requests", ge=1, le=20)
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
        return [text.strip() for text in v]

class SimilarityRequest(BaseModel):
    """Request model for similarity-based classification."""
    text: str = Field(..., description="Text to classify", min_length=1)
    reference_texts: List[str] = Field(..., description="Reference texts for similarity", min_items=1)
    similarity_threshold: float = Field(0.8, description="Similarity threshold", ge=0.0, le=1.0)
    mode: Optional[ClassificationMode] = Field(None, description="Classification mode to use")

class EmbeddingRequest(BaseModel):
    """Request model for text embeddings."""
    texts: List[str] = Field(..., description="Texts to embed", min_items=1, max_items=50)
    normalize: bool = Field(True, description="Whether to normalize embeddings")

class ClusteringRequest(BaseModel):
    """Request model for text clustering."""
    texts: List[str] = Field(..., description="Texts to cluster", min_items=3, max_items=100)
    n_clusters: int = Field(3, description="Number of clusters", ge=2, le=20)

class ClassificationResponse(BaseModel):
    """Response model for text classification."""
    class_label: str = Field(..., alias="class", description="Predicted class")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0.0, le=1.0)
    explanation: str = Field(..., description="Explanation for the classification")
    mode: str = Field(..., description="Classification mode used")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: float = Field(..., description="Unix timestamp")
    from_cache: bool = Field(False, description="Whether result was served from cache")
    embedding: Optional[List[float]] = Field(None, description="Text embedding if requested")
    alternatives: Optional[List[Dict[str, Any]]] = Field(None, description="Alternative predictions")
    ensemble_details: Optional[Dict[str, Any]] = Field(None, description="Ensemble details if applicable")
    agent_reasoning: Optional[str] = Field(None, description="Agent reasoning if applicable")
    
    class Config:
        allow_population_by_field_name = True

class BatchClassificationResponse(BaseModel):
    """Response model for batch classification."""
    results: List[ClassificationResponse] = Field(..., description="Classification results")
    total_texts: int = Field(..., description="Total number of texts processed")
    processing_time: float = Field(..., description="Total processing time")
    success_count: int = Field(..., description="Number of successful classifications")
    error_count: int = Field(..., description="Number of failed classifications")

class SimilarityResponse(BaseModel):
    """Response model for similarity-based classification."""
    classification: ClassificationResponse = Field(..., description="Classification result")
    similar_texts: List[Dict[str, Any]] = Field(..., description="Similar texts found")
    high_similarity_warning: bool = Field(False, description="Warning for high similarity")
    most_similar: Optional[Dict[str, Any]] = Field(None, description="Most similar text")

class EmbeddingResponse(BaseModel):
    """Response model for text embeddings."""
    embeddings: List[List[float]] = Field(..., description="Text embeddings")
    dimension: int = Field(..., description="Embedding dimension")
    model_name: str = Field(..., description="Embedding model used")
    processing_time: float = Field(..., description="Processing time")

class ClusteringResponse(BaseModel):
    """Response model for text clustering."""
    clusters: Dict[str, Any] = Field(..., description="Clustering results")
    n_clusters: int = Field(..., description="Number of clusters")
    total_texts: int = Field(..., description="Total texts clustered")
    processing_time: float = Field(..., description="Processing time")

class ServiceInfoResponse(BaseModel):
    """Response model for service information."""
    mode: str = Field(..., description="Current classification mode")
    available_labels: List[str] = Field(..., description="Available classification labels")
    metrics: Dict[str, Any] = Field(..., description="Service metrics")
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    models_info: Dict[str, Any] = Field(..., description="Model information")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")

class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    service_cache_size: int = Field(..., description="Service cache size")
    model_cache_sizes: Dict[str, int] = Field(..., description="Individual model cache sizes")
    cache_hit_rates: Dict[str, float] = Field(..., description="Cache hit rates")
    total_requests: int = Field(..., description="Total requests processed")

class ModelSwitchRequest(BaseModel):
    """Request model for switching classification mode."""
    mode: ClassificationMode = Field(..., description="New classification mode")
    clear_cache: bool = Field(False, description="Whether to clear cache after switch")

class ModelSwitchResponse(BaseModel):
    """Response model for mode switching."""
    old_mode: str = Field(..., description="Previous mode")
    new_mode: str = Field(..., description="New mode")
    success: bool = Field(..., description="Whether switch was successful")
    message: str = Field(..., description="Status message")

class LabelsResponse(BaseModel):
    """Response model for available labels."""
    labels: List[str] = Field(..., description="Available classification labels")
    count: int = Field(..., description="Number of labels")
    
class MetricsResponse(BaseModel):
    """Response model for detailed metrics."""
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    success_rate: float = Field(..., description="Success rate (0-1)")
    average_response_time: float = Field(..., description="Average response time")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    uptime_seconds: float = Field(..., description="Service uptime")
    requests_per_minute: float = Field(..., description="Requests per minute")
    error_rate: float = Field(..., description="Error rate (0-1)")