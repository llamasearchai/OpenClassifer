from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
import time

from ..services.classifier_service import ClassifierService, ClassifierMode
from ..core.logging import logger, struct_logger
from ..core.security import get_current_user, generate_request_id
from ..core.exceptions import ClassificationError, ValidationError, AuthenticationError
from ..core.middleware import limiter
from ..core.config import settings
from . import models
from .. import __version__

router = APIRouter()
api_logger = struct_logger.bind(component="api")

# Global service instance (in production, use dependency injection)
_classifier_service = None

def get_classifier_service():
    """Dependency to get classifier service."""
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = ClassifierService()
    return _classifier_service

@router.post("/classify", 
             response_model=models.ClassificationResponse,
             summary="Classify text",
             description="Classify a single text using the configured classification model(s)")
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_PERIOD}minute")
async def classify_text(
    request: models.ClassificationRequest,
    background_tasks: BackgroundTasks,
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Classify the provided text."""
    request_id = generate_request_id()
    
    try:
        api_logger.info("Classification request received", 
                       request_id=request_id, 
                       user_id=user_info.get("user_id"),
                       text_length=len(request.text))
        
        # Override service mode if specified in request
        if request.mode:
            original_mode = classifier_service.mode
            classifier_service.switch_mode(ClassifierMode(request.mode))
            # Schedule mode restoration
            background_tasks.add_task(classifier_service.switch_mode, original_mode)
        
        result = await classifier_service.classify(
            text=request.text,
            include_embeddings=request.include_embeddings,
            include_alternatives=request.include_alternatives
        )
        
        # Convert to response model
        response = models.ClassificationResponse(
            class_label=result["class"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            mode=result["mode"],
            processing_time=result["processing_time"],
            timestamp=result["timestamp"],
            from_cache=result.get("from_cache", False),
            embedding=result.get("embedding"),
            alternatives=result.get("alternatives"),
            ensemble_details=result.get("ensemble_details"),
            agent_reasoning=result.get("agent_reasoning")
        )
        
        api_logger.info("Classification completed", 
                       request_id=request_id,
                       classification=result["class"],
                       confidence=result["confidence"])
        
        return response
        
    except ValidationError as e:
        api_logger.warning("Validation error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except ClassificationError as e:
        api_logger.error("Classification error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        api_logger.error("Unexpected error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/classify/batch",
             response_model=models.BatchClassificationResponse,
             summary="Batch classify texts",
             description="Classify multiple texts concurrently")
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS//2}/{settings.RATE_LIMIT_PERIOD}minute")
async def classify_batch(
    request: models.BatchClassificationRequest,
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Classify multiple texts in batch."""
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        api_logger.info("Batch classification request", 
                       request_id=request_id,
                       user_id=user_info.get("user_id"),
                       text_count=len(request.texts))
        
        # Override service mode if specified
        if request.mode:
            classifier_service.switch_mode(ClassifierMode(request.mode))
        
        results = await classifier_service.classify_batch(
            texts=request.texts,
            max_concurrent=request.max_concurrent
        )
        
        # Convert results to response models
        classification_responses = []
        success_count = 0
        error_count = 0
        
        for result in results:
            if result.get("error"):
                error_count += 1
            else:
                success_count += 1
                
            classification_responses.append(
                models.ClassificationResponse(
                    class_label=result["class"],
                    confidence=result["confidence"],
                    explanation=result["explanation"],
                    mode=result.get("mode", classifier_service.mode.value),
                    processing_time=result.get("processing_time", 0),
                    timestamp=result.get("timestamp", time.time()),
                    from_cache=result.get("from_cache", False)
                )
            )
        
        response = models.BatchClassificationResponse(
            results=classification_responses,
            total_texts=len(request.texts),
            processing_time=time.time() - start_time,
            success_count=success_count,
            error_count=error_count
        )
        
        api_logger.info("Batch classification completed", 
                       request_id=request_id,
                       success_count=success_count,
                       error_count=error_count)
        
        return response
        
    except Exception as e:
        api_logger.error("Batch classification error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify/similarity",
             response_model=models.SimilarityResponse,
             summary="Classify with similarity analysis",
             description="Classify text with similarity analysis to reference texts")
async def classify_with_similarity(
    request: models.SimilarityRequest,
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Classify text with similarity analysis."""
    request_id = generate_request_id()
    
    try:
        api_logger.info("Similarity classification request", 
                       request_id=request_id,
                       reference_count=len(request.reference_texts))
        
        if request.mode:
            classifier_service.switch_mode(ClassifierMode(request.mode))
        
        result = await classifier_service.classify_with_similarity(
            text=request.text,
            reference_texts=request.reference_texts,
            similarity_threshold=request.similarity_threshold
        )
        
        classification_response = models.ClassificationResponse(
            class_label=result["class"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            mode=result["mode"],
            processing_time=result["processing_time"],
            timestamp=result["timestamp"],
            from_cache=result.get("from_cache", False)
        )
        
        response = models.SimilarityResponse(
            classification=classification_response,
            similar_texts=result.get("similar_texts", []),
            high_similarity_warning=result.get("high_similarity_warning", False),
            most_similar=result.get("most_similar")
        )
        
        return response
        
    except Exception as e:
        api_logger.error("Similarity classification error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings",
             response_model=models.EmbeddingResponse,
             summary="Generate text embeddings",
             description="Generate semantic embeddings for texts")
async def generate_embeddings(
    request: models.EmbeddingRequest,
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Generate embeddings for texts."""
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        api_logger.info("Embedding request", 
                       request_id=request_id,
                       text_count=len(request.texts))
        
        embeddings = classifier_service.embedding_model.encode(request.texts)
        
        if request.normalize:
            embeddings = classifier_service.embedding_model._normalize_embeddings(embeddings)
        
        response = models.EmbeddingResponse(
            embeddings=embeddings.tolist(),
            dimension=classifier_service.embedding_model.embedding_dim,
            model_name=classifier_service.embedding_model.model_name,
            processing_time=time.time() - start_time
        )
        
        return response
        
    except Exception as e:
        api_logger.error("Embedding generation error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cluster",
             response_model=models.ClusteringResponse,
             summary="Cluster texts",
             description="Cluster texts using semantic embeddings")
async def cluster_texts(
    request: models.ClusteringRequest,
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Cluster texts using embeddings."""
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        api_logger.info("Clustering request", 
                       request_id=request_id,
                       text_count=len(request.texts),
                       n_clusters=request.n_clusters)
        
        result = classifier_service.embedding_model.cluster_texts(
            texts=request.texts,
            n_clusters=request.n_clusters
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        response = models.ClusteringResponse(
            clusters=result["clusters"],
            n_clusters=result["n_clusters"],
            total_texts=result["total_texts"],
            processing_time=time.time() - start_time
        )
        
        return response
        
    except Exception as e:
        api_logger.error("Clustering error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/labels", 
            response_model=models.LabelsResponse,
            summary="Get available labels",
            description="Get list of available classification labels")
async def get_labels(
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """Get available classification labels."""
    labels = classifier_service.get_available_labels()
    return models.LabelsResponse(labels=labels, count=len(labels))

@router.get("/info",
            response_model=models.ServiceInfoResponse,
            summary="Get service information",
            description="Get comprehensive service information and statistics")
async def get_service_info(
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Get service information."""
    info = classifier_service.get_service_info()
    return models.ServiceInfoResponse(**info)

@router.get("/metrics",
            response_model=models.MetricsResponse,
            summary="Get detailed metrics",
            description="Get detailed performance metrics")
async def get_metrics(
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Get detailed service metrics."""
    info = classifier_service.get_service_info()
    metrics = info["metrics"]
    
    # Calculate additional metrics
    uptime = metrics["uptime_seconds"]
    requests_per_minute = (metrics["total_requests"] / max(uptime / 60, 1)) if uptime > 0 else 0
    error_rate = metrics["failed_requests"] / max(metrics["total_requests"], 1)
    
    return models.MetricsResponse(
        total_requests=metrics["total_requests"],
        successful_requests=metrics["successful_requests"],
        failed_requests=metrics["failed_requests"],
        success_rate=metrics["success_rate"],
        average_response_time=metrics["average_response_time"],
        cache_hit_rate=metrics["cache_hit_rate"],
        uptime_seconds=metrics["uptime_seconds"],
        requests_per_minute=requests_per_minute,
        error_rate=error_rate
    )

@router.get("/cache/stats",
            response_model=models.CacheStatsResponse,
            summary="Get cache statistics",
            description="Get detailed cache statistics")
async def get_cache_stats(
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Get cache statistics."""
    info = classifier_service.get_service_info()
    
    model_cache_sizes = {}
    cache_hit_rates = {}
    
    for model_name, model_info in info["models_info"].items():
        if "cache_size" in model_info:
            model_cache_sizes[model_name] = model_info["cache_size"]
        if "cache_hit_rate" in model_info:
            cache_hit_rates[model_name] = model_info["cache_hit_rate"]
    
    return models.CacheStatsResponse(
        service_cache_size=info["cache_stats"]["service_cache_size"],
        model_cache_sizes=model_cache_sizes,
        cache_hit_rates=cache_hit_rates,
        total_requests=info["metrics"]["total_requests"]
    )

@router.post("/cache/clear",
             summary="Clear caches",
             description="Clear all caches")
async def clear_cache(
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Clear all caches."""
    if "write" not in user_info.get("permissions", []):
        raise HTTPException(status_code=403, detail="Write permission required")
    
    classifier_service.clear_cache()
    api_logger.info("Caches cleared", user_id=user_info.get("user_id"))
    
    return {"message": "All caches cleared successfully"}

@router.post("/mode/switch",
             response_model=models.ModelSwitchResponse,
             summary="Switch classification mode",
             description="Switch the classification mode dynamically")
async def switch_mode(
    request: models.ModelSwitchRequest,
    classifier_service: ClassifierService = Depends(get_classifier_service),
    user_info: dict = Depends(get_current_user)
):
    """Switch classification mode."""
    if "write" not in user_info.get("permissions", []):
        raise HTTPException(status_code=403, detail="Write permission required")
    
    old_mode = classifier_service.mode.value
    
    try:
        classifier_service.switch_mode(ClassifierMode(request.mode))
        
        if request.clear_cache:
            classifier_service.clear_cache()
        
        api_logger.info("Mode switched", 
                       user_id=user_info.get("user_id"),
                       old_mode=old_mode,
                       new_mode=request.mode)
        
        return models.ModelSwitchResponse(
            old_mode=old_mode,
            new_mode=request.mode,
            success=True,
            message=f"Successfully switched from {old_mode} to {request.mode}"
        )
        
    except Exception as e:
        api_logger.error("Mode switch failed", 
                        user_id=user_info.get("user_id"),
                        error=str(e))
        
        return models.ModelSwitchResponse(
            old_mode=old_mode,
            new_mode=request.mode,
            success=False,
            message=f"Failed to switch mode: {str(e)}"
        )

@router.get("/health", 
            response_model=models.HealthResponse,
            summary="Health check",
            description="Check API health and model status")
async def health_check(
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """Check API health."""
    try:
        info = classifier_service.get_service_info()
        
        models_loaded = {
            "dspy": classifier_service.dspy_classifier is not None,
            "langchain": classifier_service.langchain_classifier is not None,
            "agent": classifier_service.agent is not None,
            "embeddings": classifier_service.embedding_model is not None
        }
        
        return models.HealthResponse(
            status="healthy",
            version=__version__,
            uptime=info["metrics"]["uptime_seconds"],
            models_loaded=models_loaded
        )
        
    except Exception as e:
        api_logger.error("Health check failed", error=str(e))
        return models.HealthResponse(
            status="unhealthy",
            version=__version__,
            uptime=0,
            models_loaded={}
        )

# Error handlers
@router.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=models.ErrorResponse(
            error=str(exc),
            error_type="ValidationError",
            timestamp=time.time()
        ).dict()
    )

@router.exception_handler(ClassificationError)
async def classification_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=models.ErrorResponse(
            error=str(exc),
            error_type="ClassificationError",
            timestamp=time.time()
        ).dict()
    )

@router.exception_handler(AuthenticationError)
async def auth_exception_handler(request, exc):
    return JSONResponse(
        status_code=401,
        content=models.ErrorResponse(
            error=str(exc),
            error_type="AuthenticationError",
            timestamp=time.time()
        ).dict()
    )