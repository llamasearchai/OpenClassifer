import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from .api.router import router
from .core.config import settings
from .core.logging import logger, struct_logger
from .core.middleware import setup_middleware
from .core.exceptions import OpenClassifierBaseException
from . import __version__

# Global variables for lifecycle management
classifier_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting Open Classifier API", version=__version__)
    
    try:
        # Initialize the classifier service
        from .services.classifier_service import ClassifierService
        global classifier_service
        classifier_service = ClassifierService()
        
        # Store in app state for access in routes
        app.state.classifier_service = classifier_service
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Open Classifier API")
    
    try:
        # Cleanup resources
        if classifier_service:
            # Save caches before shutdown
            if hasattr(classifier_service, 'embedding_model') and classifier_service.embedding_model:
                classifier_service.embedding_model._save_cache()
            
            logger.info("Cleanup completed successfully")
            
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Open Classifier API",
    description="""
    Advanced text classifier with DSPy and LangChain integration.
    
    ## Features
    
    * **Multiple Classification Modes**: DSPy, LangChain, Ensemble, and Agent-based classification
    * **Batch Processing**: Classify multiple texts concurrently
    * **Semantic Embeddings**: Generate and use text embeddings for similarity analysis
    * **Text Clustering**: Cluster texts using semantic embeddings
    * **Caching**: Intelligent caching for improved performance
    * **Monitoring**: Comprehensive metrics and health checks
    * **Security**: API key authentication and rate limiting
    
    ## Classification Modes
    
    * **DSPy Only**: Use DSPy framework for classification
    * **LangChain Only**: Use LangChain framework for classification  
    * **Ensemble**: Combine multiple models for improved accuracy
    * **Agent**: Use AI agents with reasoning and tool integration
    
    ## Authentication
    
    Use the `Authorization: Bearer <api-key>` header for authenticated requests.
    Demo key: `demo-key`
    """,
    version=__version__,
    contact={
        "name": "Open Classifier Team",
        "url": "https://github.com/your-org/open-classifier",
        "email": "support@openclassifier.ai"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Set up middleware
app = setup_middleware(app)

# Include API router
app.include_router(router, prefix=settings.API_PREFIX, tags=["Classification"])

# Global exception handler
@app.exception_handler(OpenClassifierBaseException)
async def classifier_exception_handler(request: Request, exc: OpenClassifierBaseException):
    """Handle custom classifier exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.message,
            "error_type": exc.__class__.__name__,
            "details": exc.details,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_type": "InternalServerError",
            "timestamp": time.time()
        }
    )

@app.get("/", 
         summary="Root endpoint",
         description="Get basic API information")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Open Classifier API",
        "version": __version__,
        "description": "Advanced text classifier with DSPy and LangChain integration",
        "docs_url": "/docs",
        "health_url": f"{settings.API_PREFIX}/health",
        "classification_url": f"{settings.API_PREFIX}/classify",
        "features": [
            "Multiple classification modes",
            "Batch processing",
            "Semantic embeddings",
            "Text clustering", 
            "Intelligent caching",
            "Comprehensive monitoring",
            "API key authentication",
            "Rate limiting"
        ],
        "modes": [
            "dspy_only",
            "langchain_only", 
            "ensemble",
            "agent"
        ]
    }

@app.get("/version",
         summary="Get API version",
         description="Get the current API version")
async def get_version():
    """Get API version information."""
    return {
        "version": __version__,
        "api_version": "v1",
        "build_info": {
            "python_version": "3.9+",
            "frameworks": ["FastAPI", "DSPy", "LangChain"],
            "models": ["OpenAI GPT", "Sentence Transformers"]
        }
    }

def run_server():
    """Run the API server with production configuration."""
    config = {
        "host": settings.API_HOST,
        "port": settings.API_PORT,
        "reload": settings.DEBUG,
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": True,
        "use_colors": True,
        "loop": "asyncio"
    }
    
    if not settings.DEBUG:
        # Production settings
        config.update({
            "workers": 1,  # Single worker for now due to model loading
            "reload": False,
            "access_log": False
        })
    
    logger.info("Starting server", config=config)
    uvicorn.run("open_classifier.main:app", **config)

if __name__ == "__main__":
    run_server()