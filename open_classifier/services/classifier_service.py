from typing import Dict, Any, List, Optional, Union
import asyncio
import time
from dataclasses import dataclass
from enum import Enum

from ..models.dspy_classifier import ClassifierModule
from ..models.langchain_classifier import LangChainClassifier
from ..models.agent import ClassificationAgent
from ..models.embeddings import EmbeddingModel
from ..core.logging import logger, struct_logger
from ..core.config import settings
from ..core.exceptions import ClassificationError, ValidationError
from ..core.security import hash_text

class ClassifierMode(Enum):
    """Available classifier modes."""
    DSPY_ONLY = "dspy_only"
    LANGCHAIN_ONLY = "langchain_only"
    ENSEMBLE = "ensemble"
    AGENT = "agent"

@dataclass
class ClassificationMetrics:
    """Metrics for classification performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    accuracy_by_label: Dict[str, float] = None

class ClassifierService:
    """Advanced classifier service with ensemble, agent, and monitoring capabilities."""
    
    def __init__(self, 
                 mode: ClassifierMode = None,
                 use_ensemble: bool = None,  # Backward compatibility
                 enable_agent: bool = None):
        """
        Initialize the classifier service.
        
        Args:
            mode: Classification mode to use
            use_ensemble: Backward compatibility flag
            enable_agent: Whether to enable agent mode
        """
        # Handle backward compatibility
        if mode is None:
            if use_ensemble is not None:
                mode = ClassifierMode.ENSEMBLE if use_ensemble else ClassifierMode.DSPY_ONLY
            elif enable_agent:
                mode = ClassifierMode.AGENT
            else:
                mode = ClassifierMode.ENSEMBLE if settings.ENSEMBLE_ENABLED else ClassifierMode.DSPY_ONLY
        
        self.mode = mode
        self.logger = struct_logger.bind(component="classifier_service")
        
        # Initialize models based on mode
        self.dspy_classifier = None
        self.langchain_classifier = None
        self.agent = None
        self.embedding_model = None
        
        try:
            if mode in [ClassifierMode.DSPY_ONLY, ClassifierMode.ENSEMBLE]:
                self.dspy_classifier = ClassifierModule()
                
            if mode in [ClassifierMode.LANGCHAIN_ONLY, ClassifierMode.ENSEMBLE]:
                self.langchain_classifier = LangChainClassifier()
                
            if mode == ClassifierMode.AGENT and settings.AGENT_ENABLED:
                self.agent = ClassificationAgent()
                
            # Always initialize embedding model for similarity features
            self.embedding_model = EmbeddingModel()
            
            # Performance tracking
            self.metrics = ClassificationMetrics()
            self._start_time = time.time()
            self._response_times = []
            
            # Cache for service-level results
            self._service_cache = {}
            self._cache_timestamps = {}
            
            self.logger.info("Classifier service initialized", 
                           mode=mode.value, 
                           agent_enabled=self.agent is not None)
            
        except Exception as e:
            self.logger.error("Failed to initialize classifier service", error=str(e))
            raise ClassificationError(f"Failed to initialize classifier service: {str(e)}")
    
    async def classify(self, text: str, 
                      include_embeddings: bool = False,
                      include_alternatives: bool = False) -> Dict[str, Any]:
        """
        Classify the input text with comprehensive options.
        
        Args:
            text: The text to classify
            include_embeddings: Whether to include text embeddings in response
            include_alternatives: Whether to include alternative predictions
            
        Returns:
            Dictionary containing classification results
        """
        if not text or not text.strip():
            raise ValidationError("Input text cannot be empty")
        
        start_time = time.time()
        self.metrics.total_requests += 1
        
        # Check service-level cache
        text_hash = hash_text(text)
        cache_key = f"{self.mode.value}_{text_hash}_{include_embeddings}_{include_alternatives}"
        
        if self._should_use_cache(cache_key):
            cached_result = self._service_cache[cache_key].copy()
            cached_result["from_service_cache"] = True
            self.logger.info("Classification served from service cache")
            return cached_result
        
        try:
            result = await self._classify_with_mode(text)
            
            # Add metadata
            result["mode"] = self.mode.value
            result["processing_time"] = time.time() - start_time
            result["timestamp"] = time.time()
            result["from_service_cache"] = False
            
            # Add embeddings if requested
            if include_embeddings:
                embedding = self.embedding_model.get_text_embedding(text)
                if embedding is not None:
                    result["embedding"] = embedding.tolist()
            
            # Add alternative predictions if requested
            if include_alternatives and self.mode == ClassifierMode.ENSEMBLE:
                result["alternatives"] = await self._get_alternative_predictions(text)
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            # Update metrics
            self.metrics.successful_requests += 1
            self._response_times.append(result["processing_time"])
            self._update_metrics()
            
            self.logger.info("Classification completed successfully",
                           classification=result["class"],
                           confidence=result["confidence"],
                           processing_time=result["processing_time"])
            
            return result
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.logger.error("Classification failed", error=str(e), text_length=len(text))
            raise ClassificationError(f"Classification failed: {str(e)}")
    
    async def _classify_with_mode(self, text: str) -> Dict[str, Any]:
        """Classify text based on the selected mode."""
        if self.mode == ClassifierMode.DSPY_ONLY:
            return self.dspy_classifier.forward(text)
            
        elif self.mode == ClassifierMode.LANGCHAIN_ONLY:
            return await self.langchain_classifier.classify(text)
            
        elif self.mode == ClassifierMode.ENSEMBLE:
            return await self._ensemble_classify(text)
            
        elif self.mode == ClassifierMode.AGENT:
            if not self.agent:
                raise ClassificationError("Agent mode not available")
            return await self.agent.classify_with_reasoning(text)
            
        else:
            raise ClassificationError(f"Unknown classification mode: {self.mode}")
    
    async def _ensemble_classify(self, text: str) -> Dict[str, Any]:
        """Classify using ensemble of multiple models."""
        try:
            # Run classifiers in parallel
            tasks = []
            if self.dspy_classifier:
                tasks.append(asyncio.to_thread(self.dspy_classifier.forward, text))
            if self.langchain_classifier:
                tasks.append(self.langchain_classifier.classify(text))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Classifier {i} failed", error=str(result))
                else:
                    valid_results.append(result)
            
            if not valid_results:
                raise ClassificationError("All classifiers failed")
            
            if len(valid_results) == 1:
                return valid_results[0]
            
            # Ensemble multiple results
            return self._ensemble_results(valid_results)
            
        except Exception as e:
            # Fallback to single classifier
            if self.dspy_classifier:
                self.logger.warning("Ensemble failed, falling back to DSPy", error=str(e))
                result = self.dspy_classifier.forward(text)
                result["ensemble_fallback"] = True
                return result
            else:
                raise ClassificationError(f"Ensemble classification failed: {str(e)}")
    
    def _ensemble_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple classifiers using advanced ensemble methods."""
        if len(results) == 1:
            return results[0]
        
        # Vote-based ensemble with confidence weighting
        class_votes = {}
        total_confidence = 0
        
        for result in results:
            class_label = result["class"].lower()
            confidence = result["confidence"]
            
            if class_label not in class_votes:
                class_votes[class_label] = {"votes": 0, "confidence_sum": 0}
            
            class_votes[class_label]["votes"] += 1
            class_votes[class_label]["confidence_sum"] += confidence
            total_confidence += confidence
        
        # Find the winning class
        best_class = None
        best_score = -1
        
        for class_label, data in class_votes.items():
            # Combined score: votes + normalized confidence
            score = data["votes"] + (data["confidence_sum"] / total_confidence)
            if score > best_score:
                best_score = score
                best_class = class_label
        
        # Calculate ensemble confidence
        winning_votes = class_votes[best_class]
        ensemble_confidence = winning_votes["confidence_sum"] / winning_votes["votes"]
        
        # Adjust confidence based on agreement
        agreement_boost = winning_votes["votes"] / len(results)
        final_confidence = min(1.0, ensemble_confidence * (0.8 + 0.2 * agreement_boost))
        
        # Create explanation
        explanations = [r["explanation"] for r in results]
        ensemble_explanation = f"Ensemble of {len(results)} classifiers. "
        if winning_votes["votes"] == len(results):
            ensemble_explanation += f"All classifiers agreed on '{best_class}'. "
        else:
            ensemble_explanation += f"{winning_votes['votes']}/{len(results)} classifiers predicted '{best_class}'. "
        
        return {
            "class": best_class,
            "confidence": final_confidence,
            "explanation": ensemble_explanation,
            "ensemble_details": {
                "num_classifiers": len(results),
                "agreement_rate": agreement_boost,
                "individual_results": results
            }
        }
    
    async def _get_alternative_predictions(self, text: str) -> List[Dict[str, Any]]:
        """Get alternative predictions from individual classifiers."""
        alternatives = []
        
        if self.dspy_classifier:
            try:
                dspy_result = self.dspy_classifier.forward(text)
                alternatives.append({
                    "classifier": "dspy",
                    "prediction": dspy_result
                })
            except Exception as e:
                self.logger.warning("Failed to get DSPy alternative", error=str(e))
        
        if self.langchain_classifier:
            try:
                langchain_result = await self.langchain_classifier.classify(text)
                alternatives.append({
                    "classifier": "langchain",
                    "prediction": langchain_result
                })
            except Exception as e:
                self.logger.warning("Failed to get LangChain alternative", error=str(e))
        
        return alternatives
    
    async def classify_batch(self, texts: List[str], 
                           max_concurrent: int = None) -> List[Dict[str, Any]]:
        """Classify multiple texts concurrently."""
        if not texts:
            return []
        
        max_concurrent = max_concurrent or settings.MAX_CONCURRENT_REQUESTS
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def classify_with_semaphore(text: str, index: int):
            async with semaphore:
                try:
                    result = await self.classify(text)
                    result["batch_index"] = index
                    return result
                except Exception as e:
                    self.logger.error("Batch classification failed", index=index, error=str(e))
                    return {
                        "class": settings.CLASS_LABELS[0],
                        "confidence": 0.0,
                        "explanation": f"Batch processing error: {str(e)}",
                        "batch_index": index,
                        "error": True
                    }
        
        tasks = [classify_with_semaphore(text, i) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        
        # Sort by original index
        results.sort(key=lambda x: x["batch_index"])
        for result in results:
            result.pop("batch_index", None)
        
        self.logger.info("Batch classification completed", total_texts=len(texts))
        return results
    
    async def classify_with_similarity(self, text: str, 
                                     reference_texts: List[str] = None,
                                     similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """Classify text with similarity analysis to reference texts."""
        result = await self.classify(text)
        
        if reference_texts:
            similarities = self.embedding_model.find_similar(text, reference_texts, top_k=3)
            result["similar_texts"] = similarities
            
            # Check if any similar text has high similarity
            if similarities and similarities[0]["similarity"] > similarity_threshold:
                result["high_similarity_warning"] = True
                result["most_similar"] = similarities[0]
        
        return result
    
    def _should_use_cache(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self._service_cache:
            return False
        
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return time.time() - timestamp < settings.CACHE_TTL
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache the classification result."""
        self._service_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
        
        # Clean old cache entries
        current_time = time.time()
        to_remove = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > settings.CACHE_TTL
        ]
        for key in to_remove:
            self._service_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def _update_metrics(self):
        """Update performance metrics."""
        if self._response_times:
            self.metrics.average_response_time = sum(self._response_times) / len(self._response_times)
        
        total_requests = self.metrics.total_requests or 1
        cache_hits = len([k for k in self._service_cache.keys()])
        self.metrics.cache_hit_rate = cache_hits / total_requests
    
    def get_available_labels(self) -> List[str]:
        """Get available classification labels."""
        return settings.CLASS_LABELS
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information."""
        info = {
            "mode": self.mode.value,
            "available_labels": self.get_available_labels(),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": self.metrics.successful_requests / max(1, self.metrics.total_requests),
                "average_response_time": self.metrics.average_response_time,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "uptime_seconds": time.time() - self._start_time
            },
            "cache_stats": {
                "service_cache_size": len(self._service_cache),
                "cache_ttl": settings.CACHE_TTL
            },
            "models_info": {}
        }
        
        # Add model-specific info
        if self.dspy_classifier:
            info["models_info"]["dspy"] = self.dspy_classifier.get_model_info()
        if self.langchain_classifier:
            info["models_info"]["langchain"] = self.langchain_classifier.get_model_info()
        if self.agent:
            info["models_info"]["agent"] = self.agent.get_agent_info()
        if self.embedding_model:
            info["models_info"]["embeddings"] = self.embedding_model.get_model_info()
        
        return info
    
    def clear_cache(self):
        """Clear all caches."""
        self._service_cache.clear()
        self._cache_timestamps.clear()
        
        if self.dspy_classifier:
            self.dspy_classifier._cache.clear()
            self.dspy_classifier._cache_timestamps.clear()
        
        if self.langchain_classifier:
            self.langchain_classifier.clear_cache()
        
        if self.embedding_model:
            self.embedding_model.clear_cache()
        
        self.logger.info("All caches cleared")
    
    def switch_mode(self, new_mode: ClassifierMode):
        """Switch classifier mode dynamically."""
        if new_mode == self.mode:
            return
        
        old_mode = self.mode
        self.mode = new_mode
        
        # Initialize new models if needed
        try:
            if new_mode in [ClassifierMode.DSPY_ONLY, ClassifierMode.ENSEMBLE] and not self.dspy_classifier:
                self.dspy_classifier = ClassifierModule()
                
            if new_mode in [ClassifierMode.LANGCHAIN_ONLY, ClassifierMode.ENSEMBLE] and not self.langchain_classifier:
                self.langchain_classifier = LangChainClassifier()
                
            if new_mode == ClassifierMode.AGENT and not self.agent and settings.AGENT_ENABLED:
                self.agent = ClassificationAgent()
                
            self.logger.info("Classifier mode switched", old_mode=old_mode.value, new_mode=new_mode.value)
            
        except Exception as e:
            self.mode = old_mode  # Revert on failure
            self.logger.error("Failed to switch mode", error=str(e))
            raise ClassificationError(f"Failed to switch to mode {new_mode.value}: {str(e)}")