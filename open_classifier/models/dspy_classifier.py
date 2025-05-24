import os
import dspy
from typing import List, Dict, Any, Optional
import numpy as np
import json
import asyncio
from functools import lru_cache
import time

from ..core.config import settings
from ..core.logging import logger, struct_logger
from ..core.exceptions import ClassificationError, ModelLoadError
from ..core.security import hash_text

class ClassifierModule(dspy.Module):
    """Enhanced DSPy-based text classifier with caching and robust error handling."""
    
    def __init__(self, labels: List[str] = None, model: str = None):
        super().__init__()
        
        self.labels = labels or settings.CLASS_LABELS
        self.model_name = model or settings.OPENAI_MODEL
        self.logger = struct_logger.bind(component="dspy_classifier")
        
        # Validation
        if not self.labels:
            raise ValueError("At least one classification label must be provided")
        
        try:
            # Set up DSPy with OpenAI LLM
            self.llm = dspy.OpenAI(
                model=self.model_name,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                timeout=settings.OPENAI_TIMEOUT
            )
            dspy.settings.configure(lm=self.llm, verbose=settings.DSPY_VERBOSE)
            
            # Define the signature for classification
            self.classify_signature = dspy.Signature(
                """
                text -> classification, confidence, explanation
                """
            )
            
            # Create the classifier predictor
            self.classifier = dspy.Predict(self.classify_signature)
            
            # Cache for storing recent classifications
            self._cache = {}
            self._cache_timestamps = {}
            
            self.logger.info("DSPy classifier initialized", model=self.model_name, labels=self.labels)
            
        except Exception as e:
            self.logger.error("Failed to initialize DSPy classifier", error=str(e))
            raise ModelLoadError(f"Failed to initialize DSPy classifier: {str(e)}")
    
    def _get_prompt_template(self) -> str:
        """Get the classification prompt template."""
        return f"""
        You are an expert text classifier. Analyze the following text and classify it into exactly one of these categories: {', '.join(self.labels)}.

        Important guidelines:
        1. Choose ONLY from the provided categories: {', '.join(self.labels)}
        2. Provide a confidence score between 0 and 100
        3. Give a clear, concise explanation for your choice
        4. Consider context, sentiment, intent, and content

        Text to classify: {{text}}

        Respond in this exact format:
        Classification: [your choice from the categories above]
        Confidence: [number from 0-100]
        Explanation: [brief explanation of your reasoning]
        """
    
    def _parse_response(self, prediction) -> Dict[str, Any]:
        """Parse the LLM response into structured format."""
        try:
            # Handle different response formats
            if hasattr(prediction, 'classification') and hasattr(prediction, 'confidence'):
                classification = prediction.classification.strip()
                confidence_raw = str(prediction.confidence).strip()
                explanation = getattr(prediction, 'explanation', '').strip()
            else:
                # Fallback parsing for string responses
                response_text = str(prediction)
                lines = response_text.split('\n')
                
                classification = "unknown"
                confidence_raw = "0"
                explanation = "Could not parse response"
                
                for line in lines:
                    line = line.strip()
                    if line.lower().startswith('classification:'):
                        classification = line.split(':', 1)[1].strip()
                    elif line.lower().startswith('confidence:'):
                        confidence_raw = line.split(':', 1)[1].strip()
                    elif line.lower().startswith('explanation:'):
                        explanation = line.split(':', 1)[1].strip()
            
            # Parse confidence score
            try:
                # Handle various confidence formats
                confidence_clean = ''.join(filter(str.isdigit, confidence_raw))
                confidence = float(confidence_clean) if confidence_clean else 0.0
                if confidence > 1.0:  # Assume percentage format
                    confidence = confidence / 100.0
            except:
                confidence = 0.0
            
            # Validate classification against allowed labels
            classification_lower = classification.lower()
            matched_label = None
            for label in self.labels:
                if label.lower() == classification_lower or label.lower() in classification_lower:
                    matched_label = label
                    break
            
            if not matched_label:
                # Fallback to most confident label
                matched_label = self.labels[0]
                explanation = f"Original classification '{classification}' not in allowed labels. Defaulted to '{matched_label}'. {explanation}"
                confidence = max(0.1, confidence * 0.5)  # Reduce confidence for fallback
            
            return {
                "class": matched_label,
                "confidence": min(1.0, max(0.0, confidence)),
                "explanation": explanation or "No explanation provided",
                "raw_response": str(prediction),
                "processing_time": time.time()
            }
            
        except Exception as e:
            self.logger.error("Error parsing classification response", error=str(e), prediction=str(prediction))
            return {
                "class": self.labels[0],  # Fallback to first label
                "confidence": 0.0,
                "explanation": f"Error parsing response: {str(e)}",
                "raw_response": str(prediction),
                "processing_time": time.time()
            }
    
    def _should_use_cache(self, text_hash: str) -> bool:
        """Check if cached result is still valid."""
        if text_hash not in self._cache:
            return False
        
        timestamp = self._cache_timestamps.get(text_hash, 0)
        return time.time() - timestamp < settings.CACHE_TTL
    
    def _cache_result(self, text_hash: str, result: Dict[str, Any]):
        """Cache the classification result."""
        self._cache[text_hash] = result
        self._cache_timestamps[text_hash] = time.time()
        
        # Clean old cache entries
        current_time = time.time()
        to_remove = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > settings.CACHE_TTL
        ]
        for key in to_remove:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    async def classify_async(self, text: str) -> Dict[str, Any]:
        """Asynchronously classify text."""
        return await asyncio.to_thread(self.forward, text)
    
    def forward(self, text: str) -> Dict[str, Any]:
        """Classify the input text into one of the predefined categories."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        start_time = time.time()
        
        # Check cache first
        text_hash = hash_text(text)
        if self._should_use_cache(text_hash):
            cached_result = self._cache[text_hash].copy()
            cached_result["from_cache"] = True
            self.logger.info("Classification served from cache", text_hash=text_hash)
            return cached_result
        
        try:
            # Prepare input with our template
            formatted_input = self._get_prompt_template().format(text=text.strip())
            
            # Get classification response
            prediction = self.classifier(text=formatted_input)
            
            # Parse and structure the result
            result = self._parse_response(prediction)
            result["model"] = self.model_name
            result["processing_time"] = time.time() - start_time
            result["from_cache"] = False
            
            # Cache the result
            self._cache_result(text_hash, result)
            
            self.logger.info(
                "Text classified successfully",
                classification=result["class"],
                confidence=result["confidence"],
                processing_time=result["processing_time"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Classification failed", error=str(e), text_length=len(text))
            raise ClassificationError(f"DSPy classification failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model_name": self.model_name,
            "labels": self.labels,
            "cache_size": len(self._cache),
            "temperature": settings.OPENAI_TEMPERATURE,
            "max_tokens": settings.OPENAI_MAX_TOKENS
        }