from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from pydantic import BaseModel, Field, validator
import asyncio
import json
import time

from ..core.config import settings
from ..core.logging import logger, struct_logger
from ..core.exceptions import ClassificationError, ModelLoadError
from ..core.security import hash_text

class ClassificationResult(BaseModel):
    classification: str = Field(description="The predicted class label")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0.0, le=1.0)
    explanation: str = Field(description="Explanation for the classification")
    
    @validator('classification')
    def validate_classification(cls, v):
        return v.strip().lower()
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))

class LangChainClassifier:
    """Enhanced LangChain-based text classifier with caching and agent capabilities."""
    
    def __init__(self, labels: List[str] = None, model: str = None):
        self.labels = labels or settings.CLASS_LABELS
        self.model_name = model or settings.OPENAI_MODEL
        self.logger = struct_logger.bind(component="langchain_classifier")
        
        # Validation
        if not self.labels:
            raise ValueError("At least one classification label must be provided")
        
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                request_timeout=settings.OPENAI_TIMEOUT,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
            # Initialize output parser
            self.output_parser = PydanticOutputParser(pydantic_object=ClassificationResult)
            
            # Create enhanced prompt template
            template = """
            You are an expert text classifier with deep understanding of language patterns, context, and sentiment.
            
            Your task is to classify the following text into exactly ONE of these categories: {labels}
            
            Classification Guidelines:
            1. Consider the overall context, sentiment, and intent
            2. Pay attention to subtle nuances and implied meanings
            3. Provide a confidence score based on how certain you are
            4. Give a detailed explanation of your reasoning
            
            Available categories: {labels}
            
            Text to classify: {text}
            
            Important: Your classification MUST be exactly one of the provided categories.
            
            {format_instructions}
            """
            
            self.prompt = PromptTemplate(
                template=template,
                input_variables=["text", "labels"],
                partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
            )
            
            # Create the classification chain
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            
            # Initialize memory for agent mode
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=settings.AGENT_MEMORY_SIZE,
                return_messages=True
            ) if settings.AGENT_ENABLED else None
            
            # Cache for storing results
            self._cache = {}
            self._cache_timestamps = {}
            
            # Performance metrics
            self._total_requests = 0
            self._total_time = 0.0
            self._cache_hits = 0
            
            self.logger.info("LangChain classifier initialized", model=self.model_name, labels=self.labels)
            
        except Exception as e:
            self.logger.error("Failed to initialize LangChain classifier", error=str(e))
            raise ModelLoadError(f"Failed to initialize LangChain classifier: {str(e)}")
    
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
    
    def _validate_classification(self, classification: str) -> str:
        """Validate and normalize classification result."""
        classification_lower = classification.lower().strip()
        
        # Direct match
        for label in self.labels:
            if label.lower() == classification_lower:
                return label
        
        # Partial match
        for label in self.labels:
            if label.lower() in classification_lower or classification_lower in label.lower():
                return label
        
        # No match found, return first label as fallback
        self.logger.warning("Classification not in allowed labels", 
                          classification=classification, 
                          allowed_labels=self.labels)
        return self.labels[0]
    
    async def classify(self, text: str) -> Dict[str, Any]:
        """Classify text using LangChain with caching and error handling."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        start_time = time.time()
        self._total_requests += 1
        
        # Check cache first
        text_hash = hash_text(text)
        if self._should_use_cache(text_hash):
            cached_result = self._cache[text_hash].copy()
            cached_result["from_cache"] = True
            self._cache_hits += 1
            self.logger.info("Classification served from cache", text_hash=text_hash)
            return cached_result
        
        try:
            # Run the classification chain
            result = await self.chain.arun(
                text=text.strip(),
                labels=", ".join(self.labels)
            )
            
            # Parse the output
            try:
                parsed_result = self.output_parser.parse(result)
                
                # Validate classification
                validated_class = self._validate_classification(parsed_result.classification)
                
                # Adjust confidence if classification was corrected
                confidence = parsed_result.confidence
                if validated_class.lower() != parsed_result.classification.lower():
                    confidence = max(0.1, confidence * 0.7)  # Reduce confidence for corrected results
                
                final_result = {
                    "class": validated_class,
                    "confidence": confidence,
                    "explanation": parsed_result.explanation,
                    "model": self.model_name,
                    "processing_time": time.time() - start_time,
                    "from_cache": False,
                    "raw_response": result
                }
                
            except Exception as parse_error:
                self.logger.warning("Failed to parse structured output, attempting fallback", error=str(parse_error))
                
                # Fallback parsing for unstructured responses
                result_text = str(result).lower()
                
                # Try to extract classification from response
                best_match = self.labels[0]
                highest_score = 0
                
                for label in self.labels:
                    if label.lower() in result_text:
                        # Simple scoring based on frequency and position
                        score = result_text.count(label.lower())
                        if result_text.find(label.lower()) < 100:  # Bonus for early appearance
                            score += 1
                        
                        if score > highest_score:
                            highest_score = score
                            best_match = label
                
                final_result = {
                    "class": best_match,
                    "confidence": 0.5,  # Medium confidence for fallback
                    "explanation": f"Fallback classification. Original response: {result[:200]}...",
                    "model": self.model_name,
                    "processing_time": time.time() - start_time,
                    "from_cache": False,
                    "raw_response": result
                }
            
            # Cache the result
            self._cache_result(text_hash, final_result)
            
            # Update metrics
            self._total_time += final_result["processing_time"]
            
            self.logger.info(
                "Text classified successfully",
                classification=final_result["class"],
                confidence=final_result["confidence"],
                processing_time=final_result["processing_time"]
            )
            
            return final_result
            
        except Exception as e:
            self.logger.error("Classification failed", error=str(e), text_length=len(text))
            raise ClassificationError(f"LangChain classification failed: {str(e)}")
    
    def classify_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous wrapper for classification."""
        return asyncio.run(self.classify(text))
    
    async def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts concurrently."""
        if not texts:
            return []
        
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        
        async def classify_with_semaphore(text: str):
            async with semaphore:
                return await self.classify(text)
        
        tasks = [classify_with_semaphore(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error("Batch classification failed for item", index=i, error=str(result))
                processed_results.append({
                    "class": self.labels[0],
                    "confidence": 0.0,
                    "explanation": f"Error in batch processing: {str(result)}",
                    "error": True
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        avg_time = self._total_time / max(1, self._total_requests)
        cache_hit_rate = self._cache_hits / max(1, self._total_requests)
        
        return {
            "model_name": self.model_name,
            "labels": self.labels,
            "cache_size": len(self._cache),
            "total_requests": self._total_requests,
            "cache_hit_rate": cache_hit_rate,
            "average_response_time": avg_time,
            "agent_enabled": settings.AGENT_ENABLED,
            "memory_enabled": self.memory is not None
        }
    
    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Classification cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_hit_rate": self._cache_hits / max(1, self._total_requests),
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits
        }